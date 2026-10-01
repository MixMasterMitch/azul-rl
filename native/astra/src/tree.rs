//! Neural Gumbel tree traversal. Rust owns all tree state; Python is called only
//! for batched policy/value inference.
use crate::game::{Game, ACTIONS, GLOBAL_DIM, SOURCE_DIM};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyDict};
use std::time::{Duration, Instant};

#[derive(Default)]
struct Edge {
    visits: u32,
    total: [f64; 4],
    outcomes: Vec<usize>,
    stochastic: bool,
}

struct Node {
    game: Game,
    prior: [f64; ACTIONS],
    raw: [f64; 4],
    legal: [bool; ACTIONS],
    player: usize,
    terminal: bool,
    // Most nodes have one or two visited actions. A bounded linear map avoids
    // hash-table control bytes and unchecked bucket addressing at this hot path.
    // Keep insertion order stable for deterministic floating-point reductions.
    edges: Vec<(u16, Edge)>,
}

impl Node {
    fn edge(&self, action: u16) -> &Edge {
        &self
            .edges
            .iter()
            .find(|(a, _)| *a == action)
            .expect("missing traversed edge")
            .1
    }

    fn edge_mut(&mut self, action: u16) -> &mut Edge {
        &mut self
            .edges
            .iter_mut()
            .find(|(a, _)| *a == action)
            .expect("missing traversed edge")
            .1
    }

    fn ensure_edge(&mut self, action: u16) -> &mut Edge {
        assert!((action as usize) < ACTIONS, "edge action out of bounds");
        let index = match self.edges.iter().position(|(a, _)| *a == action) {
            Some(index) => index,
            None => {
                assert!(self.edges.len() < ACTIONS, "too many distinct edges");
                self.edges.push((action, Edge::default()));
                self.edges.len() - 1
            }
        };
        &mut self.edges[index].1
    }
}

type Path = Vec<(usize, u16)>;
type PendingExpansion = (usize, u16, Path);

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
    fn gamma(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            return self.gamma(alpha + 1.0) * self.uniform().powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = (1.0 / (9.0 * d)).sqrt();
        loop {
            let x = self.normal();
            let v0 = 1.0 + c * x;
            if v0 <= 0.0 {
                continue;
            }
            let v = v0 * v0 * v0;
            let u = self.uniform();
            if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }
    fn gumbel(&mut self) -> f64 {
        -(-self.uniform().ln()).ln()
    }
    fn index(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
}

fn softmax(logits: &[f64; ACTIONS], legal: &[bool; ACTIONS]) -> [f64; ACTIONS] {
    let mut out = [0.0; ACTIONS];
    let max = (0..ACTIONS)
        .filter(|i| legal[*i])
        .map(|i| logits[i])
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return out;
    }
    let mut sum = 0.0;
    for i in 0..ACTIONS {
        if legal[i] {
            out[i] = (logits[i] - max).exp();
            sum += out[i];
        }
    }
    if sum > 0.0 {
        for x in &mut out {
            *x /= sum;
        }
    }
    out
}

fn completed_q(node: &Node) -> ([f64; ACTIONS], [u32; ACTIONS]) {
    let mut q = [0.0; ACTIONS];
    let mut visits = [0u32; ACTIONS];
    for (action, edge) in &node.edges {
        let action = *action;
        let i = action as usize;
        visits[i] = edge.visits;
        if edge.visits > 0 {
            q[i] = edge.total[node.player] / edge.visits as f64;
        }
    }
    let probs = softmax(&node.prior, &node.legal);
    let mut weight = 0.0;
    let mut total = 0.0;
    let mut count = 0u32;
    for i in 0..ACTIONS {
        if visits[i] > 0 {
            weight += probs[i];
            total += probs[i] * q[i];
            count += visits[i];
        }
    }
    let mixed =
        (node.raw[node.player] + count as f64 * total / weight.max(1e-30)) / (1.0 + count as f64);
    for i in 0..ACTIONS {
        if visits[i] == 0 {
            q[i] = mixed;
        }
    }
    (q, visits)
}

fn sigma(node: &Node, q_scale: f64) -> ([f64; ACTIONS], [u32; ACTIONS]) {
    let (mut q, visits) = completed_q(node);
    let lo = (0..ACTIONS)
        .filter(|i| node.legal[*i])
        .map(|i| q[i])
        .fold(f64::INFINITY, f64::min);
    let hi = (0..ACTIONS)
        .filter(|i| node.legal[*i])
        .map(|i| q[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let span = (hi - lo).max(1e-8);
    let vmax = visits.iter().copied().max().unwrap_or(0) as f64;
    for value in &mut q {
        *value = (*value - lo) / span * q_scale * (1.0 + vmax / 50.0);
    }
    (q, visits)
}

fn schedule(candidates: usize, budget: usize) -> Vec<u32> {
    if candidates == 1 {
        return (0..budget as u32).collect();
    }
    let rounds = (candidates as f64).log2().ceil() as usize;
    let mut out = Vec::with_capacity(budget);
    let mut level = 0u32;
    let mut remaining = candidates;
    while out.len() < budget {
        let repetitions = (budget / (rounds * remaining)).max(1);
        for _ in 0..repetitions {
            for _ in 0..remaining {
                if out.len() < budget {
                    out.push(level);
                }
            }
            level += 1;
        }
        remaining = (remaining / 2).max(2);
    }
    out
}

fn bytes_to_f32(value: &Bound<'_, PyAny>, expected: usize) -> PyResult<Vec<f32>> {
    let bytes: Vec<u8> = value.extract()?;
    if bytes.len() != expected * 4 {
        return Err(PyValueError::new_err("inference byte length mismatch"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect())
}

fn infer(py: Python<'_>, callback: &Py<PyAny>, games: &[Game]) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let mut global = Vec::with_capacity(games.len() * GLOBAL_DIM * 4);
    let mut source = Vec::with_capacity(games.len() * SOURCE_DIM * 4);
    let mut legal = vec![0u8; games.len() * ACTIONS];
    for (row, game) in games.iter().enumerate() {
        let mut g = [0.0; GLOBAL_DIM];
        let mut s = [0.0; SOURCE_DIM];
        game.encode(&mut g, &mut s);
        global.extend(g.into_iter().flat_map(f32::to_ne_bytes));
        source.extend(s.into_iter().flat_map(f32::to_ne_bytes));
        for action in game.public.legal() {
            legal[row * ACTIONS + action as usize] = 1;
        }
    }
    let result = callback.call1(
        py,
        (
            PyByteArray::new(py, &global),
            PyByteArray::new(py, &source),
            PyByteArray::new(py, &legal),
            games.len(),
            games[0].public.n,
        ),
    )?;
    let tuple: (Py<PyAny>, Py<PyAny>) = result.extract(py)?;
    Ok((
        bytes_to_f32(tuple.0.bind(py), games.len() * ACTIONS)?,
        bytes_to_f32(tuple.1.bind(py), games.len() * 4)?,
    ))
}

fn make_nodes(
    py: Python<'_>,
    callback: &Py<PyAny>,
    games: Vec<Game>,
    score_scaled: bool,
) -> PyResult<Vec<Node>> {
    let live: Vec<Game> = games.iter().filter(|g| !g.public.ended).cloned().collect();
    let (policy, values) = if live.is_empty() {
        (vec![], vec![])
    } else {
        infer(py, callback, &live)?
    };
    let mut live_row = 0;
    let mut nodes = Vec::with_capacity(games.len());
    for game in games {
        let terminal = game.public.ended;
        let mut prior = [-1e9; ACTIONS];
        let mut raw = [0.0; 4];
        if terminal {
            let final_values = game.final_values(score_scaled);
            for player in 0..4 {
                raw[player] = final_values[player] as f64;
            }
        } else {
            for i in 0..ACTIONS {
                prior[i] = policy[live_row * ACTIONS + i] as f64;
            }
            let cp = game.public.cp;
            for relative in 0..game.public.n {
                raw[(cp + relative) % game.public.n] = values[live_row * 4 + relative] as f64;
            }
            live_row += 1;
        }
        let mut legal = [false; ACTIONS];
        for action in game.public.legal() {
            legal[action as usize] = true;
        }
        nodes.push(Node {
            player: game.public.cp,
            game,
            prior,
            raw,
            legal,
            terminal,
            edges: Vec::new(),
        });
    }
    Ok(nodes)
}

#[pyfunction]
#[pyo3(signature = (snapshots, callback, num_simulations=64, temperature=0.25, q_scale=28.0,
    root_noise_scale=1.0, dirichlet_alpha=0.0, dirichlet_mix=0.0, reward_mode="binary",
    seed=0, max_root_candidates=16, max_depth=120, chance_samples=4, move_deadline_ms=None))]
#[allow(clippy::too_many_arguments)]
pub fn gumbel_tree(
    py: Python<'_>,
    snapshots: Vec<Vec<i64>>,
    callback: Py<PyAny>,
    num_simulations: usize,
    temperature: f64,
    q_scale: f64,
    root_noise_scale: f64,
    dirichlet_alpha: f64,
    dirichlet_mix: f64,
    reward_mode: &str,
    seed: u64,
    max_root_candidates: usize,
    max_depth: usize,
    chance_samples: usize,
    move_deadline_ms: Option<u64>,
) -> PyResult<Py<PyDict>> {
    if snapshots.is_empty()
        || num_simulations == 0
        || max_root_candidates == 0
        || max_depth == 0
        || chance_samples == 0
        || !["binary", "score_scaled"].contains(&reward_mode)
    {
        return Err(PyValueError::new_err("invalid native tree configuration"));
    }
    let games = snapshots
        .iter()
        .map(|s| Game::parse(s))
        .collect::<Result<Vec<_>, _>>()
        .map_err(PyValueError::new_err)?;
    let num_players = games[0].public.n;
    if games.iter().any(|g| g.public.n != num_players) {
        return Err(PyValueError::new_err("mixed player counts"));
    }
    let started = Instant::now();
    let deadline = move_deadline_ms.map(|ms| started + Duration::from_millis(ms));
    let mut rng = Rng(seed);
    let mut nodes = make_nodes(py, &callback, games, reward_mode == "score_scaled")?;
    let roots: Vec<usize> = (0..nodes.len()).collect();
    let mut candidates = Vec::with_capacity(roots.len());
    let mut noises = Vec::with_capacity(roots.len());
    let mut schedules = Vec::with_capacity(roots.len());
    for &root_id in &roots {
        let root = &mut nodes[root_id];
        for i in 0..ACTIONS {
            root.prior[i] /= temperature.max(1e-6);
        }
        if dirichlet_mix > 0.0 {
            let probs = softmax(&root.prior, &root.legal);
            let mut gamma = [0.0; ACTIONS];
            let mut total = 0.0;
            for (i, gamma_value) in gamma.iter_mut().enumerate() {
                if root.legal[i] {
                    *gamma_value = rng.gamma(dirichlet_alpha);
                    total += *gamma_value;
                }
            }
            for i in 0..ACTIONS {
                if root.legal[i] {
                    let mixed = (1.0 - dirichlet_mix) * probs[i]
                        + dirichlet_mix * gamma[i] / total.max(1e-30);
                    root.prior[i] = mixed.max(1e-30).ln();
                }
            }
        }
        let mut noise = [0.0; ACTIONS];
        for x in &mut noise {
            *x = rng.gumbel() * root_noise_scale;
        }
        let mut legal: Vec<usize> = (0..ACTIONS).filter(|i| root.legal[*i]).collect();
        legal.sort_unstable_by(|a, b| {
            (root.prior[*b] + noise[*b]).total_cmp(&(root.prior[*a] + noise[*a]))
        });
        let k = legal.len().min(max_root_candidates).min(num_simulations);
        legal.truncate(k);
        schedules.push(schedule(k.max(1), num_simulations));
        candidates.push(legal);
        noises.push(noise);
    }
    let mut max_depth_seen = 0usize;
    let mut chance_expansions = 0usize;
    let mut terminals = 0usize;
    for simulation in 0..num_simulations {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
        let mut pending: Vec<PendingExpansion> = Vec::new();
        for (root_row, &root_id) in roots.iter().enumerate() {
            if nodes[root_id].terminal || candidates[root_row].is_empty() {
                continue;
            }
            let (root_sigma, root_visits) = sigma(&nodes[root_id], q_scale);
            let wanted = schedules[root_row][simulation];
            let mut allowed: Vec<usize> = candidates[root_row]
                .iter()
                .copied()
                .filter(|a| root_visits[*a] == wanted)
                .collect();
            if allowed.is_empty() {
                let min = candidates[root_row]
                    .iter()
                    .map(|a| root_visits[*a])
                    .min()
                    .unwrap();
                allowed = candidates[root_row]
                    .iter()
                    .copied()
                    .filter(|a| root_visits[*a] == min)
                    .collect();
            }
            let mut action = *allowed
                .iter()
                .max_by(|a, b| {
                    (nodes[root_id].prior[**a] + noises[root_row][**a] + root_sigma[**a]).total_cmp(
                        &(nodes[root_id].prior[**b] + noises[root_row][**b] + root_sigma[**b]),
                    )
                })
                .unwrap() as u16;
            let mut node_id = root_id;
            let mut path = Vec::new();
            for depth in 0..max_depth {
                max_depth_seen = max_depth_seen.max(depth + 1);
                path.push((node_id, action));
                let expand = {
                    let edge = nodes[node_id].ensure_edge(action);
                    edge.outcomes.is_empty()
                        || (edge.stochastic && edge.outcomes.len() < chance_samples)
                };
                if expand {
                    pending.push((node_id, action, path));
                    break;
                }
                let outcomes = &nodes[node_id].edge(action).outcomes;
                node_id = outcomes[rng.index(outcomes.len())];
                if nodes[node_id].terminal || depth + 1 == max_depth {
                    let value = nodes[node_id].raw;
                    for (n, a) in path {
                        let e = nodes[n].edge_mut(a);
                        e.visits += 1;
                        for (total, backed_up) in e.total.iter_mut().zip(value) {
                            *total += backed_up;
                        }
                    }
                    break;
                }
                let (sig, counts) = sigma(&nodes[node_id], q_scale);
                let mut adjusted = nodes[node_id].prior;
                for i in 0..ACTIONS {
                    adjusted[i] += sig[i];
                }
                let improved = softmax(&adjusted, &nodes[node_id].legal);
                let denom = 1.0 + counts.iter().map(|x| *x as f64).sum::<f64>();
                action = (0..ACTIONS)
                    .filter(|i| nodes[node_id].legal[*i])
                    .max_by(|a, b| {
                        (improved[*a] - counts[*a] as f64 / denom)
                            .total_cmp(&(improved[*b] - counts[*b] as f64 / denom))
                    })
                    .unwrap() as u16;
            }
        }
        if pending.is_empty() {
            continue;
        }
        let mut child_games = Vec::with_capacity(pending.len());
        let mut stochastic = Vec::with_capacity(pending.len());
        for (parent, action, _) in &pending {
            let mut child = nodes[*parent].game.clone();
            child.rng = rng.next();
            child.step(*action, false, None);
            let round_end = !child.public.ended && child.public.empty();
            child.finalize(None);
            stochastic.push(round_end && !child.public.ended);
            child_games.push(child);
        }
        let children = make_nodes(py, &callback, child_games, reward_mode == "score_scaled")?;
        for (j, child) in children.into_iter().enumerate() {
            if stochastic[j] {
                chance_expansions += 1;
            }
            if child.terminal {
                terminals += 1;
            }
            let value = child.raw;
            let child_id = nodes.len();
            nodes.push(child);
            let (parent, action, path) = &pending[j];
            let edge = nodes[*parent].edge_mut(*action);
            edge.stochastic = stochastic[j];
            edge.outcomes.push(child_id);
            for &(n, a) in path {
                let e = nodes[n].edge_mut(a);
                e.visits += 1;
                for (total, backed_up) in e.total.iter_mut().zip(value) {
                    *total += backed_up;
                }
            }
        }
    }
    let mut actions = vec![0u16; roots.len()];
    let mut policies = Vec::with_capacity(roots.len() * ACTIONS * 4);
    let mut min_sims = usize::MAX;
    let mut max_sims = 0usize;
    for (row, &root_id) in roots.iter().enumerate() {
        let root = &nodes[root_id];
        if root.legal.iter().any(|x| *x) {
            let (sig, visits) = sigma(root, q_scale);
            let mut adjusted = root.prior;
            for i in 0..ACTIONS {
                adjusted[i] += sig[i];
            }
            let probs = softmax(&adjusted, &root.legal);
            policies.extend(
                probs
                    .into_iter()
                    .map(|x| x as f32)
                    .flat_map(f32::to_ne_bytes),
            );
            let vmax = candidates[row]
                .iter()
                .map(|a| visits[*a])
                .max()
                .unwrap_or(0);
            actions[row] = *candidates[row]
                .iter()
                .filter(|a| visits[**a] == vmax)
                .max_by(|a, b| {
                    (root.prior[**a] + noises[row][**a] + sig[**a])
                        .total_cmp(&(root.prior[**b] + noises[row][**b] + sig[**b]))
                })
                .unwrap() as u16;
            let count: usize = root.edges.iter().map(|(_, e)| e.visits as usize).sum();
            min_sims = min_sims.min(count);
            max_sims = max_sims.max(count);
        } else {
            policies.extend(std::iter::repeat_n(0u8, ACTIONS * 4));
        }
    }
    let out = PyDict::new(py);
    out.set_item("actions", actions)?;
    out.set_item("policies", PyByteArray::new(py, &policies))?;
    out.set_item(
        "simulations_min",
        if min_sims == usize::MAX { 0 } else { min_sims },
    )?;
    out.set_item("simulations_max", max_sims)?;
    out.set_item("max_depth", max_depth_seen)?;
    out.set_item("nodes", nodes.len())?;
    out.set_item("terminal_children", terminals)?;
    out.set_item("refill_children", chance_expansions)?;
    out.set_item("elapsed_s", started.elapsed().as_secs_f64())?;
    Ok(out.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_node() -> Node {
        Node {
            game: Game::new(2, 7),
            prior: [0.; ACTIONS],
            raw: [0.; 4],
            legal: [true; ACTIONS],
            player: 0,
            terminal: false,
            edges: Vec::new(),
        }
    }

    #[test]
    fn sparse_edges_preserve_backups_across_growth_and_repeated_access() {
        for seed in 0..64 {
            let mut rng = Rng(seed);
            let mut actions: Vec<u16> = (0..ACTIONS as u16).collect();
            for i in (1..actions.len()).rev() {
                actions.swap(i, rng.index(i + 1));
            }
            let mut node = empty_node();
            for &action in &actions {
                let edge = node.ensure_edge(action);
                edge.visits = action as u32 + 1;
                edge.total = [action as f64; 4];
                edge.outcomes.push(action as usize);
            }
            assert_eq!(node.edges.len(), ACTIONS);
            for action in 0..ACTIONS as u16 {
                let edge = node.ensure_edge(action);
                assert_eq!(edge.visits, action as u32 + 1);
                assert_eq!(edge.total, [action as f64; 4]);
                assert_eq!(edge.outcomes, [action as usize]);
                node.edge_mut(action).visits += 1;
                assert_eq!(node.edge(action).visits, action as u32 + 2);
            }
            assert_eq!(node.edges.len(), ACTIONS);
        }
    }

    #[test]
    #[should_panic(expected = "edge action out of bounds")]
    fn sparse_edges_reject_invalid_action_before_insertion() {
        empty_node().ensure_edge(ACTIONS as u16);
    }

    #[test]
    fn sequential_halving_schedule_matches_reference() {
        assert_eq!(
            schedule(4, 16),
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        );
        assert_eq!(schedule(1, 4), [0, 1, 2, 3]);
    }

    #[test]
    fn native_random_distributions_are_finite() {
        let mut rng = Rng(7);
        for alpha in [0.1, 0.3, 1.0, 3.0] {
            for _ in 0..100 {
                let value = rng.gamma(alpha);
                assert!(value.is_finite() && value > 0.0);
                assert!(rng.gumbel().is_finite());
            }
        }
    }
}

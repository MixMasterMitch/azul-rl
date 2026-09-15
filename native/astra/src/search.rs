use crate::eval::{components, values};
use crate::state::{Board, State};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub struct Config {
    pub nodes: u64,
    pub center_nodes: u64,
    pub time_ms: u64,
    pub depth: u8,
    pub width: usize,
    pub weights: [f64; 11],
    pub seed: u64,
    pub rollout: bool,
}

#[derive(Clone)]
struct Entry {
    value: [f64; 4],
    pv: Vec<u16>,
    solved: bool,
    round_complete: bool,
}

// Evaluation uses total remaining tiles, not their distribution over sources or
// the acting seat. Keep this key in sync with evaluator information dependencies.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct EvaluationKey {
    n: usize,
    first: usize,
    ended: bool,
    resolved: bool,
    available: [u8; 5],
    boards: [Board; 4],
}

impl From<&State> for EvaluationKey {
    fn from(s: &State) -> Self {
        Self {
            n: s.n,
            first: s.first,
            ended: s.ended,
            resolved: s.resolved,
            available: std::array::from_fn(|c| {
                s.center[c] + s.factories.iter().map(|f| f[c]).sum::<u8>()
            }),
            boards: s.boards,
        }
    }
}
pub struct ResultData {
    pub action: u16,
    pub nodes: u64,
    pub depth: u8,
    pub elapsed: f64,
    pub reason: String,
    pub pv: Vec<u16>,
    pub value: [f64; 4],
    pub components: [[f64; 6]; 4],
    pub solved: bool,
    pub evaluations: u64,
    pub tt_hits: u64,
    pub evaluation_cache_hits: u64,
}

struct Search {
    cfg: Config,
    started: Instant,
    nodes: u64,
    tt: HashMap<(State, u8), Entry>,
    cutoff: &'static str,
    evaluations: Cell<u64>,
    tt_hits: u64,
    evaluation_cache: RefCell<HashMap<EvaluationKey, [f64; 4]>>,
    evaluation_cache_hits: Cell<u64>,
    cache_enabled: bool,
}

pub(crate) fn tie_key(a: u16, seed: u64) -> u64 {
    let mut x = (a as u64)
        .wrapping_add(seed)
        .wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

impl Search {
    fn static_value(&self, s: &State) -> [f64; 4] {
        if !self.cache_enabled {
            self.evaluations.set(self.evaluations.get() + 1);
            return values(s, &self.cfg.weights);
        }
        let key = EvaluationKey::from(s);
        if let Some(value) = self.evaluation_cache.borrow().get(&key).copied() {
            self.evaluation_cache_hits
                .set(self.evaluation_cache_hits.get() + 1);
            return value;
        }
        self.evaluations.set(self.evaluations.get() + 1);
        let value = values(s, &self.cfg.weights);
        let mut cache = self.evaluation_cache.borrow_mut();
        if cache.len() < 200_000 {
            cache.insert(key, value);
        }
        value
    }

    fn leaf_value(&mut self, s: &State) -> Option<[f64; 4]> {
        if !self.cfg.rollout || s.resolved || s.ended {
            return Some(self.static_value(s));
        }
        let mut position = *s;
        while !position.resolved && !position.ended {
            if !self.check() {
                return None;
            }
            self.nodes += 1;
            let (choices, _) = self.ordered(&position, true);
            position = choices.first()?.1;
        }
        Some(self.static_value(&position))
    }
    fn check(&mut self) -> bool {
        if self.nodes >= self.cfg.nodes {
            self.cutoff = "nodes";
            return false;
        }
        if self.started.elapsed() >= Duration::from_millis(self.cfg.time_ms) {
            self.cutoff = "time";
            return false;
        }
        true
    }

    fn ordered(&self, s: &State, root: bool) -> (Vec<(u16, State, [f64; 4])>, bool) {
        let mut children: Vec<_> = s
            .legal()
            .into_iter()
            .map(|a| {
                let child = s.apply(a, true);
                let v = self.static_value(&child);
                (a, child, v)
            })
            .collect();
        children.sort_by(|a, b| {
            b.2[s.cp]
                .total_cmp(&a.2[s.cp])
                .then_with(|| tie_key(a.0, self.cfg.seed).cmp(&tie_key(b.0, self.cfg.seed)))
        });
        let full = children.len();
        if !root && !s.factories_empty() && self.cfg.width > 0 {
            let mut rank = 0;
            children.retain(|(a, child, _)| {
                rank += 1;
                let r = *a as usize % 6;
                rank <= self.cfg.width
                    || r == 5
                    || child.ended
                    || (r < 5
                        && (child.boards[s.cp].count[r] == (r + 1) as u8
                            // Round resolution clears completed pattern lines.
                            // A new wall cell still proves this action completed one.
                            || ((child.boards[s.cp].wall ^ s.boards[s.cp].wall)
                                & (31 << (r * 5))) != 0))
            });
        }
        let unpruned = children.len() == full;
        (children, unpruned)
    }

    fn visit(
        &mut self,
        s: &State,
        depth: u8,
        root: bool,
        mut alpha: f64,
        mut beta: f64,
        pre_evaluated: Option<[f64; 4]>,
    ) -> Option<Entry> {
        let original_alpha = alpha;
        let original_beta = beta;
        if !self.check() {
            return None;
        }
        self.nodes += 1;
        if s.ended || s.resolved || depth == 0 {
            let value = if !self.cfg.rollout || s.ended || s.resolved {
                match pre_evaluated {
                    Some(value) => value,
                    None => self.leaf_value(s)?,
                }
            } else {
                self.leaf_value(s)?
            };
            return Some(Entry {
                value,
                pv: vec![],
                solved: s.ended || s.resolved,
                round_complete: s.ended || s.resolved,
            });
        }
        if !root {
            if let Some(v) = self.tt.get(&(*s, depth)) {
                self.tt_hits += 1;
                return Some(v.clone());
            }
        }
        let (children, mut solved) = self.ordered(s, root);
        let mut round_complete = true;
        let mut best: Option<Entry> = None;
        let mut exact = true;
        for (a, child, evaluated) in children {
            let mut result = self.visit(&child, depth - 1, false, alpha, beta, Some(evaluated))?;
            solved &= result.solved;
            round_complete &= result.round_complete;
            // Two-player alpha-beta consistently maximizes/minimizes seat-0
            // utility; multiplayer Max-N uses the acting seat's own utility.
            let better = best.as_ref().is_none_or(|b| {
                if s.n == 2 {
                    if s.cp == 0 {
                        result.value[0] > b.value[0]
                    } else {
                        result.value[0] < b.value[0]
                    }
                } else {
                    result.value[s.cp] > b.value[s.cp]
                }
            });
            if better {
                result.pv.insert(0, a);
                best = Some(result);
            }
            if s.n == 2 {
                let v = best.as_ref().unwrap().value[0];
                if s.cp == 0 {
                    alpha = alpha.max(v);
                } else {
                    beta = beta.min(v);
                }
                if beta <= alpha && !root {
                    exact = false;
                    break;
                }
            }
        }
        let mut best = best?;
        best.solved = solved;
        best.round_complete = round_complete;
        // Fail-low/high results are bounds even if this node did not cut off:
        // a descendant may have returned a bound under the incoming window.
        let inside_window =
            s.n != 2 || (best.value[0] > original_alpha && best.value[0] < original_beta);
        if exact && inside_window && !root {
            self.tt.insert((*s, depth), best.clone());
        }
        // Alpha-beta bounds derived entirely from resolved leaves are proofs:
        // an excluded branch cannot improve this root decision. Heuristic
        // width pruning and unresolved depth leaves still prevent "solved".
        Some(best)
    }
}

pub fn run(s: &State, cfg: Config) -> ResultData {
    let mut cfg = cfg;
    if s.factories_empty() && cfg.center_nodes > 0 {
        cfg.nodes = cfg.center_nodes;
    }
    let mut search = Search {
        cfg,
        started: Instant::now(),
        nodes: 0,
        tt: HashMap::new(),
        cutoff: "depth",
        evaluations: Cell::new(0),
        tt_hits: 0,
        evaluation_cache: RefCell::new(HashMap::new()),
        evaluation_cache_hits: Cell::new(0),
        cache_enabled: s.n > 2 && !s.factories_empty(),
    };
    let (children, _) = search.ordered(s, true);
    let first = &children[0];
    let mut result = ResultData {
        action: first.0,
        nodes: 0,
        depth: 0,
        elapsed: 0.0,
        reason: "depth".into(),
        pv: vec![first.0],
        value: first.2,
        components: components(&first.1, &cfg.weights),
        solved: false,
        evaluations: 0,
        tt_hits: 0,
        evaluation_cache_hits: 0,
    };
    // Root children are all evaluated even when a tiny node budget prevents a
    // full search iteration. Node counts count visits, not ordering evaluations.
    for depth in 1..=cfg.depth {
        let Some(entry) = search.visit(s, depth, true, f64::NEG_INFINITY, f64::INFINITY, None)
        else {
            break;
        };
        result.action = entry.pv[0];
        result.pv = entry.pv;
        result.value = entry.value;
        result.depth = depth;
        result.solved = entry.solved;
        // Further depth cannot change this retained tree once every needed
        // bound comes from a round-resolved leaf. Width-pruned replies still
        // prevent an exact solved claim.
        if entry.round_complete {
            search.cutoff = if result.solved {
                "solved"
            } else {
                "pruned_round"
            };
            break;
        }
    }
    result.nodes = search.nodes;
    result.elapsed = search.started.elapsed().as_secs_f64();
    result.reason = search.cutoff.into();
    result.evaluations = search.evaluations.get();
    result.tt_hits = search.tt_hits;
    result.evaluation_cache_hits = search.evaluation_cache_hits.get();
    result.components = components(&s.apply(result.action, true), &cfg.weights);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Board;

    #[test]
    fn cached_evaluation_is_invariant_to_source_distribution_and_actor() {
        for n in 2..=4 {
            for seed in 0..20 {
                let mut s = State {
                    n,
                    cp: seed % n,
                    first: 0,
                    center_first: true,
                    ended: false,
                    resolved: false,
                    factories: [[0; 5]; 9],
                    center: [1; 5],
                    boards: [Board {
                        color: [-1; 5],
                        ..Board::default()
                    }; 4],
                };
                for p in 0..n {
                    s.boards[p].score = (20 + seed + p * 5) as i32;
                    s.boards[p].floor = (p % 3) as u8;
                    for r in 0..5 {
                        s.boards[p].wall |= 1 << (r * 5 + (r + p) % 5);
                    }
                    s.boards[p].count[2] = 1;
                    s.boards[p].color[2] = ((p + 1) % 5) as i8;
                }
                for f in 0..s.nf() {
                    for tile in 0..4 {
                        s.factories[f][(f + tile + seed) % 5] += 1;
                    }
                }
                let mut equivalent = s;
                // Transfer every tile in this factory to the center. Sources
                // differ tactically, but the static evaluator sees identical totals.
                equivalent.cp = (s.cp + 1) % n;
                for color in 0..5 {
                    equivalent.center[color] += equivalent.factories[0][color];
                    equivalent.factories[0][color] = 0;
                }
                assert!(EvaluationKey::from(&s) == EvaluationKey::from(&equivalent));
                for pressure in [0.0, 1.0] {
                    let mut weights = crate::eval::DEFAULT_WEIGHTS;
                    weights[10] = pressure;
                    assert_eq!(values(&s, &weights), values(&equivalent, &weights));
                }
            }
        }
    }

    #[test]
    fn reply_pruning_keeps_completions_cleared_by_round_resolution() {
        let mut s = State {
            n: 2,
            cp: 0,
            first: 0,
            center_first: true,
            ended: false,
            resolved: false,
            factories: [[0; 5]; 9],
            center: [0; 5],
            boards: [Board {
                color: [-1; 5],
                ..Board::default()
            }; 4],
        };
        s.factories[0][0] = 4;
        let search = Search {
            cfg: Config {
                nodes: 4000,
                center_nodes: 0,
                time_ms: 2000,
                depth: 32,
                width: 1,
                weights: crate::eval::DEFAULT_WEIGHTS,
                seed: 0,
                rollout: false,
            },
            started: Instant::now(),
            nodes: 0,
            tt: HashMap::new(),
            cutoff: "depth",
            evaluations: Cell::new(0),
            tt_hits: 0,
            evaluation_cache: RefCell::new(HashMap::new()),
            evaluation_cache_hits: Cell::new(0),
            cache_enabled: false,
        };
        let (children, _) = search.ordered(&s, false);
        for action in 0..4 {
            let child = children.iter().find(|entry| entry.0 == action).unwrap();
            assert!(child.1.resolved);
            assert_eq!(child.1.boards[0].count[action as usize], 0);
        }
        assert!(children.iter().any(|entry| entry.0 == 5));
    }
}

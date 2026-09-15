//! Persistent CPU batches; Python crosses the boundary once per batch operation.
use crate::game::{Game, ACTIONS, GLOBAL_DIM, SOURCE_DIM};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use std::collections::HashMap;

type Buffers = (Py<PyByteArray>, Py<PyByteArray>, Py<PyByteArray>);

#[pyclass(module = "azul_astra")]
#[derive(Clone)]
pub struct BatchEngine {
    games: Vec<Game>,
    #[pyo3(get)]
    num_players: usize,
}

fn invalid(message: &str) -> PyErr {
    PyValueError::new_err(message.to_owned())
}

impl BatchEngine {
    fn validate_draws(&self, draws: &Option<Vec<Vec<f32>>>) -> PyResult<()> {
        if let Some(rows) = draws {
            let width = (2 * self.num_players + 1) * 4;
            if rows.len() != self.games.len()
                || rows.iter().any(|row| {
                    row.len() != width
                        || row
                            .iter()
                            .any(|x| !x.is_finite() || !(0.0..1.0).contains(x))
                })
            {
                return Err(invalid("draw_uniforms must have shape (batch_size, num_factories*4) and finite float32 values in [0, 1)"));
            }
        }
        Ok(())
    }

    fn validate_actions(&self, actions: &[u16]) -> PyResult<()> {
        if actions.len() != self.games.len() {
            return Err(invalid("One action required per game"));
        }
        for (game, action) in self.games.iter().zip(actions) {
            if !game.public.ended && !game.public.legal().contains(action) {
                return Err(invalid("Illegal action; batch was not modified"));
            }
        }
        Ok(())
    }
}

#[pymethods]
impl BatchEngine {
    #[new]
    #[pyo3(signature = (batch_size, num_players=2, seed=0, game_seeds=None))]
    fn new(
        py: Python<'_>,
        batch_size: usize,
        num_players: usize,
        seed: u64,
        game_seeds: Option<Vec<u64>>,
    ) -> PyResult<Self> {
        if !(2..=4).contains(&num_players) {
            return Err(invalid("Expected 2–4 players"));
        }
        let seeds = game_seeds.unwrap_or_else(|| {
            (0..batch_size)
                .map(|i| seed.wrapping_add(i as u64))
                .collect()
        });
        if seeds.len() != batch_size {
            return Err(invalid("One seed required per game"));
        }
        Ok(py.detach(|| Self {
            games: seeds
                .into_iter()
                .map(|s| Game::new(num_players, s))
                .collect(),
            num_players,
        }))
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.games.len()
    }

    #[getter]
    fn current_player(&self) -> Vec<i8> {
        self.games.iter().map(|g| g.public.cp as i8).collect()
    }

    #[getter]
    fn ended(&self) -> Vec<bool> {
        self.games.iter().map(|g| g.public.ended).collect()
    }

    #[getter]
    fn scores(&self) -> Vec<Vec<i32>> {
        self.games
            .iter()
            .map(|g| g.public.boards.iter().map(|b| b.score).collect())
            .collect()
    }

    fn get_winners(&self) -> Vec<i8> {
        self.games.iter().map(Game::winner).collect()
    }

    #[pyo3(signature = (reward_mode="binary"))]
    fn final_values_buffer(&self, py: Python<'_>, reward_mode: &str) -> PyResult<Py<PyByteArray>> {
        if !["binary", "score_scaled"].contains(&reward_mode) {
            return Err(invalid("Expected reward_mode binary or score_scaled"));
        }
        let bytes: Vec<u8> = py.detach(|| {
            self.games
                .iter()
                .flat_map(|g| g.final_values(reward_mode == "score_scaled"))
                .flat_map(f32::to_ne_bytes)
                .collect()
        });
        Ok(PyByteArray::new(py, &bytes).unbind())
    }

    fn total_tile_count(&self) -> Vec<usize> {
        self.games
            .iter()
            .map(|g| g.inventory().iter().sum())
            .collect()
    }

    /// Snapshot version 2 includes private simulation data and exact RNG state.
    fn snapshots(&self, py: Python<'_>) -> Vec<Vec<i64>> {
        py.detach(|| self.games.iter().map(Game::pack).collect())
    }

    fn public_snapshot(&self, index: usize) -> PyResult<Vec<i32>> {
        self.games
            .get(index)
            .map(|g| g.public.pack())
            .ok_or_else(|| invalid("Game index out of range"))
    }

    fn round_done(&self) -> Vec<bool> {
        self.games
            .iter()
            .map(|g| !g.public.ended && g.public.empty())
            .collect()
    }

    /// Number of draws required by each pending refill, for legacy RNG replay.
    fn refill_counts(&self, py: Python<'_>) -> Vec<usize> {
        py.detach(|| {
            self.games
                .iter()
                .map(|g| {
                    if g.public.ended || !g.public.empty() {
                        return 0;
                    }
                    let mut copy = g.clone();
                    copy.finalize(None);
                    if copy.public.ended {
                        0
                    } else {
                        copy.public
                            .factories
                            .iter()
                            .flatten()
                            .map(|n| *n as usize)
                            .sum()
                    }
                })
                .collect()
        })
    }

    fn append(&mut self, py: Python<'_>, other: &Self) -> PyResult<()> {
        if self.num_players != other.num_players {
            return Err(invalid("Player counts differ"));
        }
        py.detach(|| self.games.extend_from_slice(&other.games));
        Ok(())
    }

    /// Tensor-compatible state snapshots for policies/UI, without Python integer lists.
    fn state_buffers(&self, py: Python<'_>) -> HashMap<String, Py<PyByteArray>> {
        let buffers = py.detach(|| {
            let names = [
                "factory_tiles",
                "center_tiles",
                "center_first",
                "pattern_count",
                "pattern_color",
                "wall",
                "floor_count",
                "floor_tiles",
                "floor_slots",
                "floor_first",
                "scores",
                "bag",
                "box_lid",
                "current_player",
                "first_player",
                "active_mask",
                "ended",
            ];
            let widths = [45, 5, 1, 20, 20, 100, 4, 20, 28, 4, 8, 5, 5, 1, 1, 4, 1];
            let mut data: Vec<Vec<u8>> = widths
                .iter()
                .map(|w| Vec::with_capacity(w * self.games.len()))
                .collect();
            for g in &self.games {
                let s = &g.public;
                data[0].extend(s.factories.iter().flatten());
                data[1].extend(s.center);
                data[2].push(u8::from(s.center_first));
                for p in 0..4 {
                    let b = &s.boards[p];
                    data[3].extend(b.count);
                    data[4].extend(b.color.map(|c| c as u8));
                    data[5].extend((0..25).map(|k| u8::from(b.wall & (1 << k) != 0)));
                    data[6].push(b.floor);
                    data[7].extend(g.floor_tiles[p]);
                    data[8].extend(g.floor_slots[p].map(|c| c as u8));
                    data[9].push(u8::from(b.marker));
                    data[10].extend((b.score as i16).to_ne_bytes());
                    data[15].push(u8::from(p < s.n));
                }
                data[11].extend(g.bag);
                data[12].extend(g.lid);
                data[13].push(s.cp as u8);
                data[14].push(s.first as u8);
                data[16].push(u8::from(s.ended));
            }
            names.into_iter().zip(data).collect::<Vec<_>>()
        });
        buffers
            .into_iter()
            .map(|(name, bytes)| (name.to_owned(), PyByteArray::new(py, &bytes).unbind()))
            .collect()
    }

    #[staticmethod]
    fn from_snapshots(
        py: Python<'_>,
        num_players: usize,
        snapshots: Vec<Vec<i64>>,
    ) -> PyResult<Self> {
        if !(2..=4).contains(&num_players) {
            return Err(invalid("Expected 2–4 players"));
        }
        let games = py
            .detach(|| {
                snapshots
                    .iter()
                    .map(|s| Game::parse(s))
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(PyValueError::new_err)?;
        if games.iter().any(|g| g.public.n != num_players) {
            return Err(invalid("All games must have the same player count"));
        }
        Ok(Self { games, num_players })
    }

    #[pyo3(name = "clone")]
    fn copy(&self, py: Python<'_>) -> Self {
        py.detach(|| Clone::clone(self))
    }

    fn index_select(&self, py: Python<'_>, indices: Vec<usize>) -> PyResult<Self> {
        if indices.iter().any(|i| *i >= self.games.len()) {
            return Err(invalid("Game index out of range"));
        }
        Ok(py.detach(|| Self {
            games: indices.iter().map(|i| self.games[*i].clone()).collect(),
            num_players: self.num_players,
        }))
    }

    fn repeat_interleave(&self, py: Python<'_>, repeats: usize) -> PyResult<Self> {
        if repeats == 0 || self.games.len().checked_mul(repeats).is_none() {
            return Err(invalid("Invalid repeat count"));
        }
        Ok(py.detach(|| Self {
            games: self
                .games
                .iter()
                .flat_map(|g| std::iter::repeat_n(g.clone(), repeats))
                .collect(),
            num_players: self.num_players,
        }))
    }

    /// Replace future random streams without changing any visible state.
    fn reseed(&mut self, seeds: Vec<u64>) -> PyResult<()> {
        if seeds.len() != self.games.len() {
            return Err(invalid("One seed required per game"));
        }
        for (game, seed) in self.games.iter_mut().zip(seeds) {
            game.rng = seed;
        }
        Ok(())
    }

    #[pyo3(signature = (actions, finalize_round=true, draw_uniforms=None))]
    fn step(
        &mut self,
        py: Python<'_>,
        actions: Vec<u16>,
        finalize_round: bool,
        draw_uniforms: Option<Vec<Vec<f32>>>,
    ) -> PyResult<()> {
        self.validate_draws(&draw_uniforms)?;
        self.validate_actions(&actions)?;
        py.detach(|| {
            for (i, (game, action)) in self.games.iter_mut().zip(actions).enumerate() {
                game.step(
                    action,
                    finalize_round,
                    draw_uniforms.as_ref().map(|d| d[i].as_slice()),
                );
            }
        });
        Ok(())
    }

    #[pyo3(signature = (draw_uniforms=None))]
    fn finalize_round(
        &mut self,
        py: Python<'_>,
        draw_uniforms: Option<Vec<Vec<f32>>>,
    ) -> PyResult<()> {
        self.validate_draws(&draw_uniforms)?;
        py.detach(|| {
            for (i, game) in self.games.iter_mut().enumerate() {
                game.finalize(draw_uniforms.as_ref().map(|d| d[i].as_slice()));
            }
        });
        Ok(())
    }

    /// Fused B×K child copy/step. Supply seeds to decouple search from live draws.
    #[pyo3(signature = (actions, game_seeds=None))]
    fn expand(
        &self,
        py: Python<'_>,
        actions: Vec<Vec<u16>>,
        game_seeds: Option<Vec<u64>>,
    ) -> PyResult<Self> {
        if actions.len() != self.games.len() {
            return Err(invalid("One candidate row required per game"));
        }
        let k = actions.first().map_or(0, Vec::len);
        if actions.iter().any(|row| row.len() != k) {
            return Err(invalid("Candidate rows must have equal width"));
        }
        for (game, row) in self.games.iter().zip(&actions) {
            let legal = game.public.legal();
            if !game.public.ended && row.iter().any(|a| !legal.contains(a)) {
                return Err(invalid("Illegal candidate action"));
            }
        }
        let count = self
            .games
            .len()
            .checked_mul(k)
            .ok_or_else(|| invalid("Too many children"))?;
        if game_seeds.as_ref().is_some_and(|s| s.len() != count) {
            return Err(invalid("One seed required per child"));
        }
        Ok(py.detach(|| {
            let mut children = Vec::with_capacity(count);
            for (i, (game, row)) in self.games.iter().zip(actions).enumerate() {
                for (j, action) in row.into_iter().enumerate() {
                    let mut child = game.clone();
                    if let Some(seeds) = &game_seeds {
                        child.rng = seeds[i * k + j];
                    }
                    child.step(action, true, None);
                    children.push(child);
                }
            }
            Self {
                games: children,
                num_players: self.num_players,
            }
        }))
    }

    fn legal_mask_buffer(&self, py: Python<'_>) -> Py<PyByteArray> {
        let mask = py.detach(|| {
            let mut mask = vec![0u8; self.games.len() * ACTIONS];
            for (i, game) in self.games.iter().enumerate() {
                for a in game.public.legal() {
                    mask[i * ACTIONS + a as usize] = 1;
                }
            }
            mask
        });
        PyByteArray::new(py, &mask).unbind()
    }

    /// Owned, native-endian float32 bytearrays and a u8 legal mask; no Python float lists.
    fn encode_buffers(&self, py: Python<'_>) -> Buffers {
        let (global, source, mask) = py.detach(|| {
            let mut global = Vec::with_capacity(self.games.len() * GLOBAL_DIM * 4);
            let mut source = Vec::with_capacity(self.games.len() * SOURCE_DIM * 4);
            let mut mask = vec![0u8; self.games.len() * ACTIONS];
            for (i, game) in self.games.iter().enumerate() {
                let mut g = [0.0; GLOBAL_DIM];
                let mut s = [0.0; SOURCE_DIM];
                game.encode(&mut g, &mut s);
                global.extend(g.into_iter().flat_map(f32::to_ne_bytes));
                source.extend(s.into_iter().flat_map(f32::to_ne_bytes));
                for a in game.public.legal() {
                    mask[i * ACTIONS + a as usize] = 1;
                }
            }
            (global, source, mask)
        });
        (
            PyByteArray::new(py, &global).unbind(),
            PyByteArray::new(py, &source).unbind(),
            PyByteArray::new(py, &mask).unbind(),
        )
    }
}

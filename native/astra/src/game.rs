//! Complete simulator state. Kept separate from the public-only tactical search.
use crate::state::{cell, Board, State};

pub const GLOBAL_DIM: usize = 275;
pub const SOURCE_DIM: usize = 50;
pub const ACTIONS: usize = 300;
pub const FULL_VERSION: i64 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Game {
    pub public: State,
    pub bag: [u8; 5],
    pub lid: [u8; 5],
    pub floor_tiles: [[u8; 5]; 4],
    pub floor_slots: [[i8; 7]; 4],
    pub rng: u64,
}

impl Game {
    pub fn new(n: usize, seed: u64) -> Self {
        let board = Board {
            color: [-1; 5],
            ..Board::default()
        };
        let mut game = Self {
            public: State {
                n,
                cp: 0,
                first: 0,
                center_first: true,
                ended: false,
                resolved: false,
                factories: [[0; 5]; 9],
                center: [0; 5],
                boards: [board; 4],
            },
            bag: [20; 5],
            lid: [0; 5],
            floor_tiles: [[0; 5]; 4],
            floor_slots: [[-1; 7]; 4],
            rng: seed,
        };
        game.fill(None);
        game
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64: independent, serializable per-game streams. Not PyTorch's RNG.
        self.rng = self.rng.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn draw_rank(&mut self, total: u64) -> usize {
        // Rejection avoids modulo bias for bags whose size does not divide 2^64.
        let threshold = total.wrapping_neg() % total;
        loop {
            let sample = self.next_u64();
            if sample >= threshold {
                return (sample % total) as usize;
            }
        }
    }

    fn fill(&mut self, draws: Option<&[f32]>) {
        for slot in 0..self.public.nf() * 4 {
            let mut total: usize = self.bag.iter().map(|x| *x as usize).sum();
            if total == 0 {
                self.bag = self.lid;
                self.lid = [0; 5];
                total = self.bag.iter().map(|x| *x as usize).sum();
            }
            if total == 0 {
                return;
            }
            let rank = match draws {
                Some(values) => (values[slot] * total as f32).floor() as usize,
                None => self.draw_rank(total as u64),
            };
            let mut cumulative = 0;
            for color in 0..5 {
                cumulative += self.bag[color] as usize;
                if rank < cumulative {
                    self.bag[color] -= 1;
                    self.public.factories[slot / 4][color] += 1;
                    break;
                }
            }
        }
    }

    /// The caller validates legal actions for the whole batch before mutation.
    pub fn step(&mut self, action: u16, finalize: bool, draws: Option<&[f32]>) {
        if self.public.ended {
            return;
        }
        let p = self.public.cp;
        let src = action as usize / 30;
        let color = action as usize % 30 / 6;
        let row = action as usize % 6;
        let qty = if src == self.public.nf() {
            self.public.center[color]
        } else {
            self.public.factories[src][color]
        };
        let mut floor = self.public.boards[p].floor as usize;
        if src == self.public.nf() && self.public.center_first && floor < 7 {
            self.floor_slots[p][floor] = 5;
            floor += 1;
        }
        let pattern = if row < 5 {
            qty.min(row as u8 + 1 - self.public.boards[p].count[row])
        } else {
            0
        };
        let excess = qty - pattern;
        let placed = excess.min((7 - floor) as u8);
        for slot in floor..floor + placed as usize {
            self.floor_slots[p][slot] = color as i8;
        }
        self.floor_tiles[p][color] += placed;
        self.lid[color] += excess - placed;
        self.public = self.public.apply(action, false);
        if finalize {
            self.finalize(draws);
        }
    }

    pub fn finalize(&mut self, draws: Option<&[f32]>) {
        if self.public.ended || !self.public.empty() {
            return;
        }
        for p in 0..self.public.n {
            let board = &self.public.boards[p];
            for row in 0..5 {
                if board.count[row] == row as u8 + 1 {
                    self.lid[board.color[row] as usize] += board.count[row] - 1;
                }
            }
            for color in 0..5 {
                self.lid[color] += self.floor_tiles[p][color];
            }
            self.floor_tiles[p] = [0; 5];
            self.floor_slots[p] = [-1; 7];
        }
        self.public.resolve();
        if !self.public.ended {
            self.public.resolved = false;
            self.public.center_first = true;
            self.public.cp = self.public.first;
            self.fill(draws);
        }
    }

    pub fn winner(&self) -> i8 {
        if !self.public.ended {
            return -1;
        }
        let winners = self.public.winners();
        if winners.iter().filter(|x| **x).count() > 1 {
            return -2;
        }
        winners.iter().position(|x| *x).unwrap() as i8
    }

    pub fn final_values(&self, score_scaled: bool) -> [f32; 4] {
        let mut values = [-1.0; 4];
        if !self.public.ended {
            return values;
        }
        let winners = self.public.winners();
        let winner_score = self.public.boards[..self.public.n]
            .iter()
            .map(|b| b.score)
            .max()
            .unwrap()
            .max(1) as f32;
        let loss_base = -1.0 / (self.public.n - 1) as f32;
        for p in 0..self.public.n {
            if winners[p] {
                values[p] = 1.0;
            } else if score_scaled {
                let ratio = self.public.boards[p].score as f32 / winner_score;
                values[p] = (loss_base + ratio * ratio).clamp(-1.0, 1.0);
            }
        }
        values
    }

    pub fn inventory(&self) -> [usize; 5] {
        std::array::from_fn(|c| {
            let mut total = self.bag[c] as usize
                + self.lid[c] as usize
                + self.public.center[c] as usize
                + self
                    .public
                    .factories
                    .iter()
                    .map(|f| f[c] as usize)
                    .sum::<usize>();
            for p in 0..self.public.n {
                total += self.floor_tiles[p][c] as usize;
                for r in 0..5 {
                    total += usize::from(cell(self.public.boards[p].wall, r, (r + c) % 5));
                    if self.public.boards[p].color[r] == c as i8 {
                        total += self.public.boards[p].count[r] as usize;
                    }
                }
            }
            total
        })
    }

    pub fn pack(&self) -> Vec<i64> {
        let mut data: Vec<i64> = self.public.pack().into_iter().map(i64::from).collect();
        data[0] = FULL_VERSION;
        data.extend(self.bag.map(i64::from));
        data.extend(self.lid.map(i64::from));
        for p in 0..self.public.n {
            data.extend(self.floor_tiles[p].map(i64::from));
            data.extend(self.floor_slots[p].map(i64::from));
        }
        data.extend([(self.rng >> 32) as i64, (self.rng & 0xffff_ffff) as i64]);
        data
    }

    pub fn parse(data: &[i64]) -> Result<Self, String> {
        if data.len() < 2 || data[0] != FULL_VERSION || !(2..=4).contains(&data[1]) {
            return Err("Expected full simulator snapshot version 2 and 2–4 players".into());
        }
        let n = data[1] as usize;
        let public_len = 57 + 14 * n;
        if data.len() != public_len + 12 + 12 * n {
            return Err("Invalid full snapshot length".into());
        }
        let mut public_data = data[..public_len]
            .iter()
            .map(|x| i32::try_from(*x))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "Public field exceeds i32")?;
        public_data[0] = 1;
        let mut public = State::parse(&public_data)?;
        // Public snapshots omit inactive boards; use the simulator's canonical padding.
        for board in &mut public.boards[n..] {
            board.color = [-1; 5];
        }
        if public.resolved != public.ended {
            return Err("Full snapshots must be before resolution or after complete setup".into());
        }
        let mut game = Self {
            public,
            bag: [0; 5],
            lid: [0; 5],
            floor_tiles: [[0; 5]; 4],
            floor_slots: [[-1; 7]; 4],
            rng: 0,
        };
        for c in 0..5 {
            for (offset, target) in [(public_len, &mut game.bag), (public_len + 5, &mut game.lid)] {
                if !(0..=20).contains(&data[offset + c]) {
                    return Err("Invalid bag/discard count".into());
                }
                target[c] = data[offset + c] as u8;
            }
        }
        for p in 0..n {
            let offset = public_len + 10 + p * 12;
            let mut counts = [0u8; 5];
            let mut markers = 0;
            for slot in 0..7 {
                let value = data[offset + 5 + slot];
                if !(-1..=5).contains(&value)
                    || (slot < game.public.boards[p].floor as usize) != (value >= 0)
                {
                    return Err("Invalid ordered floor slots".into());
                }
                game.floor_slots[p][slot] = value as i8;
                if (0..5).contains(&value) {
                    counts[value as usize] += 1;
                }
                if value == 5 {
                    markers += 1;
                }
            }
            for (c, count) in counts.iter().enumerate() {
                if data[offset + c] != i64::from(*count) {
                    return Err("Floor counts do not match ordered slots".into());
                }
            }
            let board = &game.public.boards[p];
            if markers > 1
                || (markers == 1 && !board.marker)
                || (board.marker && markers == 0 && board.floor != 7)
            {
                return Err("Invalid floor marker ownership".into());
            }
            if game.public.ended
                && (board.floor != 0
                    || board.marker
                    || board
                        .count
                        .iter()
                        .enumerate()
                        .any(|(r, c)| *c == r as u8 + 1))
            {
                return Err("Terminal snapshot must have resolved floors and pattern lines".into());
            }
            game.floor_tiles[p] = counts;
        }
        let rng = &data[data.len() - 2..];
        if rng.iter().any(|x| !(0..=u32::MAX as i64).contains(x)) {
            return Err("Invalid RNG words".into());
        }
        game.rng = ((rng[0] as u64) << 32) | rng[1] as u64;
        if game.inventory() != [20; 5] {
            return Err("Full snapshot must conserve exactly 20 tiles of every color".into());
        }
        if game.public.ended
            && (!game.public.empty()
                || !game.public.boards[..n]
                    .iter()
                    .any(|b| crate::state::rows(b.wall) > 0))
        {
            return Err("Invalid terminal game".into());
        }
        Ok(game)
    }

    /// Same 275 global and 10×5 source features as agent.net.encoder, in f32.
    pub fn encode(&self, global: &mut [f32], sources: &mut [f32]) {
        let s = &self.public;
        global.fill(0.0);
        sources.fill(0.0);
        for c in 0..5 {
            global[c] = s.center[c] as f32;
            global[9 + c] = self.bag[c] as f32 / 20.0;
            global[14 + c] = self.lid[c] as f32 / 20.0;
        }
        global[5] = u8::from(s.center_first) as f32;
        global[6 + s.n - 2] = 1.0;
        for f in 0..9 {
            for c in 0..5 {
                sources[f * 5 + c] = s.factories[f][c] as f32;
            }
        }
        for c in 0..5 {
            sources[s.nf() * 5 + c] = s.center[c] as f32;
        }
        for offset in 0..s.n {
            let p = (s.cp + offset) % s.n;
            let b = &s.boards[p];
            let base = 19 + offset * 64;
            for r in 0..5 {
                global[base + r * 6] = b.count[r] as f32 / (r + 1) as f32;
                if b.color[r] >= 0 {
                    global[base + r * 6 + 1 + b.color[r] as usize] = 1.0;
                }
                for c in 0..5 {
                    global[base + 30 + r * 5 + c] = u8::from(cell(b.wall, r, c)) as f32;
                }
            }
            global[base + 55] = b.floor as f32 / 7.0;
            for c in 0..5 {
                global[base + 56 + c] = self.floor_tiles[p][c] as f32 / 7.0;
            }
            global[base + 61] = u8::from(b.marker) as f32;
            global[base + 62] = b.score as f32 / 100.0;
            global[base + 63] = u8::from(offset == 0) as f32;
        }
    }
}

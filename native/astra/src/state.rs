//! Public information only. No bag, discard contents, or random generator.
pub const FLOOR: [i32; 8] = [0, -1, -2, -4, -6, -8, -11, -14];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Board {
    pub wall: u32,
    pub score: i32,
    pub floor: u8,
    pub marker: bool,
    pub count: [u8; 5],
    pub color: [i8; 5],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub n: usize,
    pub cp: usize,
    pub first: usize,
    pub center_first: bool,
    pub ended: bool,
    pub resolved: bool,
    pub factories: [[u8; 5]; 9],
    pub center: [u8; 5],
    pub boards: [Board; 4],
}

pub fn cell(wall: u32, r: usize, c: usize) -> bool {
    wall & (1 << (r * 5 + c)) != 0
}

pub fn wall_score(wall: u32, r: usize, c: usize) -> i32 {
    let mut h = 1;
    let mut v = 1;
    for x in (0..c).rev() {
        if !cell(wall, r, x) {
            break;
        }
        h += 1;
    }
    for x in c + 1..5 {
        if !cell(wall, r, x) {
            break;
        }
        h += 1;
    }
    for y in (0..r).rev() {
        if !cell(wall, y, c) {
            break;
        }
        v += 1;
    }
    for y in r + 1..5 {
        if !cell(wall, y, c) {
            break;
        }
        v += 1;
    }
    if h == 1 && v == 1 {
        1
    } else {
        (if h > 1 { h } else { 0 }) + (if v > 1 { v } else { 0 })
    }
}

pub fn rows(wall: u32) -> usize {
    (0..5).filter(|r| (wall >> (5 * r)) & 31 == 31).count()
}

pub fn bonus(wall: u32) -> i32 {
    let cols = (0..5).filter(|c| (0..5).all(|r| cell(wall, r, *c))).count();
    let colors = (0..5)
        .filter(|c| (0..5).all(|r| cell(wall, r, (r + c) % 5)))
        .count();
    (2 * rows(wall) + 7 * cols + 10 * colors) as i32
}

impl State {
    pub fn parse(data: &[i32]) -> Result<Self, String> {
        if data.len() < 7 || data[0] != 1 || !(2..=4).contains(&data[1]) {
            return Err("Expected snapshot version 1 and 2–4 players".into());
        }
        let n = data[1] as usize;
        if data.len() != 57 + 14 * n {
            return Err("Invalid snapshot length".into());
        }
        if !(0..n as i32).contains(&data[2]) || !(0..n as i32).contains(&data[3]) {
            return Err("Invalid current/first player".into());
        }
        if data[4..7].iter().any(|v| !(0..=1).contains(v)) {
            return Err("Invalid boolean".into());
        }
        let mut s = Self {
            n,
            cp: data[2] as usize,
            first: data[3] as usize,
            center_first: data[4] == 1,
            ended: data[5] == 1,
            resolved: data[6] == 1,
            factories: [[0; 5]; 9],
            center: [0; 5],
            boards: [Board::default(); 4],
        };
        for f in 0..9 {
            for c in 0..5 {
                let v = data[7 + 5 * f + c];
                if !(0..=4).contains(&v) {
                    return Err("Invalid factory count".into());
                }
                s.factories[f][c] = v as u8;
            }
            if s.factories[f].iter().map(|v| *v as u16).sum::<u16>() > 4
                || (f >= s.nf() && s.factories[f] != [0; 5])
            {
                return Err("Invalid factory contents".into());
            }
        }
        for c in 0..5 {
            let v = data[52 + c];
            if !(0..=20).contains(&v) {
                return Err("Invalid center count".into());
            }
            s.center[c] = v as u8;
        }
        let mut markers = usize::from(s.center_first);
        for p in 0..n {
            let x = &data[57 + p * 14..57 + (p + 1) * 14];
            if !(0..(1 << 25)).contains(&x[0])
                || !(0..=32767).contains(&x[1])
                || !(0..=7).contains(&x[2])
                || !(0..=1).contains(&x[3])
            {
                return Err("Invalid wall, score, floor, or marker".into());
            }
            let b = &mut s.boards[p];
            b.wall = x[0] as u32;
            b.score = x[1];
            b.floor = x[2] as u8;
            b.marker = x[3] == 1;
            if b.marker && b.floor == 0 {
                return Err("A floor marker requires a floor slot".into());
            }
            markers += usize::from(b.marker);
            for r in 0..5 {
                let count = x[4 + r];
                let color = x[9 + r];
                if !(0..=r as i32 + 1).contains(&count)
                    || !(-1..=4).contains(&color)
                    || (count == 0) != (color == -1)
                    || (count > 0 && cell(b.wall, r, (r + color as usize) % 5))
                {
                    return Err("Invalid pattern line".into());
                }
                b.count[r] = count as u8;
                b.color[r] = color as i8;
            }
        }
        if markers > 1 || (s.resolved && !s.empty()) || (s.ended && !s.resolved) {
            return Err("Invalid round/marker state".into());
        }
        // Each color has 20 tiles. Validate visible tile counts without hidden information.
        for c in 0..5 {
            let mut total =
                s.center[c] as usize + s.factories.iter().map(|f| f[c] as usize).sum::<usize>();
            for b in &s.boards[..n] {
                for r in 0..5 {
                    total += usize::from(cell(b.wall, r, (r + c) % 5));
                    if b.color[r] == c as i8 {
                        total += b.count[r] as usize;
                    }
                }
            }
            if total > 20 {
                return Err("Visible color count exceeds tile supply".into());
            }
        }
        Ok(s)
    }

    pub fn pack(&self) -> Vec<i32> {
        let mut v = vec![
            1,
            self.n as i32,
            self.cp as i32,
            self.first as i32,
            self.center_first as i32,
            self.ended as i32,
            self.resolved as i32,
        ];
        v.extend(self.factories.iter().flatten().map(|x| *x as i32));
        v.extend(self.center.iter().map(|x| *x as i32));
        for b in &self.boards[..self.n] {
            v.extend([b.wall as i32, b.score, b.floor as i32, b.marker as i32]);
            v.extend(b.count.iter().map(|x| *x as i32));
            v.extend(b.color.iter().map(|x| *x as i32));
        }
        v
    }
    pub fn nf(&self) -> usize {
        2 * self.n + 1
    }
    pub fn factories_empty(&self) -> bool {
        self.factories.iter().flatten().all(|x| *x == 0)
    }
    pub fn empty(&self) -> bool {
        self.factories_empty() && self.center == [0; 5]
    }
    pub fn legal(&self) -> Vec<u16> {
        let mut legal = Vec::with_capacity(100);
        if self.ended || self.resolved {
            return legal;
        }
        let b = &self.boards[self.cp];
        for src in 0..=self.nf() {
            let tiles = if src == self.nf() {
                &self.center
            } else {
                &self.factories[src]
            };
            for (c, qty) in tiles.iter().enumerate() {
                if *qty == 0 {
                    continue;
                }
                for r in 0..6 {
                    if r == 5
                        || (b.count[r] < (r + 1) as u8
                            && (b.color[r] == -1 || b.color[r] == c as i8)
                            && !cell(b.wall, r, (r + c) % 5))
                    {
                        legal.push((src * 30 + c * 6 + r) as u16);
                    }
                }
            }
        }
        legal
    }
    pub fn apply(&self, a: u16, resolve: bool) -> Self {
        let mut s = *self;
        let src = a as usize / 30;
        let color = a as usize % 30 / 6;
        let r = a as usize % 6;
        let qty;
        if src == s.nf() {
            qty = s.center[color];
            s.center[color] = 0;
            if s.center_first {
                s.center_first = false;
                s.boards[s.cp].marker = true;
                s.boards[s.cp].floor = (s.boards[s.cp].floor + 1).min(7);
            }
        } else {
            qty = s.factories[src][color];
            for c in 0..5 {
                if c != color {
                    s.center[c] += s.factories[src][c];
                }
            }
            s.factories[src] = [0; 5];
        }
        let b = &mut s.boards[s.cp];
        let placed = if r < 5 {
            qty.min((r + 1) as u8 - b.count[r])
        } else {
            0
        };
        if r < 5 {
            b.count[r] += placed;
            b.color[r] = color as i8;
        }
        b.floor = (b.floor + qty - placed).min(7);
        s.cp = (s.cp + 1) % s.n;
        if resolve && s.empty() {
            s.resolve();
        }
        s
    }
    pub fn resolve(&mut self) {
        if self.resolved || self.ended {
            return;
        }
        for b in &mut self.boards[..self.n] {
            for r in 0..5 {
                if b.count[r] == (r + 1) as u8 {
                    let c = (r + b.color[r] as usize) % 5;
                    b.score += wall_score(b.wall, r, c);
                    b.wall |= 1 << (r * 5 + c);
                    b.count[r] = 0;
                    b.color[r] = -1;
                }
            }
            b.score = (b.score + FLOOR[b.floor as usize]).max(0);
            b.floor = 0;
        }
        for p in 0..self.n {
            if self.boards[p].marker {
                self.first = p;
                self.boards[p].marker = false;
            }
        }
        self.ended = self.boards[..self.n].iter().any(|b| rows(b.wall) > 0);
        if self.ended {
            for b in &mut self.boards[..self.n] {
                b.score += bonus(b.wall);
            }
        }
        self.resolved = true;
        // A nonterminal round boundary is held before setup; no hidden draw occurs.
    }
    pub fn winners(&self) -> [bool; 4] {
        let best = self.boards[..self.n]
            .iter()
            .map(|b| (b.score, rows(b.wall)))
            .max()
            .unwrap();
        std::array::from_fn(|p| {
            p < self.n && (self.boards[p].score, rows(self.boards[p].wall)) == best
        })
    }
}

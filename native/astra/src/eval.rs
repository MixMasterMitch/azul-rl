use crate::state::{bonus, cell, rows, wall_score, State};

pub const DEFAULT_WEIGHTS: [f64; 11] = [1.0, 1.0, 0.25, 0.65, 0.6, 0.35, 1.0, 1.0, 0.0, 0.0, 0.0];
pub const COMPONENTS: [&str; 6] = [
    "projected_score",
    "bonuses",
    "partial_lines",
    "adjacency",
    "flexibility",
    "initiative",
];

fn bonus_potential(wall: u32, horizon: f64, prior: f64) -> f64 {
    let mut result = 0.0;
    for r in 0..5 {
        let count = ((wall >> (r * 5)) & 31).count_ones() as f64;
        result += 2.0 * (count / 5.0).powi(4);
    }
    for c in 0..5 {
        let count = (0..5).filter(|r| cell(wall, *r, c)).count() as f64;
        let colors = (0..5).filter(|r| cell(wall, *r, (r + c) % 5)).count() as f64;
        let classic = 7.0 * (count / 5.0).powi(3) + 10.0 * (colors / 5.0).powi(4);
        let mut col_probability = 1.0;
        let mut color_probability = 1.0;
        for (r, rate) in [1.0, 0.95, 0.8, 0.5, 0.35].iter().enumerate() {
            let missing = 5 - ((wall >> (5 * r)) & 31).count_ones();
            let chance = (horizon * rate / missing.max(1) as f64).min(0.96);
            if !cell(wall, r, c) {
                col_probability *= chance;
            }
            if !cell(wall, r, (r + c) % 5) {
                color_probability *= chance;
            }
        }
        result +=
            (1.0 - prior) * classic + prior * (7.0 * col_probability + 10.0 * color_probability);
    }
    result
}

#[allow(clippy::needless_range_loop)] // Absolute seats index several parallel arrays.
pub fn components(s: &State, w: &[f64; 11]) -> [[f64; 6]; 4] {
    let mut projected = *s;
    if !projected.resolved && !projected.ended {
        projected.resolve();
    }
    let mut out = [[0.0; 6]; 4];
    let remaining_rounds = projected.boards[..s.n]
        .iter()
        .flat_map(|b| (0..5).map(move |r| 5 - ((b.wall >> (r * 5)) & 31).count_ones()))
        .min()
        .unwrap_or(0) as f64;
    let horizon = (remaining_rounds / w[7]).clamp(0.0, 5.0);
    let available: [usize; 5] = std::array::from_fn(|c| {
        s.center[c] as usize + s.factories.iter().map(|f| f[c] as usize).sum::<usize>()
    });
    for p in 0..s.n {
        let b = &projected.boards[p];
        out[p][0] = b.score as f64;
        if s.ended {
            continue;
        }
        // Projection applies end bonuses if an already completed line ends this round.
        // Other players may still add placements before that happens.
        if !projected.ended {
            out[p][1] = w[1] * bonus_potential(b.wall, horizon, w[8]);
        }
        for r in 0..5 {
            let count = b.count[r] as usize;
            if count > 0 {
                let c = b.color[r] as usize;
                let need = r + 1 - count;
                let contest = (0..s.n)
                    .filter(|q| {
                        *q != p
                            && s.boards[*q].color.iter().enumerate().any(|(rr, cc)| {
                                *cc == c as i8 && s.boards[*q].count[rr] < (rr + 1) as u8
                            })
                    })
                    .count();
                let can_finish = available[c] >= need;
                let investment = (count as f64 / (r + 1) as f64).powf(1.5);
                let chance = if s.resolved || !can_finish {
                    0.0
                } else {
                    (available[c] as f64 / (need as f64 * (1.0 + contest as f64))).min(1.0)
                        * 0.70
                        * investment
                };
                let future = if horizon > 0.0 {
                    0.45 * investment
                } else {
                    0.0
                };
                let mut value = wall_score(b.wall, r, (r + c) % 5) as f64 + 0.7;
                if w[9] > 0.0 {
                    let completed = b.wall | (1 << (r * 5 + (r + c) % 5));
                    let gain = if projected.ended {
                        (bonus(completed) - bonus(b.wall)) as f64
                    } else {
                        bonus_potential(completed, horizon, w[8])
                            - bonus_potential(b.wall, horizon, w[8])
                    };
                    value += w[9] * w[1] * gain;
                }
                out[p][2] += w[0] * value * (chance + (1.0 - chance) * future);
                out[p][4] -=
                    w[3] * (1.0 - chance) * need as f64 / (r + 1) as f64 * horizon.min(2.0);
            }
            if horizon > 0.0 {
                for c in 0..5 {
                    if cell(b.wall, r, c) {
                        continue;
                    }
                    let neighbors = usize::from(r > 0 && cell(b.wall, r - 1, c))
                        + usize::from(r < 4 && cell(b.wall, r + 1, c))
                        + usize::from(c > 0 && cell(b.wall, r, c - 1))
                        + usize::from(c < 4 && cell(b.wall, r, c + 1));
                    // Cheap upper lines are more reusable than the bottom lines.
                    out[p][3] +=
                        w[2] * neighbors as f64 / (1.0 + r as f64 * 0.3) * horizon.min(2.0);
                }
            }
        }
        // Safe capacity matters while tiles remain, not as a blanket floor aversion.
        if !s.resolved {
            let original = &s.boards[p];
            for (c, avail) in available.iter().enumerate() {
                if *avail == 0 {
                    continue;
                }
                let capacity = (0..5)
                    .filter(|r| original.color[*r] == -1 || original.color[*r] == c as i8)
                    .filter(|r| !cell(original.wall, *r, (r + c) % 5))
                    .map(|r| r + 1 - original.count[r] as usize)
                    .max()
                    .unwrap_or(0);
                if capacity == 0 {
                    out[p][4] -= w[5] * (*avail as f64 / s.n as f64).min(3.0);
                }
            }
        }
        let marker = s.boards[p].marker || (s.resolved && s.first == p);
        if marker && horizon > 0.0 {
            out[p][5] = w[4] * (1.0 + 1.0 / horizon.max(1.0));
        }
    }
    out
}

pub fn values(s: &State, w: &[f64; 11]) -> [f64; 4] {
    if s.ended {
        let winners = s.winners();
        let k = winners.iter().filter(|v| **v).count() as f64;
        if s.n == 2 {
            let v = 1_000_000.0 * (2.0 * f64::from(winners[0]) / k - 1.0)
                + (s.boards[0].score - s.boards[1].score) as f64 * 0.001;
            return [v, -v, 0.0, 0.0];
        }
        let best = s.boards[..s.n].iter().map(|b| b.score).max().unwrap();
        return std::array::from_fn(|p| {
            if p >= s.n {
                0.0
            } else {
                1_000_000.0 * (s.n as f64 * f64::from(winners[p]) / k - 1.0)
                    + (s.boards[p].score - best) as f64 * 0.001
            }
        });
    }
    let parts = components(s, w);
    let raw: [f64; 4] = std::array::from_fn(|p| parts[p].iter().sum());
    std::array::from_fn(|p| {
        if p >= s.n {
            return 0.0;
        }
        let mut rival = (0..s.n)
            .filter(|q| *q != p)
            .map(|q| raw[q])
            .fold(f64::NEG_INFINITY, f64::max);
        if s.n > 2 && w[10] > 0.0 {
            // A table with several close rivals is harder to beat than one with
            // the same leader and weak trailing players. Smoothly aggregate the
            // rival scores at a manually chosen eight-point uncertainty scale.
            // Log-mean-exp agrees with max when opponents are equally strong;
            // subtracting the maximum before exp prevents numeric overflow.
            let pressure = (0..s.n)
                .filter(|q| *q != p)
                .map(|q| ((raw[q] - rival) / 8.0).exp())
                .sum::<f64>()
                / (s.n - 1) as f64;
            rival += w[10] * 8.0 * pressure.ln();
        }
        // Two-player minimax requires the same zero-sum objective at both seats.
        // Opponent weighting and field aggregation are multiplayer parameters.
        let opponent_weight = if s.n == 2 { 1.0 } else { w[6] };
        (raw[p] - opponent_weight * rival).clamp(-100_000.0, 100_000.0)
    })
}

#[allow(dead_code)]
pub fn earned_bonus(s: &State, p: usize) -> (i32, usize) {
    (bonus(s.boards[p].wall), rows(s.boards[p].wall))
}

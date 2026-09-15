use crate::state::{bonus, rows, wall_score, Board, State};

#[test]
fn scoring_counts_chains_and_intersections() {
    assert_eq!(wall_score(0, 2, 2), 1);
    let wall = (1 << 10) | (1 << 11) | (1 << 7) | (1 << 17);
    assert_eq!(wall_score(wall, 2, 2), 6);
    assert_eq!(rows(31), 1);
    assert_eq!(bonus((1 << 25) - 1), 95);
}

#[test]
fn terminal_tiebreak_uses_rows_then_shared_win() {
    let mut s = State {
        n: 2,
        cp: 0,
        first: 0,
        center_first: false,
        ended: true,
        resolved: true,
        factories: [[0; 5]; 9],
        center: [0; 5],
        boards: [Board::default(); 4],
    };
    s.boards[0].score = 40;
    s.boards[1].score = 40;
    s.boards[0].wall = 31;
    assert_eq!(s.winners(), [true, false, false, false]);
    s.boards[1].wall = 31 << 5;
    assert_eq!(s.winners(), [true, true, false, false]);
}

#[test]
fn two_player_terminal_utilities_are_zero_sum() {
    let mut s = small_round(2, 0);
    s.ended = true;
    s.boards[0].score = 35;
    s.boards[1].score = 30;
    let v = crate::eval::values(&s, &crate::eval::DEFAULT_WEIGHTS);
    assert_eq!(v[0], -v[1]);
    s.boards.swap(0, 1);
    let flipped = crate::eval::values(&s, &crate::eval::DEFAULT_WEIGHTS);
    assert_eq!(v[0], flipped[1]);
}

#[test]
fn two_player_position_utilities_remain_zero_sum_for_all_configurations() {
    let s = small_round(2, 3);
    let expected = crate::eval::values(&s, &crate::eval::DEFAULT_WEIGHTS);
    for weight in [0.0, 0.25, 1.0, 1.4, 100.0] {
        let mut weights = crate::eval::DEFAULT_WEIGHTS;
        weights[6] = weight;
        let actual = crate::eval::values(&s, &weights);
        assert_eq!(actual, expected);
        assert_eq!(actual[0], -actual[1]);
    }
}

#[test]
fn multiplayer_pressure_accounts_for_other_close_rivals() {
    let mut s = small_round(3, 0);
    s.resolved = true;
    s.center = [0; 5];
    s.boards[1] = s.boards[0];
    s.boards[2] = s.boards[0];
    s.boards[0].score = 40;
    s.boards[1].score = 50;
    s.boards[2].score = 50;
    let mut weights = crate::eval::DEFAULT_WEIGHTS;
    let old = crate::eval::values(&s, &weights)[0];
    weights[10] = 1.0;
    assert_eq!(crate::eval::values(&s, &weights)[0], old);
    s.boards[2].score = 25;
    assert!(crate::eval::values(&s, &weights)[0] > old);
    weights[10] = 0.0;
    assert_eq!(crate::eval::values(&s, &weights)[0], old);
    s.n = 2;
    let old_two = crate::eval::values(&s, &weights);
    weights[10] = 1.0;
    assert_eq!(crate::eval::values(&s, &weights), old_two);
}

fn small_round(n: usize, seed: usize) -> State {
    let mut s = State {
        n,
        cp: seed % n,
        first: 0,
        center_first: true,
        ended: false,
        resolved: false,
        factories: [[0; 5]; 9],
        center: [1, 2, 2, 1, 0],
        boards: [Board::default(); 4],
    };
    for p in 0..n {
        s.boards[p].color = [-1; 5];
        s.boards[p].score = (p * 3 + seed % 7) as i32;
        s.boards[p].floor = ((seed + p) % 4) as u8;
        for r in 0..5 {
            let c = (r + p + seed) % 5;
            s.boards[p].wall |= 1 << (r * 5 + c);
        }
    }
    s
}

fn exhaustive(s: &State, depth: u8) -> f64 {
    if s.ended || s.resolved || depth == 0 {
        return crate::eval::values(s, &crate::eval::DEFAULT_WEIGHTS)[0];
    }
    let vs = s
        .legal()
        .into_iter()
        .map(|a| exhaustive(&s.apply(a, true), depth - 1));
    if s.cp == 0 {
        vs.fold(f64::NEG_INFINITY, f64::max)
    } else {
        vs.fold(f64::INFINITY, f64::min)
    }
}

#[test]
fn alpha_beta_matches_exhaustive_unpruned_rounds() {
    for seed in 0..8 {
        let s = small_round(2, seed);
        let cfg = crate::search::Config {
            nodes: 10_000_000,
            center_nodes: 0,
            time_ms: 2000,
            depth: 4,
            width: 0,
            weights: crate::eval::DEFAULT_WEIGHTS,
            seed: 31,
            rollout: false,
        };
        let result = crate::search::run(&s, cfg);
        let expected = exhaustive(&s, 4);
        let chosen = exhaustive(&s.apply(result.action, true), 3);
        assert!(result.solved);
        assert!(
            (chosen - expected).abs() < 1e-9,
            "seed {seed}: got {chosen}, expected {expected}"
        );
    }
}

fn exhaustive_vectors(s: &State, depth: u8, seed: u64) -> [f64; 4] {
    if s.ended || s.resolved || depth == 0 {
        return crate::eval::values(s, &crate::eval::DEFAULT_WEIGHTS);
    }
    // Independent enumerator uses the documented ordering only for equal-valued
    // actions: Max-N ties can change other players' backed-up utilities.
    let mut children: Vec<_> = s
        .legal()
        .into_iter()
        .map(|a| {
            let child = s.apply(a, true);
            (
                a,
                child,
                crate::eval::values(&child, &crate::eval::DEFAULT_WEIGHTS),
            )
        })
        .collect();
    children.sort_by(|a, b| {
        b.2[s.cp]
            .total_cmp(&a.2[s.cp])
            .then_with(|| crate::search::tie_key(a.0, seed).cmp(&crate::search::tie_key(b.0, seed)))
    });
    let mut best = None;
    for (_, child, _) in children {
        let value = exhaustive_vectors(&child, depth - 1, seed);
        if best.is_none_or(|b: [f64; 4]| value[s.cp] > b[s.cp]) {
            best = Some(value);
        }
    }
    best.unwrap()
}

#[test]
fn max_n_matches_exhaustive_rounds_for_every_actor() {
    for n in [3, 4] {
        for seat in 0..n {
            let s = small_round(n, seat);
            let cfg = crate::search::Config {
                nodes: 10_000_000,
                center_nodes: 0,
                time_ms: 2000,
                depth: 4,
                width: 0,
                weights: crate::eval::DEFAULT_WEIGHTS,
                seed: 31,
                rollout: false,
            };
            let result = crate::search::run(&s, cfg);
            let expected = exhaustive_vectors(&s, 4, 31);
            assert_eq!(result.depth, 4);
            assert!(result.solved);
            assert_eq!(result.value, expected);
        }
    }
}

#[test]
fn multiplayer_shared_victories_use_the_symmetric_baseline() {
    for n in [3, 4] {
        let mut s = small_round(n, 0);
        s.ended = true;
        s.resolved = true;
        s.center = [0; 5];
        for p in 0..n {
            s.boards[p].score = if p < 2 { 40 } else { 30 };
        }
        let v = crate::eval::values(&s, &crate::eval::DEFAULT_WEIGHTS);
        assert!(v[0] > 100_000.0 && v[1] > 100_000.0);
        assert!(v[2] < -100_000.0);
        for p in 0..n {
            s.boards[p].score = 40;
        }
        let tied = crate::eval::values(&s, &crate::eval::DEFAULT_WEIGHTS);
        assert_eq!(&tied[..n], &vec![0.0; n]);
    }
}

#[test]
fn positional_estimates_cannot_outweigh_terminal_victory() {
    let mut s = small_round(4, 0);
    for p in 0..4 {
        s.boards[p].score = 32000;
    }
    let mut weights = crate::eval::DEFAULT_WEIGHTS;
    weights[6] = 100.0;
    assert!(crate::eval::values(&s, &weights)
        .iter()
        .all(|v| v.abs() <= 100_000.0));
}

#[test]
fn completed_pruned_round_stops_without_claiming_an_exact_solution() {
    let mut s = small_round(2, 1);
    s.center = [0, 0, 2, 1, 0];
    s.factories[0] = [4, 0, 0, 0, 0];
    let cfg = crate::search::Config {
        nodes: 1_000_000,
        center_nodes: 0,
        time_ms: 1950,
        depth: 32,
        width: 1,
        weights: crate::eval::DEFAULT_WEIGHTS,
        seed: 31,
        rollout: false,
    };
    let result = crate::search::run(&s, cfg);
    assert_eq!(result.reason, "pruned_round");
    assert_eq!(result.depth, 3);
    assert!(!result.solved);
    assert_eq!(result.action, 1);
    let unpruned = crate::search::run(&s, crate::search::Config { width: 0, ..cfg });
    assert!(unpruned.solved);
    assert_eq!(unpruned.reason, "solved");
}

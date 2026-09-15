use crate::game::Game;

#[test]
fn full_games_preserve_every_color_and_resume_exactly() {
    for players in 2..=4 {
        for seed in [0, 17, u64::MAX] {
            let mut game = Game::new(players, seed);
            let mut restored = Game::parse(&game.pack()).unwrap();
            for turn in 0..1000 {
                assert_eq!(game.inventory(), [20; 5]);
                assert_eq!(game, restored);
                if game.public.ended {
                    break;
                }
                let legal = game.public.legal();
                // Vary choices without coupling policy sampling to the game's draw RNG.
                let action = legal[(turn * 17 + seed as usize % 101) % legal.len()];
                game.step(action, true, None);
                restored.step(action, true, None);
                if turn % 11 == 0 {
                    restored = Game::parse(&restored.pack()).unwrap();
                }
            }
            assert!(game.public.ended);
            let final_state = game.clone();
            game.finalize(None);
            game.step(299, true, None);
            assert_eq!(game, final_state);
        }
    }
}

#[test]
fn truncated_and_overflowing_snapshots_return_errors() {
    let game = Game::new(4, 3);
    let data = game.pack();
    for len in 0..data.len() {
        assert!(Game::parse(&data[..len]).is_err());
    }
    for i in 0..data.len() {
        let mut broken = data.clone();
        broken[i] = i64::MAX;
        assert!(Game::parse(&broken).is_err(), "field {i}");
    }
}

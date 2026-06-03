---
name: Tuning Failure Analysis
overview: Analyze why the current phase-2 Optuna run is plateauing around 2.0k to 2.2k projected rating, identify likely correctness bugs versus tuning artifacts, and propose a validation-first remediation path.
todos:
  - id: fix-value-indexing
    content: Fix MCTS root child value indexing and self-play value target rotation, with regression tests.
    status: pending
  - id: round-end-search
    content: Add a round-ending child evaluation test and either fix conditional wall-tiling or quantify the approximation.
    status: pending
  - id: eval-consistency
    content: Repair tuning opponent naming and align baseline/session evaluation configs.
    status: pending
  - id: diagnostic-rerun
    content: Run a small corrected fixed-config diagnostic and compare raw winrates to the current DB/log baseline.
    status: pending
  - id: retune-after-fixes
    content: Launch a narrower post-fix tuning study only after diagnostics show positive learning slope.
    status: pending
isProject: false
---

# Tuning Failure Analysis

## Current Evidence

- Active study: `agent/runs/optuna_azul-tune-curve-1023x32-r2.db` has 5 completed trials and trial 5 running.
- Best completed trial is `#4`: projected `2175.2`, last-session `2169.6`.
- Shared baseline from `agent/runs/tune_baseline_eval.json` is already `2123.7` from `agent/runs/league/ckpt_00025_i650.pt`, so tuning is mostly flat or degrading relative to the initializer.
- Session winrates remain far below heuristic bots: best sessions are roughly `8-10%` vs `heuristic` and `2-7%` vs `opus`, despite near-100% vs random.
- The current league reference anchors put 2p `heuristic` around `2925.3` and `heuristic_opus` around `3159.9`; a `4500` expectation is probably not comparable to the current anchored scale, but the low heuristic winrates are real.

## Primary Hypotheses

- Correctness bug: MCTS child value lookup uses `% MAX_PLAYERS` instead of `% num_players` in [`agent/search/gumbel_mcts.py`](agent/search/gumbel_mcts.py). In 2p, a child after player 0 moves can read padded value column 3 instead of opponent-relative column 1, corrupting Q estimates and improved policy targets.
- Correctness bug: self-play target rotation uses the full target tensor width in [`agent/train/selfplay.py`](agent/train/selfplay.py), so `(B, MAX_PLAYERS)` value targets rotate through inactive padded players instead of only active seats. The acting-player target is often right, but opponent columns trained by [`agent/train/learner.py`](agent/train/learner.py) can be wrong.
- Search bug or approximation: MCTS child expansion calls `child_engine.step(..., finalize_round=False)` in [`agent/search/gumbel_mcts.py`](agent/search/gumbel_mcts.py). Round-ending moves are evaluated in a state that real play would immediately score and wall-tile, which is exactly where Azul heuristics matter most.
- Evaluation artifact: tuning records HeuristicOpusBot as `opus` in [`agent/scripts/tune.py`](agent/scripts/tune.py), while Bradley-Terry anchors use `heuristic_opus`. That disconnects the strongest heuristic anchor from tuning match results.
- Objective artifact: baseline eval uses `q_scale=25` and `max_turns=300`, while trial sessions use sampled `q_scale` and `cfg.eval_max_turns=200`. The 7-point curve mixes unlike eval protocols.
- Objective artifact: projecting from 3 hours of noisy 256-game evaluations out to `t=75h` is too optimistic/noisy for picking production hyperparameters.
- Tuning params: current ranges search high `q_scale`, high `dirichlet_mix`, and high `entropy_bonus`. Those are especially harmful while Q estimates are corrupted; they may amplify bad search targets rather than improve exploration.
- Model/search limitation: the model is probably not the first blocker, but [`agent/net/encoder.py`](agent/net/encoder.py) encodes only pattern-line fill ratio and whether a line has any color, not the specific color. If corrected training still plateaus, add pattern-line color features and consider deeper search.

## Remediation Plan

1. Fix learning-signal bugs first.
   - Change `_root_value_index` to use `num_players` and pass it from `_evaluate_root_children_batched`.
   - Change `_rotate_for_cp` to rotate only active player columns, with tests covering production-shaped `(B, MAX_PLAYERS)` values.
   - Update tests in [`agent/tests/test_mcts_value_index.py`](agent/tests/test_mcts_value_index.py) and [`agent/tests/test_selfplay_value_perspective.py`](agent/tests/test_selfplay_value_perspective.py) so they assert the intended perspective behavior rather than the current bug.

2. Fix or measure round-ending child evaluation.
   - Add a regression test for a child move that empties the offer phase and should trigger wall tiling.
   - Prefer a conditional finalize path for children that end the round; if performance is unacceptable, at minimum measure how often this occurs and mark it as a known approximation.

3. Repair tuning/eval comparability.
   - Map tuning opponent `opus` to `heuristic_opus` before Bradley-Terry fitting in [`agent/scripts/tune.py`](agent/scripts/tune.py), or rename the tournament opponent consistently in [`agent/eval/tournament.py`](agent/eval/tournament.py).
   - Use identical `q_scale`, `max_turns`, `temperature`, `rating_games`, and `rating_sims` for the baseline and session evals.
   - Store session winrates as trial user attributes, not only log events, so Optuna analysis can inspect them directly.

4. Make the next tuning study diagnostic rather than broad.
   - Stop trusting `azul-tune-curve-1023x32-r2` as a final selector; it is useful evidence of plateauing, not a clean optimization result.
   - Run a small fixed-config ablation after correctness fixes: baseline checkpoint, corrected code, same phase-2 defaults, fixed `q_scale=25`, and `rating_games>=512`.
   - Compare `last_session` and raw winrates before relying on +72h extrapolated objective.

5. Retune with safer ranges only after the signal is corrected.
   - Narrow `q_scale` initially, for example `8-18`, until value/Q behavior is verified.
   - Lower `dirichlet_mix` and `entropy_bonus` during debugging.
   - Keep `score_scaled` vs `binary` as an explicit ablation, not a broad TPE dimension at first.
   - Consider larger self-play budgets only after corrected 1023x32 shows a positive slope.

6. Revisit architecture only if corrected training still cannot climb.
   - Add pattern-line color encoding to [`agent/net/encoder.py`](agent/net/encoder.py) and adjust [`agent/net/model.py`](agent/net/model.py) input dimensions.
   - Evaluate whether 1-ply Gumbel root search is enough, or whether a real multi-ply MCTS is needed.

## Validation Gates

- Unit tests cover 2p/3p/4p value indexing and value rotation with `MAX_PLAYERS`-wide tensors.
- A targeted MCTS child test confirms round-ending actions are evaluated against the same phase as real `engine.step`.
- A checkpoint re-eval reports stable raw winrates vs `random`, `heuristic`, `heuristic_opus`, and top league opponent under a single fixed eval config.
- A short corrected training run improves over the `2123.7` baseline in last-session rating and heuristic winrate before launching another long Optuna study.
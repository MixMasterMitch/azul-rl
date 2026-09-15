# Astra in training evaluations

Astra is a useful fixed benchmark because the [expanded ranking campaign](RANKINGS.md)
placed it above the available neural agents. New training evaluations include the
production Astra policy in **12.5% of opponent seats, in expectation**. The remaining
seats retain the existing mix of random, heuristic, Opus, and sampled league models.
The total evaluation game budget stays the same.

```bash
# Default: approximately 64 Astra games in a 512-game two-player evaluation.
azul-train --eval-games 512 --eval-astra-fraction 0.125

# Other useful settings: 0 disables Astra; 1 evaluates exclusively against Astra.
azul-train --eval-astra-fraction 1 --eval-games 64 --eval-workers 4
```

The setting is `LoopConfig.eval_astra_fraction` in Python and
`UnifiedEvalConfig.astra_opponent_fraction` in the evaluator. Fractions outside
`[0, 1]` are rejected. Astra uses the same per-player-count production settings as
play: 256,000 nodes for two players and 4,000 for three/four players, with a 1,950 ms
native search ceiling. Its snapshot adapter supports the evaluation engine's
sub-batches and performs no network calls. Extra CPU time depends on how many Astra
seats are sampled; the fraction and existing evaluation worker count control it.

This adds an evaluation opponent. It does not add Astra trajectories to the
learner or change the self-play opponent mixture. The existing training loop
evaluates two-player games; the unified evaluator also supports three/four-player
weights and selects the matching Astra production configuration automatically.

## Recorded results

`metrics.jsonl` reports the evaluated checkpoint's outcomes:

| Metric | Meaning |
| --- | --- |
| `games_2p_vs_astra` | Scheduled two-player games against Astra |
| `finished_2p_vs_astra` | Games that reached an official outcome |
| `unfinished_2p_vs_astra` | Games that reached the turn cap |
| `eval_win_share_2p_vs_astra` | Checkpoint win share over finished Astra games |
| `win_share_sum_2p_vs_astra` | Raw credit used when merging worker results |

For multiplayer, the suffix is `3p_with_astra` or `4p_with_astra`: these are whole
table outcomes with at least one Astra seat, potentially alongside other agents.
They are not head-to-head estimates. A sole winner receives one credit and each
of `k` shared winners receives `1/k`. Always inspect the finished/unfinished
counts; a zero-sample ratio is zero and is not evidence of losses.

Worker results merge by summing credit and finished counts, rather than averaging
worker percentages. Metrics are persisted on normal collection, before the next
evaluation, and at training shutdown. `unified_eval_done` events also record Astra's
full configuration, native API version, and installed binary SHA-256 hashes.
Inconsistent Astra identities across workers reject the evaluation. Missing or
stale native extensions produce explicit evaluation errors; disabling Astra skips
loading its policy. Production training still needs the native game engine.

League entries gain `games_2p_vs_astra` and `winrate_2p_vs_astra` diagnostics, with
ties worth half. Astra is fitted as a floating league entity, not assigned a fixed
rating from the separate ranking campaign. Existing rating anchors stay the same.
The league's multiplayer pairwise diagnostics follow its winner-versus-opponents
convention; use the evaluation metrics above for table win share.

This change also fixes reciprocal half-win ties being rounded to zero during
league ingestion, and uses exact multiplayer shared-win credit. Previously lost
tie data cannot be reconstructed from historical aggregate counts. Compare
per-opponent results across evaluation mixtures; aggregate win rates before and
after introducing Astra measure different opponent fields.

## Paired arena and competitive campaigns

The paired arena accepts `astra` directly and records its configuration/binary
identity alongside checkpoint and search provenance:

```bash
python -m agent.scripts.arena agent/runs/competitive/baselines/v4_latest.pt astra \
  --device cpu --games 256 --sims 64 --seed 916100 \
  --report agent/runs/manual-astra-evaluation.json
```

New competitive screening passes include a 256-game Astra match alongside Opus,
heuristic, champion, and historical checkpoints. Astra configuration or binary
changes invalidate cached Astra matches. The campaign's existing promotion
criteria are unchanged; Astra is an additional diagnostic.

Already-frozen competitive experiments preserve their original evaluation mix on
resume. An older experiment budget/completion record without `eval_astra_fraction`
means zero, and changing it still requires a new run ID. Fresh experiments default
to 0.125. No running process was restarted or live league amended for this change.

## Validation

The CPU agent suite passed **248 tests, 30 skipped** (CUDA was hidden from that
test process to avoid competing with active GPU training). The targeted evaluation
suite passed 38 tests. The integration tests cover configuration, inclusion/disabled
mode, native provenance, seat sampling, shared outcomes, unfinished games, worker
aggregation, league tie ingestion, arena selection, cache invalidation, and
compatibility with frozen campaign configurations.

A real smoke run completed **12/12 games, zero unfinished**, in 29.22 seconds:
six unified-evaluation games across 2p/3p/4p with two workers, two paired-arena
games, and four evaluation games from a one-iteration training run with two
workers. It used a small probe network to validate plumbing, not estimate strength.
The isolated records are in `agent/runs/astra-training-eval-smoke/`; the active
competitive league and frozen ranking datasets were not used as output targets.

Repeat with a new output directory:

```bash
python -m agent.scripts.smoke_astra_training --output agent/runs/astra-eval-smoke-new
pytest agent/tests/test_astra_training_eval.py
```

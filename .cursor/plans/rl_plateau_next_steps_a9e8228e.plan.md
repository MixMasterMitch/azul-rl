---
name: RL Plateau Next Steps
overview: Diagnose the plateau with targeted search/eval and representation experiments before spending compute on a much larger model. The highest-probability path is to validate the current tuning signal, add cheap Azul-specific features, then only scale architecture once we know the bottleneck is capacity.
todos:
  - id: encoder-ablation
    content: Add Azul-specific encoder features first, especially pattern-line color, floor composition, and floor first-player marker status.
    status: completed
  - id: start-v3
    content: Stop the current tuning job and launch a v3 training run from the v2 i1950 checkpoint using the richer encoder.
    status: completed
  - id: reeval-signal
    content: Re-evaluate baseline and top tuning trials under one high-confidence opus-heavy eval protocol.
    status: pending
  - id: search-sweep
    content: Run fixed-checkpoint `q_scale`, `num_sims`, and temperature sweeps to separate search weakness from learned policy weakness.
    status: pending
  - id: focused-retune
    content: Run a focused binary-reward Optuna study with cleaner objective and narrower optimizer/search ranges.
    status: pending
  - id: capacity-ladder
    content: Only after the above gates, compare `attn 384` and `attn 512` against the richer `attn 256` baseline.
    status: pending
isProject: false
---

# RL Plateau Investigation Plan

## Recommendation
Do not jump straight to `attn 512`. The current stack is an `attn` trunk with `hidden=256`, but the search is a one-ply Gumbel-root evaluator and the encoder omits important Azul state detail. The next work should be staged so each experiment answers a specific bottleneck question.

Important context from the repo:

- [`agent/search/gumbel_mcts.py`](agent/search/gumbel_mcts.py) is explicitly one-ply: prior logits, Gumbel top-K root actions, one child expansion, then value-network scoring.
- [`agent/net/encoder.py`](agent/net/encoder.py) encodes pattern lines as fill ratio plus `has_color`, not the actual committed color.
- [`agent/scripts/tune.py`](agent/scripts/tune.py) fixes self-play at `1023 × 32`, while tuning mostly optimizer, reward, Dirichlet, entropy, and `q_scale`.
- [`agent/eval/tournament.py`](agent/eval/tournament.py) evaluates with `temperature=0.25` and default `q_scale=10.0`, while training defaults lean toward `q_scale=25.0` unless tuning overrides it.

## Phase 1: Complete The Encoder State
Add targeted encoder features before scaling to `512`. This is likely high ROI because pattern-line color is central to Azul planning and legality/value, but currently hidden behind a boolean.

First implementation:
- Add pattern-line color one-hot per row in [`agent/net/encoder.py`](agent/net/encoder.py), replacing the current `has_color`-only representation.
- Add per-seat floor composition: five color counts from `floor_tiles`, normalized by floor size.
- Add per-seat `floor_first` marker status. This is enough first-player-marker information during the offer phase because the marker is either still in center (`center_first`) or on a player's floor (`floor_first`).
- Do not add `first_player` directly at first. At next-round setup, [`agent/env/batched_engine.py`](agent/env/batched_engine.py) copies `first_player` into `current_player`, and the encoder is already perspective-based around `current_player`.
- Keep `hidden=256` initially so the comparison isolates representation quality.

Validation:
- Update encoder dimension tests to reflect the richer global feature shape.
- Add or update an encoder test showing committed pattern-line colors and floor marker/color counts are visible in the encoded tensor.
- Run the relevant encoder/model tests before starting the v3 training session.

## Phase 2: Start V3 From V2 i1950
Stop the current tuning process, then launch a new v3 training session from the prior v2 best checkpoint (`i1950`) using the richer encoder. This run should be treated as a representation ablation, not a broad retune.

Suggested v3 defaults:
- `hidden=256`, `arch=attn`
- `reward_mode=binary`
- LR near the current best trial, around `9.7e-4`
- `q_scale=28.0`
- Entropy near the current best trial, around `0.02`
- Same self-play budget as v2 unless throughput changes materially

## Phase 3: Trust The Signal
Keep the current tuning run as useful evidence, but do not choose a production config from the projected 72h objective alone. The eval points are noisy at 256 games, and the curve is extrapolating from only a few hours of training.

Actions:
- Re-evaluate baseline, current production, and top 2-3 trial checkpoints with the same protocol.
- Use at least 768-1024 games vs `HeuristicOpusBot` for the final comparison.
- Report raw opus winrate with binomial confidence intervals, not just projected objective.
- Sweep eval-only `q_scale` and `num_sims` on the same checkpoints before retraining.

## Phase 4: Search Quality Before Bigger Nets
The current improved policy target only has support over up to `num_sims` Gumbel-selected actions. If search is too shallow or miscalibrated, the learner is distilling weak targets no matter how wide the trunk is.

Experiments:
- Fixed-checkpoint sweep: `num_sims` in `16, 32, 64` where feasible.
- Fixed-checkpoint sweep: `q_scale` in `8, 12, 18, 25, 32`.
- Compare eval temperature `0.1, 0.25, 0.5`.
- If 64 sims improves substantially at eval time, prioritize deeper/better search or fewer games with more sims over model width.

## Phase 5: Training Mix And Objective
The current result that binary reward beats score-scaled is believable. I would narrow around binary and stop spending trials on broad optimizer ranges until the eval/search protocol is cleaner.

Next tuning shape:
- Fix `reward_mode=binary` for a short focused study.
- Use narrower LR around the best observed region, roughly `7e-4` to `1.5e-3`.
- Lower entropy range than phase-2 if best trials prefer it, roughly `0.005` to `0.025`.
- Tune `bot_selfplay_opus_prob`, `training_cycle_length`, and replay freshness, since the plateau is specifically vs opus.
- Use `opus_winrate` or an opus-heavy objective, not blended rating, while investigating this plateau.

## Phase 6: Capacity Scaling Only After Gates
Try `attn 512` only after we know one of these is true:

- Search sweeps show little benefit from more sims or better `q_scale`.
- Encoder feature ablations help but then saturate.
- Training and validation losses suggest underfitting rather than noisy or weak targets.
- Throughput profiling says the larger trunk will not cut self-play volume enough to erase the capacity gain.

Suggested capacity ladder:
- `attn 256 + richer encoder`
- `attn 384 + richer encoder`
- `attn 512 + richer encoder`

Avoid changing encoder, reward, search, and width all at once; otherwise the next plateau will be hard to diagnose.

## Decision Rule
If eval-only `num_sims`/`q_scale` sweeps produce a large winrate jump, invest in search. If richer encoder improves learning at fixed compute, invest in representation. If neither helps and losses/diagnostics show underfitting, then scale the model.
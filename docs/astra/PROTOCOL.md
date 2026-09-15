# Evaluation protocol

This protocol was recorded before final-test games. Development and promotion
records may guide the selected configuration; final records may not.

## Fixed field and matched games

The competitive field contains homogeneous Opus tables, mixed tables drawn from
Opus/basic heuristic/random, and the frozen checkpoint with 32-simulation search.
The checkpoint was selected from the available league by per-player-count rating,
falling back to overall rating when that evidence was absent. The multiplayer
fallback is weak, so Opus and the mixed field remain essential comparisons.

Every block uses one independently initialized engine seed and all designated
seats. Opponent types rotate with the designated seat. Each game receives
separate bot seeds; candidate and incumbent use identical schedules. Engines,
reference policies, checkpoint bytes, and inference sources are frozen and
hash-verified. The historical league is never modified.

Screening uses 32 development blocks. Promotion uses 128 fresh blocks and requires
the paired 95% lower confidence bound on average competitive-field win-share
improvement to exceed zero, without a significant Opus regression. An incomplete
or failed game prevents promotion. A larger node budget is a separate candidate,
with unchanged evaluator weights. Inconclusive evidence retains the incumbent.

## Finalist ablations

At the selected search budget, remove one active feature at a time: partial-line
credit, bonus potential, adjacency, unfinished-line penalty, initiative, or safe
placement capacity. At multiplayer counts also remove opponent pressure or the
attainability prior, and restore the original strategic horizon. Test depth zero
as the no-search control. Inactive experimental features are not ablated.

Use 32 fresh development blocks against Opus, mixed tables, and the checkpoint.
Compare paired block differences with the matching full-feature baseline. These
are exploratory effects, without multiple-comparison correction; a promising
removal still needs a separate 128-block promotion before changing production.

## Final holdout

Freeze the selected code, native binary, and per-count configurations before the
first final game. Launch every final job from that immutable runtime, including
the parent orchestrator. Development, promotion, and final seed ranges are
disjoint; the final split begins at 200,000,000 before the player-count offset.

For each of two, three, and four players, run:

- 256 blocks against homogeneous Opus tables: 512, 768, and 1,024 games.
- 64 mixed-field blocks with both Astra and an Opus control in the designated
  seat: 256, 384, and 512 games including controls.
- Secondary random/basic/checkpoint/self-play diagnostics when time permits.

Use at most four evaluation workers, with one PyTorch thread each. If runtime
threatens the development budget, reduce secondary diagnostics first and report
any reduction in planned final samples. Do not relax the acceptance criteria.

The primary statistic is win share: 1 for a sole win, 1/k for a k-way shared win,
and 0 for a completed loss. Resample complete seed/seat blocks 10,000 times for
95% intervals. A capped game is unfinished, has no assigned winner, and remains
visible in the report. Report scores, margins, floor losses before clamping,
unfinished lines, rounds, turns, failures, illegal actions, and move latency.

Claim superiority to Opus separately by player count only when the final lower
confidence bound exceeds 1/2, 1/3, or 1/4. Mixed-table paired differences are a
separate result. Fresh random-anchored Bradley–Terry ratings are secondary;
they are not compared with historical ratings or used for final tuning.
Comparisons ignore arithmetic residue within 1e-12 of zero or the symmetric
baseline. This prevents an exact three-way shared draw from being labeled an
improvement because of floating-point rounding; it does not relax the criteria.

## Runtime validation

Measure equivalent Python/Rust transitions and round resolution separately from
snapshot conversion, binding transport, and full move selection. Do not infer a
search speed multiplier from transition benchmarks. Record wall-clock cutoffs.

Install the release wheel and select moves in both actual Lambda runtime images
(Python 3.12/AL2023 and the original 3.11/AL2 target). Test complete handler games
and Runtime Interface Emulator requests, including cold initialization and
consecutive bot turns. Local storage and local container timings do not estimate
AWS network, DynamoDB, or deployed cold-start latency. No live AWS deployment is
part of this work.

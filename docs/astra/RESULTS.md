# Astra Heuristic: measured results

Astra is a hand-coded Rust search agent with a Python adapter. These results use the frozen final configuration, independent holdout seeds, and the fixed reference field. No final-test outcome was used for another strategy revision.

## Final win share against Opus

| Players | Games | Astra win share | Block-bootstrap 95% CI | Symmetric baseline | Superiority established |
| --- | ---: | ---: | --- | ---: | --- |
| 2 | 512 | 99.02% | 98.05–99.80% | 50.00% | Yes |
| 3 | 768 | 56.08% | 52.80–59.40% | 33.33% | Yes |
| 4 | 1,024 | 47.12% | 44.19–50.05% | 25.00% | Yes |

One Astra plays against n−1 Opus opponents. Sole wins earn one; k-way shared wins earn 1/k. Intervals resample all seat rotations in a seed block together, 10,000 times. The test criterion was fixed in [PROTOCOL.md](PROTOCOL.md).

## Mixed-table control

| Players | Games per policy | Astra share | Opus control share | Paired Astra improvement (95% CI) |
| --- | ---: | ---: | ---: | --- |
| 2 | 128 | 100.00% | 75.39% | +24.61 pp (+18.75 to +30.47) |
| 3 | 192 | 66.93% | 46.35% | +20.57 pp (+10.16 to +30.73) |
| 4 | 256 | 59.57% | 37.11% | +22.46 pp (+14.45 to +30.47) |

Astra and an Opus control occupy the same designated seat under matching deals and opponent rotations. Other seats cycle through Opus, basic heuristic, and random. The control comparison is separate from the homogeneous Opus claim.

## Game quality and latency

The following means describe Astra in the final homogeneous Opus games. Floor losses are scheduled penalties before score clamping. Margin compares Astra with the highest-scoring opponent.

| Players | Score | Margin | Floor loss | Unfinished lines | Rounds | Turns | Mean / p95 / max move (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 61.31 | +29.19 | 8.29 | 1.36 | 5.00 | 52.34 | 169.13 / 668.45 / 1343.13 |
| 3 | 66.42 | +1.79 | 7.79 | 1.27 | 5.00 | 70.17 | 3.23 / 6.06 / 10.48 |
| 4 | 69.29 | -0.87 | 7.98 | 1.18 | 5.00 | 87.13 | 3.65 / 6.14 / 11.64 |

Across 4,608 final games: 0 unfinished, 0 stalled, 0 agent failures, and 0 illegal actions. Unfinished games receive no winner and remain in the raw records.

| Players | Mean visited nodes | Mean completed depth | Exact rounds solved | Time cutoffs / searches |
| --- | ---: | ---: | ---: | ---: |
| 2 | 105699 | 5.39 | 6,509 | 0 / 13,187 |
| 3 | 3022 | 2.30 | 4,964 | 0 / 18,208 |
| 4 | 3242 | 2.13 | 4,916 | 0 / 23,316 |

A solved round ends before a random refill; it does not solve the full game. A `pruned_round` stop is never reported as exact. Fixed node limits are reproducible within the frozen runtime; wall-clock cutoffs can vary with host load.

## Frozen configuration

| Players | Regular nodes | Reply width | Terminal nodes | Initiative | Horizon multiplier (`urgency`) | Attainability prior |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 256,000 | 12 | 0 | 0.6 | 1 | 0 |
| 3 | 4,000 | 12 | 0 | 0.6 | 0.7 | 1.0 |
| 4 | 4,000 | 12 | 0 | 1.2 | 0.7 | 1.0 |

All selected configurations reserve 50 ms of the two-second ceiling for the boundary, requesting at most 1,950 ms of native search. Root actions are all evaluated. Two players use alpha-beta minimax; multiplayer uses Max-N. No neural model, automated parameter optimizer, runtime LLM, or future draw stream is used by Astra.

Frozen at **2026-09-14 06:57:24 UTC**, 7.89 hours after development began. The original Python 3.11 Lambda handler passed complete-game checks before the two-hour feasibility checkpoint.

## Equivalent Python and Rust operations

The compact Python oracle and Rust execute the same public-state transition or pre-refill resolution. Native operation times include validation and bindings. These measurements do **not** imply a Python-versus-Rust search speed multiplier.
Measurements used an AMD Ryzen 7 5800X host, Python 3.12.3, and the recorded
research environment. The benchmark ran after the tournament workers finished;
wall-clock timings remain specific to the host and its load.

| Players / position | Python transition (µs) | Rust + binding (µs) | Ratio | Tensor snapshot (µs) | Play snapshot (µs) | Play-engine selection (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 / opening | 12.24 | 1.68 | 7.30× | 25.04 | 0.72 | 749.51 |
| 2 / late_round | 3.58 | 1.62 | 2.21× | 25.23 | 0.74 | 0.09 |
| 2 / middle | 11.65 | 1.64 | 7.10× | 25.32 | 0.73 | 533.41 |
| 3 / opening | 21.61 | 2.00 | 10.82× | 26.30 | 0.81 | 4.70 |
| 3 / late_round | 4.61 | 1.87 | 2.46× | 26.27 | 0.78 | 2.41 |
| 3 / middle | 20.06 | 1.96 | 10.22× | 27.01 | 0.81 | 3.21 |
| 4 / opening | 26.60 | 2.29 | 11.64× | 23.34 | 0.88 | 5.09 |
| 4 / late_round | 4.53 | 2.09 | 2.16× | 23.15 | 0.94 | 0.23 |
| 4 / middle | 24.31 | 2.27 | 10.69× | 23.78 | 0.93 | 3.14 |

Pre-refill round resolution ranged from 3.12× to 15.60× faster in Rust in these positions. Integer-array binding roundtrips ranged from 1.21 to 1.66 µs. The full benchmark JSON retains each snapshot, five complete-move timing samples, nodes, depth, values, and principal variation.

## Local Lambda responses

Both images use x86-64 portable instructions, pinned Rust/PyO3/maturin, release wheels, and no runtime Rust compiler or build directory. Each validation container was limited to one CPU and 2 GiB, with a read-only root filesystem and temporary local storage.

| Runtime | Players | Complete AI requests | Mean / p95 / max response (ms) | Longest consecutive bot sequence (ms) |
| --- | ---: | ---: | --- | ---: |
| Python 3.12 / AL2023 | 2 | 25 | 214.34 / 835.81 / 1206.90 | 1206.90 |
| Python 3.12 / AL2023 | 3 | 46 | 4.15 / 8.09 / 8.98 | 16.73 |
| Python 3.12 / AL2023 | 4 | 68 | 4.53 / 7.51 / 8.76 | 24.06 |
| Python 3.11 / AL2 | 2 | 27 | 207.16 / 771.64 / 1287.13 | 1287.13 |
| Python 3.11 / AL2 | 3 | 47 | 4.43 / 7.35 / 8.34 | 15.31 |
| Python 3.11 / AL2 | 4 | 70 | 5.12 / 7.78 / 9.39 | 25.00 |

Python 3.12: the first resumed AI request through a restarted Runtime Interface Emulator took **1835.47 ms**, including application initialization; the warm opening-response maximum was **1272.52 ms**.

Python 3.11: the first resumed AI request through a restarted Runtime Interface Emulator took **2210.76 ms**, including application initialization; the warm opening-response maximum was **765.64 ms**.

These are complete local handler/HTTP responses and consecutive turns. They exclude AWS network, DynamoDB, and deployed cold starts. No AWS deployment was performed.
**The Python 3.11 cold response exceeded two seconds.** The agent's search and
warm responses stayed within the move ceiling; that ceiling does not bound
application initialization or complete deployed cold-start latency.

## Secondary opponents and fresh ratings

| Players | Random share | Basic heuristic share | Frozen checkpoint search32 share | Self-play share |
| --- | ---: | ---: | ---: | ---: |
| 2 | 100.00% | 100.00% | 89.06% | 50.00% |
| 3 | 100.00% | 47.40% | 100.00% | 33.33% |
| 4 | 100.00% | 43.75% | 100.00% | 25.00% |

Each secondary matchup uses 32 seed blocks: 64, 96, and 128 games at two, three, and four players. The two-player checkpoint result is 89.06% (95% block-bootstrap CI 79.69–96.88%). All-success bootstrap intervals are degenerate and do not prove perfect play. The available strongest checkpoint was selected by per-count rating, with an overall fallback for multiplayer. Its multiplayer heads are weak; their results provide limited evidence against competitive multiplayer neural agents. The checkpoint bytes and inference sources are frozen and recorded in each manifest.

| Player count | Fresh Bradley–Terry ratings (random = 1000) |
| --- | --- |
| 2 | astra: 5967; checkpoint: 5055; opus: 3905; heuristic: 3355; random: 1000 |
| 3 | astra: 5401; opus: 5097; heuristic: 5037; random: 1000; checkpoint: 673 |
| 4 | astra: 5845; opus: 5560; heuristic: 5408; random: 1000; checkpoint: 765 |

Ratings use final score/row pairwise placements, half-credit ties, base-10 scale 1000, and a weak Gaussian prior to keep separated results finite. Correlated multiplayer pairs and very weak opponents limit their interpretation. They are not comparable to historical league rating numbers.

## Revisions, ablations, and remaining weaknesses

See [ABLATIONS.md](ABLATIONS.md) for the complete matched feature removals, [REVISIONS.md](REVISIONS.md) for each manually specified hypothesis and retention decision, and [LOSS_REVIEW.md](LOSS_REVIEW.md) for recorded failure analysis. The accompanying experiment index and raw records contain paired intervals and latency for every screen and promotion. Finalist feature removals are exploratory; a positive screen alone never changes production.

The strongest retained ideas were less optimistic unfinished-line credit, attainable bonus estimates in multiplayer, a longer multiplayer horizon, and separately measured search budgets. Larger general search, greedy round rollouts, stronger generic adjacency/bonus weights, and several opponent-pressure variants did not reliably improve the fixed field.

The search sees only the current round. Future supply, game length, and bonus attainability remain heuristic. Multiplayer Max-N models opponents as independently maximizing its evaluator; actual opponents can choose differently. There is no universal-strength or professional-level claim beyond the measured field and player counts. Strategy sources and access limitations are documented in [SOURCES.md](SOURCES.md).

## Reproduction and validation

Build/install and resume commands are in [README.md](README.md) and the native build guide. The delivery bundle contains raw experiment records, manifests, final frozen runtimes/checkpoints, release wheels, validation evidence, and these reports. The existing league and unrelated workspace changes were preserved.

Validation evidence includes the complete Python suite, Rust unit tests and lint, differential pre-refill checks, tactical fixtures, clean wheel installation without Torch/NumPy, frontend production build, browser games, both Lambda runtime images, and the local Runtime Interface Emulator. Exact logs and artifact hashes accompany the delivery.
The final Python suite passed **314 tests**. The unchanged Rust source passed
**14 unit tests**, Clippy with warnings denied, and the formatting check. Browser
acceptance completed three games across all player counts with no page errors;
both Lambda images completed three handler games each. Clean wheel checks verified
the frozen configuration hash. `validation-summary.json` records these checks,
image identities, and the cold-response limitation.

Development, evaluation, and validation completed in approximately **8.48 hours**,
within the requested budget. The experiment index contains **127 experiments and
86,418 game records**, including the 4,608 held-out final games.

The portable delivery archive passed ZIP integrity and path checks. All four
exported final runtimes passed their recorded source, native-extension, and
checkpoint hashes. Removing one game from an extracted copy and resuming it
reproduced the complete game and search results exactly, excluding elapsed times;
the original final records were unchanged. The archive is
`agent/runs/astra/astra-delivery.zip`; its SHA-256 checksum is provided alongside it.

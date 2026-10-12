# Shared-core 2p/3p/4p experiment

This experiment consolidates the three retained width-256 specialists into one
fully trainable source-attention backbone and separate policy/value heads. Two-player
move choices may change; retention depends on playing strength. The specialists
remain immutable references and fallbacks. There is no automatic deployment.

## Consolidation and online training

Initialization uses the 2p backbone and copies each specialist's own heads and
player-count embedding. It uses a fresh optimizer. Donated observations come from
the preserved 2p smoke fork and the final full 3p/4p resumes. The selected 3p
teacher is the earlier 120-minute weights file; using later replay observations
is an explicit dataset donation, not an exact training resume.

For each count, 100,000 training and 4,096 validation positions are sampled without
index overlap from its donor. These states are relabelled with the selected
specialist's full policy distribution and value vector. Fresh mixed-table teacher
games add coverage. Validation states can be correlated with training states
from the same historical games; these imitation diagnostics do not measure strength.

Consolidation gives each count equal loss weight and fits policy KL plus active-seat
value MSE. It uses no dropout, a 0.0001 learning rate, at most one hour / 20,000 joint
updates, and checks held-out imitation loss every 250 updates. The best worst-count
loss checkpoint is retained. Early stopping uses KL < 0.025 and value MSE < 0.006
for every count, or eight checks without improvement.

Online learning uses three separate, fresh 200,000-position replay buffers. Each
round generates 96 games per count; odd rounds use current-model self-play, even
rounds use mixed tables sampled by the panel weights. Historical shared checkpoints
replace specialist opponents on alternate mixed rounds. Search uses 64 simulations,
or 256 every fourth round, and Rust tree search with inference caching disabled.
Every fourth round reanalyses 256 positions per count. Only finished-game candidate
positions enter mixed-table replay. Rewards remain +1/0/−1; terminal targets in this
campaign are undiscounted for all counts.

Every joint optimizer step accumulates one homogeneous minibatch per count with
equal 1/3 weights. Each count mixes 85% RL loss with 15% frozen teacher policy/value
imitation from an independent anchor batch. The learning rate is 0.00015, batch size
256 per count, and each round performs 72 joint steps. Policy targets from fewer
than 256 simulations receive the existing 0.25 relative weight. The checkpoint
contains all heads, one optimizer, all three buffers, counters, and RNG states.
Corpora are separately immutable and saved once. A full resume is verified by tests.

## Mixed tables and weighted win rates

Each table has one candidate seat. Evaluation replaces only that seat when comparing
the shared network with a specialist. All other players, checkpoint identities,
search budgets, draw seeds, and table weights stay fixed. Every seed block contains
all unique seat permutations, including opponent orderings. This gives 2, 3/6, or
12/24 games per seed depending on repeated opponents.

The score is win share: 1 for a sole win, 1/k for one of k actual shared winners,
and 0 for a loss. The primary weighted win rate is the fixed-weight mean of table
win shares. Individual table results and sole-win rates remain available.

| Count | Opponent table | Weight |
|---|---|---:|
| 2p | specialist | 50% |
| 2p | Astra | 35% |
| 2p | Opus | 15% |
| 3p | Astra + specialist | 40% |
| 3p | Astra + Opus | 30% |
| 3p | specialist + Opus | 20% |
| 3p | Astra + Astra | 10% |
| 4p | Astra + Opus + specialist | 40% |
| 4p | Astra + Astra + specialist | 30% |
| 4p | Astra + Opus + Opus | 20% |
| 4p | specialist + specialist + Astra | 10% |

These are a declared opponent distribution, not universal playing-strength weights.
There is no result-dependent rating reweighting. More games in a table do not
increase its headline weight. Seed blocks are jointly resampled across tables;
shared-versus-specialist differences are paired on identical blocks. Incomplete
blocks invalidate the evaluation instead of silently changing weights. Intervals
capture game sampling uncertainty, not variation across independent training seeds.

The main profile is 1,024 simulations for all neural seats. Each specialist first
passed 24 CPU timing positions at an internal 1.8-second / outer 2-second limit.
The selected shared model receives 48 fresh CPU timing positions per count.
Batched CUDA evaluation measures the same fixed simulation profile; latency
qualification is a separate CPU test and is not a universal worst-case guarantee.

Development uses 12 seed blocks per table and fixed seeds across checkpoints.
Screens occur after consolidation and approximately every three hours of online
training. Checkpoints with all three point differences at least −2 percentage points
are eligible for selection; the highest mean per-count difference selects one.
If none qualify, the distilled checkpoint is evaluated as an experimental fallback.
A resolved loss greater than five points in any consolidation screen prevents
online RL from starting. Selection screens are diagnostic and noisy.

Confirmation uses fresh seeds and 128 blocks per table for 2p, 64 for 3p/4p.
A separate 2,048-game 2p direct match compares the selected shared model with the
2p specialist. Experimental retention requires every weighted difference's lower
95% bound above −2 percentage points, direct 2p match-score lower bound above
48%, and every CPU timing profile qualified. This two-point practical tolerance
is predeclared for the experiment, not a claim of exact equal strength. Failure
or inconclusive results retain the specialists. Confirmation never selects an
alternative checkpoint. Positive gains are reported with their own uncertainty.

The authorized 24-hour wall window includes setup and validation. Online training
ends six hours before the deadline to reserve time for confirmation. The service
has no automatic restart and reserves five minutes for graceful saving. Disk,
frozen teacher hashes, and the native extension are checked throughout. Full states
are atomically compressed; only one latest resume is retained, plus small weights
milestones and reports. See the training journal for launch and measured outcomes.

## Completing an interrupted confirmation

`agent.scripts.complete_shared_evaluation` completes the frozen selection's missing
confirmations without calling a training loop, loading replay into a learner, or
selecting a new candidate. It has a separate manifest and service; the original
campaign plan, wall deadline, stop record, saved training state, and completed
2p/3p reports remain unchanged. The October 10 continuation runs the originally
specified 2,048-game direct 2p match first, then pairs the candidate and specialist
at each remaining 4p table. It retains the original seeds, 96-game batch size,
1,024-simulation search, table weights, and all-count acceptance criteria.

Evaluation batches are saved atomically and fsynced before acknowledging progress.
Each batch is bound to checkpoint identities, source/native hashes, search settings,
and the complete seat/draw schedule. A resume validates a contiguous sequence of
finished original batches and starts at the original global batch offset, preserving
search RNG seeds. The current batch may need repeating after a forced stop; completed
batches and completed table reports are reused. Partial or inconsistent seed blocks
cannot enter a final weighted summary. A file lock prevents duplicate runners.

Original-versus-updated evaluation and interrupted-versus-uninterrupted games were
compared directly. These are reproducibility checks, not evidence of model strength.
The completion has an 18-hour maximum runtime with a 15-minute graceful-stop reserve
for roughly 12 hours of expected evaluation work. It writes the final `result.json`,
`FINAL_REPORT.md`, and journal closeout without deploying a model.

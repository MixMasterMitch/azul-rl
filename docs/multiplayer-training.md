# Multiplayer foundation training

The first 3p/4p campaign trains independent width-256 source-attention models.
The existing 2p checkpoint remains an immutable route in a player-count bundle.
Separate policy/value heads alone would not protect 2p: training the shared
backbone changes its inputs. Independent models allow the whole backbone to
adapt to each new game without changing two-player weights or serving behavior.
This campaign does not deploy any model or modify the play server.

## Initialization and training

Each specialist copies the 2p backbone, source-aware policy head, and player-count
embedding into its target player-count slot. It copies the value head's hidden
layer but resets the final readout: all active seats initially predict `2/N - 1`,
the symmetric no-tie expectation of the +1/0/−1 reward. An unused extra value
output from the 2p model is not evidence about a third or fourth player.

Both specialists start with fresh optimizers, independent empty 500,000-position
replay buffers, and separate leagues. They use the established 64/256-simulation
Rust search recipe, 256 games per iteration, 72 learner updates, 0.25 fast-policy
loss weight, and 256 policy reanalyses every four iterations. Replay sampling is
uniform. Auxiliary score targets and teacher distillation are disabled. Training
cycles through ordinary self-play, its own league, and bot games (50% Astra,
25% Opus, 25% heuristic). Stalled games are excluded from replay. Native inference
caching remains disabled.

The runner alternates player counts at 30-minute, 120-minute, and final cumulative
training milestones, retaining full optimizer/replay/RNG state. After the smoke
checks, 60% of the remaining wall window is split equally between the two player
counts. The rest covers development screens, held-out evaluations, saves, and
CPU timing. A persistent systemd service supplies a runtime cap and graceful-stop
window; no automatic restart is configured. Each stage verifies source and frozen
2p checkpoint hashes, the installed native binary, and a 6 GiB free-space floor.

## Evaluation

The multiplayer arena seats one candidate against `N-1` copies of a fixed
opponent. Every draw seed receives all candidate-seat rotations. Whole rotations,
not individual seats, are resampled for uncertainty intervals. Opponent bot RNGs
are separated by their seat relative to the candidate.

The primary result is **win share**: a sole winner gets 1, each of `k` actual
shared winners gets `1/k`, and everyone else gets 0. Equal-strength reference
shares are 1/3 and 1/4. Sole-win rate and shared-win count are also recorded.
This evaluation measure does not change training rewards: shared winners get 0
and losers get −1, including losers in games with a shared first place.

Development screens use 96 games per player count/opponent at 256 simulations
against heuristic, Opus, and Astra, using the same seeds across checkpoints.
The mean win share across this fixed panel selects one checkpoint per player
count. Fresh held-out seeds then compare that frozen selection and its transferred
initializer against Opus and Astra, 384 games per model/opponent. No alternative
checkpoint is selected using confirmation results. These games measure progress
over initialization, not universal multiplayer superiority or robustness across
training seeds. Mixed-opponent tables are future work.

Finally, opening/middle/late positions from eight CPU reference games per player
count test the 256-simulation profile under a two-second outer turn budget and
1.8-second internal stop. Qualification requires the complete requested search
budget on all 24 positions. It is a local sample-based timing check, not a universal
latency guarantee. The candidate bundle and reports are experimental artifacts;
timing or strength results do not automatically deploy a model.

See the [training journal](training-journal.md) for live run links and results.

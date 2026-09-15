# Competitive two-player training

The first milestone is a corrected v4 baseline and a controlled comparison of the
existing `attn` model with `source_attn`. Model width remains 256. The code supports
the later experiments, but their outcomes must be measured before changing the
default champion. Three- and four-player learned agents remain outside this campaign.

New evaluation jobs include production Astra in 12.5% of opponent seats, and new
competitive screening passes include a paired Astra match. Existing frozen
experiments retain their original opponent mix on resume. See
[Astra evaluation controls and metrics](astra/TRAINING_EVALUATION.md).

## Corrected baseline, September 13, 2026

These are fresh development screens with CPU simulation, CUDA inference, 64
one-ply candidates, temperature 0.25, Q scale 28, binary terminal rewards, and
256 games per match. Each consecutive pair swaps seats and shares a draw seed.
Shared victories count as half a point for evaluation and +1 for each winning
player's binary training utility. No games were unfinished.

| Checkpoint | Opus W / L / shared | Opus match score | Standard heuristic W / L / shared |
| --- | --- | --- | --- |
| v3 iteration 5700 | 212 / 42 / 2 | 83.20% | 219 / 36 / 1 |
| v4 iteration 2350 | 219 / 36 / 1 | 85.74% | 226 / 29 / 1 |
| v4 historical rating peak, iteration 1350 | 213 / 41 / 2 | 83.59% | 220 / 32 / 4 |

v4 iteration 2350 remains the provisional champion. Its paired bootstrap 95%
interval against Opus is 81.45–89.84%. This screen does not establish a statistically
conclusive ordering of all three checkpoints or competitive-human strength.

The frozen weights and report are in `agent/runs/competitive/baselines/` and
`agent/runs/competitive/baseline.json`. Detailed evaluation JSON files include
checkpoint SHA-256 hashes, the full protocol, per-game outcomes, seats, draw seeds,
value calibration, and code/dependency provenance. Earlier audit matches used a
different protocol and the old search transition, so their results should not be
pooled with these corrected matches.

All six matches were repeated with identical per-game records. Final validation
completed with 202 passing pytest tests and a successful frontend production build.

## Implementation

- All production modes now use `agent.env.engine.GameEngine`: persistent Rust
  simulation and feature encoding on CPU, with CPU/CUDA network inference and
  learning. Install the extension with `python -m pip install ./native/astra`.
  Existing weights and replay remain compatible. Native RNG streams change
  same-seed trajectories relative to the PyTorch baseline; historical measurements
  below retain their original backend/protocol. See the [simulator guide](../native/astra/SIMULATOR.md).
- Search delegates round completion to `BatchedEngine.finalize_round()`. Terminal
  values come directly from the same shared reward helper used by self-play.
- Trajectory parity compares scores, boards, floor state, turn ownership, winners,
  and deterministic source/discard state before synchronization. Only stochastic
  refill distributions are synchronized, after checking per-color inventory.
  Tests cover 2–4 players on CPU and CUDA.
- `source_attn` retains individual contextual source embeddings. A shared 30-logit
  head acts on each source plus player/global context. A center type embedding
  distinguishes the center, inactive source slots are masked, and value prediction
  retains pooled context. Factory permutations permute policy blocks and preserve
  values. The source policy input is normalized, with small output initialization
  to tolerate the large residual embeddings in pretrained v4 weights.
- The `attn` → `source_attn` initializer copies compatible trunk/value weights and
  initializes the new policy head. Each architecture arm starts with fresh replay
  and optimizer state. Old encoder checkpoints still use the compatibility loader.
- `one_ply` is the corrected breadth-only baseline. `gumbel_tree` adds Gumbel root
  sequential halving, repeated traversal, visit/value backups, mixed-value Q
  completion, and interior improved-policy matching. Values are stored in absolute
  player coordinates. Refill edges average a bounded set of independent sampled
  outcomes; neither backend reads the live game's future draw stream.
- `SearchConfig` is shared across training, evaluation, and play. Leaf inference
  is batched. Deadline checks occur between batches; an already-running tensor
  operation cannot be preempted. The serving qualification rejects profiles that
  cannot complete their requested budget on sampled opening/middle/end positions
  within five seconds. This is an empirical qualification, not a real-time OS
  guarantee under arbitrary machine load.
- Evaluation jobs retain their checkpoint/opponent identity, preserve partial
  worker results, aggregate counts, and reject failed or incomplete jobs. Scheduled
  CUDA evaluations run between training phases and preserve the learner's RNG.
- Tuning defaults to observed final evaluation, in a separate study namespace.
  Legacy projections remain available only through explicit `--selection-metric projected`.

The architectural motivation follows [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html).
Tree selection and Q completion follow [Danihelka et al., 2022](https://openreview.net/forum?id=bERaNdoegnO)
and the [DeepMind mctx reference](https://github.com/google-deepmind/mctx).
The sampled refill treatment is this project's stochastic-game adaptation.

## Local diagnostics

Hardware: RTX 3080 Ti (12 GB), Ryzen 5800X, 32 GB RAM. The environment uses Python
3.12.3 and PyTorch 2.12.0+cu130. Learning remains FP32 with TF32 enabled; the AMP
flag does not enable mixed-precision learner updates.

A warm-started `source_attn` diagnostic with a million-position replay allocation
completed 1,023/1,023 games in 9.41 seconds and 72 learner updates in 0.67 seconds.
Peak CUDA allocation was 5.70 GiB. All updates and weights were finite, with no
skipped updates. Mean policy entropy was 2.33 nats. This is a short pipeline and
throughput check, not evidence of improvement after training.

Single-game CPU inference on six positions from two deterministic heuristic games:

| Search | p95 move time | Full requested budget completed |
| --- | ---: | --- |
| One-ply, 64 candidates | 0.020 seconds | 6/6 |
| Tree, 64 simulations | 0.205 seconds | 6/6 |
| Tree, 256 simulations | 1.079 seconds | 6/6 |
| Tree, 1024 simulations | 3.894 seconds | 6/6 |

Diagnostics are stored in `agent/runs/competitive/diagnostics/`. Tree strength
remains unmeasured until the search experiment; deeper traversal alone is not a
promotion criterion. The campaign reruns latency qualification on its actual
candidate. In a 256-game mixed bot phase, scalar Opus selection consumed 3.63 of
7.66 seconds over 2,743 moves; neural search consumed 2.47 seconds. The phase
completed every game. Opus has its own timing and position counters now, making
this Python path a concrete target before further GPU optimization.

## Run the experiments

Run from the repository root with `.venv` activated. Each invocation locks its
campaign directory, resumes completed work, and enforces an eight-hour process
ceiling. A stop request arrives at 478 minutes to allow an iteration-boundary save;
an unresponsive process is killed at 480 minutes, preserving its last atomic
checkpoint. Each default policy/tree-training arm receives 235 training minutes;
each learner arm receives 115. The remaining time covers evaluation and storage.

```bash
source .venv/bin/activate
python -m agent.scripts.competitive baseline --device cuda
python -m agent.scripts.competitive policy --device cuda
python -m agent.scripts.competitive learner --device cuda
python -m agent.scripts.competitive search --device cuda
python -m agent.scripts.competitive tree-training --device cuda
python -m agent.scripts.competitive replicate --replicate-stage policy --device cuda
python -m agent.scripts.competitive status
```

Run one training stage per night. `replicate` accepts `policy`, `learner`, or
`tree_training` and repeats the winning configuration from its original initializer
with another training seed. It requires successful held-out confirmation first.
Use a separate `--root` for another campaign. `--minutes-per-arm` supports shorter
manual experiments; changing an existing experiment's configuration requires a
new run ID/root. An interrupted stage resumes with the same command and budget.

The policy stage compares both architectures under the same saved v4 optimizer
hyperparameters and exploration settings. The learner stage compares 72/144/288
updates. The search stage evaluates full fixed budgets and independently checks
the five-second serving profile; it does not share one five-second allowance
across an entire arena batch. Tree training runs only after tree search passes its
gate, comparing 64 tree simulations with the 32-candidate one-ply teacher.

Every screen uses 256 development games against Opus, the standard heuristic,
the frozen champion, and v3. Both policy arms also receive greedy Opus evaluation.
The selected finalist receives 1,024 held-out games per opponent; confirmation
and replication seeds are separate from development seeds. Promotion requires:

- At least 55% match score against the frozen champion, with a paired 95% interval
  entirely above 50%.
- A paired Opus difference interval whose lower bound is at least −3 percentage
  points relative to the champion under the same seeds and protocol.
- Complete, ordered seat-swapped game records, matching checkpoint identities and
  search settings, legal actions, and zero unfinished confirmation games.
- Successful confirmation after repeating the training configuration with another
  seed, plus serving latency qualification.

No promotion occurs solely from a league rating or development screen. The local
95%-against-Opus milestone and competitive-human claims need further evidence.

## Storage, recovery, and monitoring

A million-position FP32 replay snapshot contains approximately 2.63 GiB of raw
tensors. Resume checkpoints now use lossless ZIP compression: a full-buffer probe
populated from 55,352 real self-play positions produced a **246 MiB** checkpoint,
with exact tensor recovery and successful atomic replacement. Sizes vary with
replay contents. The launcher keeps approximately **3.14 GiB free** as initial
headroom, and each compressed write checks remaining space before replacing the
old checkpoint. Compression uses temporary host RAM; the local 32 GB setup passed
the full-buffer probe. Training precision, capacity, and hyperparameters are
unchanged. The campaign does not delete historical v2/v3/v4 checkpoints.

Compressed checkpoints still load with ordinary `torch.load` and the project
checkpoint helpers. Do not use `mmap=True` on compressed archives. Lightweight
weight snapshots retain the original uncompressed format.

Active experiments keep one full `latest_resume.pt`, two recent model/optimizer
snapshots, and an isolated league with its frozen initializer pinned. Completed
experiments keep a lightweight finalist and retire their own full replay snapshot.
Ordinary interruption saves at an iteration boundary and updates state/heartbeat
after checkpoint replacement. Forced termination resumes from the last successful
checkpoint and may lose work since that checkpoint.

Checkpoints contain Python, NumPy, Torch, and CUDA RNG states, replay counters and
sample ages, effective configuration, model/encoder versions, dependency versions,
git revision, and a hash of dirty tracked/untracked code. Exact continuation is
tested for the same execution environment; cross-version/device bitwise identity
is not promised. Source changes during a campaign should be avoided even though
their provenance is recorded.

Inspect `status.json` for the current campaign stage, each experiment's
`heartbeat.json` for progress, and `metrics.jsonl`/`events.log` for losses, entropy,
value bias/sign accuracy, replay reuse/age, unfinished games, phase throughput,
and CUDA memory. Full timing/counter profiles are collected every 100 iterations.
Evaluation reports add value calibration and paired confidence intervals.

### September 14 recovery and quiet supervision

The first policy attempt stopped at iteration 100 after its gradient norms grew
rapidly. A replay of the saved iteration-50 state reproduced an unstable batch at
iteration 78. Pooling attention's fused CUDA backward produced a gradient norm of
249,065; explicit softmax produced 1.30. With dropout disabled, the same forward
loss produced norms of 21.29 versus 1.23546, and the math attention reference also
produced 1.23546. The large source residuals and pooling projections inherited from
v4 expose numerical cancellation in the fused backward path. Pooling now uses the
explicit attention implementation used by the original v4 training. Transformer
self-attention, model dimensions, optimizer settings and replay capacity remain
the same. Invalid learner updates now log their failure stage and stop immediately,
before replacing a durable checkpoint.

The failed arm and its resume checkpoint are archived under
`diagnostics/failed_sdpa_policy_attn_seed20260913/`. Both architecture arms restart
from the frozen v4 initializer under the corrected code, preserving the matched
comparison. The captured batch and diagnostic reports are in `diagnostics/`.

Disk pressure principally came from local Docker build/deployment images. Cleanup
removed four unused build containers, dangling images, and 36 old local deployment
images. Available space increased from approximately 0.46 GiB to 10.13 GiB before
subsequent local builds and checkpoint writes. Tagged current release/validation
images, active Python environments and historical training checkpoints were kept.
The deletion audit is `diagnostics/disk_cleanup_20260914.json`.

The local user service `azul-training.service` runs
`python -m agent.scripts.supervise_training`. It sends no notifications or external
messages. Full health checks run every **five minutes**, switching to **30 minutes**
after three consecutive checks show advancing four-phase cycles, finite learner
metrics and no skipped updates. A new process or loss of stable progress returns
checks to five minutes. Process exit wakes the supervisor immediately, independent
of the health-check interval.

The supervisor resumes the same experiment after a crash, gracefully restarts a
stalled child, checks storage, and removes old dangling Docker images and abandoned
checkpoint temporary files when space is tight. It preserves checkpoint/model
settings and never promotes a model. Four failed attempts without durable progress
stop retrying instead of looping indefinitely over the same invalid state. All
attempts share one persisted eight-hour deadline; restarts cannot reset the budget.

```bash
systemctl --user status azul-training.service
systemctl --user stop azul-training.service  # finish the iteration and save
systemctl --user start azul-training.service # resume within the existing window
cat agent/runs/competitive/health.json
cat agent/runs/competitive/supervisor.json
```

`supervisor_events.jsonl` records health checks, recovery attempts and interval
changes. `health.json` includes the next check time, checkpoint age/size, free disk,
learner loss/gradient norm and iteration. Completion or expiry ends this overnight
window; starting another experimental stage requires a new supervisor window.

## Play the frozen or promoted model

The registry stores immutable model artifacts, checksums, supported player counts,
and the evaluated search profile. `baseline` registers the provisional v4 model;
only successful replication changes it to a promoted model. Local/release catalogs
may have their own default registry, so select this campaign explicitly:

```bash
AZUL_MODEL_REGISTRY="$PWD/agent/runs/competitive/models/registry.json" azul-play
```

Restart the server after promotion so its catalog loads the new registry. The
frontend discovers compatible opponents from `/api/opponents`. The competitive
registry exposes learned agents for two-player games only. Move selection uses
the common search implementation and a maximum configured deadline of five seconds.
Game inputs and action indices are validated by the play API.

For an independent report without modifying leagues:

```bash
python -m agent.scripts.arena agent/runs/competitive/baselines/v4_latest.pt opus \
  --device cuda --games 1024 --seed 303030 --split confirmation \
  --report agent/runs/competitive/evaluations/manual_opus.json
```

Use `--backend gumbel_tree --sims 256` to test fixed tree budgets. An explicit
`--deadline` evaluates each game separately to preserve per-move deadline semantics
and can be substantially slower than fixed-budget batched evaluation.

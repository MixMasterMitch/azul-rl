# Full Rust simulator

`agent.env.engine.GameEngine` is the default engine for all production modes.
It wraps the low-level CPU `agent.env.rust_engine.RustEngine`, backed by
`azul_astra.BatchEngine`. It holds each complete game in Rust across turns:
public boards/sources, bag and discard inventories, ordered floor slots,
first-player marker ownership, and a reproducible per-game random stream.
Moves, wall tiling, penalties, bonuses and tiebreaks reuse Astra's existing
public game mechanics. Nonterminal rounds recycle tiles and refill factories.

This is separate from the public-only `State` used by `HeuristicAstraBot`.
The tactical bot's version-1 snapshot and its round-boundary behavior are
unchanged. Full simulator snapshots include private information and must not
be passed to human clients or substituted for the bot's public snapshot.

## CPU example

```python
import torch
from agent.env.rust_engine import RustEngine
from agent.net.model import AzulNet

engine = RustEngine(64, num_players=3, seed=123)
model = AzulNet(hidden=32, arch="flat").to("cpu").eval()
with torch.no_grad():
    global_feat, source_feat, legal = engine.encode_state_with_legal()
    logits, values = model(global_feat, source_feat, legal, engine.num_players)
    engine.step(logits.argmax(dim=1))
```

The global feature tensor is `(B, 275)` float32, source tensor `(B, 10, 5)`
float32, and legal mask `(B, 300)` bool. Encoding matches `agent.net.encoder`,
including current-player perspective, padded inactive seats, bag/discard,
floor colors, scores and source alignment. Rust returns owned byte buffers;
PyTorch views retain their owners. Later steps cannot invalidate or modify
an already returned observation, and changing an observation does not mutate
the engine. Actions and inputs must be CPU tensors or Python sequences.
The engine does not select or initialize a GPU.

## Production CPU/CUDA adapter

```python
from agent.env.engine import GameEngine
from agent.net.encoder import encode_state_with_legal

engine = GameEngine(64, num_players=3, device="cuda", seed=123)
global_feat, source_feat, legal = encode_state_with_legal(engine)
engine.step(legal.long().argmax(1))
```

`device` controls returned tensors and network inference; simulation remains
in native CPU memory. Training, bot and league self-play, tuning, evaluation,
one-ply/tree search, and the play server all use this adapter. Encoding and
outcomes dispatch directly to Rust. One-ply expansion fuses native state copying
and stepping; tree search stacks native states. Both reseed hypothetical children
independently of live RNG streams. Profiling uses the equivalent unfused path to
count round boundaries.

Raw tensor attributes are cached **read-only observations by convention**, not
writable game state. Mutating them does not update the simulator. Use `step`,
`finalize_round`, or restore a validated `state_dict()` to change state. Bot and
UI reads use lazy tensor buffers; scalar bots can use `cpu_view()` to avoid GPU
scalar transfers. `from_batched`/`to_batched` copy CPU reference state for migration,
diagnostics and parity tests; they are not used inside normal stepping or search.

New play saves use schema 2 and contain native state plus RNG words. Schema-1
PyTorch saves migrate on load and preserve their global/per-game draw streams
using explicit refill uniforms. Migrated games still execute all rules in Rust.
Model/replay checkpoint formats are unchanged. Run provenance records the Rust
backend and installed extension version. Resuming an old training run starts new
self-play games with native RNG, so same-seed trajectories differ from old runs.

## Batches and search

- `step(actions, finalize_round=True)` accepts one action per game. Terminal
  rows are ignored. Invalid actions in live rows reject the entire batch
  before mutation.
- `step(..., finalize_round=False)` defers wall tiling and refill;
  `finalize_round()` subsequently resolves only empty-source, nonterminal
  games. Repeating finalization after a completed setup or terminal game is
  inert. Supply exhaustion can leave partially filled factories.
- `clone()`, `index_select(indices)` and `repeat_interleave(k)` copy both
  game and RNG state. Repeated games therefore have identical future draws
  when given identical actions.
- `expand(actions, game_seeds=None)` fuses state copying and fully finalized
  stepping for a `(B, K)` candidate array. Children are ordered parent-major.
  It leaves parents untouched. For independent search outcomes, pass **B×K
  fresh seeds** to avoid using the live game's future RNG state. `reseed`
  changes future random streams without changing the current board.
- `current_player`, `ended`, `scores`, `get_winners()` and
  `total_tile_count()` return CPU tensors. Winners follow existing engine
  conventions: -1 unfinished, -2 shared victory, otherwise absolute seat.
- `final_values("binary")` and `final_values("score_scaled")` return `(B, 4)`
  float32 training targets in absolute seat order, matching the existing
  outcome helpers (including shared victories and -1 inactive-seat padding).
- Zero-sized batches and selections are supported.

Native batch loops release Python's GIL; they are currently serial CPU loops.
No worker pool is started. This keeps thread use predictable and leaves
parallel scheduling to the caller.

## Randomness and exact parity

The simulator uses SplitMix64 with rejection sampling for unbiased integer
ranks in the current bag. Each game has an independent stream. The same
seed is **not** expected to reproduce PyTorch or Python `random` draws.
Selecting/reordering a batch does not change a game's future stream.

For reproducible differential tests, `step` and `finalize_round` accept
`draw_uniforms`: a `(B, num_factories*4)` array of float32 values in `[0, 1)`.
On refill, row b consumes these values by factory slot using
`floor(float32(u * bag_total))`, with bag recycling at the same point as the
reference engine. Unused draws are ignored. Overrides do not advance the
internal RNG. All supplied values and dimensions are checked before mutation.

The parity suite feeds identical uniforms to Rust, `BatchedEngine` and
`SingleEngine`. It compares every represented state field after every move
through complete games, including exact factory refills, per-color inventory,
floor order against the batched engine, scores, winners and encoded features.
It never resynchronizes state after initialization.

## Checkpoint format

`snapshots()` returns one version-2 integer list per game.
`RustEngine.from_snapshots(num_players, snapshots)` restores the exact game
and future random stream. This is a private simulator format, independent of
training checkpoint formats.

For N players, each row contains:

1. The version-1 public snapshot layout (57 + 14×N integers), with its version
   field changed to **2**. See `agent.eval.heuristic_astra.snapshot`.
2. Five bag counts and five discard counts.
3. For each active player: five floor color counts, then seven ordered floor
   slots (-1 empty, 0–4 colors, 5 marker).
4. The high and low unsigned 32-bit words of the SplitMix64 state.

The total row length is 69 + 26×N. Restoring validates dimensions, value
ranges, pattern/wall consistency, floor order and marker ownership, terminal
state consistency and exactly 20 tiles of each color. Restore is all-or-nothing.
Native and PyTorch RNG formats differ, so `from_batched(..., seed=...)` and
`to_batched(..., seed=...)` seed new destination streams rather than claiming
to preserve the source RNG. Use native snapshots for exact continuation.

## Validation without touching a GPU or the installed extension

From the repository root:

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CARGO_BUILD_JOBS=2
export CARGO_TARGET_DIR=/tmp/azul-rust-simulator-target
cargo +1.90.0 test --locked --manifest-path native/astra/Cargo.toml
cargo +1.90.0 clippy --locked --manifest-path native/astra/Cargo.toml --all-targets -- -D warnings
PYO3_PYTHON="$PWD/.venv/bin/python" cargo +1.90.0 build --release --locked \
  --features extension-module --manifest-path native/astra/Cargo.toml
mkdir -p /tmp/azul-rust-simulator-runtime
cp "$CARGO_TARGET_DIR/release/libazul_astra.so" /tmp/azul-rust-simulator-runtime/azul_astra.so
PYTHONPATH="/tmp/azul-rust-simulator-runtime:$PWD" .venv/bin/python -m pytest \
  agent/tests/test_rust_engine.py agent/tests/test_astra.py
```

The last two commands use the Linux extension filename. This isolated build
avoids replacing a native library imported by another process. Installing a
wheel or using `maturin develop` is a separate step after active users of the
old extension have finished.

## Default-backend validation (September 13, 2026)

The default switch passed 277 Python tests across the full suite and targeted
round-boundary checks, seven Rust tests, Clippy with warnings denied, and the
three-iteration CUDA training smoke test. Coverage includes 2p/3p/4p, both search
backends, bot/league self-play, learner updates, saved-game migration, and complete
games with exact legacy draw continuation. A separate CUDA parity run compared
24 complete games (351 batch turns) using identical refill uniforms.

On the RTX 3080 Ti, three paired trials with the existing v4 checkpoint,
1,023 games, 32 candidates, and 72 learner updates gave median self-play times
of 8.779 s (PyTorch CUDA simulator) and 8.090 s (Rust + CUDA inference).
Median self-play-plus-learner cycle time was 9.322 s versus 8.603 s, a **1.084×
speedup**. Every game finished. Timing includes transfers; each engine uses its
own RNG after the identical initial states. This measures throughput, not learned
playing strength. The separate profiled runs are excluded from these medians.

Reproduce with the production adapter and explicit PyTorch reference:

```bash
python -m agent.scripts.benchmark_rust_training --repeats 3 \
  --output agent/runs/rust_benchmark/gpu_default_engine.json
```

Pass `--checkpoint` and `--config` when the default local v4 artifacts are absent.

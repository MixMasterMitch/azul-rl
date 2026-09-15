# Astra native core

This extension provides the default full-game simulator and Astra, a hand-coded
Azul evaluator and bounded tactical search. The tactical search receives only
public information and simulates the current drafting round; it never receives
the simulator's private bag or future draw stream.

The separate [full simulator](SIMULATOR.md) adds persistent CPU batches,
bag/discard/refill state, reproducible RNG, native neural feature encoding,
and complete-game parity tests. It is the default backend for training, evaluation,
and play. The public-only tactical search interface is unchanged.

## Development

From the repository root, activate the Python environment and put Cargo on PATH:

```bash
source .venv/bin/activate
source "$HOME/.cargo/env"
python -m pip install maturin==1.9.6
rustup toolchain install 1.90.0 --profile minimal --component clippy --component rustfmt
RUSTUP_TOOLCHAIN=1.90.0 maturin develop --release --manifest-path native/astra/Cargo.toml
cargo +1.90.0 test --manifest-path native/astra/Cargo.toml --locked
cargo +1.90.0 clippy --manifest-path native/astra/Cargo.toml --all-targets --locked -- -D warnings
pytest agent/tests/test_astra.py agent/tests/test_astra_tournament.py play/tests/test_astra_play.py
```

The pinned toolchain is Rust 1.90.0. When commands are run from the repository
root, use that toolchain as the default or supply `cargo +1.90.0`. The nested
toolchain file is discovered automatically when working inside this directory.

For a regular wheel installation, run `python -m pip install ./native/astra`.
The root project retains its setuptools build. Rust is not needed to install a
prebuilt wheel, and unrelated Python tools do not import the extension eagerly.

`agent.scripts.smoke_astra_wheels` verifies a root wheel and native wheel in a
temporary clean environment. It checks explicit missing-extension errors, bundled
production configurations, and an adapter move without installing PyTorch or
NumPy:

```bash
python -m agent.scripts.smoke_astra_wheels \
  --root-wheel PATH_TO_ROOT_WHEEL --native-wheel PATH_TO_NATIVE_WHEEL \
  --output agent/runs/astra/wheel-check.json
```

The checker uses pip 22.3 or newer to target a clean interpreter without an
`ensurepip` bootstrap or dependency downloads.

## Interface

`HeuristicAstraBot(seed=None, config=None)` implements the usual scalar `Bot`
protocol. `analyze(engine, game_idx)` additionally returns the chosen move,
principal variation, node visits, completed depth, evaluation components, native
elapsed time, evaluator-call and transposition-hit counts, cutoff reason, and a
conservative round-solved flag.

`components` describes the public position immediately after the chosen move;
`values` contains the backed-up search utility from the completed iteration.
The principal variation stops at the current round boundary, before any refill.
A `pruned_round` cutoff means further depth cannot change the retained search
tree; excluded replies still keep `solved=false`. It is distinct from an exact
round solution.

Static evaluation caching is scoped to one move and capped at 200,000 entries.
It is enabled for multiplayer searches that begin with factories still present;
profiling found no benefit at two players or in small center-only positions.
The cache key includes boards, scores, floors, marker/round state, and total
available colors. It excludes source distribution and the acting seat because
the static evaluator does not depend on them; tactical search still uses the
complete state and its separate transposition table.

The snapshot layout is documented in `agent.eval.heuristic_astra.snapshot`.
Snapshot format version 1 contains only ordinary integers. It omits the bag,
discard counts, floor colors, and all random generators. Native validation
checks dimensions, ranges, line consistency, active factories, and visible
colored-tile supply. The Python adapter checks the returned move against the
authoritative legality mask.

Native `legal_actions`, `transition`, and `resolve_round` functions are exposed
for differential testing. `transition(..., resolve=False)` stops after drafting;
the default resolves the round if sources become empty. Resolution stops before
the next random refill and is idempotent.

Weights are named in `AstraWeights`; their order is the native array order.
Configuration JSON can omit weights to use defaults. `depth=0` disables search
for ablations. `width=0` disables heuristic reply pruning. Terminal winners use
the official score, then complete-row, then shared-victory rules.
The `opponent` and `field_pressure` weights apply to multiplayer evaluation;
two-player utility always uses the symmetric score-estimate difference required
by minimax.

With no explicit configuration, the adapter reads the independently selected
per-player-count defaults from `agent/eval/astra_configs/production.json`.
`center_nodes=0` inherits the regular node budget; a positive value sets a
separate budget for moves starting with empty factories. This permits focused
round solving without increasing every opening search. `rollout` is experimental
and disabled in the selected configurations.
`terminal_nodes=0` disables an optional Python budget policy. A positive value
raises the node budget only once a completed pattern line guarantees a horizontal
row this round, including positions with factories still active. The policy uses
the same public snapshot and makes the same single native call; the time ceiling
does not change. Its matched screens did not improve the competitive-field
aggregate, so it remains disabled in production.

The deadline is checked between search nodes. Selected configurations reserve
50 milliseconds of the two-second move ceiling for boundary and scheduling work.
Explicit configurations can request up to 2,000 milliseconds of native search;
boundary and move-ordering work add a small amount of overhead. A legal greedy
move is available even if no iteration completes. Fixed node limits and seeds
are reproducible in the same runtime; wall-clock-limited searches can stop at different depths on
different machines.

## Experiments

```bash
python -m agent.scripts.eval_astra \
  --config agent/eval/astra_configs/v7-column-prior.json \
  --name v7-column-prior --players 2 3 4 --blocks 32 \
  --output agent/runs/astra/example

python -m agent.scripts.benchmark_astra \
  --output agent/runs/astra/benchmark.json --include-play-engine
```

An experiment creates `manifest.json`, `sources.zip`, a private `frozen/` runtime,
copied checkpoint files, append-only `games.jsonl`, and `report.json`. Repeating
the exact command resumes missing games. It rejects changed code, native binary,
or experiment settings; an interrupted final JSONL append is safely discarded.
The private worker runtime prevents unrelated workspace edits from changing a
running experiment. Use a new output directory for each candidate/protocol.

Pass `--reference-experiment PATH` to reuse a previous experiment's frozen
engine, reference bots, neural code, and checkpoint bytes across a whole campaign.
This prevents concurrent workspace changes from moving the comparison field.
The runner verifies reference artifacts, archives the overlaid source actually
used by workers, and locks each output directory against concurrent writers.

After the working tree changes, explicitly resume with the archived runtime:

```bash
python -m agent.scripts.resume_astra agent/runs/astra/example --workers 4
```

This verifies archived files before importing them and requires the recorded
Python, PyTorch, and NumPy versions. It restores missing games with the archived
decision code, rather than treating the current candidate as interchangeable.
For loss review, record trajectories with `eval_astra --trace`, then use
`python -m agent.scripts.inspect_astra RUN/games.jsonl --output review.json`.
Deeper recommendations remain hypotheses unless a current-round result is solved.
`--terminal-round-only --centers-only` focuses the review on endings already
forced by a completed pattern line, with no factory reply pruning.

Shared victories count as fractional win share. Confidence intervals resample
whole seed/seat blocks. Unfinished games remain visible, produce conservative
outcome bounds, and block superiority/promotion claims. Checkpoint policies are
explicitly labeled `greedy` or `search32`. Promotion comparisons also verify the
actual engine/bot seed and seat schedules from the raw records.

## Lambda

The full application image also validates its registered neural opponent. With
the league and checkpoint files referenced by `play/release.json` available,
stage that existing application release before building:

```bash
python -m play.scripts.prepare_release
```

The staging command writes local `play/artifacts`; it does not deploy anything.
Astra's standalone extension and adapter require no checkpoint.

```bash
docker build --platform linux/amd64 -f infra/lambda.Dockerfile \
  -t azul-astra:validation .
docker run --rm --platform linux/amd64 --entrypoint python \
  azul-astra:validation -c 'import azul_astra; print(azul_astra.SNAPSHOT_VERSION)'
```

The builder and runtime use matching Lambda base images. The current application
defaults to Python 3.12/Amazon Linux 2023. The original Python 3.11/Amazon Linux 2
target can also be built with `--build-arg LAMBDA_PYTHON=3.11`; build a wheel inside
that image rather than assuming a wheel compiled on a newer OS will load there.
The final stage installs the release wheel. CDK explicitly selects x86-64.
The final image contains no Rust compiler or target directory. CPU PyTorch and
NumPy are pinned to compatible wheels; `--only-binary` prevents unexpected source
builds in the runtime image. The regular Flask Lambda handler remains the entry
point. Building/testing the image does not deploy AWS resources.

Run complete local handler games with `python -m play.scripts.smoke_astra` inside
the image. For Lambda Runtime Interface Emulator tests, override the image CMD
with `play.scripts.smoke_astra.local_handler`. That test adapter injects a
temporary local store and calls the real handler; it avoids requiring DynamoDB.
The production image CMD and production store selection remain unchanged.

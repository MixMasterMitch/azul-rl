# Azul RL

A full-stack system for training, evaluating, and deploying an Azul-playing AI agent.

## Architecture

- **Training** — Gumbel AlphaZero self-play with a batched Rust game engine and PyTorch inference/learning
- **Evaluation** — Tournament play against reference bots and historical checkpoints
- **League** — Persistent checkpoint pool with Bradley-Terry ratings
- **Web App** — Flask/Lambda play server where humans play against trained agents and bots
- **Deployment** — AWS CDK stack with Lambda, DynamoDB, S3, CloudFront, and API Gateway

## Quick Start

```bash
# Install dependencies (Rust 1.90+ is needed to build the native wheel)
pip install ./native/astra
pip install -e ".[dev,play,aws]"

# Run tests
pytest

# Smoke test training (CPU, ~30 seconds)
python -m agent.scripts.smoke_train

# Full training run (auto = CUDA if available, else CPU; MPS not supported)
azul-train --device auto --max-iters 500 --max-wall-minutes 60

# Start play server locally
azul-play
```

## Project Structure

```
azul-rl/
├── agent/
│   ├── env/           # Rust game engine + PyTorch/single references
│   ├── net/           # Neural network (encoder + model)
│   ├── search/        # Gumbel MCTS
│   ├── eval/          # Evaluation bots and tournament
│   ├── train/         # Training loop, self-play, learner, league
│   ├── scripts/       # CLI entry points
│   └── tests/         # Unit tests
├── play/
│   ├── server.py      # Flask app
│   ├── service.py     # Game orchestration
│   └── views.py       # API routes
├── webapp/            # React/Vite frontend
├── infra/             # AWS CDK infrastructure
├── deploy.sh          # One-command deployment
└── pyproject.toml     # Project configuration
```

## Game Engine

`agent.env.engine.GameEngine` (also exported there as `BatchedEngine`) is the
default for training, tuning, evaluation, both search modes, and interactive play.
Game state, rules, refills, and feature encoding run in Rust on CPU; `--device`
selects CPU or CUDA for network inference and learning. The native extension is
required; rebuild it with `pip install ./native/astra` after native code changes.
The deployment image builds and installs this wheel automatically.

Existing model and replay checkpoints remain compatible. New games use independent
native RNG streams, so their draws differ from old PyTorch runs with the same seed.
Saved play sessions migrate on load while preserving their future draws. The
explicit PyTorch reference remains in `agent.env.batched_engine` for parity tests
and benchmarks. See [the simulator guide](native/astra/SIMULATOR.md) for API details.

The Azul engine implements the full game rules:
- 2-4 players, 5 tile colors, 100 tiles total
- Factory displays (5/7/9 depending on player count)
- Pattern lines, wall-tiling with adjacency scoring
- Floor penalties, end-game bonuses
- Fixed 5×5 Latin square wall pattern

Action space: 300 actions (10 sources × 5 colors × 6 targets)

## ML Approach

- **Algorithm**: Gumbel AlphaZero (Danihelka et al., 2022)
- **Architecture**: Attention-based policy+value network with per-player-count heads
- **Training**: Self-play → replay buffer → KL + MSE loss with entropy bonus
- **Search**: Corrected one-ply baseline plus a Gumbel tree backend with deeper traversal
- **Source-aware policy**: Versioned `source_attn` model with factory-equivariant action outputs

See [the competitive training guide](docs/competitive-training.md) for corrected
baseline results, GPU diagnostics, reproducible overnight experiments, promotion
gates, checkpoint storage requirements, and model-registry serving.

## Deployment

Hosted app: [Play Azul](https://d3cp3zwu04q8a1.cloudfront.net).

```bash
# Stage the pinned trained opponent for local play or deployment.
python -m play.scripts.prepare_release

# Build, test, and synthesize without making any AWS changes.
./deploy.sh --dry-run

# Deploy to the authenticated AWS account, separate from Splendor.
aws login
./deploy.sh
```

Requires Docker, Node/npm, the CDK CLI, and the Python AWS/dev/play extras.
The default target is `AzulStack` in `us-west-2`; `--account`, `--region`, and
`--stack-name` select another target. `--account` also checks the authenticated
account. `--skip-frontend` reuses an existing `webapp/dist` build. The script uses
`.venv/bin/python` by default; set `AZUL_PYTHON` to override it.

The release workflow freezes the current source into `.releases/<release>/source`,
including any intentional uncommitted changes. It runs backend and CDK tests on CPU,
builds a Linux amd64 image with CPU PyTorch and Astra, and completes 2p/3p/4p games
inside a container with a read-only root filesystem. It then deploys a candidate
Lambda version and checks real DynamoDB persistence, model availability, and
latency. The warm AI-request p95 must be below 2 seconds and representative cold
requests below 25 seconds. A first deployment exposes no API routes until these
checks pass. Updates keep the previous API and frontend live during validation.

After validation, the workflow switches the `live` Lambda alias and CloudFront's
frontend release together, then smoke-tests the public URL. Failed public checks
restore the previous release (or withdraw routes on a first deployment). The
CloudFront URL is printed when deployment succeeds.

Infrastructure includes private, versioned S3 frontend and release buckets,
CloudFront origin access control, an uncached API Gateway path, Lambda at 2,048 MB,
and on-demand DynamoDB game/user tables with point-in-time recovery. Data tables,
release artifacts, and published Lambda versions are retained. CloudWatch alarms
cover Lambda errors, throttling, request duration, and API Gateway 5xx responses;
application logs expire after 30 days. No alarm notification destination is
configured by default. There is no provisioned concurrency or GPU serving cost.

### Models and promotion

`play/release.json` explicitly selects the initial `net:league:294` checkpoint and
its source checksum. The preparation script exports weights and compatibility
metadata into `play/artifacts/registry.json` and content-addressed `.pt` files.
These generated files are ignored by Git, included in the Lambda image, and backed
up in the private release bucket. Training buffers, optimizer state, and unrelated
training runs are excluded. `--validate-only` checks an already-staged release.

To publish another checkpoint, add a new immutable model ID and its source
checksum to `play/release.json`, then set `default_model_id` and run `./deploy.sh`.
Changing weights or search settings requires a new ID. Previously deployed models
are recovered from the previous release archive and kept for unfinished games;
removing an ID from the selection hides it from new games without deleting its
weights. Deployment refuses to strand an active game's pinned model. For local
competitive agents, `AZUL_MODEL_REGISTRY` can point to an existing registry instead.

The shipped network uses 64-candidate Gumbel-root/one-ply search at temperature
0.25 and q-scale 28, with no Dirichlet noise. Its verified player count is two.
Three- and four-player tables expose random and heuristic opponents, including
Astra when its native extension is available. Unavailable ratings are omitted.

See [Astra's development and evaluation guide](docs/astra/README.md) for its
hand-coded strategy, versioned configurations, reproducible experiments, and
[native build instructions](native/astra/README.md).

### Games, identity, and ratings

Enter a screen name to play, resume games, and see history and ratings. Names are
case-sensitive and do not verify identity; the same name retrieves the same
games. The UI remembers the name locally and identifies requests using
`X-Azul-Username`. Names contain 1–32 ASCII letters, digits, underscores or hyphens.

Game mutations require `expected_revision`. Conflicts return HTTP 409 and the UI
reloads authoritative state. Each AI request makes one move, so refreshes and
interrupted requests can resume safely. The internal snapshot includes bag,
discard, and RNG state; those private fields are never returned to the browser.
Local play uses one atomically replaced JSON database; Lambda uses conditional
DynamoDB transactions. Completed games update ratings exactly once. Shared wins
use Azul's score/complete-row tiebreak, and abandoned games do not affect ratings.
Abandonment is available until every player has taken four turns. Older saved
games without individual turn counts use four times the player count as their
total-move cutoff.

Human ratings appear after five first-place finishes (including shared wins).
They use the existing Bradley–Terry fit against per-player-count bot anchors,
with ties split evenly and calibrated ratings combined by physical games played.
Leaderboard reads use stored human summaries and bundled bot ratings, without
refitting the league. The original file-based `HumanRatingStore` remains available
for older offline callers; hosted play uses transactional rating updates.

The API retains `/api/game`, `/api/game/<id>`, `/action`, `/step-ai`, and
`/api/opponents`, and adds `/api/me`, `/api/games`, `/api/leaderboard`,
`/api/game/<id>/abandon`, and `/api/ready`. History accepts `status` (`active`,
`completed`, or `abandoned`), `limit`, and an opaque `cursor`. Responses include
`next_cursor`; a filtered DynamoDB page can be empty while another cursor remains.

### Verification and rollback

```bash
pytest play/tests infra/tests
python -m play.scripts.smoke                    # complete games through the WSGI adapter
pip install -e ".[browser]"
# Start isolated Flask/Vite servers, then run UI acceptance tests:
python webapp/tests/hosted_play.py --url http://127.0.0.1:5177
./deploy.sh --rollback <previous-release-id>
```

The browser test uses an installed Chrome (`CHROME_BIN` overrides its path), or a
Playwright Chromium installation. It completes games at every player count,
refreshes mid-game, checks history and the leaderboard, and saves desktop/mobile
screenshots under `.hosting-checks/browser`. Set `PORT` and `AZUL_PLAY_DATA` for an
isolated Flask instance; `AZUL_API_PROXY` selects that backend for Vite.

Release metadata, frozen source, model artifacts, and latency reports are stored
in the private release bucket and locally under `.releases`. Rollback selects the
previous Lambda version and matching frontend using CDK, without deleting games
or ratings. It checks compatibility with active games first. DynamoDB snapshots
use schema version 1; future breaking schema changes must include a migration or
backward-compatible reader before release. There was no existing Azul AWS stack
or persisted hosted data to migrate for the initial release.

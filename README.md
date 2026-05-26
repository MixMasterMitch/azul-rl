# Azul RL

A full-stack system for training, evaluating, and deploying an Azul-playing AI agent.

## Architecture

- **Training** — Gumbel AlphaZero self-play with a vectorized PyTorch game engine
- **Evaluation** — Tournament play against reference bots and historical checkpoints
- **League** — Persistent checkpoint pool with Bradley-Terry ratings
- **Web App** — Flask/Lambda play server where humans play against trained agents and bots
- **Deployment** — AWS CDK stack with Lambda, DynamoDB, S3, CloudFront, and API Gateway

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

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
│   ├── env/           # Game engine (batched + single reference)
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
- **Search**: Gumbel root + 1-ply value expansion

## Deployment

```bash
# Deploy to AWS (requires CDK, Docker, AWS credentials)
./deploy.sh
```

The deployment creates:
- CloudFront CDN serving the React SPA
- API Gateway → Lambda (Docker with PyTorch) for game API
- DynamoDB for game records and user ratings

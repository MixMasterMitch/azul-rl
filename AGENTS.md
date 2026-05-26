# Azul RL — Agent Guide

## Project Overview

This repo trains, evaluates, and serves an Azul-playing AI using Gumbel AlphaZero self-play. The system has four main subsystems:

1. **Game engine** (`agent/env/`) — vectorized PyTorch Azul implementation
2. **Training** (`agent/train/`, `agent/search/`, `agent/net/`) — self-play + MCTS + neural net
3. **Play server** (`play/`) — Flask API for human vs bot games
4. **Frontend** (`webapp/`) — React/Vite SPA

## Architecture Decisions

- **Single action space**: 300 flat actions (10 sources × 5 colors × 6 targets). Legality enforced via mask, not separate action types.
- **Batched engine**: All training uses `BatchedEngine` with tensor state. Play uses `batch_size=1`.
- **Per-player-count heads**: The neural net has separate policy/value heads for 2p, 3p, and 4p games.
- **Perspective encoding**: Network always sees the current player as seat 0.
- **Bradley-Terry ratings**: Anchored to `random=1000`, fit via L-BFGS on full pairwise history.

## Conventions

### Python
- Python 3.11+, PyTorch 2.x
- Type hints on all function signatures
- No framework for the play server beyond Flask
- Tests use pytest; run with `pytest agent/tests/`

### Game Engine
- `single_engine.py` is the reference implementation (Python data structures)
- `batched_engine.py` is the production engine (PyTorch tensors)
- Both must agree on legal actions and game outcomes — test with parity tests
- Wall pattern: `wall_color(row, col) = (col - row) % 5`

### Neural Net
- Encoder produces `(global_feat, source_feat)` — never raw engine tensors
- Model forward always takes `(global_feat, source_feat, legal_mask, num_players)`
- Illegal actions are masked to `-1e9` before softmax

### Training
- Self-play generates improved policies via Gumbel MCTS
- Value targets: +1 winner, -1 losers, -1 all on stall (max_turns exceeded)
- Loop is resumable: `state.json` + checkpoint files

### Frontend
- React 18 with JSX transform (no `import React`)
- TypeScript strict mode
- Vite dev server proxies `/api` to Flask backend

## Key Files

| File | Purpose |
|------|---------|
| `agent/env/actions.py` | Action space definition + wall pattern |
| `agent/env/batched_engine.py` | Production game engine |
| `agent/env/single_engine.py` | Reference engine for testing |
| `agent/net/model.py` | `AzulNet` neural network |
| `agent/net/encoder.py` | State → feature tensors |
| `agent/search/gumbel_mcts.py` | MCTS action selection |
| `agent/train/loop.py` | Main training loop |
| `play/service.py` | Game orchestration |
| `play/views.py` | API routes |
| `infra/stack.py` | AWS CDK stack |

## Running

```bash
source .venv/bin/activate
pytest                                    # run tests
python -m agent.scripts.smoke_train       # verify training pipeline
azul-train --device auto                  # full training
azul-play                                 # local play server
cd webapp && npm run dev                  # frontend dev server
```

## Azul Rules Quick Reference

- 2-4 players, 5 colors (Blue/Yellow/Red/Black/White), 100 tiles (20 each)
- Factories: 5 (2p), 7 (3p), 9 (4p) — each filled with 4 random tiles
- Turn: pick ALL tiles of ONE color from ONE source → place on ONE pattern line (or floor)
- Remaining factory tiles go to center; first to take from center gets first-player marker (floor penalty)
- Round ends when all sources empty → wall-tiling: completed lines move one tile to wall
- Scoring: isolated=1pt; adjacent=count horizontal + count vertical linked tiles
- Floor penalties: -1, -1, -2, -2, -2, -3, -3
- Game ends when any player completes a horizontal wall row
- Bonuses: +2/complete row, +7/complete column, +10/complete color set

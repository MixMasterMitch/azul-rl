#!/usr/bin/env bash
# attn_256_v4: trial #8 hyperparams from azul-tune-v3-focused-opus-v1, warm-start i5700.
#
# Trial #8 (projected opus 86.8%): binary reward, lower opus bot mix, q_scale ~27.
# Async eval uses the same q_scale; eval_sims=64 matches the tuning / Phase 4 protocol.

set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ ! -d .venv ]]; then
  echo "error: .venv not found" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

INIT_FROM="${INIT_FROM:-agent/runs/attn_256_v3/checkpoints/iter_005700.pt}"
RUN_ID="${RUN_ID:-attn_256_v4}"
LOG_PATH="${LOG_PATH:-/tmp/${RUN_ID}.log}"
MAX_ITERS="${MAX_ITERS:-2500}"
MAX_WALL_MINUTES="${MAX_WALL_MINUTES:-720}"

if [[ ! -f "${INIT_FROM}" ]]; then
  echo "error: INIT_FROM not found: ${INIT_FROM}" >&2
  exit 1
fi

echo "Run: ${RUN_ID}"
echo "Init: ${INIT_FROM}"
echo "Log: ${LOG_PATH}"
echo "Wall budget: ${MAX_WALL_MINUTES} min, max_iters=${MAX_ITERS}"
echo ""
echo "Starting in 3s (Ctrl-C to abort)..."
sleep 3

exec python -u -m agent.scripts.train \
  --run-id "${RUN_ID}" \
  --init-from "${INIT_FROM}" \
  --num-players 2 \
  --device auto \
  --hidden 256 \
  --arch attn \
  --selfplay-games 1023 \
  --selfplay-sims 32 \
  --max-turns 200 \
  --turns-per-player 60 \
  --replay-capacity 1000000 \
  --learner-batch 256 \
  --learner-steps 72 \
  --entropy-bonus 0.014274964936918592 \
  --checkpoint-every 50 \
  --lr 0.0009217470483295072 \
  --weight-decay 8.96240535348826e-05 \
  --max-iters "${MAX_ITERS}" \
  --max-wall-minutes "${MAX_WALL_MINUTES}" \
  --dirichlet-alpha 0.25668356196635866 \
  --dirichlet-mix 0.5332343378344558 \
  --q-scale 26.747517709474458 \
  --time-discount 0.999444440976462 \
  --reward-mode binary \
  --training-cycle-length 4 \
  --bot-opus-prob 0.36801279738712916 \
  --eval-games 512 \
  --eval-sims 64 \
  --eval-workers 1 \
  --bot-policy batched \
  2>&1 | tee -a "${LOG_PATH}"

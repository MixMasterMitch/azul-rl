#!/usr/bin/env bash
# Optuna study: 2-hour trials, evaluated every 30 minutes, objective = projected win rate vs opus.
# Prepared for manual launch; nothing runs until this script is executed.
#
# Flow per study:
#   1. Baseline opus-only eval on --init-from (t=0), cached separately from rating baselines
#   2. Each trial: 4x30min training sessions, each followed by opus-only eval
#   3. Objective: logit-curve projected vs_opus_winrate at t_last + 72 hours
#
# Default checkpoint: best fixed-eval checkpoint from attn_256_v1, league/ckpt_00100_i1325.pt.

set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ ! -d .venv ]]; then
  echo "error: .venv not found" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

STUDY_NAME="${STUDY_NAME:-azul-tune-i1325-opus-2h-v1}"
N_TRIALS="${N_TRIALS:-8}"
SESSIONS="${SESSIONS:-4}"
SESSION_MINUTES="${SESSION_MINUTES:-30}"
RATING_GAMES="${RATING_GAMES:-256}"
RATING_SIMS="${RATING_SIMS:-32}"
EXTRAPOLATE_HOURS="${EXTRAPOLATE_HOURS:-72}"
INIT_FROM="${INIT_FROM:-agent/runs/league/ckpt_00100_i1325.pt}"
LOG_PATH="${LOG_PATH:-/tmp/tune_i1325_opus_2h_v1.log}"
FORCE_BASELINE="${FORCE_BASELINE:-0}"

if [[ ! -f "${INIT_FROM}" ]]; then
  echo "error: INIT_FROM not found: ${INIT_FROM}" >&2
  exit 1
fi

force_args=()
if [[ "${FORCE_BASELINE}" == "1" ]]; then
  force_args+=(--force-baseline)
fi

echo "Study: ${STUDY_NAME}"
echo "Init: ${INIT_FROM}"
echo "Trials: ${N_TRIALS}"
echo "Per trial: ${SESSIONS}x${SESSION_MINUTES}min train/eval = 2h default"
echo "Eval/objective: opus only, maximize projected vs_opus_winrate at +${EXTRAPOLATE_HOURS}h"
echo "Eval games/sims: ${RATING_GAMES}/${RATING_SIMS}"
echo "DB: agent/runs/optuna_${STUDY_NAME}.db"
echo "Log: ${LOG_PATH}"
echo ""
echo "Starting in 3s (Ctrl-C to abort)..."
sleep 3

exec python -u -m agent.scripts.tune \
  --study-name "${STUDY_NAME}" \
  --n-trials "${N_TRIALS}" \
  --sessions-per-trial "${SESSIONS}" \
  --session-minutes "${SESSION_MINUTES}" \
  --extrapolate-hours "${EXTRAPOLATE_HOURS}" \
  --objective opus_winrate \
  --init-from "${INIT_FROM}" \
  --rating-games "${RATING_GAMES}" \
  --rating-sims "${RATING_SIMS}" \
  --device auto \
  --output-dir agent/runs \
  "${force_args[@]}" \
  2>&1 | tee -a "${LOG_PATH}"

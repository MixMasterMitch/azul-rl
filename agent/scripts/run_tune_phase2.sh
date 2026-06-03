#!/usr/bin/env bash
# Phase-2 Optuna: 7-point log-curve objective (baseline + 6×30min sessions).
# Prepared for manual launch — does not auto-start.
#
# Flow per study:
#   1. Baseline 2p rating eval on --init-from (t=0), cached in tune_baseline_eval.json
#   2. Each trial: 6×30min training (1023×32, 4-step cycle) + rating eval → 6 points
#   3. Fit rating ≈ a + b·log(t); objective = predicted rating at t_last + 3 days
#
# Prerequisites:
#   source .venv/bin/activate
#   --init-from checkpoint (default: league/ckpt_00025_i650.pt)
#   agent/runs/league/league.json for top-league eval matchup

set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ ! -d .venv ]]; then
  echo "error: .venv not found" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

STUDY_NAME="${STUDY_NAME:-azul-tune-curve-1023x32}"
N_TRIALS="${N_TRIALS:-12}"
SESSIONS="${SESSIONS:-6}"
SESSION_MINUTES="${SESSION_MINUTES:-30}"
EXTRAPOLATE_HOURS="${EXTRAPOLATE_HOURS:-72}"
RATING_GAMES="${RATING_GAMES:-256}"
INIT_FROM="${INIT_FROM:-agent/runs/league/ckpt_00025_i650.pt}"

if [[ ! -f "${INIT_FROM}" ]]; then
  echo "error: INIT_FROM not found: ${INIT_FROM}" >&2
  exit 1
fi

echo "Study: ${STUDY_NAME}"
echo "Init: ${INIT_FROM}"
echo "Trials: ${N_TRIALS}"
echo "Per trial: baseline + ${SESSIONS}×${SESSION_MINUTES}min train/eval → log fit → +${EXTRAPOLATE_HOURS}h extrapolation"
echo "Self-play: 1023×32, 4-step cycle"
echo "DB: agent/runs/optuna_${STUDY_NAME}.db"
echo ""
echo "Starting in 3s (Ctrl-C to abort)…"
sleep 3

exec python -m agent.scripts.tune \
  --study-name "${STUDY_NAME}" \
  --n-trials "${N_TRIALS}" \
  --sessions-per-trial "${SESSIONS}" \
  --session-minutes "${SESSION_MINUTES}" \
  --extrapolate-hours "${EXTRAPOLATE_HOURS}" \
  --init-from "${INIT_FROM}" \
  --rating-games "${RATING_GAMES}" \
  --rating-sims 32 \
  --device auto \
  --output-dir agent/runs

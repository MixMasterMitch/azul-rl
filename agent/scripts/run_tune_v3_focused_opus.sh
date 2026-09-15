#!/usr/bin/env bash
# Phase 5: focused binary-reward Optuna study vs opus from best v3 checkpoint.
#
# Prerequisites:
#   source .venv/bin/activate
#   Phase 3 report (best checkpoint) — defaults to league/ckpt_00222_i3450.pt
#   Phase 4 report (best q_scale for eval) — defaults to training q_scale=28

set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ ! -d .venv ]]; then
  echo "error: .venv not found" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

STUDY_NAME="${STUDY_NAME:-azul-tune-v3-focused-opus-v1}"
N_TRIALS="${N_TRIALS:-10}"
SESSIONS="${SESSIONS:-4}"
SESSION_MINUTES="${SESSION_MINUTES:-30}"
RATING_GAMES="${RATING_GAMES:-512}"
RATING_SIMS="${RATING_SIMS:-64}"
EXTRAPOLATE_HOURS="${EXTRAPOLATE_HOURS:-72}"
PHASE3_REPORT="${PHASE3_REPORT:-agent/runs/phase3_opus_eval.json}"
PHASE4_REPORT="${PHASE4_REPORT:-agent/runs/phase4_search_sweep.json}"
LOG_PATH="${LOG_PATH:-/tmp/tune_v3_focused_opus_v1.log}"
FORCE_BASELINE="${FORCE_BASELINE:-1}"

mapfile -t _phase_cfg < <(python3 <<PY
import json
import pathlib

phase3 = pathlib.Path("${PHASE3_REPORT}")
phase4 = pathlib.Path("${PHASE4_REPORT}")
init_from = "agent/runs/league/ckpt_00222_i3450.pt"
q_scale = 28.0
if phase3.exists():
    payload = json.loads(phase3.read_text())
    init_from = payload.get("best_checkpoint", init_from)
if phase4.exists():
    payload = json.loads(phase4.read_text())
    q_scale = float(payload.get("best", {}).get("q_scale", q_scale))
print(init_from)
print(q_scale)
PY
)
INIT_FROM="${_phase_cfg[0]}"
BASELINE_Q_SCALE="${_phase_cfg[1]}"

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
echo "Baseline eval q_scale: ${BASELINE_Q_SCALE}"
echo "Trials: ${N_TRIALS}"
echo "Per trial: ${SESSIONS}x${SESSION_MINUTES}min train/eval"
echo "Objective: opus_winrate (projected +${EXTRAPOLATE_HOURS}h)"
echo "Search: --focused-ranges (binary reward, opus_prob, cycle length)"
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
  --baseline-q-scale "${BASELINE_Q_SCALE}" \
  --rating-games "${RATING_GAMES}" \
  --rating-sims "${RATING_SIMS}" \
  --focused-ranges \
  --device auto \
  --output-dir agent/runs \
  "${force_args[@]}" \
  2>&1 | tee -a "${LOG_PATH}"

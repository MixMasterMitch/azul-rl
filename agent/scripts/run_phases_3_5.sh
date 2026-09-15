#!/usr/bin/env bash
# Run plateau plan phases 3–5 sequentially:
#   3. High-confidence opus eval (1024 games × 3 checkpoints)
#   4. Search sweep on best checkpoint
#   5. Focused Optuna study (background via nohup unless RUN_FOREGROUND=1)

set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ ! -d .venv ]]; then
  echo "error: .venv not found" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

CKPT_I3450="agent/runs/league/ckpt_00222_i3450.pt"
CKPT_I5250="agent/runs/league/ckpt_00258_i5250.pt"
CKPT_I5700="agent/runs/attn_256_v3/checkpoints/iter_005700.pt"

echo "=== Phase 3: high-confidence opus eval ==="
python -u -m agent.scripts.eval_checkpoints_opus \
  "${CKPT_I3450}" "${CKPT_I5250}" "${CKPT_I5700}" \
  --num-games 1024 \
  --num-sims 64 \
  --q-scale 28.0 \
  --device auto \
  --report agent/runs/phase3_opus_eval.json

BEST_CKPT="$(python3 -c "import json; print(json.load(open('agent/runs/phase3_opus_eval.json'))['best_checkpoint'])")"
echo "Phase 3 best checkpoint: ${BEST_CKPT}"

echo ""
echo "=== Phase 4: search sweep on ${BEST_CKPT} ==="
python -u -m agent.scripts.search_sweep \
  "${BEST_CKPT}" \
  --num-games 512 \
  --q-scales "8,12,18,25,32" \
  --num-sims-list "16,32,64" \
  --device auto \
  --report agent/runs/phase4_search_sweep.json

echo ""
echo "=== Phase 5: focused Optuna study ==="
if [[ "${RUN_FOREGROUND:-0}" == "1" ]]; then
  bash agent/scripts/run_tune_v3_focused_opus.sh
else
  LOG="/tmp/tune_v3_focused_opus_v1.log"
  nohup bash agent/scripts/run_tune_v3_focused_opus.sh > "${LOG}" 2>&1 &
  echo "Phase 5 started in background (pid $!, log ${LOG})"
fi

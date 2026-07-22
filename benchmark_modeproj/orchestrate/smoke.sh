#!/bin/bash
# smoke.sh - budget-first smoke task (Section 5).
# Runs 1 arch x 1 seed x 1 member END TO END with --smoke (few epochs, few
# molecules) and confirms it converges and writes artifacts BEFORE the full
# array is launched. Exit non-zero if the expected artifacts are absent.
set -euo pipefail

module purge
source ~/.bashrc
conda activate pyg

OUT_ROOT="/srv/scratch/z5076150/benchmark_modeproj"
cd "$(dirname "$0")/.."

ARCH="${1:-candidate}"; SEED="${2:-0}"; MEMBER="${3:-0}"

echo "SMOKE: $ARCH seed=$SEED member=$MEMBER (--smoke)"
python train.py --arch "$ARCH" --seed "$SEED" --member "$MEMBER" \
    --prereg "$OUT_ROOT/prereg.json" --out-root "$OUT_ROOT" --smoke

RUN="$OUT_ROOT/$ARCH/seed$SEED/member$MEMBER"
if [ ! -f "$RUN/checkpoint.pt" ] || [ ! -f "$RUN/test_sis.npz" ]; then
    echo "SMOKE FAILED: expected artifacts missing in $RUN"; exit 1
fi
if ! grep -q "\"status\": \"ok\"" "$OUT_ROOT/runs.jsonl" 2>/dev/null; then
    echo "SMOKE FAILED: no ok run recorded in runs.jsonl"; exit 1
fi
echo "SMOKE OK: artifacts present and run recorded. Safe to qsub the full array."

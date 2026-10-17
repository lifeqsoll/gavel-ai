#!/usr/bin/env bash
# Overnight: re-extract hands (bushon train), retrain, gate, local ship.
# No git commit/push. Wrap with keep_awake.sh for no sleep/lock.
set -euo pipefail

ROOT="${ROOT:-/mnt/disk-e/gavel-data}"
REPO="${REPO:-/home/emosensei/gavel-ai}"
PY="${PY:-$HOME/gavel/train-venv/bin/python}"
OUT="${OUT:-$REPO/models/exp_night_hands}"
OUT_B="${OUT_B:-$REPO/models/exp_night_hands_b}"
MAX_A="${MAX_REAL_HANDS_A:-40000}"
MAX_B="${MAX_REAL_HANDS_B:-80000}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"

mkdir -p "$OUT" "$ROOT/logs"
LOG="$OUT/overnight.log"
REPORT="$OUT/overnight_report.txt"
exec > >(tee -a "$LOG") 2>&1

cd "$REPO"
echo "=== overnight start $(date -Is) ==="
echo "ROOT=$ROOT REPO=$REPO OUT=$OUT"

run_gates() {
  local weights=$1 thresholds=$2 tag=$3
  echo "=== gates $tag ==="
  $PY -u training/eval_samples_gate.py \
    --weights "$weights" --thresholds "$thresholds" \
    --strict-hand --require-l0-real-gym
  $PY -u training/eval_hardneg_local.py \
    --weights "$weights" --thresholds "$thresholds" \
    --wiki-n 16 --strict-bushon
  $PY -u training/eval_faces_local.py \
    --weights "$weights" --thresholds "$thresholds"
  $PY -m pytest -q
}

# 0) Hygiene: drop eval filenames from bushon train dir
EVAL="$ROOT/real/hands/bushon_eval"
TRAIN="$ROOT/real/hands/bushon"
if [[ -d "$EVAL" && -d "$TRAIN" ]]; then
  for f in "$EVAL"/*; do
    [[ -f "$f" ]] || continue
    rm -f "$TRAIN/$(basename "$f")"
  done
  echo "bushon train count: $(find "$TRAIN" -type f | wc -l); eval: $(find "$EVAL" -type f | wc -l)"
fi

# 1) Extract
if [[ "$SKIP_EXTRACT" != "1" ]]; then
  echo "=== extract real_hands $(date -Is) ==="
  REAL_HANDS_PER_DIR_LIMIT="${REAL_HANDS_PER_DIR_LIMIT:-3000}" \
    $PY -u training/extract_features.py --source real_hands --root "$ROOT" \
    | tee -a "$ROOT/logs/extract_real_hands_night.log"
  $PY - <<'PY'
import numpy as np
from collections import Counter
from pathlib import Path
p = Path("/mnt/disk-e/gavel-data/features_normalized/real_hands.npz")
d = np.load(p, allow_pickle=True)
print("n=", len(d["labels"]), Counter(map(str, d["generators"].tolist())))
PY
else
  echo "SKIP_EXTRACT=1"
fi

# 2) Retrain A
echo "=== retrain A max=$MAX_A $(date -Is) ==="
MAX_REAL_HANDS="$MAX_A" OUT="$OUT" ROOT="$ROOT" PY="$PY" \
  bash training/retrain_when_hands_ready.sh

PASS=0
if run_gates "$OUT/detector_weights.npz" "$OUT/thresholds.json" "passA"; then
  PASS=1
  SHIP_DIR="$OUT"
else
  echo "=== pass A failed; retrain B max=$MAX_B ==="
  MAX_REAL_HANDS="$MAX_B" OUT="$OUT_B" ROOT="$ROOT" PY="$PY" \
    bash training/retrain_when_hands_ready.sh
  if run_gates "$OUT_B/detector_weights.npz" "$OUT_B/thresholds.json" "passB"; then
    PASS=1
    SHIP_DIR="$OUT_B"
  fi
fi

{
  echo "overnight_report $(date -Is)"
  echo "pass=$PASS ship=${SHIP_DIR:-none}"
} > "$REPORT"

if [[ "$PASS" -eq 1 ]]; then
  cp "$SHIP_DIR/detector_weights.npz" "$REPO/models/detector_weights.npz"
  cp "$SHIP_DIR/thresholds.json" "$REPO/models/thresholds.json"
  cp "$SHIP_DIR/metrics.json" "$REPO/models/metrics.json"
  echo "LOCAL SHIP -> $REPO/models (no git)"
  echo "shipped" >> "$REPORT"
else
  echo "NO SHIP" >> "$REPORT"
  exit 1
fi

echo "=== overnight done $(date -Is) ==="

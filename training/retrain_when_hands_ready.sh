#!/usr/bin/env bash
# Run after you drop real hand/close-up photos and extract features.
# Expected feature file: $ROOT/features_normalized/real_hands.npz
# (label=0). Build it with:
#   python training/extract_features.py --source real_hands --root "$ROOT"
# once that source exists, OR place a compatible npz yourself.
#
# Does NOT ship weights. Prints eval gate; copy manually if GATE PASS + metrics ok.
set -euo pipefail
ROOT="${ROOT:-/mnt/disk-e/gavel-data}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$HOME/gavel/train-venv/bin/python}"
OUT="${OUT:-$REPO/models/exp_hands}"

HANDS_NPZ="$ROOT/features_normalized/real_hands.npz"
if [[ ! -f "$HANDS_NPZ" ]]; then
  echo "missing $HANDS_NPZ"
  echo "Put permissive real hand/close-up images, add extract source real_hands, then re-run."
  exit 1
fi

cd "$REPO"
# Cap each real source (esp. HaGRID ~600k) so hands hard-negs help without
# drowning coco/scenes — full 605k hands failed florence (FP) while gym stayed veto-only.
MAX_PER_REAL="${MAX_PER_REAL:-20000}"
$PY -u training/train_detector.py --root "$ROOT" \
  --real-sources coco,open_images,wikimedia,coco_scenes,wikimedia_scenes,wikimedia_faces,wikimedia_extra,real_hands \
  --gen-sources nano_banana,cf_commercial_0,ecommerce_cc0,hmnsh_fake,ai_vs_real_fake,diffusiondb,synthetic_characters \
  --max-per-real-source "$MAX_PER_REAL" \
  --out-dir "$OUT"

echo "=== sample gate (strict hand) ==="
$PY -u training/eval_samples_gate.py \
  --weights "$OUT/detector_weights.npz" \
  --thresholds "$OUT/thresholds.json" \
  --strict-hand || true

echo "Candidate in $OUT — ship only if GATE PASS and uncertain mass on coco_real is sane."
echo "Ship manually:"
echo "  cp $OUT/detector_weights.npz $REPO/models/detector_weights.npz"
echo "  cp $OUT/thresholds.json $REPO/models/thresholds.json"
echo "  cp $OUT/metrics.json $REPO/models/metrics.json"

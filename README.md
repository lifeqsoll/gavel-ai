# Gavel AI

Русская версия: [README.ru.md](README.ru.md)

**Apache-2.0** open-source prototype that looks for *signs* of image generation. It combines a learned global detector with local symbolic layers (anatomy, artifacts, geometry). It does **not** claim that an image is AI-generated with certainty — every label is a **signal to verify manually**.

## Contents

1. [What it does](#what-it-does)
2. [Screenshots](#screenshots)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Web interface](#web-interface)
7. [Learned detector (L0)](#learned-detector-l0)
8. [Layers and fusion](#layers-and-fusion)
9. [Protocol / overlay](#protocol--overlay)
10. [HTTP API](#http-api)
11. [Training and eval](#training-and-eval)
12. [Known limitations](#known-limitations)
13. [Tests](#tests)
14. [Image sources and licenses](#image-sources-and-licenses)

## What it does

| Layer | Role | Votes in fusion? |
|---|---|---|
| **L0 global** | Frozen CLIP ViT-L/14 + linear probe → score and `generated` / `uncertain` / `real` | Yes (calibrated high ≈ **0.910**) |
| **L1 anatomy** | Face Mesh + Pose + Hands map; 2D distal hand WARNING only | Yes when hand evidence fires |
| **L2 artifacts** | Flat-region residual kurtosis (soft cue) | Soft: with L0 uncertain/generated; **cannot overturn L0=`real`** |
| **L3 geometry** | LSD (Hough fallback) + RANSAC vanishing points | Yes on **CRITICAL** only (default variance thr **0.05**) |
| **L4 physics** | Lighting / shadow rays (classical scene provider) | Cue only (no fusion vote; classical FP on real hands) |

**Fusion A:** final decision is `generated` if L0=`generated` **or** a hard layer votes.  
Exceptions:

- **`anatomy_hand_veto`** — clean face-free hand close-up (anatomy score ≤ 0.1) can veto an L0-only false positive (stock gym-hand case).
- **Soft artifacts** — ignored when L0 already decided `real` (stops night/texture photos from flipping the verdict alone).

Wide images (side-by-side diptychs) also score **left/right** tiles; the highest L0 probability wins.

## Screenshots

<p align="center">
  <img src="assets/readme/01_street_real.jpg" width="48%" alt="Real street → no generation signal" />
  <img src="assets/readme/05_hand_gym_veto_real.jpg" width="48%" alt="Real gym hand → anatomy veto of L0 FP" />
</p>

<p align="center">
  <em>Left: real Florence street → <code>real</code> (L0 low). Right: real gym hand → L0 high but final <code>real</code> via <code>anatomy_hand_veto</code>.</em>
</p>

<p align="center">
  <img src="assets/readme/02_stylized_generated.jpg" width="48%" alt="Stylized AI portrait → generated" />
  <img src="assets/readme/04_portrait_owl_generated.jpg" width="48%" alt="Photoreal AI portrait → generated" />
</p>

<p align="center">
  <img src="assets/readme/06_steampunk_cat_generated.jpg" width="48%" alt="AI steampunk cat → generated" />
  <img src="assets/readme/03_layers_breakdown.jpg" width="48%" alt="Per-layer breakdown in the UI" />
</p>

## Architecture

```
gavel-ai/
├── agents/                 perception (LSD, MediaPipe anatomy, classical shadows)
├── core/
│   ├── pipeline.py         orchestrates L0 + layers → EvidenceProtocol + overlay
│   ├── learned_detector.py CLIP features → linear probe (multi-tile on wide images)
│   ├── verdict.py          Fusion A + anatomy hand veto + soft-artifact rule
│   ├── normalize.py        shared resize / center-crop / JPEG q90 (train = infer)
│   ├── registry.py         Pydantic EvidenceProtocol, LayerScores, DetectorResult
│   └── validators/         perspective, anatomy (2D), artifacts, halves, edges, shadows
├── visualizer/overlay.py   cyan/green/amber map + red/yellow symbolic marks
├── app.py                  Gradio UI
├── analyze.py              CLI batch
├── api.py                  HTTP API (API key auth)
├── training/               extract / train / sample+hardneg gates / calibrate artifacts
├── models/                 shipped weights + thresholds (experiments gitignored)
├── assets/readme/          UI screenshots
├── samples/real/           licensed real test photos
└── tests/
```

Data flow:

```
image ─┬─► L0 CLIP probe ──────────────────────────────┐
       ├─► AnatomyAgent → AnatomyValidator (hands 2D) ─┤
       ├─► ArtifactValidator (flat kurtosis) ───────────┤──► Fusion A ──► verdict
       ├─► GeometryAgent → PerspectiveValidator ────────┤
       └─► LightAgent (optional) → ShadowValidator ─────┘
                         │
                         ▼
              EvidenceProtocol (JSON) + annotated overlay
```

If a stage cannot run (no MediaPipe, no hands, no scene provider), it is `skipped` with a reason; other layers still run.

## Installation

Python **3.11+**. The learned detector needs `torch` and `open_clip_torch` (in `requirements.txt`). On first use, open_clip downloads ViT-L/14 (~1.7 GB). Anatomy needs MediaPipe **0.10.x** with the legacy `mp.solutions` API.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full install with hands/face map (recommended for the Gradio demo):
# uv venv -p 3.11 .venv311
# uv pip install -p .venv311/bin/python -r requirements.txt "mediapipe==0.10.14"
```

```bash
python -c "import cv2, numpy, scipy, pydantic; print('ok')"
python -c "import mediapipe; print(mediapipe.__version__)"   # if installed
```

## Quick start

```bash
python -m pytest -q
python app.py                  # local Gradio
python app.py --share          # temporary public link
python analyze.py samples/real/*.jpg --out-dir results
GAVEL_API_KEYS=dev-key python api.py   # HTTP API on :8000
python mock_run.py             # synthetic scene smoke test
```

Shipped probe files:

- `models/detector_weights.npz` — linear head
- `models/thresholds.json` — L0 high/low
- `models/layer_thresholds.json` — per-layer vote thresholds
- `models/metrics.json` — last reported train/eval metrics

## Web interface

`app.py` Gradio app:

1. Upload an image or pick a sample.
2. Optional: line detector (`lsd` / `hough`), perspective variance slider (default **0.05**).
3. **Analyze** → annotated overlay + markdown summary (final decision, fusion reason, per-layer scores).

Overlay banner shows **Final** decision (after fusion) separately from raw L0 score. Cyan/green/amber lines are an **anatomy map**, not a verdict. Red/yellow marks are symbolic findings.

## Learned detector (L0)

- Backbone: OpenCLIP **ViT-L/14** (`laion2b_s32b_b82k`), frozen.
- Head: logistic regression on L2-normalized embeddings.
- Preprocess: `normalize_pil` — scale so min side = 512, center-crop 448, JPEG quality 90 (same in training and inference).
- Thresholds (shipped): **high ≈ 0.910** (≈1% FPR on real calibration incl. capped hands), **low ≈ 0.014**.
- Wide images: also score left/right halves; keep max probability.
- **2×2 quadrants** when L0 is still in the uncertain band (skip when already decisive — latency).
- Extra probes: **face crop** (primary max), **hand crops** (veto / disagree only), **halves** composite (diptych cue), **plastic skin** (face Laplacian ± clipping), **edge coherence** (LSD density soft cue).

L0 is a **semantic** probe, not a pixel-forensics detector. It can false-positive on polished stock close-ups and false-negative on OOD generators; diptychs and local AI patches rely on the probes above until harder negatives are trained in.

## Layers and fusion

| Rule | Behavior |
|---|---|
| L0 `generated` | Final → generation signal (unless hand veto applies and no other hard votes) |
| Geometry CRITICAL | Votes even if L0=`real` |
| Anatomy hand WARNING | Votes (score ≥ 0.5) |
| Artifacts high score | Soft vote only if L0 is not firmly `real` |
| Hand veto | Hands present, no face, anatomy ≤ 0.1, large `hand_span` → suppress L0-only FP |
| Hand-crop disagree | Full L0 vs hand crop clash → `uncertain` |
| Halves / plastic / edges | Soft generation lean on diptych, plastic face, or over-coherent edges when L0 is not already generated |

Primary L0 = max over full / left-right / quads / face (hand crops excluded from the max).

UI labels:

- `Possible generation signal (verify manually)`
- `Inconclusive`
- `No generation signal`

## Protocol / overlay

Output type: `EvidenceProtocol` (Pydantic) — image metadata, checks, evidences, optional `detector`, `anatomy_map`, `layers`.

```bash
python analyze.py path/to/image.jpg --out-dir results
# writes results/<name>.json and results/<name>_overlay.png
```

## HTTP API

For Telegram bots, websites, or any HTTP client. Auth: `X-API-Key` or `Authorization: Bearer <key>`. Keys from env `GAVEL_API_KEYS` (comma-separated). Server refuses to start if that env is empty.

```bash
GAVEL_API_KEYS=dev-key,other-key python api.py
# listens on 0.0.0.0:8000 (override with --host / --port)
```

```bash
curl -s -X POST http://127.0.0.1:8000/v1/analyze \
  -H "X-API-Key: dev-key" \
  -F "file=@samples/real/real_street_florence.jpg" \
  -F "include_overlay=false"
```

`GET /health` needs no key. Response of `/v1/analyze` includes `decision`, `label`, `fusion`, `probability`, full `protocol`, and optional `overlay_png_base64`.

## Training and eval

Training data and feature caches live **outside** the git repo (e.g. `/mnt/disk-e/gavel-data`). Experiment dirs `models/exp_*` are gitignored.

Canonical trainer: `training/train_detector.py`. Research-only (not used in inference): `train_cnn.py`, `extract_layers.py`, `train_head.py`.

Do **not** mix `ai_vs_real_real` into training reals.

```bash
python training/eval_samples_gate.py --strict-hand
python training/eval_hardneg_local.py
python training/eval_faces_local.py

# Cap hands retrain then gate before ship
MAX_REAL_HANDS=20000 OUT=models/exp_hands_cap20k ./training/retrain_when_hands_ready.sh
python training/eval_samples_gate.py \
  --weights models/exp_hands_cap20k/detector_weights.npz \
  --thresholds models/exp_hands_cap20k/thresholds.json \
  --strict-hand

python training/calibrate_artifacts.py --root /mnt/disk-e/gavel-data
```

## Known limitations

1. **Stock close-ups** — L0 can still score high on real gym hands; fusion uses `anatomy_hand_veto` when the anatomy map is clean.
2. **Artifacts** — kurtosis still lights up on some night/texture scenes; soft rule prevents overturning L0=`real`.
3. **Geometry** — aggressive thresholds (0.02) FP on near-parallel real edges; default is **0.05**.
4. **Physics** — classical HSV/contour pairing; layer runs for overlay but does not vote in fusion (FP on real hands/scenes).
5. **OOD AI** — remaining semantic FP/FN without new weights; diptychs / local AI patches are partly covered by halves / quads / edges probes.
6. **Not proof** — do not use as sole evidence for moderation, legal, or journalism claims.

## Tests

```bash
python -m pytest -q
```

Covers validators, fusion (including hand veto and soft artifacts), thresholds, normalization, pipeline wiring.

## Image sources and licenses

**Code** is Apache-2.0 ([LICENSE](LICENSE)). Training corpora and CLIP weights are **not** redistributed in this repo.

| Sample | Source | License |
|---|---|---|
| `samples/real/real_hand_gym.jpg` | [Wikimedia: Close-up Hand holding dumbbell in gym](https://commons.wikimedia.org/wiki/File:Close-up_Hand_holding_dumbbell_in_gym.jpg) | CC BY 2.0 |
| `samples/real/real_street_florence.jpg` | [Wikimedia: Florence - Vespa Vanishing Point](https://commons.wikimedia.org/wiki/File:Florence_-_Vespa_Vanishing_Point.jpg) | CC0 |
| `samples/real/real_corridor_jama_masjid.jpg` | [Wikimedia: Perspective at Jama Masjid](https://commons.wikimedia.org/wiki/File:Perspective_at_Jama_Masjid.jpg) | CC BY-SA 4.0 |

`samples/ai_generated/` is **gitignored** (upstream dataset license unclear) — keep local only; do not redistribute without checking rights.

UI screenshots under `assets/readme/` are from the Gradio demo and may be committed with the docs.

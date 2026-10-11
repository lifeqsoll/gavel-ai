# Gavel AI

Russian version: [README.ru.md](README.ru.md)

Gavel AI is a prototype that looks for signs of image generation. It has two parts:

- A learned detector: a linear probe on frozen CLIP ViT-L/14 features. It returns a score and a
  decision (`generated`, `uncertain`, `real`) at thresholds calibrated on real and generated photos.
- Symbolic checks: perspective geometry, hand anatomy, and light consistency. They run on NumPy and
  SciPy and give exact, visually grounded explanations. They do not decide the label.

The result is a strictly typed JSON object (`EvidenceProtocol`) and an annotated image.

The detector flags some real photos, so Gavel AI does not claim that an image is AI-generated. Its
labels describe a signal. The project is a prototype. This README describes what is implemented, how
the detector was trained and tested, and where it fails.

## Contents

1. [Architecture](#architecture)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Web interface](#web-interface)
5. [Learned detector](#learned-detector)
6. [Checks](#checks)
7. [Protocol format](#protocol-format)
8. [Threshold configuration](#threshold-configuration)
9. [Results on sample images](#results-on-sample-images)
10. [Known limitations](#known-limitations)
11. [Tests](#tests)
12. [Image sources and licenses](#image-sources-and-licenses)

## Architecture

```
gavel-ai/
├── agents/                      perception layer (feature extractors)
│   ├── geometry_agent.py        LSD (fallback: Hough) -> line segments
│   ├── anatomy_agent.py         MediaPipe Hands -> 21 keypoints per hand (3D and px)
│   ├── light_agent.py           shadow objects from a pluggable scene provider
│   └── synthetic.py             synthetic hand and shadow generators (tests, mock run)
├── core/
│   ├── registry.py              Pydantic schemas: Evidence, EvidenceProtocol, payloads
│   ├── observations.py          agent -> validator contracts (HandObservation, LightObservation)
│   ├── pipeline.py              orchestrator: detector, agents, validators, protocol, visualization
│   ├── learned_detector.py      CLIP ViT-L/14 features -> linear probe -> score and decision
│   ├── normalize.py             shared resize, crop and JPEG round trip (training and inference)
│   ├── verdict.py               detector decision (or heuristic score) -> verdict label
│   └── validators/              symbolic layer (deterministic math)
│       ├── homogeneous.py       homogeneous coordinates, normalized lines
│       ├── perspective.py       RANSAC vanishing points, DLT, intersection variance
│       ├── anatomy.py           phalanx proportions, signed joint angles, finger spread
│       └── shadows.py           least-squares ray intersection, directional case
├── visualizer/
│   └── overlay.py               OpenCV: line extensions, joint boxes, yellow shadow rays
├── app.py                       Gradio web interface (upload, overlay, verdict, protocol)
├── analyze.py                   CLI for arbitrary image files
├── mock_run.py                  mock run on a synthetic scene (scenarios: mixed, clean)
├── models/                      detector_weights.npz, thresholds.json, metrics.json
├── training/                    feature extraction, probe training, Wikimedia download, controls
├── tests/test_validators.py     10 unit tests for validators and schemas
├── tests/test_verdict.py        8 tests for the score and verdict labels
├── tests/test_learned_detector.py  18 tests for thresholds, normalization, verdicts, pipeline
├── samples/                     test images (see the licenses section)
├── results/                     output of analyze.py on samples/
└── output/                      output of mock_run.py
```

Data flow:

```
image --> GeometryAgent --> segments --> PerspectiveValidator --> GEO evidence
image --> AnatomyAgent  --> hands    --> AnatomyValidator     --> ANAT evidence
image --> LightAgent    --> shadows  --> ShadowValidator      --> LIGHT evidence
                                  |
                                  v
                    EvidenceProtocol (JSON) + annotated image
```

Each stage is isolated. If an agent cannot run (MediaPipe is missing, no scene provider is
configured, the frame contains no hands), the stage is marked `skipped` with a reason, and the other
stages still run. This lets the LLM distinguish "no anomaly found" from "check not performed".

## Installation

Requirements: Python 3.11 or newer. The learned detector needs `torch` and `open_clip_torch`
(listed in `requirements.txt`). On first use open_clip downloads the ViT-L/14 backbone, about 1.7 GB.
The anatomy check needs MediaPipe 0.10.14 with the legacy `mp.solutions` API, which works on Python
3.11 and older. Without torch or MediaPipe, the other checks still run, and the detector is skipped.

Base installation (geometry and light):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Full installation with the anatomy check (Python 3.11):

```bash
uv venv -p 3.11 .venv311
uv pip install -p .venv311/bin/python -r requirements.txt "mediapipe==0.10.14"
```

Verify the installation:

```bash
python -c "import cv2, numpy, scipy, pydantic; print('ok')"
python -c "import mediapipe; print(mediapipe.__version__)"   # full installation only
```

## Quick start

Run the tests:

```bash
python -m pytest -q
```

Mock run on a synthetic scene. The `mixed` scenario contains three deliberately planted anomalies,
the `clean` scenario contains only consistent cues:

```bash
python mock_run.py --scenario mixed --out-dir output
python mock_run.py --scenario clean --out-dir output
```

Expected output: `mixed` reports three `CRITICAL` evidences (`GEO-02`, `ANAT-01`, `LIGHT-01`) and
`clean` reports none. `output/overlay_mixed.png` shows the red and yellow markings for them.
The unit tests should report `36 passed`.

Analyze real images (MediaPipe is required for the anatomy check):

```bash
python analyze.py samples/real/*.jpg samples/ai_generated/*.jpg --out-dir results
```

For each file the script writes `<name>_overlay.png` (annotated image) and `<name>.json` (protocol),
and prints a summary of checks and found evidence.

Programmatic use:

```python
import cv2
from core.pipeline import GavelPipeline

image = cv2.imread("photo.jpg")
result = GavelPipeline().analyze(image, image_name="photo.jpg")

print(result.protocol.has_critical)
print(result.protocol.to_json())
cv2.imwrite("photo_overlay.png", result.annotated_image)
```

## Web interface

Start the interface from the project root:

```bash
python app.py                  # open http://127.0.0.1:7860
python app.py --share          # temporary public link (Gradio tunnel)
```

Gradio is listed in `requirements.txt`. Use the Python 3.11 environment with MediaPipe if the anatomy
check is needed.

The page has two columns:

- Left: image upload (or clipboard paste), a list of sample images, the line detector
  (`lsd` or `hough`), and the perspective variance threshold. The `Analyze` button runs the pipeline.
- Right: the annotated overlay, the detector score with the verdict label, the list of checks with
  their status and reason, the list of evidence items, the full `EvidenceProtocol` as JSON, and a
  downloadable JSON file.

The same analysis is available over the API as the endpoint `analyze` (for example through
`gradio_client`).

### Verdict

`core/verdict.py` produces the label. It works in two modes.

With the learned detector (default in `app.py` and `analyze.py`), the label follows the detector
decision at two thresholds from `models/thresholds.json`:

| Decision | Condition | Label |
|---|---|---|
| `generated` | score at or above `high` (0.4435) | `Possible generation signal (verify manually)` |
| `uncertain` | score between `low` (0.0184) and `high` | `Inconclusive` |
| `real` | score below `low` | `No generation signal` |

Symbolic findings are listed as explanations. They do not change the label.

Without the detector (torch is missing, or `--no-detector`), the heuristic score is used. Each evidence
adds `CRITICAL` 0.60 or `WARNING` 0.25, combined as `1 - prod(1 - w_i)`. Labels: no evidence and all
checks ran gives `No anomalies found`; no evidence with skipped checks gives
`No anomalies found (limited coverage)`; no check ran gives `Not analyzed`; evidence with a score below
50% gives `Suspicious`; 50% or more gives `Anomalous`. This score is not calibrated.

No label says that an image is AI-generated. The detector score is not the probability that an image
is generated. A low score does not prove that a photo is real, and a high score on a real photo is
possible (see [Results on sample images](#results-on-sample-images)).

## Learned detector

### Method

The image is normalized (shorter side resized to 512, center crop 448, JPEG quality 90) so that format
and size carry no label signal. The normalized image is embedded with frozen CLIP ViT-L/14
(`laion2b_s32b_b82k`). The embedding is L2-normalized, and a logistic regression (`C = 10`,
balanced classes) gives the logit. The score is `sigmoid(logit)`.

Training data, split 70/10/20 per source:

| Data | Role | License | Note |
|---|---|---|---|
| nano-banana images | generated, train | MIT (dataset card) | |
| FLUX-schnell (CommunityForensics commercial, shard 0) | generated, train | Apache 2.0 | |
| COCO 2017 validation and test images | real, train | annotations CC BY 4.0, images under Flickr terms | 20,000 extracted |
| Open Images subset (`bitmind/open-images-v7-subset`, 20 shards) | real, train | per-image terms | 15,000 extracted; terms not verified |
| Wikimedia Commons photos (800) | real, train | CC0, public domain, CC BY, CC BY-SA per file | list in the manifest (`training/fetch_wikimedia.py`) |

Held-out evaluation only, not used for training: DALL-E 3, Firefly 2, Firefly 3, Ideogram (from
CommunityForensics commercial, restricted output terms). Excluded entirely: FLUX-dev (non-commercial)
and CommunityForensics-Small (CC BY-NC-SA).

The three sample photos in `samples/real/` are not in the Wikimedia training set. They serve as the
final check and were not used for threshold selection.

Thresholds are taken from the calibration split only. `high` is the score at which 1% of real
calibration photos are flagged. `low` is the score below which 0.1% of generated calibration images
fall.

### Results on held-out test splits

| Test set | AUC | TPR at 1% FPR |
|---|---|---|
| Generated vs COCO real | 0.9995 | 0.988 |
| Generated vs Open Images real | 0.9987 | 0.975 |
| Generated vs Wikimedia real | 0.9931 | 0.902 |
| Held-out DALL-E 3 vs real | 0.9986 | 0.975 |
| Held-out Firefly 2 vs real | 0.9867 | 0.810 |
| Held-out Firefly 3 vs real | 0.9945 | 0.913 |
| Held-out Ideogram vs real | 0.9963 | 0.930 |

AUC looks high, but the decision shares below show the real problem.

Decision shares on the test splits (`generated` / `uncertain` / `real`):

| Test set | generated | uncertain | real |
|---|---|---|---|
| Generated, in distribution | 98.1% | 1.8% | 0.1% |
| COCO real | 0.4% | 9.0% | 90.7% |
| Open Images real | 1.5% | 12.1% | 86.3% |
| Wikimedia real | 11.3% | 33.1% | 55.6% |
| Held-out DALL-E 3 | 97.7% | 2.3% | 0.0% |
| Held-out Firefly 2 | 81.6% | 15.7% | 2.7% |
| Held-out Firefly 3 | 91.6% | 7.6% | 0.8% |
| Held-out Ideogram | 93.3% | 6.1% | 0.6% |

The Wikimedia row is the important one. Real photos with a polished composition are flagged at a rate
far above the 1% design target. Adding 560 Wikimedia photos to training did not remove this. Two
earlier versions (COCO only, and COCO with Open Images) were also tested. COCO only flagged 11% of Open
Images photos at the COCO threshold, which showed that the probe partly learned the source of the
photos.

### Why this is not enough

CLIP features respond to composition, subject and finish. A professionally lit stock photo can look
like a generated one to the probe. The normalization that removes format and size confounds also
removes low-level traces of generation, so the probe mostly sees content and style. Better calibration
on more real photos reduces the problem but does not remove it.

## Checks

### 1. Perspective geometry (`geometry.perspective`)

`GeometryAgent` detects line segments with LSD (`cv2.createLineSegmentDetector`). If LSD is
unavailable, `cv2.HoughLinesP` is used. `PerspectiveValidator`:

1. Converts segments to normalized homogeneous lines `l = (a, b, c)` with `a^2 + b^2 = 1`. Then
   `|l . p| / |p_w|` is the Euclidean distance in pixels from a point to the line.
2. Drops segments shorter than 3% of the image diagonal.
3. Runs an iterative RANSAC for up to three vanishing points. A hypothesis `v = l_i x l_j` is built
   from a random pair of lines whose angle is at least 8 degrees. A line is an inlier if
   `|l . v| / |v_w| < 4 px`. Consensus is weighted by segment length. All hypotheses are scored in
   one matrix operation of shape `(N, H)`.
4. Refines each cluster with DLT: `v` is the right singular vector of `diag(sqrt(w)) * L` associated
   with the smallest singular value. Iterations repeat while the inlier mask changes.
5. Computes all pairwise intersections of the cluster lines. The robust variance is
   `(1.4826 * MAD_x)^2 + (1.4826 * MAD_y)^2`, normalized by the image diagonal.
6. Pairs that intersect outside a region of radius 10 diagonals around the image center are counted
   as divergent.

Segments that join no cluster are reported as random (`rejected_segment_indices`).

Severity:
- `CRITICAL`: robust spread above `variance_threshold_norm` (2%), or divergent pair fraction above 50%.
- `WARNING`: values above half of the threshold.

The `GEOMETRY_PERSPECTIVE_VIOLATION` evidence contains the vanishing point, the cluster segments, and
the erroneous crossings. The visualizer extends both segments of each erroneous crossing to the
crossing point with thin red lines.

### 2. Hand anatomy (`anatomy.hands`)

`AnatomyAgent` runs MediaPipe Hands in static-image mode and returns 21 keypoints per hand in two
coordinate systems: meters (world landmarks) and image pixels.

`AnatomyValidator` works with 3D coordinates, so proportions do not depend on scale:

1. Phalanx length ratios `distal/proximal` and `middle/proximal` are compared with approximate
   norms (average adult phalanx lengths, table `ANATOMICAL_PHALANX_MM`). A deviation above 35%
   gives `CRITICAL`, above 20% gives `WARNING`.
2. Signed joint flexion `alpha = arccos(a . b / (|a| |b|))`, where `a = joint - prev` and
   `b = next - joint`. Sign: `sign((a x b) . (f x n))`, where `f` is the hand axis (wrist to the base
   of the middle finger) and `n` is the palm normal. Positive sign means flexion toward the palm,
   negative sign means hyperextension. Allowed ranges:
   - PIP: -15 to 110 degrees;
   - DIP: -20 to 90 degrees;
   - thumb MCP: -20 to 70 degrees;
   - thumb IP: -20 to 90 degrees.
   A value outside its range gives `CRITICAL`.
3. Angle between the proximal phalanges of adjacent fingers (index and middle, middle and ring, ring
   and pinky). Above 45 degrees gives `CRITICAL`.

The `ANATOMY_HAND_DEFORMATION` evidence contains one pixel box per flagged joint. The visualizer draws
red rectangles around these joints.

### 3. Light source consistency (`lighting.consistency`)

This check works with objects for which the vertex `V` and the shadow tip `S` are known. The ray
`L -> V -> S` is straight, so a point light `L` lies on the line `SV` of every object.
`ShadowValidator`:

1. Builds the lines `SV` for all objects and finds their least-squares intersection `L` via SVD.
2. Computes the RMS residual `|l . L| / |L_w|` and normalizes it by the diagonal. A value above 2%
   gives `CRITICAL`.
3. Checks that `L` lies beyond the vertex as seen from the shadow tip: `(L - V) . (V - S) > 0`.
   A violation gives `CRITICAL`, because the configuration is physically impossible.
4. If the light estimate goes to infinity (directional light), checks that all shadow directions agree
   within 5 degrees.

The `LIGHT_CONSISTENCY_VIOLATION` evidence contains the object boxes, their points, the light
estimate, and the pairwise intersections. The visualizer draws yellow rays from each shadow tip
through its vertex toward the light estimate.

**Important:** the project does not include an extractor that finds objects and shadows on an
arbitrary frame. `LightAgent` without a configured scene provider returns `skipped`. As a result, this
check is not run on real images at the moment. It has only been verified on synthetic data.

## Protocol format

`EvidenceProtocol` is serialized to JSON with `result.protocol.to_json()`.

Top level:

| Field | Type | Description |
|---|---|---|
| `protocol_version` | string | version of the protocol schema |
| `image` | object | `file_name`, `width`, `height`, `analyzed_at_utc` |
| `checks` | array | one entry per check: `check`, `status` (`passed`, `violation`, `skipped`), `reason` |
| `evidences` | array | found evidence |
| `detector` | object or null | learned detector output: `model`, `probability`, `logit`, `decision`, `high_threshold`, `low_threshold` (null when the detector did not run) |

Each evidence item (`Evidence`):

| Field | Description |
|---|---|
| `evidence_id` | `GEO-NN`, `ANAT-NN`, or `LIGHT-01` |
| `anomaly_type` | `GEOMETRY_PERSPECTIVE_VIOLATION`, `ANATOMY_HAND_DEFORMATION`, or `LIGHT_CONSISTENCY_VIOLATION` |
| `severity` | `CRITICAL` or `WARNING` |
| `description` | human-readable description (Russian) |
| `metrics` | dictionary of finite numbers with exact measurements and thresholds |
| `geometry` / `anatomy` / `lighting` | visualization payload; exactly one is set, matching the type |

The schema is strict: `extra="forbid"`, and `Evidence` rejects a payload that does not match its
anomaly type. `NaN` and `inf` values in metrics are not allowed.

## Threshold configuration

All thresholds are set through dataclass configurations passed to the validators.

`PerspectiveConfig` (`core/validators/perspective.py`):

| Parameter | Default | Meaning |
|---|---|---|
| `min_segment_length_ratio` | 0.03 | minimum segment length, fraction of the diagonal |
| `inlier_threshold_px` | 4.0 | RANSAC inlier tolerance, pixels |
| `min_sample_angle_deg` | 8.0 | minimum angle of a line pair sampled by RANSAC |
| `min_inliers` | 4 | minimum cluster size |
| `ransac_iterations` | 2000 | number of hypotheses |
| `max_vanishing_points` | 3 | maximum vanishing points per frame |
| `max_region_factor` | 10.0 | radius of the admissible region, in diagonals |
| `variance_threshold_norm` | 0.02 | robust spread threshold, fraction of the diagonal |
| `divergent_fraction_threshold` | 0.5 | threshold for the fraction of divergent pairs |
| `random_seed` | 42 | RANSAC seed for reproducibility |

`AnatomyConfig` (`core/validators/anatomy.py`):

| Parameter | Default | Meaning |
|---|---|---|
| `proportion_critical_deviation` | 0.35 | proportion deviation for `CRITICAL` |
| `proportion_warning_deviation` | 0.20 | proportion deviation for `WARNING` |
| `adjacent_max_deg` | 45.0 | maximum angle between adjacent fingers |
| `bbox_padding_ratio` | 0.12 | half-size of a joint box, fraction of hand size |

`ShadowConfig` (`core/validators/shadows.py`):

| Parameter | Default | Meaning |
|---|---|---|
| `min_shadow_length_ratio` | 0.02 | minimum shadow length, fraction of the diagonal |
| `consistency_threshold_norm` | 0.02 | light RMS error threshold, fraction of the diagonal |
| `directional_threshold_deg` | 5.0 | direction disagreement threshold for directional light |
| `far_light_factor` | 20.0 | light estimates beyond this radius are treated as directional light |

Detector thresholds are stored in `models/thresholds.json` and set by `training/train_detector.py`.
They are not edited by hand. Calibration reruns on the training data.

`VerdictConfig` (`core/verdict.py`) sets the weights and thresholds of the heuristic score, used only
when the detector did not run:

| Parameter | Default | Meaning |
|---|---|---|
| `critical_weight` | 0.60 | score contribution of one `CRITICAL` evidence |
| `warning_weight` | 0.25 | score contribution of one `WARNING` evidence |
| `anomalous_threshold` | 0.50 | score at or above which the label is `Anomalous` |

## Results on sample images

Run with `python analyze.py samples/real/*.jpg samples/ai_generated/*.jpg --out-dir results` on
Python 3.11 with MediaPipe 0.10.14 and the learned detector (CUDA). Full protocols and overlays are in
`results/`. The light check is skipped on every image because there is no scene provider.

| File | Size | Geometry | Anatomy | Detector score | Evidence |
|---|---|---|---|---|---|
| `real/real_corridor_jama_masjid.jpg` | 1280x1707 | passed | skipped | 0.663 generated | none |
| `real/real_hand_gym.jpg` | 1280x1917 | passed | violation | 0.990 generated | ANAT-01 CRITICAL |
| `real/real_street_florence.jpg` | 1280x960 | violation | skipped | 0.040 uncertain | GEO-01 CRITICAL, GEO-03 CRITICAL |
| `ai_generated/ai_01.jpg` | 880x440 | passed | skipped | 0.011 real | none |
| `ai_generated/ai_02.jpg` | 2940x1960 | skipped | skipped | 1.000 generated | none |
| `ai_generated/ai_03.jpg` | 1067x1690 | violation | skipped | 1.000 generated | GEO-01 WARNING |
| `ai_generated/ai_04.jpg` | 800x800 | passed | skipped | 0.846 generated | none |
| `ai_generated/ai_05.jpg` | 1200x600 | violation | skipped | 0.071 uncertain | GEO-01 CRITICAL, GEO-02 CRITICAL |
| `ai_generated/ai_06.jpg` | 1664x1120 | passed | skipped | 0.634 generated | none |
| `ai_generated/ai_07.jpg` | 1021x574 | violation | skipped | 0.991 generated | GEO-03 WARNING |
| `ai_generated/ai_08.jpg` | 1400x933 | passed | skipped | 1.000 generated | none |
| `ai_generated/ai_09.jpg` | 1024x688 | passed | skipped | 0.952 generated | none |
| `ai_generated/ai_10.jpg` | 1600x1600 | passed | violation | 0.614 generated | ANAT-01 CRITICAL |
| `ai_generated/ai_11.jpg` | 600x600 | passed | skipped | 1.000 generated | none |
| `ai_generated/ai_12.jpg` | 1800x1200 | violation | skipped | 1.000 generated | GEO-03 CRITICAL |

Interpretation:

- The sample is small (3 real and 12 generated images). These numbers are not an accuracy estimate.
- On the 12 generated images, the detector gives a `generated` decision for 10, `uncertain` for `ai_05`
  (a web-page screenshot whose source label is doubtful), and `real` for `ai_01`.
- On the 3 real photos, the detector gives a `generated` decision for 2: `real_hand_gym.jpg` (0.990) and
  `real_corridor_jama_masjid.jpg` (0.663). This is the main failure. The label text says "possible
  signal", but the signal is still on for real photos.
- The symbolic findings on real photos are also false positives. The anatomy check on
  `real_hand_gym.jpg` gives a `CRITICAL` from a flexion sign that looks inverted for real MediaPipe world
  coordinates. The geometry check on `real_street_florence.jpg` reads cobblestone texture and scooter edges
  as perspective lines. The 2% threshold was calibrated on synthetic data only.
- `ai_02.jpg` has only one line segment longer than 3% of the diagonal, so the geometry check is skipped.
- The light check was not run on any image because there is no scene provider.

## Known limitations

1. **The flexion sign is not verified on real data.** The palm orientation convention in
   `palm_frame` was verified only on synthetic hands. On a real image all angles came out negative,
   which points to a systematic inversion. A fix requires checking several real photos (open palm,
   fist, gestures) and probably flipping the sign for MediaPipe world coordinates.
2. **Phalanx proportions are unreliable under strong flexion and oblique viewing.** MediaPipe
   estimates depth heuristically, and lengths of bent fingers are distorted. The proportion check
   should only trigger on hands with small flexion, or be cross-checked against the 2D projection.
3. **The geometry threshold is calibrated on synthetic data.** Real perspective photos produce a
   spread of 2-3% because of textures, lens distortion, and edges. The threshold needs to be tuned on
   a labeled dataset.
4. **The light check has no extractor.** A segmenter for objects and shadows, or manual annotation,
   is required. Without it the check returns `skipped`.
5. **No accuracy evaluation.** There is no labeled set of real and generated images, so accuracy,
   recall, and false positive rate cannot be claimed.
6. **MediaPipe 0.10.14 does not run on Python 3.12 or newer.** The anatomy check needs Python 3.11 or
   older.
7. **RANSAC sampling.** When a cluster's angular spread is smaller than `min_sample_angle_deg` (8
   degrees), no hypothesis can be sampled and the cluster is not found. Such frames return `passed`
   rather than `skipped`.
8. **The detector flags real photos.** Two of three real sample photos get a `generated` decision.
   On 800 Wikimedia photos, 11% of the held-out real set gets `generated` and 33% get `uncertain`. The
   probe responds to composition and finish, and the normalization removes low-level generation traces.
   It cannot serve as a standalone verdict.
9. **Generator coverage is narrow.** Positives are Nano Banana and FLUX-schnell. Held-out tests cover
   DALL-E 3, Firefly and Ideogram. Midjourney, Imagen, Stable Diffusion and newer models are not tested.
10. **Per-source false positive rates differ.** At the shared threshold the `generated` share on real
    photos is 0.4% for COCO, 1.5% for Open Images and 11% for Wikimedia.
11. **Dataset labels and licenses are not verified.** `ai_05.jpg` is a web page screenshot, so some
    source labels may be wrong. Open Images per-image terms are not checked. Wikimedia photos under
    CC BY-SA may carry share-alike terms that could extend to the weights. These points must be resolved
    before any release of the weights.
12. **The heuristic score is not a probability.** It is used only when the detector did not run
    (see [Verdict](#verdict)). It must not be quoted as the chance that an image is generated.

## Tests

`tests/test_verdict.py` contains 8 tests for the heuristic score and its labels: clean, partial coverage,
not analyzed, combination of independent evidence (parametrized), and the upper bound of 1.0.

`tests/test_learned_detector.py` contains 18 tests: decision boundaries, ordered thresholds, loading
weights with thresholds, a missing thresholds file, normalization output size, verdicts for each
decision, symbolic findings that do not change the detector label, and the pipeline with a fake detector,
a failing detector, and no detector.

`tests/test_validators.py` contains 10 tests:

- RANSAC finds a converging family of lines and produces no evidence for it;
- an inconsistent family of near-parallel lines produces `CRITICAL`;
- a normal hand produces no evidence;
- a distal phalanx proportion violation produces `CRITICAL`;
- hyperextension yields a negative signed angle;
- positive flexion is measured correctly;
- a consistent light source produces no evidence and is recovered to within 1 px;
- an inconsistent light source produces `CRITICAL`;
- light behind the vertices produces `CRITICAL`;
- the `Evidence` schema rejects a payload that does not match its type.

The full suite (36 tests) was run on Python 3.11 with MediaPipe 0.10.14 and torch.

## Image sources and licenses

The project code is licensed separately from the images in `samples/`. The images come from open
sources and are used only to test the pipeline.

| File | Source | License | Requirement |
|---|---|---|---|
| `samples/real/real_hand_gym.jpg` | [Close-up Hand holding dumbbell in gym](https://commons.wikimedia.org/wiki/File:Close-up_Hand_holding_dumbbell_in_gym.jpg), Wikimedia Commons | CC BY 2.0 | credit the author and license |
| `samples/real/real_street_florence.jpg` | [Florence - Vespa Vanishing Point](https://commons.wikimedia.org/wiki/File:Florence_-_Vespa_Vanishing_Point.jpg), Wikimedia Commons | CC0 | none |
| `samples/real/real_corridor_jama_masjid.jpg` | [Perspective at Jama Masjid](https://commons.wikimedia.org/wiki/File:Perspective_at_Jama_Masjid.jpg), Wikimedia Commons | CC BY-SA 4.0 | credit the author and license; derivatives under the same license |
| `samples/ai_generated/ai_01.jpg` to `ai_12.jpg` | [Hemg/AI-Generated-vs-Real-Images-Datasets](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets), Hugging Face | the dataset card declares no license | use for testing only; do not redistribute without checking rights |

Author details for each photograph are on the Wikimedia Commons pages linked above.

Training photos for the learned detector are listed under [Learned detector](#learned-detector). The
Wikimedia training photos and their licenses are recorded in the manifest written by
`training/fetch_wikimedia.py`.

## License

The source code is licensed under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The license does not cover the sample images in `samples/` or any training data. Model weights trained
on third-party data inherit the terms of that data. See [Image sources and licenses](#image-sources-and-licenses).

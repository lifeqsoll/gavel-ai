"""Shared image normalization for training features and inference.

Every image is resized, center-cropped and JPEG round-tripped to the same
format and size, so format and resolution carry no label signal. Training
(training/extract_features.py) and inference (core/learned_detector.py) must
use this same function, otherwise the probability thresholds do not apply.
"""

from __future__ import annotations

import io

from PIL import Image

NORMALIZE_SIZE = 512
NORMALIZE_CROP = 448
NORMALIZE_QUALITY = 90


def normalize_pil(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    scale = NORMALIZE_SIZE / min(img.size)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
    left = (img.width - NORMALIZE_CROP) // 2
    top = (img.height - NORMALIZE_CROP) // 2
    img = img.crop((left, top, left + NORMALIZE_CROP, top + NORMALIZE_CROP))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=NORMALIZE_QUALITY)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

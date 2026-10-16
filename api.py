"""HTTP API for Gavel AI (Telegram bots, websites, scripts).

Auth: X-API-Key or Authorization: Bearer <key>.
Keys from env GAVEL_API_KEYS (comma-separated). Empty keys => refuse to start.

Usage:
    GAVEL_API_KEYS=dev-key python api.py
    GAVEL_API_KEYS=dev-key python api.py --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import base64
import os
import threading
from dataclasses import replace
from functools import lru_cache

import cv2
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.security import APIKeyHeader

from agents.anatomy_agent import AnatomyAgent
from agents.geometry_agent import GeometryAgent
from core.learned_detector import LearnedDetector
from core.pipeline import GavelPipeline
from core.registry import PROTOCOL_VERSION
from core.validators.perspective import PerspectiveConfig, PerspectiveValidator
from core.verdict import compute_verdict

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_run_lock = threading.Lock()
_anatomy_agent = AnatomyAgent()


def _load_api_keys() -> set[str]:
    raw = os.environ.get("GAVEL_API_KEYS", "").strip()
    if not raw:
        raise RuntimeError("GAVEL_API_KEYS is empty; set at least one key before starting the API")
    return {k.strip() for k in raw.split(",") if k.strip()}


def create_app(api_keys: set[str] | None = None) -> FastAPI:
    keys = api_keys if api_keys is not None else _load_api_keys()
    app = FastAPI(title="Gavel AI", version=PROTOCOL_VERSION)

    def require_key(
        x_api_key: str | None = Depends(_API_KEY_HEADER),
        authorization: str | None = Header(default=None),
    ) -> str:
        token = (x_api_key or "").strip()
        if not token and authorization:
            auth = authorization.strip()
            if auth.lower().startswith("bearer "):
                token = auth[7:].strip()
            else:
                token = auth
        if not token or token not in keys:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return token

    @lru_cache(maxsize=1)
    def _detector() -> LearnedDetector | None:
        try:
            return LearnedDetector()
        except FileNotFoundError:
            return None

    @lru_cache(maxsize=16)
    def _pipeline(geometry_method: str, variance_threshold: float) -> GavelPipeline:
        cfg = replace(PerspectiveConfig(), variance_threshold_norm=variance_threshold)
        return GavelPipeline(
            geometry_agent=GeometryAgent(method=geometry_method),
            perspective_validator=PerspectiveValidator(cfg),
            anatomy_agent=_anatomy_agent,
            learned_detector=_detector(),
        )

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "version": PROTOCOL_VERSION}

    @app.post("/v1/analyze")
    async def analyze(
        file: UploadFile = File(...),
        geometry_method: str = Form("lsd"),
        variance_threshold: float = Form(0.05),
        include_overlay: bool = Form(False),
        _key: str = Depends(require_key),
    ) -> dict:
        if geometry_method not in {"lsd", "hough"}:
            raise HTTPException(status_code=400, detail="geometry_method must be lsd or hough")
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty file")
        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="cannot decode image")

        pipe = _pipeline(geometry_method, float(variance_threshold))
        name = file.filename or "upload"
        with _run_lock:
            result = pipe.analyze(image, image_name=name)
        verdict = compute_verdict(result.protocol)
        body: dict = {
            "decision": verdict.decision,
            "label": verdict.label,
            "fusion": verdict.fusion,
            "probability": verdict.probability,
            "high_threshold": verdict.high_threshold,
            "protocol": result.protocol.model_dump(mode="json"),
        }
        if include_overlay:
            ok, buf = cv2.imencode(".png", result.annotated_image)
            if not ok:
                raise HTTPException(status_code=500, detail="overlay encode failed")
            body["overlay_png_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")
        return body

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Gavel AI HTTP API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

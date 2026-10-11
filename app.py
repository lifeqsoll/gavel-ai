"""Gradio web interface for Gavel AI: upload, analyze, overlay, verdict, protocol download.

Usage:
    python app.py                 # http://127.0.0.1:7860
    python app.py --share         # temporary public link
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr

from agents.anatomy_agent import AnatomyAgent
from agents.geometry_agent import GeometryAgent
from core.learned_detector import LearnedDetector
from core.pipeline import GavelPipeline
from core.validators.perspective import PerspectiveConfig, PerspectiveValidator
from core.verdict import Verdict, compute_verdict

ROOT = Path(__file__).resolve().parent
SAMPLE_DIRS = (ROOT / "samples" / "ai_generated", ROOT / "samples" / "real")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# MediaPipe Hands is shared across requests and is not safe for concurrent calls.
_anatomy_agent = AnatomyAgent()
_run_lock = threading.Lock()

EVIDENCE_HEADERS = ["ID", "Anomaly", "Severity", "Description"]
CHECK_HEADERS = ["Check", "Status", "Reason"]


def sample_paths() -> list[str]:
    paths: list[Path] = []
    for folder in SAMPLE_DIRS:
        if folder.is_dir():
            paths.extend(sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
    return [str(p) for p in paths]


@lru_cache(maxsize=1)
def _get_detector() -> LearnedDetector | None:
    try:
        return LearnedDetector()
    except FileNotFoundError as exc:
        print(f"learned detector disabled: {exc}")
        return None


@lru_cache(maxsize=16)
def _build_pipeline(geometry_method: str, variance_threshold: float) -> GavelPipeline:
    perspective_config = replace(PerspectiveConfig(), variance_threshold_norm=variance_threshold)
    return GavelPipeline(
        geometry_agent=GeometryAgent(method=geometry_method),
        perspective_validator=PerspectiveValidator(perspective_config),
        anatomy_agent=_anatomy_agent,
        learned_detector=_get_detector(),
    )


def _summary_markdown(verdict: Verdict) -> str:
    if verdict.probability is not None:
        return _detector_summary(verdict)
    lines = [
        f"### {verdict.label}",
        "",
        f"**Anomaly score: {verdict.score * 100:.0f}%**  ",
        f"Coverage: {verdict.coverage_note}",
        "",
        "The learned detector did not run, so this verdict uses the heuristic score. "
        "The score is not a calibrated probability that the image is AI-generated.",
    ]
    if verdict.findings:
        lines += ["", "**Findings**", ""]
        lines += [f"- {finding}" for finding in verdict.findings]
    return "\n".join(lines)


def _detector_summary(verdict: Verdict) -> str:
    lines = [
        f"### {verdict.label}",
        "",
        f"**Detector score: {verdict.probability * 100:.1f}%**  ",
        f"Coverage: {verdict.coverage_note}",
        "",
        "The detector score is not the probability that the image is AI-generated. "
        "It is a linear probe on frozen CLIP features. It can flag real photos with a polished "
        "composition, and it has not been validated on the generators you care about. "
        "Use it as one signal and check the image yourself.",
    ]
    if verdict.findings:
        lines += [
            "",
            "**Symbolic findings (explanation only, they do not change the verdict)**",
            "",
        ]
        lines += [f"- {finding}" for finding in verdict.findings]
    return "\n".join(lines)


def analyze_image(image_path: str | None, geometry_method: str, variance_threshold: float):
    if not image_path:
        raise gr.Error("Upload an image or pick a sample first.")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise gr.Error("The file could not be decoded as an image.")

    pipeline = _build_pipeline(geometry_method, float(variance_threshold))
    with _run_lock:
        result = pipeline.analyze(image, image_name=Path(image_path).name)

    protocol = result.protocol
    verdict = compute_verdict(protocol)
    annotated_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)

    check_rows = [[c.check, c.status, c.reason] for c in protocol.checks]
    evidence_rows = [
        [e.evidence_id, e.anomaly_type.value, e.severity.value, e.description] for e in protocol.evidences
    ]

    json_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{Path(image_path).stem}.json", delete=False, encoding="utf-8"
    )
    json_file.write(protocol.to_json())
    json_file.close()

    return (
        annotated_rgb,
        _summary_markdown(verdict),
        check_rows,
        evidence_rows,
        json.loads(protocol.to_json()),
        json_file.name,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Gavel AI") as demo:
        gr.Markdown(
            "# Gavel AI\n"
            "Neuro-symbolic checks for perspective geometry, hand anatomy and lighting. "
            "Upload an image or pick a sample, then run the analysis."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Image", sources=["upload", "clipboard"])
                geometry_method = gr.Dropdown(
                    choices=["lsd", "hough"],
                    value="lsd",
                    label="Line detector",
                    info="LSD is the default. Hough is a fallback.",
                )
                variance_threshold = gr.Slider(
                    minimum=0.005,
                    maximum=0.1,
                    value=0.02,
                    step=0.001,
                    label="Perspective variance threshold (normalized)",
                    info="Lower values flag weaker line-family conflicts.",
                )
                run_button = gr.Button("Analyze", variant="primary")
                gr.Examples(
                    examples=[[path] for path in sample_paths()],
                    inputs=[image_input],
                    label="Samples",
                    examples_per_page=15,
                )
            with gr.Column(scale=1):
                annotated_output = gr.Image(label="Annotated overlay", type="numpy")
                summary = gr.Markdown()
                with gr.Accordion("Checks", open=True):
                    checks_table = gr.Dataframe(headers=CHECK_HEADERS, interactive=False, wrap=True)
                with gr.Accordion("Evidences", open=True):
                    evidence_table = gr.Dataframe(headers=EVIDENCE_HEADERS, interactive=False, wrap=True)
                with gr.Accordion("EvidenceProtocol JSON", open=False):
                    protocol_json = gr.JSON()
                    protocol_file = gr.File(label="Download protocol")

        run_button.click(
            fn=analyze_image,
            inputs=[image_input, geometry_method, variance_threshold],
            outputs=[annotated_output, summary, checks_table, evidence_table, protocol_json, protocol_file],
            api_name="analyze",
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gavel AI web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_app().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

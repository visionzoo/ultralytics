"""Export the DMS encoder and ROI heads as an inseparable ONNX pair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .model import DMSModel
from .schema import LANDMARK_NAMES, PART_SLOTS


class EncoderExport(nn.Module):
    """ONNX wrapper: image -> global raw detection plus B2/P3 feature maps."""

    def __init__(self, model: DMSModel):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model.forward_encoder(images)
        global_raw = output.global_raw[0] if isinstance(output.global_raw, tuple) else output.global_raw
        return global_raw, output.b2, output.p3


class ROIHeadsExport(nn.Module):
    """ONNX wrapper: host-cropped B2/P3 tensors -> all per-face predictions."""

    def __init__(self, model: DMSModel):
        super().__init__()
        self.heads = model.dms_heads

    def forward(
        self, b2_rois: torch.Tensor, p3_rois: torch.Tensor, roi_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.heads(b2_rois, p3_rois, roi_valid)
        return (
            output["part_box"],
            output["part_presence_logits"],
            output["eye_state_logits"],
            output["mouth_state_logits"],
            output["landmark_heatmap"],
            output["pose_ypr"],
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_dms_onnx(
    model: DMSModel,
    output_dir: str | Path,
    image_size: int = 640,
    max_faces: int = 4,
    opset: int = 17,
) -> dict:
    """Export static-shape encoder/ROI ONNX files and their compatibility manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    image = torch.zeros(1, 3, image_size, image_size, device=next(model.parameters()).device)
    with torch.inference_mode():
        _, b2, p3 = EncoderExport(model)(image)
        # Materialize LazyConv2d before serializing the ROI graph.
        model.forward_roi(
            torch.zeros(max_faces, b2.shape[1], 32, 32, device=image.device),
            torch.zeros(max_faces, p3.shape[1], 16, 16, device=image.device),
            torch.ones(max_faces, dtype=torch.bool, device=image.device),
        )

    encoder_path, roi_path = output_dir / "dms_encoder.onnx", output_dir / "dms_roi_heads.onnx"
    torch.onnx.export(
        EncoderExport(model), image, encoder_path, opset_version=opset,
        input_names=["images"], output_names=["global_raw", "b2", "p3"],
    )
    torch.onnx.export(
        ROIHeadsExport(model),
        (
            torch.zeros(max_faces, b2.shape[1], 32, 32, device=image.device),
            torch.zeros(max_faces, p3.shape[1], 16, 16, device=image.device),
            torch.ones(max_faces, dtype=torch.bool, device=image.device),
        ),
        roi_path,
        opset_version=opset,
        input_names=["b2_rois", "p3_rois", "roi_valid"],
        output_names=["part_box", "part_presence_logits", "eye_state_logits", "mouth_state_logits", "landmark_heatmap", "pose_ypr"],
    )
    manifest = {
        "schema_version": 1,
        "image_size": image_size,
        "max_faces": max_faces,
        "roi_align": {"aligned": True, "b2_stride": 4, "p3_stride": 8, "b2_size": 32, "p3_size": 16},
        "part_slots": PART_SLOTS,
        "landmark_names": LANDMARK_NAMES,
        "state_vocabularies": {"eye": ["staring", "squint"], "mouth": ["normal", "yawn", "cigarette"]},
        "artifacts": {encoder_path.name: _sha256(encoder_path), roi_path.name: _sha256(roi_path)},
    }
    (output_dir / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

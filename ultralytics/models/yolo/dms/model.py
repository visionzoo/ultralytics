"""YOLO26 DMS model with encoder and exportable ROI boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ultralytics.nn.modules.dms import DMSROIHeads, roi_align_features
from ultralytics.nn.tasks import DetectionModel


@dataclass
class EncoderOutput:
    global_raw: torch.Tensor | tuple
    b2: torch.Tensor
    p3: torch.Tensor
    b2_stride: int
    p3_stride: int


class DMSModel(DetectionModel):
    """YOLO26 detector which exposes B2/P3 once for all face ROI tasks."""

    def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.dms_heads = DMSROIHeads()

    @staticmethod
    def _stride_for(image: torch.Tensor, feature: torch.Tensor) -> int | None:
        if feature.ndim != 4 or image.shape[-2] % feature.shape[-2] or image.shape[-1] % feature.shape[-1]:
            return None
        height_stride, width_stride = image.shape[-2] // feature.shape[-2], image.shape[-1] // feature.shape[-1]
        return height_stride if height_stride == width_stride else None

    def forward_encoder(self, image: torch.Tensor) -> EncoderOutput:
        """Execute the normal YOLO graph once and retain its final s4 and s8 maps."""
        saved, x, features = [], image, {}
        for layer in self.model:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if item == -1 else saved[item] for item in layer.f]
            x = layer(x)
            saved.append(x)
            if isinstance(x, torch.Tensor):
                stride = self._stride_for(image, x)
                if stride in (4, 8):
                    features[stride] = x
        if 4 not in features or 8 not in features:
            raise RuntimeError("YOLO graph did not expose required B2/P3 feature maps")
        return EncoderOutput(x, features[4], features[8], 4, 8)

    def forward_roi(
        self, b2_rois: torch.Tensor, p3_rois: torch.Tensor, roi_valid: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Exportable ROI-only path; it deliberately does not contain ROIAlign."""
        return self.dms_heads(b2_rois, p3_rois, roi_valid)

    def forward_dms(
        self, image: torch.Tensor, face_rois: torch.Tensor | None = None, roi_valid: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | tuple]:
        """Convenience PyTorch path; feature ROIAlign is used only when ROIs are supplied."""
        encoder = self.forward_encoder(image)
        result: dict[str, torch.Tensor | tuple] = {
            "global_raw": encoder.global_raw,
            "b2": encoder.b2,
            "p3": encoder.p3,
        }
        if face_rois is not None:
            b2_rois, p3_rois = roi_align_features(
                encoder.b2, encoder.p3, face_rois, encoder.b2_stride, encoder.p3_stride
            )
            result.update(self.forward_roi(b2_rois, p3_rois, roi_valid))
        return result

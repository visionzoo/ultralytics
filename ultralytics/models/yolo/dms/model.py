"""DMS model: preserve YOLO26 global detection and reuse B2/P3 for face ROI tasks."""

from __future__ import annotations

import torch

from ultralytics.nn.modules.dms import DMSROIHeads
from ultralytics.nn.tasks import DetectionModel


class DMSModel(DetectionModel):
    """YOLO26 detector with a separate, opt-in ROI branch for DMS tasks."""

    def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, state_classes=5, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.dms_heads = DMSROIHeads(state_classes=state_classes)

    def forward_dms(self, image: torch.Tensor, face_rois: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Run the backbone once; ``face_rois`` is Nx5 in input-image coordinates."""
        saved, x = [], image
        b2 = p3 = None
        for layer in self.model:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            x = layer(x)
            saved.append(x)
            if layer.i == 2:
                b2 = x
            elif layer.i == 16:
                p3 = x
        result = {"global_det": x, "b2": b2, "p3": p3}
        if face_rois is not None:
            result.update(self.dms_heads(b2, p3, face_rois))
        return result

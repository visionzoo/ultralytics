"""YOLO26 DMS model with encoder and exportable ROI boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ultralytics.nn.modules.dms import build_roi_head, roi_align_features
from ultralytics.nn.tasks import DetectionModel

from .loss import DMSROILoss
from ultralytics.nn.modules.dms_pose import POSE_BIN_CONFIG


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
        yaml_cfg = getattr(self, "yaml", {})
        dms_cfg = yaml_cfg.get("dms", {}) if isinstance(yaml_cfg, dict) else {}
        self.feature_strides = tuple(int(value) for value in dms_cfg.get("feature_strides", (4, 8)))
        if len(self.feature_strides) != 2 or len(set(self.feature_strides)) != 2:
            raise ValueError("dms.feature_strides must contain exactly two distinct strides")
        configured_layers = dms_cfg.get("feature_layers")
        if configured_layers is None:
            # YOLO26n's first P2/4 backbone output and the final P3/8 neck
            # output. Custom backbones must declare their own taps explicitly.
            configured_layers = (1, 16)
        self.feature_layers = tuple(int(value) for value in configured_layers)
        if len(self.feature_layers) != 2 or len(set(self.feature_layers)) != 2:
            raise ValueError("dms.feature_layers must contain exactly two distinct layer indices")
        self.roi_head_name = str(dms_cfg.get("roi_head", "default"))
        self.roi_head_kwargs = dict(dms_cfg.get("roi_kwargs", {}))
        self.dms_heads = build_roi_head(self.roi_head_name, **self.roi_head_kwargs)
        self.dms_architecture = {
            "roi_head": self.roi_head_name,
            "roi_kwargs": self.roi_head_kwargs,
            "feature_strides": list(self.feature_strides),
            "feature_layers": list(self.feature_layers),
            "landmark_decoder": "heatmap_quarter_pixel_v3",
            "pose_decoder": "per_axis_soft_bin_expectation_residual_v1",
            "pose_bin_config": POSE_BIN_CONFIG,
        }
        self.dms_criterion = DMSROILoss()

    @staticmethod
    def _stride_for(image: torch.Tensor, feature: torch.Tensor) -> int | None:
        if feature.ndim != 4 or image.shape[-2] % feature.shape[-2] or image.shape[-1] % feature.shape[-1]:
            return None
        height_stride, width_stride = image.shape[-2] // feature.shape[-2], image.shape[-1] // feature.shape[-1]
        return height_stride if height_stride == width_stride else None

    def forward_encoder(self, image: torch.Tensor) -> EncoderOutput:
        """Execute the normal YOLO graph once and retain configured feature maps."""
        saved, x, features = [], image, {}
        taps = set(self.feature_layers)
        for index, layer in enumerate(self.model):
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if item == -1 else saved[item] for item in layer.f]
            x = layer(x)
            saved.append(x)
            if index in taps:
                if not isinstance(x, torch.Tensor):
                    raise RuntimeError(f"configured DMS feature layer {index} did not return a tensor")
                stride = self._stride_for(image, x)
                expected = self.feature_strides[len(features)]
                if stride != expected:
                    raise RuntimeError(
                        f"configured DMS feature layer {index} has stride {stride}, expected {expected}"
                    )
                features[index] = x
        missing = [layer for layer in self.feature_layers if layer not in features]
        if missing:
            raise RuntimeError(f"YOLO graph did not expose configured feature layers: {missing}")
        first, second = self.feature_layers
        return EncoderOutput(x, features[first], features[second], self.feature_strides[0], self.feature_strides[1])

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

    def loss(self, batch: dict, preds=None):
        """Route global batches to native YOLO loss and ROI batches to DMS loss."""
        task = batch.get("task", "global")
        if isinstance(task, (list, tuple)):
            if len(set(task)) != 1:
                raise ValueError("a DMS batch must contain exactly one task")
            task = task[0]
        if task == "global":
            return super().loss(batch, preds)
        required = {"img", "face_rois"}
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"ROI batch is missing: {sorted(missing)}")
        outputs = self.forward_dms(batch["img"], batch["face_rois"], batch.get("roi_valid"))
        total, items = self.dms_criterion(outputs, batch)
        return total, {name: value.detach() for name, value in items.items()}

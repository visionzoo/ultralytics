"""Small DMS ROI heads built on top of already-cropped YOLO features."""

from __future__ import annotations

import importlib

import torch
from torch import nn

from .dms_pose import POSE_AXES, decode_pose, pose_bin_count


_ROI_HEAD_FACTORIES: dict[str, object] = {}


def register_roi_head(name: str):
    """Register a named ROI-head factory without changing DMSModel."""
    def decorator(factory):
        if name in _ROI_HEAD_FACTORIES:
            raise KeyError(f"ROI head already registered: {name}")
        _ROI_HEAD_FACTORIES[name] = factory
        return factory

    return decorator


def build_roi_head(name: str = "default", **kwargs) -> nn.Module:
    """Build a registered ROI head; external modules may register their own heads."""
    if ":" in name:
        module_name, class_name = name.split(":", 1)
        factory = getattr(importlib.import_module(module_name), class_name)
        return factory(**kwargs)
    try:
        factory = _ROI_HEAD_FACTORIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_ROI_HEAD_FACTORIES)) or "<none>"
        raise KeyError(f"Unknown ROI head {name!r}; available: {available}") from exc
    return factory(**kwargs)


class DWBlock(nn.Module):
    """Depthwise pointwise block used only after a face ROI has been extracted."""

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        # ROI batches alternate between DSM parts, Eye12 landmarks and
        # 300W-LP pose.  BatchNorm running statistics therefore mix three
        # unrelated image/task distributions and make eval-time outputs
        # collapse. GroupNorm keeps train/eval behavior identical.
        groups = 8 if channels % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride, 1, groups=channels, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def roi_align_features(
    b2: torch.Tensor, p3: torch.Tensor, rois: torch.Tensor, b2_stride: int, p3_stride: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training-only feature crop; deployment performs this operation in the host."""
    from torchvision.ops import roi_align

    b2_rois = roi_align(b2, rois, (32, 32), spatial_scale=1.0 / b2_stride, aligned=True)
    p3_rois = roi_align(p3, rois, (16, 16), spatial_scale=1.0 / p3_stride, aligned=True)
    return b2_rois, p3_rois


class DMSROIHeads(nn.Module):
    """Fuse B2/P3 face crops and predict fixed parts, landmarks and head pose.

    The three slots are left eye, right eye and mouth. Slots make structure
    explicit, so this head needs neither a second grid detector nor local NMS.
    """

    def __init__(self, eye_states: int = 2, mouth_states: int = 3, channels: int = 96):
        super().__init__()
        self.channels, self.eye_states, self.mouth_states = channels, eye_states, mouth_states
        self.fuse = nn.Sequential(
            nn.LazyConv2d(channels, 1, bias=False),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.SiLU(),
            DWBlock(channels),
            DWBlock(channels),
        )
        self.parts = nn.Sequential(DWBlock(channels), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 15))
        self.eye_state = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 2 * eye_states))
        self.mouth_state = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, mouth_states))
        self.landmark = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, 2, 2), nn.SiLU(), nn.Conv2d(channels // 2, 12, 1)
        )
        self.landmark_logvar = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 12))
        self.pose_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 64), nn.SiLU())
        self.pose_bin = nn.ModuleDict({axis: nn.Linear(64, pose_bin_count(axis)) for axis in POSE_AXES})
        self.pose_residual = nn.ModuleDict({axis: nn.Linear(64, 1) for axis in POSE_AXES})

    def forward(
        self, b2_rois: torch.Tensor, p3_rois: torch.Tensor, roi_valid: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Run the exportable ROI head on `[N,C,32,32]` and `[N,C,16,16]` crops."""
        count = b2_rois.shape[0]
        if count == 0:
            empty = b2_rois.new_empty((0, self.channels, 32, 32))
            return {
                "face_feature": empty,
                "part_box": empty.new_empty((0, 3, 4)),
                "part_presence_logits": empty.new_empty((0, 3)),
                "eye_state_logits": empty.new_empty((0, 2, self.eye_states)),
                "mouth_state_logits": empty.new_empty((0, 1, self.mouth_states)),
                "landmark_heatmap": empty.new_empty((0, 12, 64, 64)),
                "landmark_logvar": empty.new_empty((0, 12)),
                "pose_ypr": empty.new_empty((0, 3)),
                **{f"{axis}_bin_logits": empty.new_empty((0, pose_bin_count(axis))) for axis in POSE_AXES},
                **{f"{axis}_residual": empty.new_empty((0, 1)) for axis in POSE_AXES},
            }
        p3_rois = nn.functional.interpolate(p3_rois, size=(32, 32), mode="bilinear", align_corners=False)
        face = self.fuse(torch.cat((b2_rois, p3_rois), 1))
        part_values = self.parts(face).reshape(count, 3, 5)
        pose_feature = self.pose_pool(face)
        result = {
            "face_feature": face,
            "part_box": part_values[..., :4].sigmoid(),
            "part_presence_logits": part_values[..., 4],
            "eye_state_logits": self.eye_state(face).reshape(count, 2, self.eye_states),
            "mouth_state_logits": self.mouth_state(face).reshape(count, 1, self.mouth_states),
            "landmark_heatmap": self.landmark(face),
            "landmark_logvar": self.landmark_logvar(face).clamp(-4.0, 4.0),
            **{f"{axis}_bin_logits": self.pose_bin[axis](pose_feature) for axis in POSE_AXES},
            **{f"{axis}_residual": 0.5 * torch.tanh(self.pose_residual[axis](pose_feature)) for axis in POSE_AXES},
        }
        result["pose_ypr"] = decode_pose(result)
        if roi_valid is not None:
            valid = roi_valid.to(dtype=face.dtype).reshape(count, 1)
            result["part_presence_logits"] = result["part_presence_logits"] * valid
            result["landmark_heatmap"] = result["landmark_heatmap"] * valid[:, :, None, None]
            result["pose_ypr"] = result["pose_ypr"] * valid
            for axis in POSE_AXES:
                result[f"{axis}_bin_logits"] = result[f"{axis}_bin_logits"] * valid
                result[f"{axis}_residual"] = result[f"{axis}_residual"] * valid
        return result


register_roi_head("default")(DMSROIHeads)

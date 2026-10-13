"""Small RGB face-crop teacher used only during DMS training."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from ultralytics.nn.modules.dms_pose import POSE_AXES, decode_pose, pose_bin_count


class TeacherDWBlock(nn.Module):
    """Teacher-only block; its checkpoint is independent of student ROI norm."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CropTeacher(nn.Module):
    """128px face crop -> 96x32x32 feature, 12 heatmaps and yaw/pitch/roll."""

    def __init__(self, channels: int = 96):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels // 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(),
            nn.Conv2d(channels // 2, channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            TeacherDWBlock(channels),
            TeacherDWBlock(channels),
        )
        self.landmark = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, 2, 2), nn.SiLU(), nn.Conv2d(channels // 2, 12, 1)
        )
        self.pose_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 64), nn.SiLU())
        self.pose_bin = nn.ModuleDict({axis: nn.Linear(64, pose_bin_count(axis)) for axis in POSE_AXES})
        self.pose_residual = nn.ModuleDict({axis: nn.Linear(64, 1) for axis in POSE_AXES})

    def forward(self, face_crop: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.encoder(face_crop)
        pose_feature = self.pose_pool(feature)
        result = {
            "face_feature": feature,
            "landmark_heatmap": self.landmark(feature),
            **{f"{axis}_bin_logits": self.pose_bin[axis](pose_feature) for axis in POSE_AXES},
            **{f"{axis}_residual": 0.5 * torch.tanh(self.pose_residual[axis](pose_feature)) for axis in POSE_AXES},
        }
        result["pose_ypr"] = decode_pose(result)
        return result

    def freeze(self) -> "CropTeacher":
        self.eval()
        return self.requires_grad_(False)


def make_teacher_crops(image: torch.Tensor, face_rois: torch.Tensor, size: int = 128) -> torch.Tensor:
    """Crop letterboxed RGB faces with the same aligned ROIAlign convention as the student."""
    from torchvision.ops import roi_align

    return roi_align(image, face_rois, (size, size), spatial_scale=1.0, aligned=True)


def distillation_loss(
    student: Mapping[str, torch.Tensor], teacher: Mapping[str, torch.Tensor], valid: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Feature and heatmap KD, evaluated only for ROIs with matching labels."""
    valid = valid.bool()
    if not valid.any():
        zero = student["face_feature"].sum() * 0
        return {"kd_feature": zero, "kd_heatmap": zero}
    student_feature = nn.functional.normalize(student["face_feature"][valid], dim=1)
    teacher_feature = nn.functional.normalize(teacher["face_feature"][valid].detach(), dim=1)
    return {
        "kd_feature": nn.functional.mse_loss(student_feature, teacher_feature),
        "kd_heatmap": nn.functional.mse_loss(student["landmark_heatmap"][valid], teacher["landmark_heatmap"][valid].detach()),
    }

"""Small RGB face-crop teacher used only during DMS training."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from ultralytics.nn.modules.dms import DWBlock


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
            DWBlock(channels),
            DWBlock(channels),
        )
        self.landmark = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, 2, 2), nn.SiLU(), nn.Conv2d(channels // 2, 12, 1)
        )
        self.pose = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 64), nn.SiLU(), nn.Linear(64, 3))

    def forward(self, face_crop: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.encoder(face_crop)
        return {"face_feature": feature, "landmark_heatmap": self.landmark(feature), "pose_ypr": self.pose(feature)}

    def freeze(self) -> "CropTeacher":
        self.eval()
        return self.requires_grad_(False)


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

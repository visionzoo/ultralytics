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
    student: Mapping[str, torch.Tensor], teacher: Mapping[str, torch.Tensor], valid: torch.Tensor,
    pose_scale: torch.Tensor | None = None, pose_valid: torch.Tensor | None = None,
    pose_axis_weights: tuple[float, float, float] = (5.0, 1.0, 1.0),
    bin_temperature: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Feature, heatmap, continuous pose and bin/residual KD for matching ROIs."""
    valid = valid.bool()
    if not valid.any():
        zero = student["face_feature"].sum() * 0
        return {"kd_feature": zero, "kd_heatmap": zero, "kd_pose": zero, "kd_pose_bins": zero}
    student_feature = nn.functional.normalize(student["face_feature"][valid], dim=1)
    teacher_feature = nn.functional.normalize(teacher["face_feature"][valid].detach(), dim=1)
    result = {
        "kd_feature": nn.functional.mse_loss(student_feature, teacher_feature),
        "kd_heatmap": nn.functional.mse_loss(student["landmark_heatmap"][valid], teacher["landmark_heatmap"][valid].detach()),
    }
    # Pose KD must be a subset of the teacher-valid ROIs.  Intersecting the
    # masks prevents a malformed mixed batch from supervising padded/invalid
    # crops even when its pose_valid flag is set.
    pose_mask = valid & (pose_valid.bool() if pose_valid is not None else valid)
    if "pose_ypr" in student and "pose_ypr" in teacher and pose_mask.any():
        scale = pose_scale if pose_scale is not None else student["pose_ypr"].new_tensor((55.0, 12.0, 11.0))
        scale = scale.to(device=student["pose_ypr"].device, dtype=student["pose_ypr"].dtype).clamp_min(1.0)
        axis_weights = student["pose_ypr"].new_tensor(pose_axis_weights)
        pose_error = nn.functional.smooth_l1_loss(
            student["pose_ypr"][pose_mask] / scale,
            teacher["pose_ypr"][pose_mask].detach() / scale,
            reduction="none",
        ).mean(0)
        for index, axis in enumerate(POSE_AXES):
            result[f"kd_pose_{axis}"] = pose_error[index]
        result["kd_pose"] = (pose_error * axis_weights).sum() / axis_weights.sum()
        temperature = max(float(bin_temperature), 1e-3)
        bin_terms = []
        for index, axis in enumerate(POSE_AXES):
            student_logits = student[f"{axis}_bin_logits"][pose_mask]
            teacher_logits = teacher[f"{axis}_bin_logits"][pose_mask].detach()
            student_log_prob = (student_logits / temperature).log_softmax(-1)
            teacher_prob = (teacher_logits / temperature).softmax(-1)
            bin_error = nn.functional.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * temperature**2
            residual_error = nn.functional.smooth_l1_loss(
                student[f"{axis}_residual"][pose_mask], teacher[f"{axis}_residual"][pose_mask].detach()
            )
            result[f"kd_pose_bin_{axis}"] = bin_error
            result[f"kd_pose_residual_{axis}"] = residual_error
            bin_terms.append(bin_error + residual_error)
            result[f"kd_pose_bin_acc_{axis}"] = (
                (student_logits.argmax(-1) == teacher_logits.argmax(-1)).float().mean().detach()
            )
        result["kd_pose_bins"] = (torch.stack(bin_terms) * axis_weights).sum() / axis_weights.sum()
        result["kd_pose_yaw_mae"] = (
            (student["pose_ypr"][pose_mask, 0] - teacher["pose_ypr"][pose_mask, 0].detach()).abs().mean().detach()
        )
    else:
        result["kd_pose"] = student["face_feature"].sum() * 0
        result["kd_pose_bins"] = student["face_feature"].sum() * 0
    return result

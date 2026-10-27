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

    def __init__(self, channels: int = 96, use_context: bool = True):
        super().__init__()
        self.architecture = "crop_teacher_context_v2" if use_context else "crop_teacher_legacy_v1"
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
        # A small spatial bottleneck sees the entire face and retains coarse
        # position. Only the landmark branch changes; pose tensors stay intact.
        if use_context:
            self.landmark_context = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                nn.Linear(channels * 16, channels), nn.SiLU(),
                nn.Linear(channels, channels * 16), nn.Unflatten(1, (channels, 4, 4)),
            )
            self.landmark_fusion = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, channels), nn.SiLU(),
            )
        self.pose_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 64), nn.SiLU())
        self.pose_bin = nn.ModuleDict({axis: nn.Linear(64, pose_bin_count(axis)) for axis in POSE_AXES})
        self.pose_residual = nn.ModuleDict({axis: nn.Linear(64, 1) for axis in POSE_AXES})

    def forward(self, face_crop: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.encoder(face_crop)
        pose_feature = self.pose_pool(feature)
        landmark_feature = feature
        if hasattr(self, "landmark_context"):
            context = nn.functional.interpolate(self.landmark_context(feature), size=feature.shape[-2:],
                                                mode="bilinear", align_corners=False)
            landmark_feature = self.landmark_fusion(torch.cat((feature, context), 1))
        result = {
            "face_feature": feature,
            "landmark_heatmap": self.landmark(landmark_feature),
            **{f"{axis}_bin_logits": self.pose_bin[axis](pose_feature) for axis in POSE_AXES},
            **{f"{axis}_residual": 0.5 * torch.tanh(self.pose_residual[axis](pose_feature)) for axis in POSE_AXES},
        }
        result["pose_ypr"] = decode_pose(result)
        if hasattr(self, "landmark_context"):
            logits = result["landmark_heatmap"]
            # The student's supervised heatmaps are bounded Gaussian scores,
            # not categorical logits. Convert only the KD target, without
            # changing either model's deployed argmax coordinates.
            result["landmark_kd_heatmap"] = (logits - logits.amax((-2, -1), keepdim=True)).exp()
        return result

    @classmethod
    def from_state(cls, state: dict, metadata: dict | None = None) -> "CropTeacher":
        """Explicit legacy support for audits, without silently dropping new weights."""
        teacher = cls(channels=state["encoder.3.weight"].shape[0],
                      use_context=any(key.startswith("landmark_context.") for key in state))
        declared = (metadata or {}).get("teacher_architecture")
        if declared is not None and declared != teacher.architecture:
            raise ValueError("Teacher architecture metadata does not match its tensors")
        teacher.load_state_dict(state, strict=True)
        return teacher

    def freeze(self) -> "CropTeacher":
        self.eval()
        return self.requires_grad_(False)


def make_teacher_crops(image: torch.Tensor, face_rois: torch.Tensor, size: int = 128) -> torch.Tensor:
    """Crop letterboxed RGB faces with the same aligned ROIAlign convention as the student."""
    from torchvision.ops import roi_align

    return roi_align(image, face_rois, (size, size), spatial_scale=1.0, aligned=True)


class TeacherLoss(nn.Module):
    """Spatial KL for landmark logits; reuse the existing degree/bin pose loss.

    Do not mix bounded Gaussian MSE with raw-logit CE or coordinate regression.
    A constant heatmap has a large KL and cannot masquerade as a localized peak.
    Student ROI losses are deliberately unchanged.
    """

    def __init__(self, pose_axis_weights=(1., 1., 1.)):
        super().__init__()
        from .loss import DMSROILoss
        self.pose_loss = DMSROILoss(pose_axis_weights=pose_axis_weights)

    def forward(self, outputs, batch):
        logits = outputs["landmark_heatmap"].float()
        if "landmark_valid" not in batch:
            return self.pose_loss(outputs, batch)
        valid = batch["landmark_valid"].bool()
        annotated = batch["landmark_xy"]
        if (valid & (~torch.isfinite(annotated).all(-1) | ((annotated < 0) | (annotated > 1)).any(-1))).any():
            raise ValueError("Valid Teacher landmark targets must be finite normalized ROI coordinates")
        points = torch.where(valid[..., None], batch["landmark_xy"].float(), 0.)
        target = self.pose_loss._heatmap_target(points, valid, *logits.shape[-2:]).flatten(-2)
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
        kl = nn.functional.kl_div(logits.flatten(-2).log_softmax(-1), target, reduction="none").sum(-1)
        loss = torch.where(valid, kl, 0.).sum() / valid.sum().clamp_min(1)
        return loss, {"landmark_spatial_kl": loss.detach()}


def warm_start_teacher_pose(teacher: CropTeacher, payload: dict) -> int:
    """Copy validated encoder/pose only; never reuse the failed landmark head."""
    source = CropTeacher.from_state(payload.get("model", payload), payload.get("metadata", {})).state_dict()
    current = teacher.state_dict()
    keys = [key for key in current if key.startswith(("encoder.", "pose_pool.", "pose_bin.", "pose_residual."))]
    for key in keys:
        current[key] = source[key]
    teacher.load_state_dict(current, strict=True)
    return len(keys)


@torch.no_grad()
def landmark_teacher_quality(output: Mapping[str, torch.Tensor], batch: dict, max_nme: float = 0.10):
    """Gate landmark KD against GT in the student's actual ROI, never test data."""
    from .loss import DMSROILoss
    target, valid = batch["landmark_xy"], batch["landmark_valid"].bool()
    prediction = DMSROILoss.decode_heatmap(output["landmark_heatmap"])
    boxes = batch["face_rois"][:, 1:]
    scale = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1).unsqueeze(1)
    left = (target[:, (0, 3)] * scale).mean(1)
    right = (target[:, (6, 9)] * scale).mean(1)
    distance = (right - left).norm(dim=-1)
    errors = ((prediction - target) * scale).norm(dim=-1)
    errors = torch.where(valid, errors, 0.)
    nme = errors.sum(-1) / valid.sum(-1).clamp_min(1) / distance.clamp_min(1e-6)
    measurable = valid[:, (0, 3, 6, 9)].all(-1) & (distance > 1e-6) & torch.isfinite(nme)
    finite_maps = torch.isfinite(output["landmark_heatmap"]).flatten(-2).all(-1)
    measurable &= (finite_maps | ~valid).all(-1)
    accepted = measurable & (nme <= max_nme)
    return accepted, nme, measurable


def distillation_loss(
    student: Mapping[str, torch.Tensor], teacher: Mapping[str, torch.Tensor], valid: torch.Tensor,
    pose_scale: torch.Tensor | None = None, pose_valid: torch.Tensor | None = None,
    pose_axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    bin_temperature: float = 2.0,
    landmark_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Feature, heatmap, continuous pose and bin/residual KD for matching ROIs."""
    valid = valid.bool()
    if not valid.any():
        zero = student["face_feature"].sum() * 0
        return {"kd_feature": zero, "kd_heatmap": zero, "kd_pose": zero, "kd_pose_bins": zero}
    teacher_feature = teacher["face_feature"][valid].detach()
    if teacher_feature.shape[-2:] != student["face_feature"].shape[-2:]:
        teacher_feature = nn.functional.interpolate(teacher_feature, size=student["face_feature"].shape[-2:],
                                                    mode="bilinear", align_corners=False)
    student_feature = nn.functional.normalize(student["face_feature"][valid], dim=1)
    teacher_feature = nn.functional.normalize(teacher_feature, dim=1)
    result = {
        "kd_feature": nn.functional.mse_loss(student_feature, teacher_feature),
        "kd_heatmap": student["landmark_heatmap"].sum() * 0,
    }
    if landmark_valid is not None:
        point_mask = landmark_valid.bool() & valid[:, None]
        if point_mask.any():
            scores = teacher.get("landmark_kd_heatmap", teacher["landmark_heatmap"]).detach()
            if scores.shape[-2:] != student["landmark_heatmap"].shape[-2:]:
                scores = nn.functional.interpolate(scores, size=student["landmark_heatmap"].shape[-2:],
                                                    mode="bilinear", align_corners=True)
            result["kd_heatmap"] = nn.functional.mse_loss(
                student["landmark_heatmap"][point_mask], scores[point_mask])
    # Pose KD must be a subset of the teacher-valid ROIs.  Intersecting the
    # masks prevents a malformed mixed batch from supervising padded/invalid
    # crops even when its pose_valid flag is set.
    pose_mask = valid & (pose_valid.bool() if pose_valid is not None else torch.zeros_like(valid))
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

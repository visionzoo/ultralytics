"""Losses for DMS ROI tasks; YOLO's native detection loss remains untouched."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class DMSROILoss(nn.Module):
    """Masked fixed-part, heatmap and pose losses for partially labelled batches."""

    def __init__(self, heatmap_sigma: float = 1.5, shape_prior=None):
        super().__init__()
        self.heatmap_sigma = heatmap_sigma
        self.shape_prior = shape_prior

    @staticmethod
    def _zero(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return next(value for value in outputs.values() if isinstance(value, torch.Tensor)).sum() * 0

    @staticmethod
    def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid = valid.to(dtype=values.dtype)
        return (values * valid).sum() / valid.sum().clamp_min(1)

    def _heatmap_target(self, points: torch.Tensor, valid: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Create gaussian heatmaps from normalized ROI coordinates without NumPy."""
        y = torch.arange(height, dtype=points.dtype, device=points.device).view(1, 1, height, 1)
        x = torch.arange(width, dtype=points.dtype, device=points.device).view(1, 1, 1, width)
        px = points[..., 0].unsqueeze(-1).unsqueeze(-1) * (width - 1)
        py = points[..., 1].unsqueeze(-1).unsqueeze(-1) * (height - 1)
        target = torch.exp(-((x - px).square() + (y - py).square()) / (2 * self.heatmap_sigma**2))
        return target * valid.to(dtype=target.dtype).unsqueeze(-1).unsqueeze(-1)

    @staticmethod
    def _softargmax(logits: torch.Tensor) -> torch.Tensor:
        height, width = logits.shape[-2:]
        probabilities = logits.flatten(-2).softmax(-1).reshape(*logits.shape[:2], height, width)
        x = torch.linspace(0, 1, width, device=logits.device, dtype=logits.dtype)
        y = torch.linspace(0, 1, height, device=logits.device, dtype=logits.dtype)
        return torch.stack((probabilities.sum(-2) @ x, probabilities.sum(-1) @ y), dim=-1)

    @staticmethod
    def _state_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Avoid passing ignored ``-1`` labels to cross entropy."""
        if not valid.any():
            return logits.sum() * 0
        return nn.functional.cross_entropy(logits[valid], target[valid].long())

    def forward(self, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return a total plus named terms, omitting unavailable supervision safely."""
        zero, losses = self._zero(outputs), {}

        if "part_box" in batch and "part_box_valid" in batch:
            valid = batch["part_box_valid"].bool()
            box_error = nn.functional.smooth_l1_loss(outputs["part_box"], batch["part_box"], reduction="none").mean(-1)
            losses["part_box"] = self._masked_mean(box_error, valid)
        if "part_present" in batch and "part_present_valid" in batch:
            present_error = nn.functional.binary_cross_entropy_with_logits(
                outputs["part_presence_logits"], batch["part_present"].to(outputs["part_presence_logits"].dtype), reduction="none"
            )
            losses["part_presence"] = self._masked_mean(present_error, batch["part_present_valid"].bool())
        if "eye_state" in batch and "eye_state_valid" in batch:
            logits, target, valid = outputs["eye_state_logits"], batch["eye_state"], batch["eye_state_valid"].bool()
            losses["eye_state"] = self._state_loss(logits, target, valid)
        if "mouth_state" in batch and "mouth_state_valid" in batch:
            logits, target, valid = outputs["mouth_state_logits"], batch["mouth_state"], batch["mouth_state_valid"].bool()
            losses["mouth_state"] = self._state_loss(logits, target, valid)
        if "landmark_xy" in batch and "landmark_valid" in batch:
            logits, points, valid = outputs["landmark_heatmap"], batch["landmark_xy"], batch["landmark_valid"].bool()
            target = self._heatmap_target(points, valid, logits.shape[-2], logits.shape[-1])
            heatmap_error = nn.functional.mse_loss(logits.sigmoid(), target, reduction="none").mean((-1, -2))
            logvar = outputs.get("landmark_logvar", logits.new_zeros(logits.shape[:2])).clamp(-4.0, 4.0)
            losses["landmark"] = self._masked_mean(torch.exp(-logvar) * heatmap_error + logvar, valid)
            if self.shape_prior is not None:
                losses["shape_prior"] = self.shape_prior(self._softargmax(logits), valid)
        if "pose_ypr" in batch and "pose_valid" in batch:
            valid = batch["pose_valid"].bool()
            pose_error = nn.functional.smooth_l1_loss(outputs["pose_ypr"], batch["pose_ypr"], reduction="none").mean(-1)
            losses["pose"] = self._masked_mean(pose_error, valid)

        total = sum(losses.values(), zero)
        return total, losses

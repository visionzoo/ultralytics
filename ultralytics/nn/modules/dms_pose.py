"""Shared bin/residual pose encoding used by teacher, ROI head and loss."""

from __future__ import annotations

import torch

POSE_AXES = ("yaw", "pitch", "roll")
POSE_BIN_CONFIG = {"yaw": (-90.0, 90.0, 5.0), "pitch": (-45.0, 45.0, 5.0), "roll": (-90.0, 90.0, 5.0)}


def pose_bin_count(axis: str) -> int:
    minimum, maximum, step = POSE_BIN_CONFIG[axis]
    return int(round((maximum - minimum) / step))


def pose_centers(axis: str, *, device=None, dtype=None) -> torch.Tensor:
    minimum, _, step = POSE_BIN_CONFIG[axis]
    return minimum + (torch.arange(pose_bin_count(axis), device=device, dtype=dtype) + 0.5) * step


def decode_axis(logits: torch.Tensor, residual: torch.Tensor, axis: str) -> torch.Tensor:
    _, _, step = POSE_BIN_CONFIG[axis]
    centers = pose_centers(axis, device=logits.device, dtype=logits.dtype)
    expected = (logits.softmax(-1) * centers).sum(-1)
    return expected + residual.squeeze(-1) * step


def decode_pose(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(
        [decode_axis(outputs[f"{axis}_bin_logits"], outputs[f"{axis}_residual"], axis) for axis in POSE_AXES],
        dim=-1,
    )

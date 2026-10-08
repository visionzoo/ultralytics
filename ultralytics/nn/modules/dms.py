"""Minimal DMS ROI modules built on top of YOLO feature maps."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.ops import roi_align


class DWBlock(nn.Module):
    def __init__(self, c: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(c, c, 3, stride, 1, groups=c, bias=False), nn.BatchNorm2d(c), nn.SiLU(), nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DMSROIHeads(nn.Module):
    """Fuse B2/P3 face ROIs and predict local states, eye landmarks and head pose."""

    def __init__(self, state_classes: int = 5, channels: int = 96):
        super().__init__()
        self.channels = channels
        self.fuse = nn.Sequential(nn.LazyConv2d(channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU(), DWBlock(channels), DWBlock(channels))
        self.local = nn.Sequential(DWBlock(channels, 2), DWBlock(channels, 2))
        self.state_cls = nn.Conv2d(channels, state_classes, 1)
        self.state_box = nn.Conv2d(channels, 4, 1)
        self.state_center = nn.Conv2d(channels, 1, 1)
        self.landmark = nn.Sequential(nn.ConvTranspose2d(channels, channels // 2, 2, 2), nn.SiLU(), nn.Conv2d(channels // 2, 12, 1))
        self.pose = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 64), nn.SiLU(), nn.Linear(64, 3))

    def forward(self, b2: torch.Tensor, p3: torch.Tensor, rois: torch.Tensor) -> dict[str, torch.Tensor]:
        if rois.numel() == 0:
            empty = b2.new_empty((0, self.channels, 32, 32))
            return {"face_feature": empty, "state_cls": empty.new_empty((0, self.state_cls.out_channels, 8, 8)), "state_box": empty.new_empty((0, 4, 8, 8)), "state_center": empty.new_empty((0, 1, 8, 8)), "landmark": empty.new_empty((0, 12, 64, 64)), "pose": empty.new_empty((0, 3))}
        b2_roi = roi_align(b2, rois, (32, 32), spatial_scale=0.25, aligned=True)
        p3_roi = roi_align(p3, rois, (16, 16), spatial_scale=0.125, aligned=True)
        face = self.fuse(torch.cat((b2_roi, nn.functional.interpolate(p3_roi, scale_factor=2, mode="bilinear", align_corners=False)), 1))
        local = self.local(face)
        return {"face_feature": face, "state_cls": self.state_cls(local), "state_box": self.state_box(local), "state_center": self.state_center(local), "landmark": self.landmark(face), "pose": self.pose(face)}

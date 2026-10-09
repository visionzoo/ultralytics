"""Small DMS ROI heads built on top of already-cropped YOLO features."""

from __future__ import annotations

import torch
from torch import nn


class DWBlock(nn.Module):
    """Depthwise pointwise block used only after a face ROI has been extracted."""

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
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
            nn.BatchNorm2d(channels),
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
        self.pose = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 64), nn.SiLU(), nn.Linear(64, 3))

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
            }
        p3_rois = nn.functional.interpolate(p3_rois, size=(32, 32), mode="bilinear", align_corners=False)
        face = self.fuse(torch.cat((b2_rois, p3_rois), 1))
        part_values = self.parts(face).reshape(count, 3, 5)
        result = {
            "face_feature": face,
            "part_box": part_values[..., :4].sigmoid(),
            "part_presence_logits": part_values[..., 4],
            "eye_state_logits": self.eye_state(face).reshape(count, 2, self.eye_states),
            "mouth_state_logits": self.mouth_state(face).reshape(count, 1, self.mouth_states),
            "landmark_heatmap": self.landmark(face),
            "landmark_logvar": self.landmark_logvar(face).clamp(-4.0, 4.0),
            "pose_ypr": self.pose(face),
        }
        if roi_valid is not None:
            valid = roi_valid.to(dtype=face.dtype).reshape(count, 1)
            result["part_presence_logits"] = result["part_presence_logits"] * valid
            result["landmark_heatmap"] = result["landmark_heatmap"] * valid[:, :, None, None]
            result["pose_ypr"] = result["pose_ypr"] * valid
        return result

"""Canonical DMS labels and annotation records.

Keep dataset-specific spelling here so the model, loss and trainer never need to
know whether a label originated in DSM, Eye12 or 300W-LP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


LANDMARK_NAMES = (
    "left_eye_left_corner",
    "left_eye_upper_left",
    "left_eye_upper_right",
    "left_eye_right_corner",
    "left_eye_lower_right",
    "left_eye_lower_left",
    "right_eye_left_corner",
    "right_eye_upper_left",
    "right_eye_upper_right",
    "right_eye_right_corner",
    "right_eye_lower_right",
    "right_eye_lower_left",
)
PART_SLOTS = ("left_eye", "right_eye", "mouth")
PartName = Literal["eye", "mouth"]
AssociationStatus = Literal["matched", "orphan", "ambiguous"]


@dataclass(frozen=True)
class Box:
    """Absolute xyxy pixel box."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def contains_center_of(self, other: "Box") -> bool:
        x, y = other.center
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def iou(self, other: "Box") -> float:
        inter_w = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        inter_h = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        union = self.area + other.area - inter_w * inter_h
        return 0.0 if union <= 0 else inter_w * inter_h / union


@dataclass(frozen=True)
class DMSObject:
    label: str
    box: Box


@dataclass(frozen=True)
class PartTarget:
    part: PartName
    state: str | None
    box: Box
    raw_label: str


@dataclass(frozen=True)
class PartAssociation:
    target: PartTarget
    status: AssociationStatus
    face_index: int | None
    confidence: float
    candidate_count: int

    def to_dict(self) -> dict:
        result = asdict(self)
        result["target"]["box"] = asdict(self.target.box)
        return result


def normalized_label(label: str) -> str:
    return label.strip().lower().replace("_", "")


def is_face_label(label: str) -> bool:
    """DSM face annotations consistently use the ``*face`` suffix."""
    return normalized_label(label).endswith("face")


def part_target(obj: DMSObject) -> PartTarget | None:
    """Map a DSM local label to geometry plus an optional independent state."""
    label = normalized_label(obj.label)
    if label.endswith("mouth"):
        state = {"normalmouth": "normal", "yawnmouth": "yawn", "cigarettemouth": "cigarette"}.get(label)
        return PartTarget("mouth", state, obj.box, label)
    if label in {"staring", "squint"}:
        return PartTarget("eye", label, obj.box, label)
    return None


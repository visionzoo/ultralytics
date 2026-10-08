"""Dataset adapters for the independent DMS supervision sources."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .schema import Box, DMSObject, PartAssociation, PartTarget, is_face_label, part_target


@dataclass(frozen=True)
class VOCRecord:
    annotation: Path
    filename: str
    width: int
    height: int
    objects: tuple[DMSObject, ...]


@dataclass(frozen=True)
class Eye12Record:
    image: Path
    points: tuple[tuple[float, float], ...]
    crop_box: Box
    input_size: int
    split: str


@dataclass(frozen=True)
class PoseRecord:
    image: Path
    face_box: Box
    ypr_deg: tuple[float, float, float]


def parse_voc_xml(path: str | Path) -> VOCRecord:
    """Read one DSM Pascal-VOC XML without inferring any dataset semantics."""
    annotation = Path(path)
    root = ET.parse(annotation).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"VOC size missing: {annotation}")
    objects = []
    for node in root.findall("object"):
        name, bbox = node.findtext("name"), node.find("bndbox")
        if not name or bbox is None:
            continue
        try:
            box = Box(*(float(bbox.findtext(axis, "nan")) for axis in ("xmin", "ymin", "xmax", "ymax")))
        except ValueError as exc:
            raise ValueError(f"invalid VOC box: {annotation}") from exc
        if box.area > 0:
            objects.append(DMSObject(name, box))
    return VOCRecord(
        annotation=annotation,
        filename=root.findtext("filename", ""),
        width=int(size.findtext("width", "0")),
        height=int(size.findtext("height", "0")),
        objects=tuple(objects),
    )


def dsm_image_path(record: VOCRecord) -> Path:
    """Resolve DSM's sibling ``Annotations``/``images`` convention strictly."""
    image = record.annotation.parent.parent / "images" / record.filename
    if not image.is_file():
        raise FileNotFoundError(f"DSM image missing for {record.annotation}: {image}")
    return image


def iter_dsm(root: str | Path) -> Iterator[VOCRecord]:
    """Yield all DSM annotations in deterministic order."""
    for annotation in sorted(Path(root).glob("**/Annotations/*.xml")):
        yield parse_voc_xml(annotation)


def associate_dsm_parts(record: VOCRecord, min_coverage: float = 0.8, tie_iou: float = 1e-6) -> tuple[PartAssociation, ...]:
    """Associate local DSM labels to faces, preserving uncertainty for audit.

    A target must have its center in a face and at least ``min_coverage`` of its
    area inside it. The selected face maximizes IoU; equal best candidates are
    explicitly ambiguous instead of being assigned by XML order.
    """
    faces = [obj for obj in record.objects if is_face_label(obj.label)]
    associations = []
    for obj in record.objects:
        target = part_target(obj)
        if target is None:
            continue
        candidates: list[tuple[int, float]] = []
        for index, face in enumerate(faces):
            if not face.box.contains_center_of(target.box):
                continue
            overlap_w = max(0.0, min(face.box.x2, target.box.x2) - max(face.box.x1, target.box.x1))
            overlap_h = max(0.0, min(face.box.y2, target.box.y2) - max(face.box.y1, target.box.y1))
            coverage = overlap_w * overlap_h / target.box.area if target.box.area else 0.0
            if coverage >= min_coverage:
                candidates.append((index, face.box.iou(target.box)))
        if not candidates:
            associations.append(PartAssociation(target, "orphan", None, 0.0, 0))
            continue
        candidates.sort(key=lambda item: item[1], reverse=True)
        best_index, best_iou = candidates[0]
        tied = len(candidates) > 1 and abs(best_iou - candidates[1][1]) <= tie_iou
        status = "ambiguous" if tied else "matched"
        associations.append(PartAssociation(target, status, None if tied else best_index, best_iou, len(candidates)))
    return tuple(associations)


def iter_eye12(jsonl_path: str | Path) -> Iterator[Eye12Record]:
    """Yield validated 12-point records from the supplied Eye12 JSONL files."""
    with Path(jsonl_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            item = json.loads(line)
            points = tuple((float(x), float(y)) for x, y in item["points"])
            if len(points) != 12:
                raise ValueError(f"expected 12 points at {jsonl_path}:{line_number}")
            crop = item["crop_box_xyxy"]
            yield Eye12Record(
                Path(item["image"]), points, Box(*(float(v) for v in crop)), int(item.get("input_size", 128)), item.get("split", "")
            )


def _rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = rvec / theta
    cross = np.array(((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)))
    return np.eye(3) + math.sin(theta) * cross + (1 - math.cos(theta)) * (cross @ cross)


def rodrigues_to_ypr(rvec: np.ndarray) -> tuple[float, float, float]:
    """Return degrees using R = Rz(roll) @ Ry(yaw) @ Rx(pitch)."""
    rotation = _rodrigues_to_matrix(np.asarray(rvec, dtype=np.float64).reshape(3))
    yaw = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    pitch = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    roll = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return tuple(math.degrees(value) for value in (yaw, pitch, roll))


def parse_300wlp_mat(path: str | Path) -> PoseRecord:
    """Read a 300W-LP pair. SciPy is imported lazily for non-pose workflows."""
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError("300W-LP parsing requires scipy") from exc
    mat_path = Path(path)
    data = loadmat(mat_path)
    roi = np.asarray(data["roi"], dtype=np.float64).reshape(-1)
    pose = np.asarray(data["Pose_Para"], dtype=np.float64).reshape(-1)
    if roi.size < 4 or pose.size < 3:
        raise ValueError(f"invalid 300W-LP annotation: {mat_path}")
    return PoseRecord(mat_path.with_suffix(".jpg"), Box(*map(float, roi[:4])), rodrigues_to_ypr(pose[:3]))


def write_association_audit(path: str | Path, records: Iterator[tuple[VOCRecord, tuple[PartAssociation, ...]]]) -> None:
    """Write one JSON object per local target; callers choose which XMLs to scan."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record, associations in records:
            for association in associations:
                handle.write(json.dumps({"annotation": str(record.annotation), **association.to_dict()}, ensure_ascii=False) + "\n")


def dsm_manifest_entry(record: VOCRecord) -> dict:
    """Create one loss-agnostic DSM manifest row and preserve all raw labels."""
    associations = associate_dsm_parts(record)
    return {
        "task": "dsm",
        "image": str(dsm_image_path(record)),
        "size": [record.width, record.height],
        "objects": [{"label": obj.label, "box_xyxy": list(vars(obj.box).values())} for obj in record.objects],
        "part_associations": [item.to_dict() for item in associations],
    }

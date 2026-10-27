"""ROI datasets and collate functions backed by the DMS JSONL manifests."""

from __future__ import annotations

import json
import zlib
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.data.augment import LetterBox

from .schema import is_face_label
from .geometry import square_rois


@lru_cache(maxsize=8)
def face_scale_reference(manifest: str) -> tuple[float, ...]:
    """DSM TRAIN face-side fractions after full-frame letterbox; no val/test leakage."""
    scales = []
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("split") != "train":
            continue
        width, height = row["size"]
        for obj in row.get("objects", []):
            if is_face_label(obj["label"]):
                x1, y1, x2, y2 = obj["box_xyxy"]
                side = max(x2 - x1, y2 - y1) / max(width, height)
                if 0 < side <= 1:
                    scales.append(side)
    if not scales:
        raise ValueError(f"No train face scale reference in {manifest}")
    return tuple(scales)


def _box(values: list[float] | tuple[float, ...]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


class DMSROIDataset(Dataset):
    """One manifest task per dataset; each sample carries exactly one face ROI."""

    def __init__(self, manifest: str | Path, task: str, image_size: int = 640, split: str | None = None,
                 match_scale: bool = True):
        self.task, self.image_size = task, image_size
        self.split = split
        self.scales = face_scale_reference(str(Path(manifest).parent / "dsm.jsonl")) if match_scale and task in {"landmark", "pose"} else ()
        rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()]
        if split:
            rows = [row for row in rows if row.get("split") == split]
        self.samples = self._expand(rows)
        if not self.samples:
            suffix = f" split={split}" if split else ""
            raise ValueError(f"no {task} samples in {manifest}{suffix}")

    def _expand(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.task == "landmark":
            return [row for row in rows if row.get("task") == "landmark"]
        if self.task == "pose":
            return [row for row in rows if row.get("task") == "pose"]
        if self.task != "parts":
            raise ValueError(f"unsupported ROI task: {self.task}")
        samples = []
        for row in rows:
            if row.get("task") != "dsm":
                continue
            faces = [obj["box_xyxy"] for obj in row["objects"] if is_face_label(obj["label"])]
            grouped: dict[int, list[dict]] = defaultdict(list)
            for association in row.get("part_associations", []):
                if association["status"] == "matched" and association["face_index"] is not None:
                    grouped[association["face_index"]].append(association["target"])
            for face_index, targets in grouped.items():
                if face_index < len(faces):
                    samples.append({"task": "parts", "image": row["image"], "face_box_xyxy": faces[face_index], "targets": targets})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def pose_sampling_weights(self) -> torch.Tensor:
        """Inverse-frequency weights over yaw/pitch/roll bins for 300W-LP."""
        if self.task != "pose" or not self.samples:
            raise ValueError("pose_sampling_weights is only valid for pose datasets")
        values = np.asarray([sample["pose_ypr_deg"] for sample in self.samples], dtype=np.float32)
        weights = np.ones(len(values), dtype=np.float64)
        bin_edges = (
            np.asarray([-90, -60, -30, 0, 30, 60, 90], dtype=np.float32),
            np.asarray([-60, -40, -20, 0, 20, 40, 60], dtype=np.float32),
            np.asarray([-60, -40, -20, 0, 20, 40, 60], dtype=np.float32),
        )
        for axis in range(3):
            edges = bin_edges[axis]
            bins = np.searchsorted(edges[1:-1], values[:, axis], side="right")
            counts = np.bincount(bins, minlength=6).astype(np.float64)
            weights *= np.sqrt(counts.mean() / np.maximum(counts[bins], 1.0))
        weights = np.clip(weights, 0.25, 4.0)
        return torch.from_numpy(weights / weights.mean())

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = cv2.imread(sample["image"])
        if image is None:
            raise FileNotFoundError(sample["image"])
        source_image = image
        image_height, image_width = image.shape[:2]
        face = _box(sample["face_box_xyxy"]).reshape(1, 4)
        points = None
        if self.task == "landmark" and sample.get("image_scope") != "full_frame":
            # Eye12's ``image`` is already the canonical aligned crop (the
            # generated files are 128x128).  ``crop_box_xyxy`` and
            # ``alignment_matrix`` describe where that crop came from in the
            # original source image; applying them again to this image would
            # crop outside its bounds and corrupt the landmark geometry.
            height, width = image.shape[:2]
            points = np.asarray(sample["points"], dtype=np.float32)
            face = np.asarray([[0, 0, width, height]], dtype=np.float32)
        source_face = face[0].copy()
        # Encode the available source image once, preserving its context and
        # aspect ratio. Only features, not input pixels, are cropped for the
        # student. Eye12/300W-LP remain crop-only sources when no full frame is
        # available; DSM supplies whole-frame supervision.
        transform = LetterBox(new_shape=(self.image_size, self.image_size), auto=False, stride=32)
        params = transform.get_params({"img": image})
        image = transform(image=image)
        # ``ratio`` is an ``(rx, ry)`` tuple.  Expand it explicitly to
        # ``(rx, ry, rx, ry)`` for xyxy coordinates; avoid ambiguous Python
        # tuple multiplication and keep the mapping valid for non-square
        # source images.
        ratio = np.asarray(params["ratio"], dtype=np.float32)
        face = source_face * np.tile(ratio, 2) + np.asarray(
            [params["left"], params["top"], params["left"], params["top"]], dtype=np.float32
        )
        offset = np.asarray([params["left"], params["top"]], dtype=np.float32)
        matrix = np.asarray([[ratio[0], 0, offset[0]], [0, ratio[1], offset[1]]], dtype=np.float32)
        if self.scales and sample.get("image_scope") != "full_frame":
            # Use native image pixels, not a 320px upsampled face.  Match the
            # DSM face-cell distribution; only train samples randomize placement.
            rng = np.random if self.split == "train" else np.random.RandomState(zlib.crc32(sample["image"].encode()))
            fraction = self.scales[int(rng.randint(len(self.scales)))]
            side = max(1.0, fraction * self.image_size)
            source_side = max(float(source_face[2] - source_face[0]), float(source_face[3] - source_face[1]), 1.0)
            rotation = np.eye(2, dtype=np.float32)
            if self.task == "landmark" and sample.get("alignment_matrix") is not None:
                # Undo the known in-plane alignment only.  This does not
                # fabricate missing full-frame context or source-image identity.
                alignment = np.asarray(sample["alignment_matrix"], dtype=np.float32)[:, :2]
                angle = np.arctan2(alignment[1, 0], alignment[0, 0])
                c, s = np.cos(angle), np.sin(angle)
                rotation = np.asarray([[c, s], [-s, c]], dtype=np.float32)
                source_side *= abs(float(c)) + abs(float(s))
            scale = side / source_side
            center = (source_face[:2] + source_face[2:]) / 2
            dest = rng.uniform(side / 2, self.image_size - side / 2, size=2) if self.split == "train" else np.asarray([self.image_size / 2] * 2)
            linear = scale * rotation
            matrix = np.column_stack((linear, dest - linear @ center)).astype(np.float32)
            image = cv2.warpAffine(source_image, matrix, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
            corners = np.asarray([[source_face[0], source_face[1]], [source_face[2], source_face[1]],
                                  [source_face[2], source_face[3]], [source_face[0], source_face[3]]], dtype=np.float32)
            corners = corners @ matrix[:, :2].T + matrix[:, 2]
            face = np.concatenate((corners.min(0), corners.max(0)))
        # Apply precisely the same square-after-letterbox rule as deployment.
        roi = torch.tensor([[0, *face.tolist()]], dtype=torch.float32)
        face = square_rois(roi, self.image_size, self.image_size)[0, 1:].numpy()
        img = torch.from_numpy(np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1)))
        result: dict[str, Any] = {
            "task": self.task,
            "img": img,
            "face_box": torch.from_numpy(face),
            "source_to_input": torch.from_numpy(matrix),
        }
        if self.task == "landmark":
            points = torch.tensor(sample["points"] if points is None else points, dtype=torch.float32)
            points = points.numpy() @ matrix[:, :2].T + matrix[:, 2]
            normalized = (points - face[:2]) / (face[2:] - face[:2]).clip(min=1)
            result["landmark_xy"] = torch.from_numpy(normalized)
            annotated = torch.as_tensor(sample.get("landmark_valid", [True] * 12), dtype=torch.bool)
            result["landmark_valid"] = annotated & torch.isfinite(result["landmark_xy"]).all(-1) & ((result["landmark_xy"] >= 0) & (result["landmark_xy"] <= 1)).all(-1)
            result["landmark_xy"] = torch.nan_to_num(result["landmark_xy"], nan=0., posinf=0., neginf=0.)
        elif self.task == "pose":
            result["pose_ypr"] = torch.tensor(sample["pose_ypr_deg"], dtype=torch.float32)
            result["pose_valid"] = torch.tensor(True)
        else:
            part_box = torch.zeros(3, 4)
            part_box_valid = torch.zeros(3, dtype=torch.bool)
            part_present = torch.zeros(3)
            part_present_valid = torch.zeros(3, dtype=torch.bool)
            eye_state, eye_state_valid = torch.full((2,), -1), torch.zeros(2, dtype=torch.bool)
            mouth_state, mouth_state_valid = torch.full((1,), -1), torch.zeros(1, dtype=torch.bool)
            original_face = source_face
            for target in sample["targets"]:
                part = target["part"]
                raw_box = _box([target["box"][axis] for axis in ("x1", "y1", "x2", "y2")])
                center_x = (raw_box[0] + raw_box[2]) / 2
                slot = 2 if part == "mouth" else (0 if center_x < (original_face[0] + original_face[2]) / 2 else 1)
                transformed = raw_box.reshape(2, 2) @ matrix[:, :2].T + matrix[:, 2]
                part_box[slot] = torch.from_numpy(((transformed - face[:2]) / (face[2:] - face[:2]).clip(min=1)).reshape(4))
                part_box_valid[slot] = part_present_valid[slot] = True
                part_present[slot] = 1
                state = target.get("state")
                if part == "eye" and state in {"staring", "squint"}:
                    eye_state[slot] = {"staring": 0, "squint": 1}[state]
                    eye_state_valid[slot] = True
                if part == "mouth" and state in {"normal", "yawn", "cigarette"}:
                    mouth_state[0] = {"normal": 0, "yawn": 1, "cigarette": 2}[state]
                    mouth_state_valid[0] = True
            result.update(
                part_box=part_box,
                part_box_valid=part_box_valid,
                part_present=part_present,
                part_present_valid=part_present_valid,
                eye_state=eye_state,
                eye_state_valid=eye_state_valid,
                mouth_state=mouth_state,
                mouth_state_valid=mouth_state_valid,
            )
        if "part_box" in result:
            from .geometry import mask_cropped_parts
            result = mask_cropped_parts(result)
        return result


def collate_roi(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate only homogeneous tasks; task mixing happens between batches."""
    tasks = {sample["task"] for sample in samples}
    if len(tasks) != 1:
        raise ValueError("DMSROI collate requires a single task per batch")
    batch: dict[str, Any] = {"task": tasks.pop(), "img": torch.stack([sample["img"] for sample in samples])}
    batch["face_rois"] = torch.stack(
        [torch.cat((torch.tensor([float(index)]), sample["face_box"])) for index, sample in enumerate(samples)]
    )
    batch["roi_valid"] = torch.ones(len(samples), dtype=torch.bool)
    for key in set().union(*(sample.keys() for sample in samples)) - {"task", "img", "face_box"}:
        batch[key] = torch.stack([sample[key] for sample in samples])
    return batch

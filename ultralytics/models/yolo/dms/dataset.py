"""ROI datasets and collate functions backed by the DMS JSONL manifests."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.data.augment import LetterBox

from .schema import PART_SLOTS, is_face_label


def _box(values: list[float] | tuple[float, ...]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _letterbox(image: np.ndarray, boxes: np.ndarray, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    """Use Ultralytics LetterBox and apply the same recorded affine to xyxy boxes."""
    transform = LetterBox(new_shape=(image_size, image_size), auto=False, scaleup=True, stride=32)
    params = transform.get_params({"img": image})
    resized = transform(image=image)
    ratio, left, top = params["ratio"][0], params["left"], params["top"]
    if boxes.size:
        boxes = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio + left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio + top
    tensor = torch.from_numpy(np.ascontiguousarray(resized[..., ::-1].transpose(2, 0, 1)))
    return tensor, boxes


class DMSROIDataset(Dataset):
    """One manifest task per dataset; each sample carries exactly one face ROI."""

    def __init__(self, manifest: str | Path, task: str, image_size: int = 640):
        self.task, self.image_size = task, image_size
        rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()]
        self.samples = self._expand(rows)
        if not self.samples:
            raise ValueError(f"no {task} samples in {manifest}")

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

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = cv2.imread(sample["image"])
        if image is None:
            raise FileNotFoundError(sample["image"])
        face = _box(sample["face_box_xyxy"]).reshape(1, 4)
        img, face = _letterbox(image, face, self.image_size)
        result: dict[str, Any] = {
            "task": self.task,
            "img": img,
            "face_box": torch.from_numpy(face[0]),
        }
        if self.task == "landmark":
            points = torch.tensor(sample["points"], dtype=torch.float32)
            original_face = _box(sample["face_box_xyxy"])
            result["landmark_xy"] = (points - original_face[:2]) / (original_face[2:] - original_face[:2]).clip(min=1)
            result["landmark_valid"] = torch.ones(12, dtype=torch.bool)
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
            original_face = _box(sample["face_box_xyxy"])
            for target in sample["targets"]:
                part = target["part"]
                raw_box = _box([target["box"][axis] for axis in ("x1", "y1", "x2", "y2")])
                center_x = (raw_box[0] + raw_box[2]) / 2
                slot = 2 if part == "mouth" else (0 if center_x < (original_face[0] + original_face[2]) / 2 else 1)
                part_box[slot] = torch.from_numpy((raw_box - np.tile(original_face[:2], 2)) / np.tile((original_face[2:] - original_face[:2]).clip(min=1), 2))
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

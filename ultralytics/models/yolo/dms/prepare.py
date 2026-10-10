"""Build auditable JSONL manifests from DSM, Eye12 and 300W-LP sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

_RESEARCH_ROOT = Path(__file__).resolve().parents[6]
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

import yaml

from .data import dsm_image_path, dsm_manifest_entry, iter_dsm, iter_eye12, parse_300wlp_mat
from .schema import is_face_label, normalized_label
from dms.shape_bayes import fit_shape_prior


GLOBAL_NAMES = ("face", "phone", "cigarette")
SPLIT_NAMES = ("train", "val", "test")


def _write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _eye12_sources(root: Path) -> list[Path]:
    annotations = root / "annotations"
    mixed = sorted(annotations.glob("eye12_mixed_*.jsonl"))
    return mixed or sorted(annotations.glob("*.jsonl"))


def _global_class(label: str) -> int | None:
    label = normalized_label(label)
    if is_face_label(label):
        return 0
    if label in {"phone", "playphone"}:
        return 1
    if label in {"cigar", "cigarette"}:
        return 2
    return None


def _split_paths(paths: list[Path], seed: int) -> dict[Path, str]:
    """Assign deterministic 70/15/15 splits without relying on source ordering."""
    if not paths:
        return {}
    order = list(paths)
    random.Random(seed).shuffle(order)
    train_end = max(1, int(len(order) * 0.70))
    val_end = min(len(order), train_end + int(len(order) * 0.15))
    return {
        path: "train" if index < train_end else "val" if index < val_end else "test"
        for index, path in enumerate(order)
    }


def _link(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if mode == "symlink":
        target.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, target)
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(f"unknown link mode: {mode}")


def materialize_global_yolo(
    dsm_root: Path,
    output: Path,
    seed: int = 20260916,
    link_mode: str = "hardlink",
    records=None,
    splits: dict[Path, str] | None = None,
) -> Path:
    """Build a deterministic native-YOLO dataset from whole-frame DSM VOC XML."""
    records = list(iter_dsm(dsm_root)) if records is None else records
    if not records:
        raise ValueError(f"no DSM VOC annotations under {dsm_root}")
    split_of = _split_paths([record.annotation for record in records], seed) if splits is None else splits
    rows = []
    for record in records:
        split, image = split_of[record.annotation], dsm_image_path(record)
        stem = hashlib.sha256(str(record.annotation).encode()).hexdigest()[:16]
        image_name = stem + image.suffix.lower()
        destination = output / "images" / split / image_name
        _link(image, destination, link_mode)
        labels = []
        for obj in record.objects:
            class_id = _global_class(obj.label)
            if class_id is None:
                continue
            box = obj.box
            x_center, y_center = (box.x1 + box.x2) / 2 / record.width, (box.y1 + box.y2) / 2 / record.height
            width, height = box.width / record.width, box.height / record.height
            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        rows.append({"annotation": str(record.annotation), "image": str(image), "split": split, "global_labels": len(labels)})
    data = {"path": str(output), "train": "images/train", "val": "images/val", "test": "images/test", "names": dict(enumerate(GLOBAL_NAMES))}
    (output / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "split_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return output / "data.yaml"


def build_manifests(
    dsm_root: Path,
    eye12_root: Path,
    wlp_root: Path,
    output: Path,
    global_yolo_output: Path | None = None,
    split_seed: int = 20260916,
    link_mode: str = "hardlink",
) -> dict[str, int | str]:
    """Materialize source records without copying images or silently dropping ambiguity."""
    dsm_records = list(iter_dsm(dsm_root))
    dsm_splits = _split_paths([record.annotation for record in dsm_records], split_seed)
    pose_sources = sorted(wlp_root.glob("**/*.mat"))
    pose_splits = _split_paths(pose_sources, split_seed)
    counts = {
        "dsm": _write_jsonl(
            output / "dsm.jsonl",
            ({**dsm_manifest_entry(record), "split": dsm_splits[record.annotation]} for record in dsm_records),
        ),
        "eye12": _write_jsonl(
            output / "eye12.jsonl",
            (
                {
                    "task": "landmark",
                    "image": str(record.image),
                    "points": record.points,
                    "face_box_xyxy": [0, 0, record.input_size, record.input_size],
                    "crop_box_xyxy": list(vars(record.crop_box).values()),
                    "alignment_matrix": record.alignment_matrix,
                    "split": record.split,
                }
                for source in _eye12_sources(eye12_root)
                for record in iter_eye12(source)
            ),
        ),
        "pose": _write_jsonl(
            output / "pose.jsonl",
            (
                {
                    "task": "pose",
                    "image": str(record.image),
                    "face_box_xyxy": list(vars(record.face_box).values()),
                    "pose_ypr_deg": record.ypr_deg,
                    "split": pose_splits[source],
                }
                for source in pose_sources
                for record in (parse_300wlp_mat(source),)
                if record.image.is_file()
            ),
        ),
    }
    eye_rows = [
        {
            "points": [
                [
                    (point[0] - record.crop_box.x1) / max(record.crop_box.width, 1.0),
                    (point[1] - record.crop_box.y1) / max(record.crop_box.height, 1.0),
                ]
                for point in record.points
            ]
        }
        for source in _eye12_sources(eye12_root)
        for record in iter_eye12(source)
    ]
    (output / "shape_prior.json").write_text(
        json.dumps(fit_shape_prior(eye_rows), indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    if global_yolo_output:
        counts["global_yolo"] = str(
            materialize_global_yolo(dsm_root, global_yolo_output, split_seed, link_mode, dsm_records, dsm_splits)
        )
        (output / "summary.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsm-root", type=Path, required=True)
    parser.add_argument("--eye12-root", type=Path, required=True)
    parser.add_argument("--300wlp-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-yolo-output", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260916)
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    wlp_root = getattr(args, "300wlp_root")
    print(
        json.dumps(
            build_manifests(
                args.dsm_root,
                args.eye12_root,
                wlp_root,
                args.output,
                args.global_yolo_output,
                args.split_seed,
                args.link_mode,
            )
        )
    )


if __name__ == "__main__":
    main()

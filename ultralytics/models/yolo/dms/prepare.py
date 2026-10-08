"""Build auditable JSONL manifests from DSM, Eye12 and 300W-LP sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import dsm_manifest_entry, iter_dsm, iter_eye12, parse_300wlp_mat


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


def build_manifests(dsm_root: Path, eye12_root: Path, wlp_root: Path, output: Path) -> dict[str, int]:
    """Materialize source records without copying images or silently dropping ambiguity."""
    counts = {
        "dsm": _write_jsonl(output / "dsm.jsonl", (dsm_manifest_entry(record) for record in iter_dsm(dsm_root))),
        "eye12": _write_jsonl(
            output / "eye12.jsonl",
            (
                {
                    "task": "landmark",
                    "image": str(record.image),
                    "points": record.points,
                    "face_box_xyxy": [0, 0, record.input_size, record.input_size],
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
                }
                for source in sorted(wlp_root.glob("**/*.mat"))
                for record in (parse_300wlp_mat(source),)
                if record.image.is_file()
            ),
        ),
    }
    (output / "summary.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsm-root", type=Path, required=True)
    parser.add_argument("--eye12-root", type=Path, required=True)
    parser.add_argument("--300wlp-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wlp_root = getattr(args, "300wlp_root")
    print(json.dumps(build_manifests(args.dsm_root, args.eye12_root, wlp_root, args.output)))


if __name__ == "__main__":
    main()

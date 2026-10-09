"""Thin trainer registration; native detection training is inherited unchanged."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from collections.abc import Iterator
from pathlib import Path

_RESEARCH_ROOT = Path(__file__).resolve().parents[6]
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

import torch
from torch.utils.data import DataLoader

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import torch_safe_load
from ultralytics.utils import RANK

from .model import DMSModel
from .dataset import DMSROIDataset, collate_roi
from .loss import DMSROILoss
from .schema import LANDMARK_NAMES
from .teacher import CropTeacher, distillation_loss, make_teacher_crops
from dms.shape_bayes import ShapeBayesPrior


class DMSTrainer(DetectionTrainer):
    """Use DMSModel while retaining Ultralytics' normal detection trainer."""

    def get_model(self, cfg: str | dict | None = None, weights: torch.nn.Module | None = None, verbose: bool = True) -> DMSModel:
        model = self.set_model_names_for_load(
            DMSModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model


def _cycle(loader: DataLoader) -> Iterator[dict]:
    while True:
        yield from loader


def _cuda_devices(spec: str) -> tuple[torch.device, list[int]]:
    """Return the primary device and CUDA ids for a compact single-process trainer."""
    value = str(spec).strip()
    if value.startswith("cuda"):
        ids = value.split(":", 1)[1] if ":" in value else "0"
    elif "," in value or value.isdigit():
        ids = value
    else:
        return torch.device(value), []
    device_ids = [int(item) for item in ids.split(",") if item.strip()]
    if not device_ids:
        device_ids = [0]
    return torch.device(f"cuda:{device_ids[0]}"), device_ids


def _parallel(module: torch.nn.Module, device_ids: list[int]) -> torch.nn.Module:
    if len(device_ids) > 1:
        return torch.nn.DataParallel(module, device_ids=device_ids)
    return module


class _ROIForward(torch.nn.Module):
    """DataParallel adapter that rebuilds local ROI batch indices per GPU."""

    def __init__(self, model: DMSModel):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, face_boxes: torch.Tensor, roi_valid: torch.Tensor | None = None):
        indexes = torch.arange(face_boxes.shape[0], device=face_boxes.device, dtype=face_boxes.dtype).unsqueeze(1)
        rois = torch.cat((indexes, face_boxes), dim=1)
        return self.model.forward_dms(image, rois, roi_valid)


def _roi_loaders(
    manifest_dir: Path, image_size: int, batch_size: int, split: str, tasks: tuple[str, ...] | None = None,
    workers: int = 0,
) -> dict[str, Iterator[dict]]:
    sources = {"parts": "dsm.jsonl", "landmark": "eye12.jsonl", "pose": "pose.jsonl"}
    loaders = {}
    for task in tasks or tuple(sources):
        filename = sources[task]
        path = manifest_dir / filename
        if path.is_file() and path.stat().st_size:
            dataset = DMSROIDataset(path, task, image_size, split)
            loaders[task] = _cycle(DataLoader(
                dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_roi,
                num_workers=workers, pin_memory=torch.cuda.is_available(),
                persistent_workers=workers > 0,
            ))
    if not loaders:
        raise ValueError(f"no usable ROI manifests in {manifest_dir}")
    return loaders


def _device_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def load_dms_weights(model: DMSModel, weights: str | Path | None) -> DMSModel:
    """Load either a native Ultralytics checkpoint or a DMS ROI state-dict checkpoint."""
    if not weights:
        return model
    checkpoint, _ = torch_safe_load(weights)
    stored = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if isinstance(stored, dict):
        model.load_state_dict(stored, strict=False)
    else:
        model.load(checkpoint)
    return model


def load_teacher_weights(teacher: CropTeacher, weights: str | Path) -> CropTeacher:
    """Load a self-contained crop-teacher checkpoint and freeze it for KD*."""
    checkpoint, _ = torch_safe_load(weights)
    stored = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(stored, dict):
        raise TypeError(f"teacher checkpoint has no state_dict: {weights}")
    teacher.load_state_dict(stored)
    return teacher.freeze()


def _teacher_valid(batch: dict) -> torch.Tensor:
    """KD* only applies where the crop teacher has landmark or pose supervision."""
    if "landmark_valid" in batch:
        return batch["landmark_valid"].bool().any(dim=1)
    if "pose_valid" in batch:
        return batch["pose_valid"].bool()
    return torch.zeros(batch["img"].shape[0], device=batch["img"].device, dtype=torch.bool)


def _write_teacher_artifacts(output: Path, history: list[dict]) -> None:
    (output / "teacher_train_metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    tasks = sorted({task for row in history for task in row})
    with (output / "teacher_train_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", *tasks])
        writer.writeheader()
        for index, row in enumerate(history, 1):
            writer.writerow({"epoch": index, **row})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Landmark heatmap loss and pose angle loss have different units/scales.
        # Plot them with independent y-axes; never compare their raw heights.
        fig, axes = plt.subplots(max(1, len(tasks)), 1, figsize=(9, 4 * max(1, len(tasks))), squeeze=False)
        axes = [item for row in axes for item in row]
        for index, task in enumerate(tasks):
            values = [row.get(task) for row in history]
            axes[index].plot(range(1, len(values) + 1), values, label=task)
            axes[index].set_xlabel("Epoch")
            axes[index].set_ylabel(f"{task} loss")
            axes[index].set_title(f"DMS Teacher - {task} loss")
            axes[index].grid(True, alpha=0.3)
            axes[index].legend()
        fig.tight_layout()
        fig.savefig(output / "teacher_loss_curves.png", dpi=150)
        plt.close(fig)
        for task in tasks:
            values = [row.get(task) for row in history]
            plt.figure(figsize=(9, 4))
            plt.plot(range(1, len(values) + 1), values, label=task)
            plt.xlabel("Epoch")
            plt.ylabel(f"{task} loss")
            plt.title(f"DMS Teacher - {task} loss")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output / f"teacher_loss_{task}.png", dpi=150)
            plt.close()
    except Exception as exc:
        (output / "teacher_plot_error.txt").write_text(str(exc) + "\n", encoding="utf-8")

def train_teacher(
    manifest_dir: str | Path,
    output: str | Path,
    epochs: int,
    batch_size: int,
    image_size: int,
    steps_per_epoch: int,
    crop_size: int = 128,
    lr: float = 1e-4,
    device: str = "cpu",
    workers: int = 0,
) -> Path:
    """Train the RGB crop teacher on only landmark and pose train samples."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device_obj, device_ids = _cuda_devices(device)
    teacher = CropTeacher().to(device_obj).train()
    teacher_parallel = _parallel(teacher, device_ids)
    criterion = DMSROILoss()
    loaders = _roi_loaders(Path(manifest_dir), image_size, batch_size, "train", ("landmark", "pose"), workers)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=lr)
    history = []
    log_path = output / "teacher_train.log"
    tasks = tuple(loaders)
    log_path.write_text(f"{datetime.now().isoformat(timespec='seconds')} teacher training start\n", encoding="utf-8")
    for epoch in range(epochs):
        totals, counts = {}, {}
        for _ in range(steps_per_epoch):
            task = random.choice(tasks)
            batch = _device_batch(next(loaders[task]), device_obj)
            batch["img"] = batch["img"].float() / 255
            optimizer.zero_grad(set_to_none=True)
            outputs = teacher_parallel(make_teacher_crops(batch["img"], batch["face_rois"], crop_size))
            loss, _ = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            totals[task] = totals.get(task, 0.0) + float(loss.detach())
            counts[task] = counts.get(task, 0) + 1
        epoch_metrics = {task: totals[task] / counts[task] for task in totals}
        history.append(epoch_metrics)
        message = f"{datetime.now().isoformat(timespec='seconds')} epoch={epoch + 1}/{epochs} " + " ".join(f"{task}_loss={value:.6f}" for task, value in sorted(epoch_metrics.items()))
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        _write_teacher_artifacts(output, history)
        torch.save(
            {
                "model": teacher.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "metadata": {
                    "crop_size": crop_size,
                    "landmark_names": LANDMARK_NAMES,
                    "pose_order": "yaw_pitch_roll_deg",
                },
            },
            output / "teacher.pt",
        )
    _write_teacher_artifacts(output, history)
    return output / "teacher.pt"


def _student_roi_loss(
    model: DMSModel,
    roi_forward: torch.nn.Module | None,
    batch: dict,
    teacher: CropTeacher | None,
    teacher_crop_size: int,
    kd_feature_weight: float,
    kd_heatmap_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if roi_forward is None:
        outputs = model.forward_dms(batch["img"], batch["face_rois"], batch.get("roi_valid"))
    else:
        outputs = roi_forward(batch["img"], batch["face_rois"][:, 1:], batch.get("roi_valid"))
    total, items = model.dms_criterion(outputs, batch)
    valid = _teacher_valid(batch)
    if teacher is not None and valid.any():
        with torch.inference_mode():
            teacher_outputs = teacher(make_teacher_crops(batch["img"], batch["face_rois"], teacher_crop_size))
        kd = distillation_loss(outputs, teacher_outputs, valid)
        total = total + kd_feature_weight * kd["kd_feature"] + kd_heatmap_weight * kd["kd_heatmap"]
        items.update(kd)
    return total, items


def train_roi(
    model: DMSModel,
    manifest_dir: str | Path,
    output: str | Path,
    epochs: int,
    batch_size: int,
    image_size: int,
    steps_per_epoch: int,
    lr: float = 1e-4,
    task_weights: dict[str, float] | None = None,
    device: str = "cpu",
    teacher: CropTeacher | None = None,
    teacher_crop_size: int = 128,
    kd_feature_weight: float = 1.0,
    kd_heatmap_weight: float = 1.0,
    workers: int = 0,
) -> Path:
    """Train heterogeneous ROI batches; global detector training stays with DetectionTrainer."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device_obj, device_ids = _cuda_devices(device)
    model.to(device_obj).train()
    # ROI fine-tuning must not move the global detector/backbone.  The deploy
    # checkpoint intentionally takes those weights from global best.pt.
    model.model.eval()
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    # Materialize LazyConv2d before constructing the optimizer/DataParallel.
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device_obj)
        dummy_roi = torch.tensor([[0.0, 0.0, 0.0, float(image_size), float(image_size)]], device=device_obj)
        model.forward_dms(dummy, dummy_roi, torch.ones(1, dtype=torch.bool, device=device_obj))
    model.dms_heads.train()
    for parameter in model.dms_heads.parameters():
        parameter.requires_grad_(True)
    shape_prior = ShapeBayesPrior.from_json(Path(manifest_dir) / "shape_prior.json")
    if shape_prior is not None:
        model.dms_criterion.shape_prior = shape_prior.to(device_obj)
    if teacher is not None:
        teacher.to(device_obj).freeze()
    loaders = _roi_loaders(Path(manifest_dir), image_size, batch_size, "train", workers=workers)
    task_weights = task_weights or {task: 1.0 for task in loaders}
    tasks = [task for task in loaders if task_weights.get(task, 0) > 0]
    if not tasks:
        raise ValueError("all ROI task weights are zero")
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=lr)
    roi_forward = _parallel(_ROIForward(model), device_ids)
    history = []
    for epoch in range(epochs):
        totals, counts = {}, {}
        for _ in range(steps_per_epoch):
            task = random.choices(tasks, weights=[task_weights[item] for item in tasks], k=1)[0]
            batch = _device_batch(next(loaders[task]), device_obj)
            batch["img"] = batch["img"].float() / 255
            optimizer.zero_grad(set_to_none=True)
            loss, items = _student_roi_loss(
                model, roi_forward, batch, teacher, teacher_crop_size, kd_feature_weight, kd_heatmap_weight
            )
            loss.backward()
            optimizer.step()
            totals[task] = totals.get(task, 0.0) + float(loss.detach())
            counts[task] = counts.get(task, 0) + 1
        history.append({task: totals[task] / counts[task] for task in totals})
        checkpoint = {"model": model.state_dict(), "epoch": epoch + 1, "history": history}
        torch.save(checkpoint, output / "last.pt")
    (output / "roi_train_metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return output / "last.pt"


@torch.inference_mode()
def evaluate_roi(
    model: DMSModel, manifest_dir: str | Path, batch_size: int, image_size: int, batches_per_task: int, device: str = "cpu"
) -> dict[str, float]:
    """Evaluate mean supervised ROI loss per task; metric heads are added separately."""
    device_obj, _ = _cuda_devices(device)
    model.to(device_obj).eval()
    results = {}
    for task, loader in _roi_loaders(Path(manifest_dir), image_size, batch_size, "test").items():
        losses = []
        for _ in range(batches_per_task):
            batch = _device_batch(next(loader), device_obj)
            batch["img"] = batch["img"].float() / 255
            loss, _ = model.loss(batch)
            losses.append(float(loss))
        results[f"{task}_loss"] = sum(losses) / len(losses)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate DMS ROI heads from prepared manifests.")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--model-config", default="yolo26n.yaml")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train-teacher", action="store_true")
    parser.add_argument("--teacher-weights", type=Path)
    parser.add_argument("--teacher-crop-size", type=int, default=128)
    parser.add_argument("--kd-feature-weight", type=float, default=1.0)
    parser.add_argument("--kd-heatmap-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.train_teacher:
        print(
            train_teacher(
                args.manifest_dir,
                args.output,
                args.epochs,
                args.batch,
                args.imgsz,
                args.steps_per_epoch,
                args.teacher_crop_size,
                device=args.device,
                workers=args.workers,
            )
        )
        return
    model = load_dms_weights(DMSModel(args.model_config, nc=3, verbose=True), args.weights)
    if args.eval:
        metrics = evaluate_roi(model, args.manifest_dir, args.batch, args.imgsz, args.steps_per_epoch, args.device)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "roi_eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics))
    else:
        teacher = load_teacher_weights(CropTeacher(), args.teacher_weights) if args.teacher_weights else None
        print(
            train_roi(
                model,
                args.manifest_dir,
                args.output,
                args.epochs,
                args.batch,
                args.imgsz,
                args.steps_per_epoch,
                device=args.device,
                teacher=teacher,
                teacher_crop_size=args.teacher_crop_size,
                kd_feature_weight=args.kd_feature_weight,
                kd_heatmap_weight=args.kd_heatmap_weight,
                workers=args.workers,
            )
        )


if __name__ == "__main__":
    main()

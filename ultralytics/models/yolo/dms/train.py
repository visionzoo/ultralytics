"""Thin trainer registration; native detection training is inherited unchanged."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import torch_safe_load
from ultralytics.utils import RANK

from .model import DMSModel
from .dataset import DMSROIDataset, collate_roi


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


def _roi_loaders(manifest_dir: Path, image_size: int, batch_size: int, split: str) -> dict[str, Iterator[dict]]:
    sources = {"parts": "dsm.jsonl", "landmark": "eye12.jsonl", "pose": "pose.jsonl"}
    loaders = {}
    for task, filename in sources.items():
        path = manifest_dir / filename
        if path.is_file() and path.stat().st_size:
            dataset = DMSROIDataset(path, task, image_size, split)
            loaders[task] = _cycle(DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_roi))
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
) -> Path:
    """Train heterogeneous ROI batches; global detector training stays with DetectionTrainer."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    model.to(device_obj).train()
    loaders = _roi_loaders(Path(manifest_dir), image_size, batch_size, "train")
    task_weights = task_weights or {task: 1.0 for task in loaders}
    tasks = [task for task in loaders if task_weights.get(task, 0) > 0]
    if not tasks:
        raise ValueError("all ROI task weights are zero")
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=lr)
    history = []
    for epoch in range(epochs):
        totals, counts = {}, {}
        for _ in range(steps_per_epoch):
            task = random.choices(tasks, weights=[task_weights[item] for item in tasks], k=1)[0]
            batch = _device_batch(next(loaders[task]), device_obj)
            batch["img"] = batch["img"].float() / 255
            optimizer.zero_grad(set_to_none=True)
            loss, items = model.loss(batch)
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
    device_obj = torch.device(device)
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
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    model = load_dms_weights(DMSModel(args.model_config, nc=3, verbose=True), args.weights)
    if args.eval:
        metrics = evaluate_roi(model, args.manifest_dir, args.batch, args.imgsz, args.steps_per_epoch, args.device)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "roi_eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics))
    else:
        print(train_roi(model, args.manifest_dir, args.output, args.epochs, args.batch, args.imgsz, args.steps_per_epoch, device=args.device))

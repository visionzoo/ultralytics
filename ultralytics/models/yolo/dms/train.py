"""Thin trainer registration; native detection training is inherited unchanged."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
from datetime import datetime
from collections.abc import Iterator
from pathlib import Path

_RESEARCH_ROOT = Path(__file__).resolve().parents[6]
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import torch_safe_load
from ultralytics.utils import RANK

from .model import DMSModel
from .dataset import DMSROIDataset, collate_roi
from .loss import DMSROILoss
from .schema import LANDMARK_NAMES, ROI_INPUT_CONTRACT, DETECTION_NAMES
from .teacher import CropTeacher, TeacherLoss, distillation_loss, make_teacher_crops, landmark_teacher_quality, warm_start_teacher_pose
from .teacher_validation import evaluate_teacher_landmarks, teacher_selection, validate_teacher_checkpoint
from .geometry import matched_prediction_rois, rebase_targets
from ultralytics.nn.modules.dms import roi_align_features
from dms.shape_bayes import ShapeBayesPrior


def _atomic_torch_save(payload, target: Path) -> None:
    """Write a checkpoint atomically so interruption cannot corrupt it."""
    target = Path(target)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(payload, temporary)
    try:
        os.replace(temporary, target)
    except PermissionError:
        # CIFS/SMB mounts can reject rename-over-existing-file even when the
        # directory is writable.  Fall back to a verified copy and cleanup.
        shutil.copyfile(temporary, target)
        temporary.unlink(missing_ok=True)


def _tensorboard_writer(log_dir: str | Path | None, job_type: str):
    """TensorBoard writer that also feeds optional storage-light W&B scalars."""
    from dms.wandb_tracking import MetricWriter

    writer = MetricWriter(log_dir, job_type)
    return writer if writer else None


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
    workers: int = 0, pin_memory: bool = False,
    match_scale: bool = True,
) -> dict[str, Iterator[dict]]:
    sources = {"parts": "dsm.jsonl", "landmark": "eye12.jsonl", "pose": "pose.jsonl"}
    loaders = {}
    for task in tasks or tuple(sources):
        filename = sources[task]
        path = manifest_dir / filename
        if path.is_file() and path.stat().st_size:
            dataset = DMSROIDataset(path, task, image_size, split, match_scale=match_scale)
            loader_kwargs = {
                "batch_size": batch_size,
                "collate_fn": collate_roi,
                "num_workers": workers,
                "pin_memory": pin_memory,
                "persistent_workers": workers > 0,
            }
            if task == "pose" and split == "train":
                loader_kwargs["sampler"] = WeightedRandomSampler(
                    dataset.pose_sampling_weights(), num_samples=len(dataset), replacement=True
                )
            else:
                loader_kwargs["shuffle"] = True
            # Never fork a worker after CUDA has been initialized.  Spawned
            # workers are independent CPU readers and cannot inherit CUDA locks.
            if workers > 0:
                loader_kwargs["multiprocessing_context"] = mp.get_context("spawn")
            loaders[task] = _cycle(DataLoader(dataset, **loader_kwargs))
    if not loaders:
        raise ValueError(f"no usable ROI manifests in {manifest_dir}")
    return loaders


def _pose_train_stats(manifest_dir: Path) -> tuple[list[float], list[float]]:
    """Read train-split 300W-LP statistics without loading images."""
    values = []
    path = manifest_dir / "pose.jsonl"
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split") == "train" and row.get("pose_ypr_deg"):
                    values.append(row["pose_ypr_deg"])
    if not values:
        return [0.0, 0.0, 0.0], [55.0, 12.0, 11.0]
    tensor = torch.tensor(values, dtype=torch.float32)
    return tensor.mean(0).tolist(), tensor.std(0, unbiased=False).clamp_min(1.0).tolist()


def _device_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def roi_selection_metric(metrics: dict) -> float | None:
    """Select only on validated pose/localization quality, never loss sums."""
    if "validation_skipped" in metrics:
        return None
    if "pose/pose_mae_yaw_deg" not in metrics or "landmark/landmark_eye12_nme" not in metrics:
        raise ValueError("Missing yaw/Eye12 validation metrics: refusing loss-sum selection")
    value = float(metrics["pose/pose_mae_yaw_deg"]) / 20.0 + float(metrics["landmark/landmark_eye12_nme"])
    if not math.isfinite(value):
        raise ValueError("Non-finite validation selection metric")
    return value


@torch.inference_mode()
def _teacher_pose_mae(teacher: torch.nn.Module, manifest_dir: Path, image_size: int,
                      crop_size: int, batch_size: int, device: torch.device,
                      split: str = "val", match_scale: bool = False) -> tuple[float, float, float]:
    """Evaluate per-axis teacher pose MAE on a finite split."""
    dataset = DMSROIDataset(manifest_dir / "pose.jsonl", "pose", image_size, split, match_scale=match_scale)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda", collate_fn=collate_roi)
    was_training = teacher.training
    teacher.eval()
    error, count = torch.zeros(3, device=device), 0
    for batch in loader:
        batch = _device_batch(batch, device)
        image = batch["img"].float() / 255.0
        prediction = teacher(make_teacher_crops(image, batch["face_rois"], crop_size))["pose_ypr"]
        error += (prediction - batch["pose_ypr"].float()).abs().sum(dim=0)
        count += int(prediction.shape[0])
    if was_training:
        teacher.train()
    return tuple(float(value) for value in (error / max(count, 1)).cpu())


def load_dms_weights(model: DMSModel, weights: str | Path | None) -> DMSModel:
    """Load either a native Ultralytics checkpoint or a DMS ROI state-dict checkpoint."""
    if not weights:
        return model
    checkpoint, _ = torch_safe_load(weights)
    stored = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if isinstance(stored, dict):
        if any(key.startswith("dms_heads.") for key in stored):
            metadata = checkpoint.get("metadata", {})
            contract = metadata.get("roi_input_contract")
            if contract != ROI_INPUT_CONTRACT:
                raise ValueError(
                    "DMS ROI checkpoint uses the old pixel-crop encoder contract. "
                    "Reuse the global detector and retrain ROI heads with dms-roi-train before merging."
                )
            stored_architecture = metadata.get("dms_architecture")
            if stored_architecture != model.dms_architecture:
                raise ValueError(
                    "DMS ROI checkpoint architecture does not match the configured model: "
                    f"checkpoint={stored_architecture}, model={model.dms_architecture}"
                )
            if tuple(metadata.get("class_names", ())) != DETECTION_NAMES:
                raise ValueError("DMS checkpoint class names/order missing or incompatible")
            for name, value in metadata.get("loss_config", {}).items():
                if name not in {"heatmap_sigma", "pose_axis_weights", "pose_continuous_weight", "landmark_coord_weight", "landmark_peak_weight", "landmark_geometry_weight"}:
                    raise ValueError(f"Unknown checkpoint loss setting: {name}")
                setattr(model.dms_criterion, name, tuple(value) if name == "pose_axis_weights" else value)
        # ``strict=False`` still raises on same-name shape changes.  ROI heads
        # are intentionally replaceable (for example, a higher-capacity
        # heatmap refinement tower), so load only compatible tensors and let
        # newly introduced layers keep their initialization.
        current = model.state_dict()
        compatible = {}
        for key, value in stored.items():
            if key not in current:
                continue
            try:
                shape_ok = tuple(current[key].shape) == tuple(value.shape)
            except RuntimeError:
                # LazyConv2d has no shape until the first ROI forward.  Its
                # checkpoint tensor is exactly what should initialize it.
                shape_ok = True
            if shape_ok:
                compatible[key] = value
        skipped = sorted(set(stored) - set(compatible))
        required = {key for key in current if key.startswith(("model.", "dms_heads."))}
        if any(key.startswith("dms_heads.") for key in stored):
            missing = required - set(compatible)
            incompatible = [key for key in skipped if key.startswith(("model.", "dms_heads."))]
            if missing or incompatible:
                raise ValueError(f"Incomplete DMS checkpoint: missing={sorted(missing)[:8]}, incompatible={incompatible[:8]}")
        if skipped:
            print(f"dms_weight_skip_incompatible={skipped[:8]}", flush=True)
        model.load_state_dict(compatible, strict=False)
    else:
        names = getattr(stored, "names", {})
        names = tuple(names.values()) if isinstance(names, dict) else tuple(names)
        if names != DETECTION_NAMES:
            raise ValueError("Train the DMS global detector first: native checkpoint class names/order differ")
        model.load(checkpoint)
    model.names = dict(enumerate(DETECTION_NAMES))
    return model


def load_teacher_weights(teacher: CropTeacher, weights: str | Path) -> CropTeacher:
    """Load a self-contained crop-teacher checkpoint and freeze it for KD*."""
    checkpoint, _ = torch_safe_load(weights)
    stored = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(stored, dict):
        raise TypeError(f"teacher checkpoint has no state_dict: {weights}")
    loaded = CropTeacher.from_state(stored, checkpoint.get("metadata", {}))
    if loaded.architecture == "crop_teacher_context_v2":
        validate_teacher_checkpoint(checkpoint)
    return loaded.freeze()


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
        # Per-point errors/counts stay in CSV/JSON; avoid dozens of redundant
        # figures per epoch. Named metrics retain their own units, not "loss".
        plot_tasks = [task for task in tasks if task in {
            "landmark", "pose", "landmark_val_nme", "landmark_val_kd_accepted_fraction", "teacher_joint_score",
            "pose_val_yaw_mae_deg", "pose_val_pitch_mae_deg", "pose_val_roll_mae_deg", "pose_val_mae_deg",
            "landmark_val_pck_interocular_005", "landmark_val_pck_interocular_010"}]
        fig, axes = plt.subplots(max(1, len(plot_tasks)), 1, figsize=(9, 4 * max(1, len(plot_tasks))), squeeze=False)
        axes = [item for row in axes for item in row]
        for index, task in enumerate(plot_tasks):
            values = [row.get(task) for row in history]
            axes[index].plot(range(1, len(values) + 1), values, label=task)
            axes[index].set_xlabel("Epoch")
            axes[index].set_ylabel(task)
            axes[index].set_title(f"DMS Teacher - {task}")
            axes[index].grid(True, alpha=0.3)
            axes[index].legend()
        fig.tight_layout()
        fig.savefig(output / "teacher_loss_curves.png", dpi=150)
        plt.close(fig)
        for task in plot_tasks:
            values = [row.get(task) for row in history]
            plt.figure(figsize=(9, 4))
            plt.plot(range(1, len(values) + 1), values, label=task)
            plt.xlabel("Epoch")
            plt.ylabel(task)
            plt.title(f"DMS Teacher - {task}")
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
    tb_log_dir: str | Path | None = None,
    target_pose_mae: float | None = None,
    pose_patience: int = 0,
    pose_min_delta: float = 0.1,
    pose_task_weight: float = 1.0,
    resume_checkpoint: str | Path | None = None,
    pose_axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    target_landmark_nme: float = 0.10,
    init_checkpoint: str | Path | None = None,
    freeze_teacher_encoder: bool = False,
) -> Path:
    """Train the RGB crop teacher on only landmark and pose train samples."""
    output = Path(output)
    pose_target = 10.0 if target_pose_mae is None else float(target_pose_mae)
    if epochs < 1 or steps_per_epoch < 1 or batch_size < 1 or pose_task_weight <= 0 or pose_min_delta < 0:
        raise ValueError("Teacher epochs/steps/batch/task sampling must be positive, min_delta nonnegative")
    teacher_selection(dict(zip(("pose_val_yaw_mae_deg", "pose_val_pitch_mae_deg", "pose_val_roll_mae_deg", "landmark_val_nme"), [0.]*4)),
                      pose_target, target_landmark_nme)
    if resume_checkpoint and init_checkpoint:
        raise ValueError("Use either exact v2 resume or pose warm-start, not both")
    if not resume_checkpoint and any((output / name).exists() for name in ("teacher.pt", "teacher_last.pt", "teacher_best.pt")):
        raise ValueError("Teacher output already contains checkpoints; use a new experiment ID")
    output.mkdir(parents=True, exist_ok=True)
    # Construct CPU readers before touching CUDA.  Worker processes are
    # spawned explicitly below, so they never inherit a CUDA runtime state.
    loaders = _roi_loaders(
        Path(manifest_dir), image_size, batch_size, "train", ("landmark", "pose"), workers,
        pin_memory=False,
        match_scale=True,
    )
    device_obj, device_ids = _cuda_devices(device)
    teacher = CropTeacher()
    resume_epoch = 0
    resumed = None
    if resume_checkpoint:
        resumed = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        metadata = resumed.get("metadata", {})
        if (metadata.get("teacher_architecture") != teacher.architecture or
            metadata.get("roi_input_contract") != ROI_INPUT_CONTRACT or
            metadata.get("landmark_objective") != "spatial_kl_v1" or
            metadata.get("crop_size") != crop_size or metadata.get("image_size") != image_size or
            tuple(metadata.get("pose_axis_weights", ())) != tuple(pose_axis_weights) or
            "optimizer" not in resumed or "scaler" not in resumed):
            raise ValueError("Incompatible Teacher resume; use --teacher-init for an explicit legacy pose warm-start")
        teacher.load_state_dict(resumed["model"], strict=True)
        resume_epoch = int(resumed["epoch"])
    elif init_checkpoint:
        initial = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        # Verify the entire source first, then copy only encoder and pose. The
        # failed landmark branch is intentionally freshly initialized.
        copied = warm_start_teacher_pose(teacher, initial)
        print(f"teacher_pose_warm_start={init_checkpoint} copied={copied} landmark=fresh epoch=0", flush=True)
    teacher = teacher.to(device_obj).train()
    teacher_parallel = _parallel(teacher, device_ids)
    criterion = TeacherLoss(pose_axis_weights=pose_axis_weights)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=lr)
    amp_enabled = device_obj.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if resumed is not None:
        optimizer.load_state_dict(resumed["optimizer"])
        # Resume optimizer moments, while honoring the new run's explicitly
        # configured fine-tuning LR instead of silently restoring the old LR.
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.load_state_dict(resumed["scaler"])
    if freeze_teacher_encoder:
        teacher.encoder.requires_grad_(False)
    history = list(resumed.get("history", [])) if resumed else []
    best_score = teacher_selection(resumed["metadata"]["validation"], pose_target, target_landmark_nme)[0] if resumed else float("inf")
    best_kd_key = (float("-inf"), float("inf"), float("inf"))
    progress_score = best_score
    qualified_payload = None
    stale_epochs = 0
    if resumed is not None:
        same_limits = (resumed["metadata"].get("pose_target_deg") == pose_target and
                       resumed["metadata"].get("landmark_target_nme") == target_landmark_nme)
        same_output = output.resolve() == Path(resume_checkpoint).resolve().parent
        if same_output and not same_limits:
            raise ValueError("Changed Teacher acceptance thresholds require a new experiment directory")
        candidate = resumed
        previous_best = Path(resume_checkpoint).parent / "teacher_best.pt"
        if same_limits and previous_best.is_file():
            saved = torch.load(previous_best, map_location="cpu", weights_only=False)
            if int(saved["epoch"]) <= resume_epoch:
                CropTeacher.from_state(saved["model"], saved["metadata"])
                if teacher_selection(saved["metadata"]["validation"], pose_target, target_landmark_nme)[0] < best_score:
                    candidate = saved
        best_score = teacher_selection(candidate["metadata"]["validation"], pose_target, target_landmark_nme)[0]
        if not same_limits:
            candidate = copy.deepcopy(candidate)
            candidate["metadata"].update(pose_target_deg=pose_target, landmark_target_nme=target_landmark_nme,
                best_joint_score=best_score, qualified=teacher_selection(candidate["metadata"]["validation"], pose_target, target_landmark_nme)[1])
        monitor = resumed.get("early_stopping", {}) if same_limits else {}
        progress_score = float(monitor.get("progress_score", best_score))
        stale_epochs = int(monitor.get("stale_epochs", 0))
        if not same_output:
            _atomic_torch_save(candidate, output / "teacher_best.pt")
        # Keep the best student-distillation candidate separately from the
        # joint pose/NME checkpoint. Reweighting metrics can select different
        # epochs, so downstream work must not lose the KD-acceptance optimum.
        kd_candidate = candidate
        previous_kd = Path(resume_checkpoint).parent / "teacher_best_kd.pt"
        if previous_kd.is_file():
            saved_kd = torch.load(previous_kd, map_location="cpu", weights_only=False)
            CropTeacher.from_state(saved_kd["model"], saved_kd["metadata"])
            saved_kd_metrics = saved_kd["metadata"]["validation"]
            candidate_metrics = candidate["metadata"]["validation"]
            saved_key = (float(saved_kd_metrics["landmark_val_kd_accepted_fraction"]),
                         -float(saved_kd_metrics["landmark_val_nme"]),
                         -teacher_selection(saved_kd_metrics, pose_target, target_landmark_nme)[0])
            candidate_key = (float(candidate_metrics["landmark_val_kd_accepted_fraction"]),
                             -float(candidate_metrics["landmark_val_nme"]),
                             -teacher_selection(candidate_metrics, pose_target, target_landmark_nme)[0])
            if saved_key > candidate_key:
                kd_candidate = saved_kd
        if not same_limits:
            kd_candidate = copy.deepcopy(kd_candidate)
            kd_candidate_score, kd_candidate_qualified = teacher_selection(
                kd_candidate["metadata"]["validation"], pose_target, target_landmark_nme
            )
            kd_candidate["metadata"].update(
                pose_target_deg=pose_target, landmark_target_nme=target_landmark_nme,
                best_joint_score=kd_candidate_score, qualified=kd_candidate_qualified,
            )
        candidate_metrics = kd_candidate["metadata"]["validation"]
        best_kd_key = (
            float(candidate_metrics["landmark_val_kd_accepted_fraction"]),
            -float(candidate_metrics["landmark_val_nme"]),
            -teacher_selection(candidate_metrics, pose_target, target_landmark_nme)[0],
        )
        if not same_output:
            kd_candidate = copy.deepcopy(kd_candidate)
            kd_candidate["metadata"].update(
                selection_metric="max_kd_acceptance_then_min_nme_then_min_joint_score",
                checkpoint_role="best_kd_acceptance",
            )
            _atomic_torch_save(kd_candidate, output / "teacher_best_kd.pt")
        else:
            existing_kd = output / "teacher_best_kd.pt"
            if existing_kd.is_file():
                saved_kd = torch.load(existing_kd, map_location="cpu", weights_only=False)
                saved_metrics = saved_kd["metadata"]["validation"]
                saved_key = (float(saved_metrics["landmark_val_kd_accepted_fraction"]),
                             -float(saved_metrics["landmark_val_nme"]),
                             -teacher_selection(saved_metrics, pose_target, target_landmark_nme)[0])
                best_kd_key = max(best_kd_key, saved_key)
            else:
                kd_candidate = copy.deepcopy(kd_candidate)
                kd_candidate["metadata"].update(
                    selection_metric="max_kd_acceptance_then_min_nme_then_min_joint_score",
                    checkpoint_role="best_kd_acceptance",
                )
                _atomic_torch_save(kd_candidate, existing_kd)
    writer = _tensorboard_writer(tb_log_dir, "teacher")
    log_path = output / "teacher_train.log"
    tasks = tuple(loaders)
    # Balanced by default; these are sampling probabilities, not loss weights.
    task_weights = {"landmark": 1.0, "pose": max(float(pose_task_weight), 0.0)}
    with log_path.open("a" if resumed else "w", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat(timespec='seconds')} teacher training start architecture={teacher.architecture} "
                     f"roi_contract={ROI_INPUT_CONTRACT} objective=spatial_kl_v1 pose_target={pose_target} "
                     f"landmark_target={target_landmark_nme} lr={lr:g} resume_epoch={resume_epoch}\n")
    for epoch in range(epochs):
        totals, counts, item_totals, item_counts = {}, {}, {}, {}
        for _ in range(steps_per_epoch):
            task = random.choices(tasks, weights=[task_weights.get(item, 1.0) for item in tasks], k=1)[0]
            batch = _device_batch(next(loaders[task]), device_obj)
            batch["img"] = batch["img"].float() / 255
            if freeze_teacher_encoder:
                # Keep shared features and BatchNorm statistics fixed so pose
                # and landmark heads can refine independently without task
                # gradients shifting the common representation.
                teacher.encoder.eval()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                outputs = teacher_parallel(make_teacher_crops(batch["img"], batch["face_rois"], crop_size))
                loss, _ = criterion(outputs, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals[task] = totals.get(task, 0.0) + float(loss.detach())
            counts[task] = counts.get(task, 0) + 1
        epoch_metrics = {task: totals[task] / counts[task] for task in totals}
        pose_val_axes = _teacher_pose_mae(
            teacher_parallel, Path(manifest_dir), image_size, crop_size, batch_size, device_obj, match_scale=True
        )
        pose_val_yaw, pose_val_pitch, pose_val_roll = pose_val_axes
        pose_val_mae = sum(pose_val_axes) / 3.0
        epoch_metrics.update({
            "pose_val_yaw_mae_deg": pose_val_yaw,
            "pose_val_pitch_mae_deg": pose_val_pitch,
            "pose_val_roll_mae_deg": pose_val_roll,
            "pose_val_mae_deg": pose_val_mae,
        })
        display_epoch = resume_epoch + epoch + 1
        epoch_metrics.update(evaluate_teacher_landmarks(
            teacher_parallel, manifest_dir, image_size, crop_size, batch_size, device_obj,
            visual_dir=output / "val" / f"epoch{display_epoch:04d}"))
        score, qualified = teacher_selection(epoch_metrics, pose_target, target_landmark_nme)
        epoch_metrics["teacher_joint_score"] = score
        kd_key = (
            float(epoch_metrics["landmark_val_kd_accepted_fraction"]),
            -float(epoch_metrics["landmark_val_nme"]),
            -score,
        )
        kd_improved = kd_key > best_kd_key
        if kd_improved:
            best_kd_key = kd_key
        improved = score < best_score
        best_score = min(best_score, score)
        if score < progress_score - pose_min_delta / pose_target:
            progress_score, stale_epochs = score, 0
        else:
            stale_epochs += 1
        history.append(epoch_metrics)
        if writer:
            for task, value in epoch_metrics.items():
                if value is not None:
                    writer.add_scalar(f"teacher/{task}", value, display_epoch)
        message = f"{datetime.now().isoformat(timespec='seconds')} epoch={display_epoch}/{resume_epoch + epochs} " + " ".join(f"{task}={value:.6f}" for task, value in sorted(epoch_metrics.items()) if value is not None)
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        _write_teacher_artifacts(output, history)
        payload = {
                "model": teacher.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "early_stopping": {"progress_score": progress_score, "stale_epochs": stale_epochs},
                "epoch": display_epoch,
                "history": history,
                "metadata": {
                    "teacher_architecture": teacher.architecture,
                    "roi_input_contract": ROI_INPUT_CONTRACT,
                    "selection_split": "val", "selection_metric": "worst_normalized_pose_landmark_error",
                    "landmark_objective": "spatial_kl_v1", "qualified": qualified,
                    "pose_target_deg": pose_target, "landmark_target_nme": target_landmark_nme,
                    "validation": dict(epoch_metrics), "best_joint_score": best_score,
                    "image_size": image_size, "pose_axis_weights": list(pose_axis_weights),
                    "crop_size": crop_size,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "encoder_frozen": freeze_teacher_encoder,
                    "landmark_names": LANDMARK_NAMES,
                    "pose_order": "yaw_pitch_roll_deg",
                },
            }
        _atomic_torch_save(payload, output / "teacher_last.pt")
        if improved:
            _atomic_torch_save(payload, output / "teacher_best.pt")
        if kd_improved:
            kd_payload = copy.deepcopy(payload)
            kd_payload["metadata"].update(
                selection_metric="max_kd_acceptance_then_min_nme_then_min_joint_score",
                checkpoint_role="best_kd_acceptance",
            )
            _atomic_torch_save(kd_payload, output / "teacher_best_kd.pt")
        if qualified:
            qualified_payload = copy.deepcopy(payload)
        reached_target = qualified
        reached_patience = pose_patience > 0 and stale_epochs >= pose_patience
        if reached_target or reached_patience:
            reason = "target" if reached_target else "patience"
            stop_message = (
                f"{datetime.now().isoformat(timespec='seconds')} early_stop={reason} "
                f"pose_val_yaw_mae_deg={pose_val_yaw:.6f} "
                f"pose_val_mae_deg={pose_val_mae:.6f} landmark_val_nme={epoch_metrics['landmark_val_nme']:.6f} best_joint={best_score:.6f} qualified={qualified}"
            )
            print(stop_message, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(stop_message + "\n")
            break
    if qualified_payload is not None:
        _atomic_torch_save(qualified_payload, output / "teacher.pt")
    _write_teacher_artifacts(output, history)
    if writer:
        writer.summary({"teacher/best_joint_score": best_score, "teacher/epochs_run": len(history), "teacher/qualified": qualified_payload is not None})
        if qualified_payload is not None:
            writer.checkpoint("teacher", output / "teacher.pt")
        writer.close()
    if qualified_payload is None:
        raise RuntimeError("Teacher not jointly qualified: diagnostic teacher_last.pt/teacher_best.pt saved; teacher.pt not published")
    return output / "teacher.pt"


def _student_roi_loss(
    model: DMSModel,
    roi_forward: torch.nn.Module | None,
    batch: dict,
    teacher: CropTeacher | None,
    teacher_crop_size: int,
    kd_feature_weight: float,
    kd_heatmap_weight: float,
    kd_pose_weight: float,
    kd_pose_bin_weight: float,
    predicted_roi_probability: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if roi_forward is None:
        outputs = model.forward_dms(batch["img"], batch["face_rois"], batch.get("roi_valid"))
    else:
        outputs = roi_forward(batch["img"], batch["face_rois"][:, 1:], batch.get("roi_valid"))
    matched = 0
    if predicted_roi_probability > 0:
        rois, matched = matched_prediction_rois(outputs["global_raw"], batch["face_rois"],
                                                batch["img"].shape[-2], batch["img"].shape[-1], predicted_roi_probability)
        batch = rebase_targets(batch, rois)
        # Reuse the already encoded features: no second backbone execution.
        b2, p3 = roi_align_features(outputs["b2"], outputs["p3"], rois, *model.feature_strides)
        outputs.update(model.forward_roi(b2, p3, batch.get("roi_valid")))
    total, items = model.dms_criterion(outputs, batch)
    items["roi_predicted_fraction"] = total.new_tensor(matched / len(batch["img"]))
    valid = _teacher_valid(batch)
    if teacher is not None and valid.any():
        with torch.inference_mode():
            teacher_outputs = teacher(make_teacher_crops(batch["img"], batch["face_rois"], teacher_crop_size))
        if "landmark_valid" in batch:
            accepted, _, _ = landmark_teacher_quality(teacher_outputs, batch)
            valid = valid & accepted
            items["kd_landmark_accepted_fraction"] = accepted.float().mean()
        kd = distillation_loss(
            outputs, teacher_outputs, valid, model.dms_criterion.pose_scale, batch.get("pose_valid"),
            pose_axis_weights=model.dms_criterion.pose_axis_weights,
            landmark_valid=batch.get("landmark_valid"),
        )
        total = total + kd_feature_weight * kd["kd_feature"] + kd_heatmap_weight * kd["kd_heatmap"]
        total = total + kd_pose_weight * kd["kd_pose"] + kd_pose_bin_weight * kd["kd_pose_bins"]
        items.update(kd)
    return total, items


@torch.no_grad()
def _evaluate_roi_epoch(
    model: DMSModel, manifest_dir: Path, image_size: int, batch_size: int, device: torch.device,
    split: str = "val", batches_per_task: int = 0,
) -> dict[str, float]:
    """Finite deterministic validation; held-out test is only used explicitly."""
    model.eval()
    results: dict[str, float] = {}
    sources = {"parts": "dsm.jsonl", "landmark": "eye12.jsonl", "pose": "pose.jsonl"}
    for task, filename in sources.items():
        path = manifest_dir / filename
        if not path.is_file() or not path.stat().st_size:
            continue
        dataset = DMSROIDataset(path, task, image_size, split)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_roi, num_workers=0)
        totals, denominators, count = {}, {}, 0
        pose_sum = pose_sq = None
        for batch_index, batch in enumerate(loader):
            if batches_per_task > 0 and batch_index >= batches_per_task:
                break
            batch = _device_batch(batch, device)
            batch["img"] = batch["img"].float() / 255
            outputs = model.forward_dms(batch["img"], batch["face_rois"], batch.get("roi_valid"))
            _, items = model.dms_criterion(outputs, batch)
            # The final batch may be smaller than ``batch_size``.  Weight each
            # batch by its actual sample count, otherwise validation means are
            # biased whenever len(dataset) is not divisible by batch_size.
            batch_count = int(batch["img"].shape[0])
            count += batch_count
            for name, value in items.items():
                weight = batch_count
                if name == "landmark_eye12_nme":
                    weight = int(batch["landmark_valid"][:, (0, 3, 6, 9)].all(-1).sum())
                elif name in {"landmark_nme", "landmark_soft_nme", "landmark_pck05", "landmark_pck10"}:
                    weight = int(batch["landmark_valid"].sum())
                totals[name] = totals.get(name, 0.0) + float(value) * weight
                denominators[name] = denominators.get(name, 0) + weight
            if task == "pose":
                prediction = outputs["pose_ypr"].detach()
                pose_sum = prediction.sum(0) if pose_sum is None else pose_sum + prediction.sum(0)
                pose_sq = prediction.square().sum(0) if pose_sq is None else pose_sq + prediction.square().sum(0)
        for name, value in totals.items():
            results[f"{task}/{name}"] = value / denominators[name] if denominators[name] else float("nan")
        if task == "pose" and pose_sum is not None:
            mean = pose_sum / max(count, 1)
            std = (pose_sq / max(count, 1) - mean.square()).clamp_min(0).sqrt()
            for index, axis in enumerate(("yaw", "pitch", "roll")):
                results[f"pose/pred_{axis}_mean_deg"] = float(mean[index])
                results[f"pose/pred_{axis}_std_deg"] = float(std[index])
    model.model.eval()
    model.dms_heads.train()
    return results


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
    kd_pose_weight: float = 1.0,
    kd_pose_bin_weight: float = 1.0,
    workers: int = 0,
    amp: bool = True,
    tb_log_dir: str | Path | None = None,
    unfreeze_layers: tuple[int, ...] = (),
    backbone_lr: float = 1e-5,
    pose_weight: float = 3.0,
    landmark_weight: float = 2.0,
    parts_weight: float = 1.0,
    pose_axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    pose_continuous_weight: float = 1.0,
    landmark_coord_weight: float = 1.0,
    landmark_peak_weight: float = 0.25,
    landmark_geometry_weight: float = 4.0,
    val_interval: int = 1,
    pin_memory: bool = True,
    predicted_roi_probability: float = 0.5,
    predicted_roi_start: float = 0.2,
) -> Path:
    """Train heterogeneous ROI batches; global detector training stays with DetectionTrainer."""
    if unfreeze_layers:
        raise ValueError("ROI-only training requires frozen encoder; joint detection fine-tuning is not implemented")
    if not (0 <= predicted_roi_probability <= 1 and 0 <= predicted_roi_start <= 1):
        raise ValueError("Predicted ROI probability/start must be in [0,1]")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Build all CPU-side readers before model.to(cuda).  This ordering is
    # required when the dataset lives on a shared filesystem.
    loaders = _roi_loaders(Path(manifest_dir), image_size, batch_size, "train", workers=workers, pin_memory=pin_memory)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device_obj, device_ids = _cuda_devices(device)
    model.to(device_obj).train()
    # ROI fine-tuning must not move the global detector/backbone.  The deploy
    # checkpoint intentionally takes those weights from global best.pt.
    model.model.eval()
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    for index in unfreeze_layers:
        if index < 0 or index >= len(model.model):
            raise ValueError(f"DMS unfreeze layer out of range: {index}")
        for parameter in model.model[index].parameters():
            parameter.requires_grad_(True)
    # Materialize LazyConv2d before constructing the optimizer/DataParallel.
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device_obj)
        dummy_roi = torch.tensor([[0.0, 0.0, 0.0, float(image_size), float(image_size)]], device=device_obj)
        model.forward_dms(dummy, dummy_roi, torch.ones(1, dtype=torch.bool, device=device_obj))
    model.dms_heads.train()
    for parameter in model.dms_heads.parameters():
        parameter.requires_grad_(True)
    shape_prior = ShapeBayesPrior.from_json(Path(manifest_dir) / "shape_prior.json", expected_contract=ROI_INPUT_CONTRACT)
    if shape_prior is not None:
        model.dms_criterion.shape_prior = shape_prior.to(device_obj)
    pose_center, pose_scale = _pose_train_stats(Path(manifest_dir))
    model.dms_criterion.set_pose_stats(pose_center, pose_scale)
    if len(pose_axis_weights) != 3 or not all(math.isfinite(v) for v in pose_axis_weights) or min(pose_axis_weights) < 0 or sum(pose_axis_weights) <= 0:
        raise ValueError("pose_axis_weights must be three nonnegative weights with positive sum")
    model.dms_criterion.pose_axis_weights = tuple(pose_axis_weights)
    model.dms_criterion.pose_continuous_weight = float(pose_continuous_weight)
    model.dms_criterion.landmark_coord_weight = float(landmark_coord_weight)
    model.dms_criterion.landmark_peak_weight = float(landmark_peak_weight)
    model.dms_criterion.landmark_geometry_weight = float(landmark_geometry_weight)
    if teacher is not None:
        teacher.to(device_obj).freeze()
    task_weights = task_weights or {
        "parts": parts_weight,
        "landmark": landmark_weight,
        "pose": pose_weight,
    }
    tasks = [task for task in loaders if task_weights.get(task, 0) > 0]
    if not tasks:
        raise ValueError("all ROI task weights are zero")
    backbone_params = [p for p in model.model.parameters() if p.requires_grad]
    head_params = [p for p in model.dms_heads.parameters() if p.requires_grad]
    groups = ([{"params": backbone_params, "lr": backbone_lr}] if backbone_params else [])
    groups.append({"params": head_params, "lr": lr})
    optimizer = torch.optim.AdamW(groups)
    # The pose bin/residual heads are sensitive to a constant high learning
    # rate: yaw can improve for a few epochs and then jump back toward the
    # mean while the heatmap keeps changing.  A short warm-up is already
    # provided by the first epochs; cosine decay stabilizes the joint metric
    # without introducing task-specific dynamic weighting.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(epochs), 1), eta_min=float(lr) * 0.05
    )
    roi_forward = _parallel(_ROIForward(model), device_ids)
    amp_enabled = bool(amp and device_obj.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    validation_history = []
    writer = _tensorboard_writer(tb_log_dir, "roi")
    best_metric = float("inf")
    # Select a checkpoint using both primary ROI quality signals.  A yaw-only
    # selector can preserve a model whose Eye12 heatmaps are visibly unusable.
    # The scales make 20 degrees of yaw error comparable to 1.0 Eye12 NME.
    best_metric_name = "joint/yaw20_plus_eye12_nme"
    log_path = output / "roi_train.log"
    log_path.write_text(f"{datetime.now().isoformat(timespec='seconds')} roi training start\n", encoding="utf-8")
    print(
        f"roi_backbone_unfreeze={list(unfreeze_layers)} roi_backbone_params={len(backbone_params)} "
        f"roi_head_params={len(head_params)}",
        flush=True,
    )
    for epoch in range(epochs):
        totals, counts = {}, {}
        item_totals, item_counts = {}, {}
        for _ in range(steps_per_epoch):
            task = random.choices(tasks, weights=[task_weights[item] for item in tasks], k=1)[0]
            batch = _device_batch(next(loaders[task]), device_obj)
            batch["img"] = batch["img"].float() / 255
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                loss, items = _student_roi_loss(
                    model, roi_forward, batch, teacher, teacher_crop_size, kd_feature_weight, kd_heatmap_weight,
                    kd_pose_weight, kd_pose_bin_weight,
                    predicted_roi_probability if epoch / max(epochs, 1) >= predicted_roi_start else 0.0,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            totals[task] = totals.get(task, 0.0) + float(loss.detach())
            counts[task] = counts.get(task, 0) + 1
            for name, value in items.items():
                if name.startswith("kd_") or name == "roi_predicted_fraction":
                    item_totals[name] = item_totals.get(name, 0.0) + float(value.detach())
                    item_counts[name] = item_counts.get(name, 0) + 1
        scheduler.step()
        epoch_train = {task: totals[task] / counts[task] for task in totals}
        epoch_metrics = {name: item_totals[name] / item_counts[name] for name in item_totals}
        epoch_record = {**epoch_train, "metrics": epoch_metrics} if epoch_metrics else epoch_train
        history.append(epoch_record)
        if (epoch + 1) % max(int(val_interval), 1) == 0 or epoch == epochs - 1 or not validation_history:
            validation_history.append(_evaluate_roi_epoch(model, Path(manifest_dir), image_size, batch_size, device_obj))
        else:
            # Keep a row aligned to every training epoch for TensorBoard and
            # downstream parsers; values are explicitly marked stale.
            validation_history.append({"validation_skipped": float(val_interval)})
        if writer:
            for task, value in epoch_train.items():
                writer.add_scalar(f"roi/train/{task}", value, epoch + 1)
            for name, value in epoch_metrics.items():
                writer.add_scalar(f"roi/train/{name}", value, epoch + 1)
            for name, value in validation_history[-1].items():
                writer.add_scalar(f"roi/val/{name}", value, epoch + 1)
        message = (
            f"{datetime.now().isoformat(timespec='seconds')} epoch={epoch + 1}/{epochs} "
            f"train={history[-1]} val={validation_history[-1]}"
        )
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        # Keep metrics inspectable while a long run is still in progress.
        (output / "roi_train_metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        (output / "roi_validation_metrics.json").write_text(
            json.dumps(validation_history, indent=2) + "\n", encoding="utf-8"
        )
        # Frozen encoder provenance is checked against the Global at merge.
        merge_prefixes = ["dms_heads.", "dms_adapter.", "dms_criterion."]
        checkpoint = {
            "model": model.state_dict(),
            "metadata": {
                "roi_input_contract": ROI_INPUT_CONTRACT,
                "dms_architecture": model.dms_architecture,
                "merge_prefixes": merge_prefixes,
                "adapted_model_layers": list(unfreeze_layers),
                "selection_metric": best_metric_name,
                "selection_split": "val",
                "encoder_policy": "frozen_global",
                "roi_training_distribution": "dsm_train_scale_canvas_v1",
                "predicted_roi_probability": predicted_roi_probability,
                "predicted_roi_start_fraction": predicted_roi_start,
                "class_names": list(DETECTION_NAMES),
                "pose_axis_weights": list(pose_axis_weights),
                "task_sampling_weights": dict(task_weights),
                "landmark_kd_max_nme": 0.10,
                "shape_prior_enabled": shape_prior is not None,
                "loss_config": {name: getattr(model.dms_criterion, name) for name in (
                    "heatmap_sigma", "pose_axis_weights", "pose_continuous_weight", "landmark_coord_weight", "landmark_peak_weight", "landmark_geometry_weight")},
            },
            "epoch": epoch + 1,
            "history": history,
            "validation_history": validation_history,
        }
        _atomic_torch_save(checkpoint, output / "last.pt")
        metric = roi_selection_metric(validation_history[-1])
        if metric is None:
            continue
        checkpoint["metadata"]["joint_metric"] = metric
        if float(metric) < best_metric:
            best_metric = float(metric)
            checkpoint["metadata"]["best_metric"] = best_metric
            _atomic_torch_save(checkpoint, output / "best.pt")
    if not (output / "best.pt").is_file():
        raise ValueError("No validated best checkpoint was produced")
    if writer:
        writer.summary({"roi/best_selection_metric": best_metric, "roi/selection_metric_name": best_metric_name})
        writer.checkpoint("roi_best", output / "best.pt")
        writer.close()
    (output / "roi_train_metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output / "roi_validation_metrics.json").write_text(json.dumps(validation_history, indent=2) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ``metrics`` is a nested bookkeeping object, not a scalar loss.
        # Keep it in JSON, but do not pass it to matplotlib as a curve.
        tasks_seen = sorted({name for row in history for name, value in row.items() if isinstance(value, (int, float))})
        fig, axes = plt.subplots(max(1, len(tasks_seen)), 1, figsize=(9, 3 * max(1, len(tasks_seen))), squeeze=False)
        axes = [axis for row in axes for axis in row]
        for axis, task in zip(axes, tasks_seen):
            axis.plot(range(1, len(history) + 1), [row.get(task, float("nan")) for row in history])
            axis.set_title(f"DMS ROI - {task} loss")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss")
            axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output / "roi_loss_curves.png", dpi=150)
        plt.close(fig)
        val_names = sorted({name for row in validation_history for name in row})
        fig, axes = plt.subplots(max(1, len(val_names)), 1, figsize=(9, 3 * max(1, len(val_names))), squeeze=False)
        axes = [axis for row in axes for axis in row]
        for axis, name in zip(axes, val_names):
            axis.plot(range(1, len(validation_history) + 1), [row.get(name, float("nan")) for row in validation_history])
            axis.set_title(f"DMS ROI validation - {name}")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss")
            axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output / "roi_validation_curves.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        (output / "roi_plot_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
    return output / "last.pt"


@torch.inference_mode()
def evaluate_roi(
    model: DMSModel, manifest_dir: str | Path, batch_size: int, image_size: int, batches_per_task: int, device: str = "cpu"
) -> dict[str, float]:
    """Evaluate all named ROI test metrics for ROI OR unified checkpoints."""
    device_obj, _ = _cuda_devices(device)
    model.to(device_obj).eval()
    results = _evaluate_roi_epoch(model, Path(manifest_dir), image_size, batch_size, device_obj,
                                  split="test", batches_per_task=batches_per_task)
    model.eval()
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
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--predicted-roi-probability", type=float, default=0.5)
    parser.add_argument("--predicted-roi-start", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--pose-sample-weight", type=float, default=3.0, dest="pose_weight", help="Task sampling probability weight, not a loss multiplier")
    parser.add_argument("--landmark-sample-weight", type=float, default=2.0, dest="landmark_weight")
    parser.add_argument("--parts-sample-weight", type=float, default=1.0, dest="parts_weight")
    parser.add_argument("--pose-axis-weights", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("YAW", "PITCH", "ROLL"))
    parser.add_argument("--pose-continuous-weight", type=float, default=1.0)
    parser.add_argument("--landmark-coord-weight", type=float, default=1.0)
    parser.add_argument("--landmark-peak-weight", type=float, default=0.25)
    parser.add_argument("--landmark-geometry-weight", type=float, default=4.0)
    parser.add_argument("--unfreeze-layers", default="", help="comma-separated YOLO layer indices")
    parser.add_argument("--tb-log-dir")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train-teacher", action="store_true")
    parser.add_argument("--teacher-weights", type=Path)
    parser.add_argument("--resume-teacher", type=Path,
                        help="exact v2 continuation from teacher_last.pt/teacher_best.pt")
    parser.add_argument("--teacher-init", type=Path, help="warm-start encoder/pose only; new landmark branch and epoch zero")
    parser.add_argument("--target-landmark-nme", type=float, default=0.10)
    parser.add_argument("--teacher-crop-size", type=int, default=128)
    parser.add_argument("--target-pose-mae", type=float,
                        help="all three validation axes must pass this threshold, together with landmark NME; default 10 degrees")
    parser.add_argument("--pose-patience", type=int, default=0,
                        help="joint pose/landmark patience; 0 disables (failed qualification never publishes teacher.pt)")
    parser.add_argument("--pose-min-delta", type=float, default=0.1)
    parser.add_argument("--teacher-pose-sample-weight", type=float, default=1.0, dest="teacher_pose_task_weight")
    parser.add_argument("--teacher-freeze-encoder", action="store_true",
                        help="fine-tune independent task heads with shared encoder and BatchNorm statistics frozen")
    parser.add_argument("--kd-feature-weight", type=float, default=1.0)
    parser.add_argument("--kd-heatmap-weight", type=float, default=1.0)
    parser.add_argument("--kd-pose-weight", type=float, default=1.0)
    parser.add_argument("--kd-pose-bin-weight", type=float, default=1.0)
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
                lr=args.lr,
                tb_log_dir=args.tb_log_dir,
                target_pose_mae=args.target_pose_mae,
                pose_patience=args.pose_patience,
                pose_min_delta=args.pose_min_delta,
                pose_task_weight=args.teacher_pose_task_weight,
                resume_checkpoint=args.resume_teacher,
                pose_axis_weights=tuple(args.pose_axis_weights),
                target_landmark_nme=args.target_landmark_nme,
                init_checkpoint=args.teacher_init,
                freeze_teacher_encoder=args.teacher_freeze_encoder,
            )
        )
        return
    model = load_dms_weights(DMSModel(args.model_config, nc=len(DETECTION_NAMES), verbose=True), args.weights)
    if args.eval:
        metrics = evaluate_roi(model, args.manifest_dir, args.batch, args.imgsz, args.steps_per_epoch, args.device)
        args.output.mkdir(parents=True, exist_ok=True)
        report = {"checkpoint": str(args.weights), "split": "test", "roi_protocol": "dsm_train_scale_reference_gt", "metrics": metrics}
        (args.output / "roi_eval_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
                kd_pose_weight=args.kd_pose_weight,
                kd_pose_bin_weight=args.kd_pose_bin_weight,
                workers=args.workers,
                lr=args.lr,
                tb_log_dir=args.tb_log_dir,
                unfreeze_layers=tuple(int(item) for item in args.unfreeze_layers.split(",") if item.strip()),
                backbone_lr=args.backbone_lr,
                pose_weight=args.pose_weight,
                landmark_weight=args.landmark_weight,
                parts_weight=args.parts_weight,
                pose_axis_weights=tuple(args.pose_axis_weights),
                pose_continuous_weight=args.pose_continuous_weight,
                landmark_coord_weight=args.landmark_coord_weight,
                landmark_peak_weight=args.landmark_peak_weight,
                landmark_geometry_weight=args.landmark_geometry_weight,
                val_interval=args.val_interval,
                pin_memory=not args.no_pin_memory,
                predicted_roi_probability=args.predicted_roi_probability,
                predicted_roi_start=args.predicted_roi_start,
            )
        )


if __name__ == "__main__":
    main()

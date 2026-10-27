"""Shared Teacher landmark validation, qualification and visual diagnostics."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import DMSROIDataset, collate_roi
from .schema import ROI_INPUT_CONTRACT, LANDMARK_NAMES
from .teacher import CropTeacher, make_teacher_crops, landmark_teacher_quality
from .loss import DMSROILoss


def teacher_selection(metrics: dict, pose_target: float, landmark_target: float) -> tuple[float, bool]:
    """Minimize the worst normalized task error; qualify only when all pass."""
    limits = (pose_target, pose_target, pose_target, landmark_target)
    names = ("pose_val_yaw_mae_deg", "pose_val_pitch_mae_deg", "pose_val_roll_mae_deg", "landmark_val_nme")
    values = [float(metrics[name]) for name in names]
    if not all(math.isfinite(v) and v >= 0 for v in values) or not all(math.isfinite(v) and v > 0 for v in limits):
        raise ValueError("Teacher selection requires finite errors and positive thresholds")
    ratios = [value / limit for value, limit in zip(values, limits)]
    return max(ratios), all(value <= 1 for value in ratios)


def validate_teacher_checkpoint(payload: dict) -> None:
    """Do not skip training or publish a Teacher solely because a file exists."""
    metadata = payload.get("metadata", {})
    required = {"teacher_architecture": "crop_teacher_context_v2", "roi_input_contract": ROI_INPUT_CONTRACT,
                "selection_split": "val", "landmark_objective": "spatial_kl_v1", "qualified": True}
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("Teacher is not jointly qualified under the current architecture/ROI protocol")
    if tuple(metadata.get("landmark_names", ())) != LANDMARK_NAMES or metadata.get("pose_order") != "yaw_pitch_roll_deg":
        raise ValueError("Teacher landmark/pose output semantics do not match the current contract")
    CropTeacher.from_state(payload["model"], metadata)
    _, qualified = teacher_selection(metadata["validation"], metadata["pose_target_deg"], metadata["landmark_target_nme"])
    if not qualified or metadata["validation"].get("landmark_val_samples", 0) <= 0:
        raise ValueError("Teacher validation does not satisfy pose and landmark acceptance thresholds")


def _visualize(crops, output, batch, directory: Path):
    from PIL import Image, ImageDraw
    directory.mkdir(parents=True, exist_ok=True)
    size = crops.shape[-1]
    prediction = DMSROILoss.decode_heatmap(output["landmark_heatmap"]).cpu()
    target, valid = batch["landmark_xy"].cpu(), batch["landmark_valid"].cpu()
    grid = Image.new("RGB", (size * 4, size * 2))
    for index in range(min(8, len(crops))):
        pixels = (crops[index].detach().float().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
        canvas = Image.fromarray(pixels)
        draw = ImageDraw.Draw(canvas)
        for point in range(12):
            if not valid[index, point]:
                continue
            for xy, color in ((target[index, point], "lime"), (prediction[index, point], "red")):
                x, y = (xy * size).tolist()
                draw.ellipse((x-2, y-2, x+2, y+2), outline=color)
        grid.paste(canvas, ((index % 4) * size, (index // 4) * size))
    grid.save(directory / "landmark_predictions.jpg")
    maps = Image.new("RGB", (size * 4, size * 3))
    for point, heat in enumerate(output["landmark_heatmap"][0].detach().float().cpu()):
        pixels = ((heat - heat.min()) / (heat.max() - heat.min()).clamp_min(1e-6) * 255).numpy().astype("uint8")
        tile = Image.fromarray(pixels).convert("RGB").resize((size, size))
        draw = ImageDraw.Draw(tile)
        draw.text((3, 3), str(point), fill="yellow")
        if valid[0, point]:
            x, y = (target[0, point] * size).tolist()
            draw.ellipse((x-2, y-2, x+2, y+2), outline="lime")
        maps.paste(tile, ((point % 4) * size, (point // 4) * size))
    maps.save(directory / "landmark_heatmaps.jpg")


@torch.inference_mode()
def evaluate_teacher_landmarks(teacher, manifest_dir, image_size, crop_size, batch_size, device,
                               split="val", match_scale=True, visual_dir: Path | None = None):
    """Full finite split, deployment argmax, interocular NME/PCK and per-point error."""
    dataset = DMSROIDataset(Path(manifest_dir) / "eye12.jsonl", "landmark", image_size, split, match_scale=match_scale)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_roi)
    was_training = teacher.training
    teacher.eval()
    nmes, count, accepted, eligible_count = [], 0, 0, 0
    errors = torch.zeros(12, dtype=torch.float64)
    denominators = torch.zeros(12, dtype=torch.float64)
    pck05 = pck10 = 0
    try:
        for batch in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            crops = make_teacher_crops(batch["img"].float() / 255, batch["face_rois"], crop_size)
            output = teacher(crops)
            passed, nme, measurable = landmark_teacher_quality(output, batch)
            anchors = batch["landmark_valid"][:, (0, 3, 6, 9)].all(-1)
            target = batch["landmark_xy"]
            anchors &= (target[:, (6, 9)].mean(1)-target[:, (0, 3)].mean(1)).norm(dim=-1) > 1e-6
            if (anchors & ~measurable).any():
                raise ValueError("Nonfinite Teacher output on an evaluable landmark sample")
            eligible_count += int(anchors.sum())
            nmes.extend(nme[measurable].cpu().tolist())
            accepted += int(passed.sum())
            count += len(nme)
            prediction = DMSROILoss.decode_heatmap(output["landmark_heatmap"])
            target, valid = batch["landmark_xy"], batch["landmark_valid"] & measurable[:, None]
            scale = (batch["face_rois"][:, 3:] - batch["face_rois"][:, 1:3]).unsqueeze(1)
            interocular = ((target[:, (6, 9)] * scale).mean(1) - (target[:, (0, 3)] * scale).mean(1)).norm(dim=-1)
            point_error = ((prediction-target)*scale).norm(dim=-1) / interocular[:, None].clamp_min(1e-6)
            errors += torch.where(valid, point_error, 0.).sum(0).double().cpu()
            denominators += valid.sum(0).double().cpu()
            pck05 += int(((point_error <= .05) & valid).sum())
            pck10 += int(((point_error <= .10) & valid).sum())
            if visual_dir is not None:
                _visualize(crops, output, batch, Path(visual_dir))
                visual_dir = None
        if not nmes or not all(math.isfinite(value) for value in nmes):
            raise ValueError("No finite measurable landmark samples in Teacher validation")
        # Nonfinite output must fail selection, not disappear from its denominator.
        if len(nmes) != eligible_count:
            raise ValueError("Teacher validation metric denominator mismatch")
        return {"landmark_val_nme": sum(nmes)/len(nmes), "landmark_val_samples": count,
                "landmark_val_measured_samples": len(nmes), "landmark_val_coverage": len(nmes)/count,
                "landmark_val_kd_accepted_fraction": accepted/count,
                "landmark_val_pck_interocular_005": pck05/max(float(denominators.sum()), 1),
                "landmark_val_pck_interocular_010": pck10/max(float(denominators.sum()), 1),
                **{f"landmark_val_nme_point_{i:02d}": float(errors[i]/denominators[i]) if denominators[i] else None for i in range(12)}}
    finally:
        teacher.train(was_training)

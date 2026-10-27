"""Coordinate contract shared by datasets, runtime and host ROIAlign tests."""
from __future__ import annotations

import torch


def square_rois(rois: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Square [batch,x1,y1,x2,y2] boxes in letterboxed image coordinates."""
    result = rois.clone()
    if not len(result):
        return result
    box = rois[:, 1:]
    size = (box[:, 2:] - box[:, :2]).amax(-1).clamp(1, min(height, width))
    corner = (box[:, :2] + box[:, 2:] - size[:, None]) / 2
    limit = box.new_tensor([width, height])[None] - size[:, None]
    corner = torch.minimum(corner.clamp_min(0), limit.clamp_min(0))
    result[:, 1:3], result[:, 3:5] = corner, corner + size[:, None]
    return result


def encode_points(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    return (points - boxes[..., None, :2]) / (boxes[..., None, 2:] - boxes[..., None, :2]).clamp_min(1e-6)


def decode_points(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    return boxes[..., None, :2] + points * (boxes[..., None, 2:] - boxes[..., None, :2])


def mask_cropped_parts(batch: dict) -> dict:
    """Ignore truncated annotated parts; absence cannot be inferred from partial labels."""
    result = dict(batch)
    boxes = batch["part_box"]
    complete = torch.isfinite(boxes).all(-1) & ((boxes >= 0) & (boxes <= 1)).all(-1)
    complete &= (boxes[..., 2:] > boxes[..., :2]).all(-1)
    result["part_box_valid"] = batch["part_box_valid"] & complete
    result["part_box"] = torch.where(complete[..., None], boxes, 0.)
    result["part_present_valid"] = batch["part_present_valid"] & ((batch["part_present"] == 0) | complete)
    result["part_present"] = torch.where(complete, batch["part_present"], 0.)
    for name, slots in (("eye_state", slice(0, 2)), ("mouth_state", slice(2, 3))):
        if name + "_valid" in batch:
            result[name + "_valid"] = batch[name + "_valid"] & complete[..., slots]
    return result


def rebase_targets(batch: dict, rois: torch.Tensor) -> dict:
    """Move supervision to a new ROI without moving pixels or GT pose angles."""
    result = dict(batch)
    old, new = batch["face_rois"][:, 1:], rois[:, 1:]
    result["face_rois"] = rois
    if "landmark_xy" in batch:
        points = encode_points(decode_points(batch["landmark_xy"], old), new)
        result["landmark_xy"] = points
        result["landmark_valid"] = batch["landmark_valid"] & ((points >= 0) & (points <= 1)).all(-1)
    if "part_box" in batch:
        corners = batch["part_box"].reshape(len(old), -1, 2)
        corners = encode_points(decode_points(corners, old), new)
        result["part_box"] = corners.reshape_as(batch["part_box"])
        result = mask_cropped_parts(result)
    return result


@torch.no_grad()
def matched_prediction_rois(raw, rois, height, width, probability, min_iou=0.5):
    """Use only detector boxes matched to each GT face; unmatched stays GT."""
    from ultralytics.utils.metrics import box_iou
    from .postprocess import decode_detections
    result, matched = rois.clone(), 0
    for index, predictions in enumerate(decode_detections(raw)):
        candidates = predictions[predictions[:, 5] == 0, :4]
        if not len(candidates) or torch.rand((), device=rois.device) >= probability:
            continue
        # Both sides use deployment geometry, including boundary clamping.
        candidate_rois = torch.cat((candidates.new_zeros(len(candidates), 1), candidates), -1)
        candidates = square_rois(candidate_rois, height, width)[:, 1:]
        ious = box_iou(rois[index:index + 1, 1:], candidates)[0]
        best = ious.argmax()
        if ious[best] >= min_iou:
            result[index, 1:] = candidates[best]
            matched += 1
    return square_rois(result, height, width), matched

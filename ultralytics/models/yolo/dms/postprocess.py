"""Shared host detection postprocessing for runtime and matched training ROIs."""
from ultralytics.utils.nms import TorchNMS, non_max_suppression
from .schema import DETECTION_NAMES

HOST_NMS_IOU = 0.4


def suppress(detections, iou_threshold=HOST_NMS_IOU, max_det=100):
    if not len(detections):
        return detections.reshape(0, 6)
    boxes = detections[:, :4] + detections[:, 5:6] * 7680.0
    return detections[TorchNMS.nms(boxes, detections[:, 4], iou_threshold)[:max_det]]


def decode_detections(raw, conf=0.25):
    return [suppress(rows) for rows in non_max_suppression(raw, conf_thres=conf, nc=len(DETECTION_NAMES), max_det=300)]

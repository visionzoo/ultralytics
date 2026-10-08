"""Thin trainer registration; native detection training is inherited unchanged."""

from __future__ import annotations

import torch

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import RANK

from .model import DMSModel


class DMSTrainer(DetectionTrainer):
    """Use DMSModel while retaining Ultralytics' normal detection trainer."""

    def get_model(self, cfg: str | dict | None = None, weights: torch.nn.Module | None = None, verbose: bool = True) -> DMSModel:
        model = self.set_model_names_for_load(
            DMSModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model

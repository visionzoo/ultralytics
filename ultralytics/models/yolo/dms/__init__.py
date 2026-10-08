from .model import DMSModel
from .loss import DMSROILoss
from .schema import LANDMARK_NAMES, PART_SLOTS
from .teacher import CropTeacher, distillation_loss, make_teacher_crops
from .export import export_dms_onnx

__all__ = (
    "DMSModel",
    "DMSROILoss",
    "CropTeacher",
    "distillation_loss",
    "make_teacher_crops",
    "export_dms_onnx",
    "LANDMARK_NAMES",
    "PART_SLOTS",
)


def __getattr__(name: str):
    """Keep ``python -m ...dms.train`` from importing the module twice."""
    if name == "DMSTrainer":
        from .train import DMSTrainer

        return DMSTrainer
    raise AttributeError(name)

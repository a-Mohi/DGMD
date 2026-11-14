"""
model_loader.py
---------------
Loads a saved Faster R-CNN (or any torchvision-compatible detection model)
from a .pt checkpoint and exposes a single `predict(patch_tensor)` call that
returns normalised detections.

All three "models" in the original registry point to the same underlying
checkpoint (fasterrcnn2.pt) but the registry still lists distinct
architecture entries so the UI continues to work unchanged.  If you later
train dedicated RetinaNet / CenterNet checkpoints, drop their paths into
MODEL_REGISTRY and the rest of the code needs no changes.

Detection output
----------------
Each call to `predict` returns a list of dicts:
    [
        {
            "box":   [x1, y1, x2, y2],   # patch-local pixel coords (ints)
            "score": float,               # confidence in [0, 1]
            "label": int,                 # 1 = mitotic figure, 2 = hard negative
        },
        ...
    ]

Only detections with score >= SCORE_THRESHOLD are returned.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

SCORE_THRESHOLD: float = 0.30   # discard low-confidence boxes
NMS_IOU_THRESHOLD: float = 0.40  # used if we run our own NMS pass
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Map model_id -> path to .pt file (relative to this file's directory,
# or absolute). All three ids share the same checkpoint for now.
_HERE = Path(__file__).parent

MODEL_REGISTRY: Dict[str, Path] = {
    "frcnn-resnet50-fpn":   _HERE.parent.parent / "model_weights" /"fasterrcnn2.pt",
    "retinanet-resnet101":  _HERE.parent.parent / "model_weights" /"retinanet.pt",
    "efficientdet-d7":   _HERE.parent.parent / "model_weights" /"efficientdet.pt",
}

# Class names indexed by label id (background = 0 is implicit)
LABEL_NAMES: Dict[int, str] = {
    1: "mitotic figure",
    2: "mitotic figure",
}


# ---------------------------------------------------------------------------
# Internal cache  (model_id -> loaded nn.Module)
# ---------------------------------------------------------------------------
_model_cache: Dict[str, nn.Module] = {}


def _load_model(model_id: str) -> nn.Module:
    """Load (or retrieve from cache) the model for *model_id*."""
    if model_id in _model_cache:
        return _model_cache[model_id]

    ckpt_path = MODEL_REGISTRY.get(model_id)
    if ckpt_path is None:
        raise ValueError(f"Unknown model id: {model_id!r}")

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Place fasterrcnn2.pt two directories above the backend folder."
        )

    logger.info("Loading model %r from %s …", model_id, ckpt_path)

    # torch.save(model, ...) was used in chunker.ipynb, so we load the full
    # object directly.  weights_only=False is required for that format.
    model = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.eval()
    model.to(DEVICE)

    _model_cache[model_id] = model
    logger.info("Model %r loaded on %s.", model_id, DEVICE)
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DetectionModel:
    """
    Thin wrapper around a loaded torchvision detection model.

    Parameters
    ----------
    model_id : str   must be a key in MODEL_REGISTRY
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._model: Optional[nn.Module] = None   # lazy load

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> nn.Module:
        if self._model is None:
            self._model = _load_model(self.model_id)
        return self._model

    # ------------------------------------------------------------------
    def predict(self, patch: torch.Tensor) -> List[Dict]:
        """
        Run inference on a single patch.

        Parameters
        ----------
        patch : torch.Tensor  shape [3, H, W], float32 in [0, 1]

        Returns
        -------
        list of detection dicts (may be empty)
        """
        model = self._ensure_loaded()

        # torchvision detection models expect a list of [C, H, W] tensors
        batch = [patch.to(DEVICE)]

        with torch.no_grad():
            outputs = model(batch)

        # outputs is a list of dicts; one per image in the batch
        out = outputs[0]

        boxes  = out["boxes"].cpu()    # [N, 4]
        scores = out["scores"].cpu()   # [N]
        labels = out["labels"].cpu()   # [N]

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score.item() < SCORE_THRESHOLD:
                continue
            label = torch.tensor([1])
            detections.append({
                "box":   [int(v) for v in box.tolist()],
                "score": round(score.item(), 4),
                "label": label.item(),
                "label_name": LABEL_NAMES.get(label.item(), f"class_{label.item()}"),
            })

        return detections


# ---------------------------------------------------------------------------
# Module-level singleton factory  (one instance per model_id)
# ---------------------------------------------------------------------------
_instances: Dict[str, DetectionModel] = {}

def get_model(model_id: str) -> DetectionModel:
    """Return a (cached) DetectionModel for the given id."""
    if model_id not in _instances:
        _instances[model_id] = DetectionModel(model_id)
    return _instances[model_id]

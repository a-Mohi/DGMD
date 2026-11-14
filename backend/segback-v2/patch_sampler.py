"""
patch_sampler.py
----------------
Slides a 256×256 window over a PIL image with a stride of 224 pixels
(giving 32-pixel overlap on every edge), yielding one patch at a time
together with its grid position so the caller can track progress and
re-assemble detections in image-space coordinates.

Usage
-----
    sampler = SlidingWindowSampler(pil_image, patch_size=256, stride=224)
    for patch_tensor, meta in sampler:
        # patch_tensor : Float32 [3, 256, 256]  values in [0, 1]
        # meta         : PatchMeta(col, row, x1, y1, x2, y2, grid_cols, grid_rows)
        ...

The grid dimensions are also exposed as:
    sampler.grid_cols
    sampler.grid_rows
    len(sampler)          # total patch count
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generator, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image


@dataclass(frozen=True)
class PatchMeta:
    """Carries spatial metadata for a single patch."""
    col: int          # grid column index  (0-based, left → right)
    row: int          # grid row index     (0-based, top  → bottom)
    x1: int           # left   pixel in original image
    y1: int           # top    pixel in original image
    x2: int           # right  pixel in original image  (exclusive)
    y2: int           # bottom pixel in original image  (exclusive)
    grid_cols: int    # total columns in the grid
    grid_rows: int    # total rows    in the grid

    @property
    def patch_index(self) -> int:
        return self.row * self.grid_cols + self.col

    @property
    def total_patches(self) -> int:
        return self.grid_cols * self.grid_rows


class SlidingWindowSampler:
    """
    Deterministic, convolutional-style patch sampler.

    Parameters
    ----------
    image       : PIL.Image  (any mode; converted to RGB internally)
    patch_size  : int        side length of each square patch   (default 256)
    stride      : int        step between patch origins          (default 224)
    """

    def __init__(
        self,
        image: Image.Image,
        patch_size: int = 256,
        stride: int = 224,
    ) -> None:
        self.image      = image.convert("RGB")
        self.patch_size = patch_size
        self.stride     = stride

        self.img_w, self.img_h = self.image.size          # PIL: (width, height)

        # Number of steps needed so the window covers the full axis.
        # We always include a patch that starts at position 0, and we keep
        # adding steps until the *end* of the window reaches the image edge.
        self.grid_cols = max(1, math.ceil((self.img_w - patch_size) / stride) + 1)
        self.grid_rows = max(1, math.ceil((self.img_h - patch_size) / stride) + 1)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.grid_cols * self.grid_rows

    def _origin(self, col: int, row: int) -> Tuple[int, int]:
        """Return (x1, y1) for a grid cell, clamped so the patch fits."""
        x1 = min(col * self.stride, self.img_w - self.patch_size)
        y1 = min(row * self.stride, self.img_h - self.patch_size)
        return max(0, x1), max(0, y1)

    def get_patch(self, col: int, row: int) -> Tuple[torch.Tensor, PatchMeta]:
        """Return a single (tensor, meta) pair for a grid position."""
        x1, y1 = self._origin(col, row)
        x2 = x1 + self.patch_size
        y2 = y1 + self.patch_size

        crop = self.image.crop((x1, y1, x2, y2))
        tensor = TF.to_tensor(crop)           # [3, H, W], float32 in [0, 1]

        meta = PatchMeta(
            col=col, row=row,
            x1=x1,  y1=y1,
            x2=x2,  y2=y2,
            grid_cols=self.grid_cols,
            grid_rows=self.grid_rows,
        )
        return tensor, meta

    def __iter__(self) -> Generator[Tuple[torch.Tensor, PatchMeta], None, None]:
        """Iterate left-to-right, top-to-bottom (matches the frontend scan order)."""
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                yield self.get_patch(col, row)

    # ------------------------------------------------------------------
    # Utility: map model boxes back to full-image coordinates
    # ------------------------------------------------------------------

    @staticmethod
    def boxes_to_image_space(
        boxes: torch.Tensor,   # [N, 4]  x1y1x2y2 in patch-local coords
        meta: PatchMeta,
    ) -> torch.Tensor:
        """Shift patch-local bounding boxes to full-image coordinates."""
        if boxes.numel() == 0:
            return boxes
        offset = torch.tensor([meta.x1, meta.y1, meta.x1, meta.y1],
                               dtype=boxes.dtype)
        return boxes + offset

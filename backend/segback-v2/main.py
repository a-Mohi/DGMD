import asyncio
import io
import os
import time
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageDraw

from patch_sampler import SlidingWindowSampler, PatchMeta
from model_loader import get_model, LABEL_NAMES

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mitosis Detection API — Real Inference")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
PATCH_DIR  = Path("patches")
UPLOAD_DIR.mkdir(exist_ok=True)
PATCH_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Models registry  (same structure as original — UI depends on this shape)
# ---------------------------------------------------------------------------

MODELS = {
    "Faster R-CNN ResNet-50 FPN": {
        "id":         "frcnn-resnet50-fpn",
        "params":     "41.8M",
        "backbone":   "ResNet-50",
        "fpn":        True,
        "input_size": "256×256",
        "dataset":    "MIDOG++ / MiDeSEC / CCMCT / CMC",
    },
    "RetinaNet ResNet-101 FPN": {
        "id":         "retinanet-resnet101",
        "params":     "53.2M",
        "backbone":   "ResNet-101",
        "fpn":        True,
        "input_size": "256×256",
        "dataset":    "MIDOG++ / MiDeSEC / CCMCT / CMC",
    },
    "EfficientDet-D7": {
        "id":         "efficientdet-d7",
        "params":     "52.4M",
        "backbone":   "MobileNetV3-Large",
        "fpn":        True,
        "input_size": "256×256",
        "dataset":    "MIDOG++ / MiDeSEC / CCMCT / CMC",
    },
}

# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------

state: Dict[str, Any] = {
    "image_path":          None,
    "selected_model_id":   None,
    "selected_model_name": None,
    # Grid dimensions — set by SlidingWindowSampler after upload
    "grid_x":              0,    # columns
    "grid_y":              0,    # rows
    # Scan progress
    "current_x":           0,
    "current_y":           0,
    "run_complete":        False,
    "running":             False,
    # Results
    "detected_patches":    [],   # list of {"path": str, "meta": dict, "detections": list}
    "run_stats":           {},
    "start_time":          None,
}

# ---------------------------------------------------------------------------
# Background inference task
# ---------------------------------------------------------------------------

async def run_scan() -> None:
    """
    Iterate every patch produced by SlidingWindowSampler, run real model
    inference on each, and record patches that contain at least one detection
    above the score threshold.
    """
    image_path = state["image_path"]
    model_id   = state["selected_model_id"]

    state["run_complete"]     = False
    state["running"]          = True
    state["detected_patches"] = []
    state["run_stats"]        = {}
    state["start_time"]       = time.time()
    state["current_x"]        = 0
    state["current_y"]        = 0

    # Clear old patch thumbnails
    for f in PATCH_DIR.glob("*.png"):
        f.unlink(missing_ok=True)

    # ---- Load image -------------------------------------------------------
    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logger.error("Failed to open image %s: %s", image_path, exc)
        state["running"] = False
        return

    # ---- Build sampler ----------------------------------------------------
    sampler = SlidingWindowSampler(pil_image, patch_size=256, stride=224)
    state["grid_x"] = sampler.grid_cols
    state["grid_y"] = sampler.grid_rows
    total_patches   = len(sampler)

    logger.info(
        "Starting scan: %d×%d grid (%d patches) with model %r",
        sampler.grid_cols, sampler.grid_rows, total_patches, model_id,
    )

    # ---- Load model (lazy, cached) ----------------------------------------
    detector = get_model(model_id)

    # Pre-warm: pay the JIT / CUDA init cost before the timed loop
    try:
        first_tensor, _ = sampler.get_patch(0, 0)
        await asyncio.get_event_loop().run_in_executor(
            None, detector.predict, first_tensor
        )
    except Exception as exc:
        logger.warning("Model warm-up failed (will retry during scan): %s", exc)

    # ---- Scan loop --------------------------------------------------------
    for patch_tensor, meta in sampler:
        state["current_x"] = meta.col
        state["current_y"] = meta.row

        # Yield to the event loop so /get-status polling stays responsive
        await asyncio.sleep(0)

        try:
            detections = await asyncio.get_event_loop().run_in_executor(
                None, detector.predict, patch_tensor
            )
        except Exception as exc:
            logger.warning(
                "Inference failed at patch (%d,%d): %s", meta.col, meta.row, exc
            )
            detections = []

        if detections:
            thumb_path = _save_patch_thumbnail(pil_image, meta, detections)
            state["detected_patches"].append({
                "path":       str(thumb_path),
                "meta": {
                    "col": meta.col, "row": meta.row,
                    "x1":  meta.x1,  "y1":  meta.y1,
                    "x2":  meta.x2,  "y2":  meta.y2,
                },
                "detections": detections,
            })

    # ---- Finalise ---------------------------------------------------------
    elapsed    = round(time.time() - state["start_time"], 2)
    detected   = len(state["detected_patches"])
    total_dets = sum(len(p["detections"]) for p in state["detected_patches"])

    state["run_complete"] = True
    state["running"]      = False
    state["run_stats"]    = {
        "total_patches":    total_patches,
        "detected_patches": detected,
        "total_detections": total_dets,
        "detection_rate":   f"{round(detected / total_patches * 100, 1)}%",
        "elapsed_seconds":  elapsed,
        "grid":             f"{sampler.grid_cols}×{sampler.grid_rows}",
        "patch_size":       "256×256",
        "stride":           224,
        "model":            state["selected_model_name"] or "—",
        "status":           "complete",
    }

    logger.info(
        "Scan complete — %d/%d patches with detections, %d mitoses, %.1fs elapsed",
        detected, total_patches, total_dets, elapsed,
    )


# ---------------------------------------------------------------------------
# Helper: save patch thumbnail with bounding-box overlay
# ---------------------------------------------------------------------------

_LABEL_COLOURS: Dict[int, tuple] = {
    1: (255, 80,  80),   # mitotic figure — red
    2: (80,  180, 255),  # hard negative  — blue
}
_DEFAULT_COLOUR = (255, 200, 50)


def _save_patch_thumbnail(
    full_image: Image.Image,
    meta:       PatchMeta,
    detections: List[Dict],
) -> Path:
    """
    Crop the patch from the original image, draw detection bounding boxes
    with label + confidence text, and save as a PNG thumbnail.
    """
    crop  = full_image.crop((meta.x1, meta.y1, meta.x2, meta.y2)).copy()
    draw  = ImageDraw.Draw(crop)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label           = det["label"]
        score           = det["score"]
        colour          = _LABEL_COLOURS.get(label, _DEFAULT_COLOUR)
        label_name      = LABEL_NAMES.get(label, f"cls{label}")

        draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
        draw.text(
            (x1 + 2, y1 + 2),
            f"{label_name} {score:.2f}",
            fill=colour,
        )

    path = PATCH_DIR / f"patch_{meta.col}_{meta.row}_{uuid.uuid4().hex[:6]}.png"
    crop.save(path)
    return path


# ---------------------------------------------------------------------------
# Endpoints  — identical contracts to the original mock backend
# ---------------------------------------------------------------------------

@app.get("/get-progress-dim")
def get_progress_dim():
    """Returns grid dimensions (columns × rows)."""
    return {"x": state["grid_x"], "y": state["grid_y"]}


@app.get("/get-status")
def get_status():
    """Returns current scan position and completion flag."""
    return {
        "x":            state["current_x"],
        "y":            state["current_y"],
        "run_complete": state["run_complete"],
    }


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Accept an image upload, store it, and compute grid dimensions."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Remove previous upload
    if state["image_path"] and Path(state["image_path"]).exists():
        Path(state["image_path"]).unlink(missing_ok=True)

    ext  = Path(file.filename).suffix or ".jpg"
    dest = UPLOAD_DIR / f"image_{uuid.uuid4().hex}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    state["image_path"] = str(dest)

    # Derive grid dimensions from the sampler so the UI is always in sync
    try:
        pil_image = Image.open(dest).convert("RGB")
        sampler   = SlidingWindowSampler(pil_image, patch_size=256, stride=224)
        state["grid_x"] = sampler.grid_cols
        state["grid_y"] = sampler.grid_rows
    except Exception:
        state["grid_x"] = 4
        state["grid_y"] = 7

    # Reset run state
    state["run_complete"]     = False
    state["current_x"]        = 0
    state["current_y"]        = 0
    state["run_stats"]        = {}
    state["detected_patches"] = []

    return {
        "message":  "Image uploaded successfully.",
        "filename": dest.name,
        "grid_x":   state["grid_x"],
        "grid_y":   state["grid_y"],
    }


@app.get("/get-models")
def get_models():
    """Returns available detection models."""
    return MODELS


@app.get("/set-model")
def set_model(model: str):
    """Sets the active model by id."""
    matched = next(
        ((name, m) for name, m in MODELS.items() if m["id"] == model),
        None,
    )
    if not matched:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found.")
    name, meta = matched
    state["selected_model_id"]   = model
    state["selected_model_name"] = name
    return {"message": f"Model set to '{name}'.", "model": meta}


@app.get("/start")
async def start(background_tasks: BackgroundTasks):
    """Starts the detection run. Returns 400 if image or model not set."""
    if not state["image_path"]:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    if not state["selected_model_id"]:
        raise HTTPException(status_code=400, detail="No model selected.")
    if state["running"]:
        raise HTTPException(status_code=409, detail="Detection already running.")

    background_tasks.add_task(run_scan)
    return {"message": "Detection started."}


@app.get("/get-patch-count")
def get_patch_count():
    """Returns the number of patches that contain at least one detection."""
    return {"count": len(state["detected_patches"])}


@app.get("/get-patch")
def get_patch(id: int):
    """Returns patch thumbnail image by index."""
    patches = state["detected_patches"]
    if id < 0 or id >= len(patches):
        raise HTTPException(status_code=404, detail="Patch index out of range.")
    path = Path(patches[id]["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Patch file not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/get-patch-detections/{patch_id}")
def get_patch_detections(patch_id: int):
    """
    Returns raw detection data (boxes, scores, labels) for a detected patch.
    Extension endpoint — not required by the current frontend, but useful for
    building a richer overlay view later.
    """
    patches = state["detected_patches"]
    if patch_id < 0 or patch_id >= len(patches):
        raise HTTPException(status_code=404, detail="Patch index out of range.")
    entry = patches[patch_id]
    return {
        "patch_id":   patch_id,
        "meta":       entry["meta"],
        "detections": entry["detections"],
    }


@app.get("/get-run-stats")
def get_run_stats():
    """Returns statistics from the last completed run."""
    if not state["run_stats"]:
        return {"status": "no run completed yet"}
    return state["run_stats"]


@app.get("/health")
def health():
    return {"status": "ok"}

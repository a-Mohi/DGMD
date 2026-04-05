import asyncio
import os
import time
import uuid
import math
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="Segmentation Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
PATCH_DIR = Path("patches")
UPLOAD_DIR.mkdir(exist_ok=True)
PATCH_DIR.mkdir(exist_ok=True)

# ── Models registry ──────────────────────────────────────────────────────────

MODELS = {
    "Faster R-CNN ResNet-50 FPN": {
        "id": "frcnn-resnet50-fpn",
        "params": "41.8M",
        "backbone": "ResNet-50",
        "fpn": True,
        "input_size": "800×1333",
        "dataset": "COCO",
        "map_50": "60.0",
    },
    "Faster R-CNN ResNet-101 FPN": {
        "id": "frcnn-resnet101-fpn",
        "params": "60.9M",
        "backbone": "ResNet-101",
        "fpn": True,
        "input_size": "800×1333",
        "dataset": "COCO",
        "map_50": "62.5",
    },
    "Faster R-CNN MobileNet V3": {
        "id": "frcnn-mobilenet-v3",
        "params": "19.4M",
        "backbone": "MobileNetV3-Large",
        "fpn": True,
        "input_size": "800×1333",
        "dataset": "COCO",
        "map_50": "55.0",
    },
}

# ── Shared state ─────────────────────────────────────────────────────────────

state = {
    "image_path": None,           # Path to uploaded image
    "selected_model_id": None,    # Currently selected model id
    "selected_model_name": None,
    "grid_x": 4,                  # Columns
    "grid_y": 7,                  # Rows
    "current_x": 0,               # Current patch col
    "current_y": 0,               # Current patch row
    "run_complete": False,
    "running": False,
    "detected_patches": [],       # Paths of saved patch images
    "run_stats": {},
    "start_time": None,
}

# ── Background scan task ─────────────────────────────────────────────────────

async def run_scan():
    """Simulates scanning the image grid patch by patch."""
    cols = state["grid_x"]
    rows = state["grid_y"]
    total = cols * rows
    state["detected_patches"] = []
    state["run_complete"] = False
    state["running"] = True
    state["start_time"] = time.time()

    # Clear old patches
    for f in PATCH_DIR.glob("*.png"):
        f.unlink()

    for row in range(rows):
        for col in range(cols):
            state["current_x"] = col
            state["current_y"] = row

            await asyncio.sleep(0.3)  # Simulate processing time per patch

            # Simulate detection: ~40% chance a patch has a detection
            import random
            if random.random() < 0.4:
                patch_path = _save_mock_patch(col, row)
                state["detected_patches"].append(patch_path)

    # Mark complete
    state["run_complete"] = True
    state["running"] = False
    elapsed = round(time.time() - state["start_time"], 2)

    detected = len(state["detected_patches"])
    state["run_stats"] = {
        "total_patches": total,
        "detected_patches": detected,
        "detection_rate": f"{round(detected / total * 100, 1)}%",
        "elapsed_seconds": elapsed,
        "grid": f"{cols}×{rows}",
        "model": state["selected_model_name"] or "—",
        "status": "complete",
    }


def _save_mock_patch(col: int, row: int) -> Path:
    """Creates a tiny solid-color PNG as a mock detected patch."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import random

        colors = [
            (220, 80, 80),
            (80, 180, 80),
            (80, 130, 220),
            (200, 160, 50),
            (160, 80, 200),
        ]
        color = random.choice(colors)
        img = Image.new("RGB", (64, 64), color=color)
        draw = ImageDraw.Draw(img)
        draw.text((4, 4), f"{col},{row}", fill=(255, 255, 255))
        path = PATCH_DIR / f"patch_{col}_{row}_{uuid.uuid4().hex[:6]}.png"
        img.save(path)
        return path
    except ImportError:
        # Pillow not available — write a raw 1×1 red pixel PNG
        import struct, zlib

        def _chunk(name, data):
            c = zlib.crc32(name + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + name + data + struct.pack(">I", c)

        png = (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
            + _chunk(b"IEND", b"")
        )
        path = PATCH_DIR / f"patch_{col}_{row}_{uuid.uuid4().hex[:6]}.png"
        path.write_bytes(png)
        return path


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/get-progress-dim")
def get_progress_dim():
    """Returns the grid dimensions (columns × rows)."""
    return {"x": state["grid_x"], "y": state["grid_y"]}


@app.get("/get-status")
def get_status():
    """Returns current scan position and completion flag."""
    return {
        "x": state["current_x"],
        "y": state["current_y"],
        "run_complete": state["run_complete"],
    }


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Accepts an image upload and stores it to disk."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Clear old upload
    if state["image_path"] and Path(state["image_path"]).exists():
        Path(state["image_path"]).unlink()

    ext = Path(file.filename).suffix or ".jpg"
    dest = UPLOAD_DIR / f"image_{uuid.uuid4().hex}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    state["image_path"] = str(dest)

    # Auto-set grid dimensions based on image size if Pillow available
    try:
        from PIL import Image
        with Image.open(dest) as img:
            w, h = img.size
            state["grid_x"] = max(2, min(10, math.ceil(w / 100)))
            state["grid_y"] = max(2, min(12, math.ceil(h / 100)))
    except Exception:
        state["grid_x"] = 4
        state["grid_y"] = 7

    # Reset run state
    state["run_complete"] = False
    state["current_x"] = 0
    state["current_y"] = 0
    state["run_stats"] = {}
    state["detected_patches"] = []

    return {
        "message": "Image uploaded successfully.",
        "filename": dest.name,
        "grid_x": state["grid_x"],
        "grid_y": state["grid_y"],
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
    state["selected_model_id"] = model
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
    """Returns the number of detected patches."""
    return {"count": len(state["detected_patches"])}


@app.get("/get-patch")
def get_patch(id: int):
    """Returns patch image by index."""
    patches = state["detected_patches"]
    if id < 0 or id >= len(patches):
        raise HTTPException(status_code=404, detail="Patch index out of range.")
    path = Path(patches[id])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Patch file not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/get-run-stats")
def get_run_stats():
    """Returns statistics from the last completed run."""
    if not state["run_stats"]:
        return {"status": "no run completed yet"}
    return state["run_stats"]


@app.get("/health")
def health():
    return {"status": "ok"}

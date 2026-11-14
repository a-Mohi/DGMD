# Mitosis Detection Backend

FastAPI backend for the image Mitosis detection dashboard.

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Docs available at: http://localhost:8000/docs

## Endpoints

| Method | Endpoint            | Description                              |
|--------|---------------------|------------------------------------------|
| GET    | /get-progress-dim   | Grid dimensions (x cols, y rows)         |
| GET    | /get-status         | Current scan position + run_complete     |
| POST   | /upload-image       | Upload image (multipart/form-data)       |
| GET    | /get-models         | List available models                    |
| GET    | /set-model?model=id | Select active model                      |
| GET    | /start              | Start detection (400 if not configured)  |
| GET    | /get-patch-count    | Number of detected patches               |
| GET    | /get-patch?id=N     | Get patch image by index                 |
| GET    | /get-run-stats      | Stats from last completed run            |
| GET    | /health             | Health check                             |

## Plugging in a real model

Replace the `run_scan()` function in `main.py` with your actual inference logic.
The function should:
1. Split the image into `grid_x × grid_y` patches
2. Run inference on each patch
3. Update `state["current_x"]` and `state["current_y"]` as it processes
4. Save detected patch images to the `patches/` directory
5. Append their paths to `state["detected_patches"]`
6. Set `state["run_complete"] = True` when done

## File structure

```
segmentation-backend/
├── main.py
├── requirements.txt
├── uploads/      ← uploaded images saved here
└── patches/      ← detected patch images saved here
```

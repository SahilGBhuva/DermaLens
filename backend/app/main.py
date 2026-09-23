import json
import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .model import DermaLensModel

app = FastAPI(title="DermaLens API", version="0.5.0")
model = DermaLensModel()
ROOT_DIR = Path(__file__).resolve().parents[2]
EVALUATION_PATH = ROOT_DIR / "models" / "evaluation.json"

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "DermaLens API",
        "status": "ok",
        "demo_mode": model.demo_mode,
        "medical_device": False,
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "demo_mode": model.demo_mode,
        "model_loaded": not model.demo_mode,
    }


async def load_image(file: UploadFile) -> Image.Image:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Upload a JPEG, PNG, or WebP image.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be under 10 MB.")

    try:
        check = Image.open(BytesIO(raw))
        check.verify()
        image = Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    if image.width < 32 or image.height < 32:
        raise HTTPException(status_code=400, detail="Image is too small to analyze.")

    return image


@app.get("/research-status")
def research_status():
    evaluation = None
    if EVALUATION_PATH.exists():
        try:
            raw = json.loads(EVALUATION_PATH.read_text())
            evaluation = {
                "accuracy": raw.get("accuracy"),
                "balanced_accuracy": raw.get("balanced_accuracy"),
                "macro_f1": raw.get("macro_f1"),
                "weighted_f1": raw.get("weighted_f1"),
                "macro_ovr_roc_auc": raw.get("macro_ovr_roc_auc"),
            }
        except (OSError, json.JSONDecodeError):
            evaluation = None

    return {
        "model_loaded": not model.demo_mode,
        "evaluation_available": evaluation is not None,
        "evaluation": evaluation,
        "implemented": {
            "lesion_level_split": True,
            "class_weighted_training": True,
            "held_out_test_pipeline": True,
            "grad_cam": True,
            "robustness_lab": True,
        },
        "note": (
            "Metrics are only shown when a real held-out evaluation file exists. "
            "DermaLens does not invent performance numbers."
        ),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await load_image(file)
    result = model.predict(image)

    return {
        "top_class": result.top_class,
        "confidence": result.confidence,
        "uncertainty": result.uncertainty,
        "entropy": result.entropy,
        "probabilities": result.probabilities,
        "demo_mode": result.demo_mode,
        "heatmap_data_url": result.heatmap_data_url,
        "disclaimer": (
            "DermaLens is an educational research tool and is not a diagnosis or medical device. "
            "A clinician should evaluate any concerning lesion."
        ),
    }


@app.post("/stress-test")
async def stress_test(file: UploadFile = File(...)):
    image = await load_image(file)
    return model.stress_test(image)

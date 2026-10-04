import asyncio
import json
import os
import warnings
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .model import CONFIG_PATH, MODEL_PATH, DermaLensModel, sha256
from .weights import ensure_file

ROOT_DIR = Path(__file__).resolve().parents[2]
EVALUATION_PATH = ROOT_DIR / "models" / "evaluation.json"

# In the cloud, fetch hosted weights/evaluation before the model loads.
ensure_file(MODEL_PATH, "MODEL_URL", "MODEL_SHA256")
ensure_file(CONFIG_PATH, "CONFIG_URL")
ensure_file(EVALUATION_PATH, "EVALUATION_URL")

# Upload limits. A small, highly compressed file can decode to a huge image
# ("decompression bomb"), so the pixel count is checked before decoding.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000  # ~5000 x 5000; dermoscopy images are far smaller
WORKING_SIZE = 1024  # larger uploads are shrunk first; HAM10000 images are 600 x 450
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
warnings.simplefilter("error", Image.DecompressionBombWarning)

# One analysis at a time by default: in a 512 MB container, two concurrent
# Grad-CAM passes exceeded the limit and the process was OOM-killed. Each request
# takes ~1 s, so others simply queue. Raise this on larger instances.
INFERENCE_SLOTS = asyncio.Semaphore(int(os.getenv("INFERENCE_CONCURRENCY", "1")))
QUEUE_TIMEOUT_SECONDS = 30

app = FastAPI(title="DermaLens API", version="0.7.0")
model = DermaLensModel()

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,  # the API uses no cookies or auth headers
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def reject_oversized_bodies(request: Request, call_next):
    """Refuse oversized uploads from the declared length, before parsing them."""
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 64 * 1024:
        return JSONResponse({"detail": "Image must be under 10 MB."}, status_code=413)
    return await call_next(request)


async def run_inference(fn, image):
    """Run model work off the event loop, with a bounded number at once."""
    try:
        await asyncio.wait_for(INFERENCE_SLOTS.acquire(), timeout=QUEUE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=503, detail="DermaLens is busy. Please try again shortly.") from exc
    try:
        return await run_in_threadpool(fn, image)
    finally:
        INFERENCE_SLOTS.release()


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

    # Read at most one byte past the limit, never the whole stream.
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be under 10 MB.")

    try:
        check = Image.open(BytesIO(raw))
        width, height = check.size  # header only; nothing decoded yet
        if width * height > MAX_PIXELS:
            raise HTTPException(status_code=413, detail="Image dimensions are too large (max 25 megapixels).")
        check.verify()
        image = Image.open(BytesIO(raw))
        image.draft("RGB", (WORKING_SIZE, WORKING_SIZE))  # JPEG: decode at reduced size
        image = image.convert("RGB")
        image.thumbnail((WORKING_SIZE, WORKING_SIZE))
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=413, detail="Image dimensions are too large (max 25 megapixels).") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    if image.width < 32 or image.height < 32:
        raise HTTPException(status_code=400, detail="Image is too small to analyze.")

    return image


def per_class_summary(raw: dict):
    """Sensitivity, specificity and support per class, from evaluation.json."""
    rates = raw.get("per_class_sensitivity_specificity") or {}
    report = raw.get("classification_report") or {}
    rows = []
    for label, values in rates.items():
        rows.append(
            {
                "label": label,
                "sensitivity": values.get("sensitivity"),
                "specificity": values.get("specificity"),
                "support": int((report.get(label) or {}).get("support") or 0),
            }
        )
    return rows


def test_image_count(raw: dict):
    matrix = raw.get("confusion_matrix")
    if not matrix:
        return None
    return int(sum(sum(row) for row in matrix))


def evaluation_matches_model(raw: dict) -> bool:
    """Only publish metrics that were measured on the weights being served."""
    if model.demo_mode:
        return False
    expected_file = model.config.get("evaluation_sha256")
    if expected_file and sha256(EVALUATION_PATH) != expected_file:
        return False
    measured_on = (raw.get("settings") or {}).get("weights_sha256")
    return not measured_on or measured_on == getattr(model, "weights_sha256", None)


@app.get("/research-status")
def research_status():
    evaluation = None
    if EVALUATION_PATH.exists():
        try:
            raw = json.loads(EVALUATION_PATH.read_text())
            if not evaluation_matches_model(raw):
                raise ValueError("evaluation does not belong to the loaded model")
            evaluation = {
                "accuracy": raw.get("accuracy"),
                "balanced_accuracy": raw.get("balanced_accuracy"),
                "macro_f1": raw.get("macro_f1"),
                "weighted_f1": raw.get("weighted_f1"),
                "macro_ovr_roc_auc": raw.get("macro_ovr_roc_auc"),
                "expected_calibration_error": raw.get("expected_calibration_error"),
                "multiclass_brier_score": raw.get("multiclass_brier_score"),
                "per_class": per_class_summary(raw),
                "test_images": test_image_count(raw),
                # Rows are true classes, columns predicted, in model CLASSES order.
                "confusion_matrix": raw.get("confusion_matrix"),
                "classes": model.config["classes"],
            }
        except (OSError, ValueError):
            evaluation = None

    return {
        "model_loaded": not model.demo_mode,
        "model": model.info(),
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
    result = await run_inference(model.predict, image)

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
    return await run_inference(model.stress_test, image)

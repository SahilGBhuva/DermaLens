import asyncio
import ctypes
import gc
import os
import sys
import warnings
from functools import partial
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .guards import RequestGuard
from .model import CONFIG_PATH, MODEL_PATH, MODELS_DIR
from .registry import ModelEntry, build_registry
from .weights import ensure_file

# Single-model deployments may still set MODEL_URL/CONFIG_URL/... (files land in
# models/); multi-model deployments set MODEL_RELEASES (see registry.py).
ensure_file(MODEL_PATH, "MODEL_URL", "MODEL_SHA256")
ensure_file(CONFIG_PATH, "CONFIG_URL")
ensure_file(MODELS_DIR / "evaluation.json", "EVALUATION_URL")
ensure_file(MODELS_DIR / "training_history.json", "HISTORY_URL")

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

# Only the formats the site accepts are ever handed to an image decoder,
# whatever Content-Type the client claims (each extra decoder is attack surface).
DECODERS = ["JPEG", "PNG", "WEBP"]

PRODUCTION = os.getenv("DERMALENS_ENV", "development") == "production"
app = FastAPI(
    title="DermaLens API",
    version="0.8.0",
    # Interactive docs are handy locally but not needed on the public API.
    docs_url=None if PRODUCTION else "/docs",
    redoc_url=None,
    openapi_url=None if PRODUCTION else "/openapi.json",
)
registry = build_registry(MODELS_DIR)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

# Added before CORS so CORS wraps it: 413/429 replies still carry CORS headers
# and the browser can show the message instead of a generic network error.
app.add_middleware(
    RequestGuard,
    max_body_bytes=MAX_UPLOAD_BYTES + 64 * 1024,  # file plus multipart framing
    limited_paths=("/predict", "/stress-test"),
    rate_limit=int(os.getenv("RATE_LIMIT_REQUESTS", "30")),
    rate_window_seconds=float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "600")),
    trust_proxy_headers=os.getenv("TRUST_PROXY_HEADERS") == "1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,  # the API uses no cookies or auth headers
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


async def run_inference(fn, image):
    """Run model work off the event loop, with a bounded number at once."""
    try:
        await asyncio.wait_for(INFERENCE_SLOTS.acquire(), timeout=QUEUE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=503, detail="DermaLens is busy. Please try again shortly.") from exc
    try:
        return await run_in_threadpool(fn, image)
    finally:
        release_memory()
        INFERENCE_SLOTS.release()


_LIBC = ctypes.CDLL("libc.so.6") if sys.platform.startswith("linux") else None


def release_memory():
    """Return freed heap pages to the OS after each analysis (glibc only)."""
    gc.collect()
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


# The models were trained on dermatoscopic images (HAM10000). Everyday photos
# are accepted, analysed the same way, and clearly flagged as less reliable.
IMAGE_TYPES = {"dermoscopy", "photo"}
PHOTO_NOTE = (
    "Regular photo: DermaLens was trained on dermoscopy images, so results on everyday "
    "photos are less accurate, and that accuracy has not been measured yet."
)


def check_image_type(image_type: str) -> str:
    if image_type not in IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="image_type must be 'dermoscopy' or 'photo'.")
    return image_type


def default_entry() -> ModelEntry:
    return registry.get(None)


def resolve(version: str | None) -> ModelEntry:
    entry = registry.get(version)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown model version. Available: {', '.join(registry.order)}.")
    return entry


@app.get("/")
def root():
    return {
        "name": "DermaLens API",
        "status": "ok",
        "demo_mode": default_entry().model.demo_mode,
        "medical_device": False,
    }


@app.get("/health")
def health():
    loaded = [e.version for e in registry if not e.model.demo_mode]
    return {
        "ok": True,
        "demo_mode": not loaded,
        "model_loaded": bool(loaded),
        "models": loaded,
        "default_model": registry.default if loaded else None,
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
        check = Image.open(BytesIO(raw), formats=DECODERS)
        width, height = check.size  # header only; nothing decoded yet
        if width * height > MAX_PIXELS:
            raise HTTPException(status_code=413, detail="Image dimensions are too large (max 25 megapixels).")
        check.verify()
        image = Image.open(BytesIO(raw), formats=DECODERS)
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


def evaluation_summary(entry: ModelEntry):
    raw = entry.evaluation()
    if raw is None:
        return None
    return {
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
        "classes": entry.model.config["classes"],
    }


def model_status(entry: ModelEntry) -> dict:
    evaluation = evaluation_summary(entry)
    return {
        "version": entry.version,
        "model_loaded": not entry.model.demo_mode,
        "model": {**entry.model.info(), "version": entry.version},
        "training": entry.training(),
        "evaluation_available": evaluation is not None,
        "evaluation": evaluation,
    }


@app.get("/research-status")
def research_status():
    models = [model_status(entry) for entry in registry]
    default = models[0]  # registry order starts with the default
    return {
        # Top-level fields describe the default model (original response shape).
        "model_loaded": default["model_loaded"],
        "model": default["model"],
        "training": default["training"],
        "evaluation_available": default["evaluation_available"],
        "evaluation": default["evaluation"],
        "default_model": registry.default,
        "models": models,
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
async def predict(
    file: UploadFile = File(...),
    model: str | None = Query(None, max_length=32),
    image_type: str = Query("dermoscopy", max_length=16),
    explain: bool = Query(True, description="false skips Grad-CAM (scores only)"),
):
    image_type = check_image_type(image_type)
    entry = resolve(model)
    image = await load_image(file)
    result = await run_inference(partial(entry.model.predict, explain=explain), image)

    return {
        "model_version": entry.version,
        "image_type": image_type,
        "reliability_note": PHOTO_NOTE if image_type == "photo" else None,
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
async def stress_test(
    file: UploadFile = File(...),
    model: str | None = Query(None, max_length=32),
    image_type: str = Query("dermoscopy", max_length=16),
):
    image_type = check_image_type(image_type)
    entry = resolve(model)
    image = await load_image(file)
    return {
        "model_version": entry.version,
        "image_type": image_type,
        **await run_inference(entry.model.stress_test, image),
    }

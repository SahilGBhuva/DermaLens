from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .model import DermaLensModel

app = FastAPI(title="DermaLens API", version="0.1.0")
model = DermaLensModel()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "demo_mode": model.demo_mode}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Upload a JPEG, PNG, or WebP image.")

    raw = await file.read()

    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be under 10 MB.")

    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    result = model.predict(image)

    return {
        "top_class": result.top_class,
        "confidence": result.confidence,
        "uncertainty": result.uncertainty,
        "probabilities": result.probabilities,
        "demo_mode": result.demo_mode,
        "disclaimer": (
            "DermaLens is an educational research tool and is not a diagnosis or medical device. "
            "A clinician should evaluate any concerning lesion."
        ),
    }

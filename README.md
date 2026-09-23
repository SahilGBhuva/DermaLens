# DermaLens

DermaLens is an educational/research web app for experimenting with machine-learning analysis of skin-lesion images.

> **Important:** DermaLens is not a medical device and does not diagnose cancer. Its outputs are for education and model research only.

## What it does

- Upload a skin-lesion image
- Run an image classifier
- Show class probabilities and uncertainty
- Generate a Grad-CAM style explanation when supported by the loaded model
- Keep medical-safety language visible in the UI
- Provide training/evaluation scripts for a HAM10000-style dataset

## Stack

- **Frontend:** Next.js + TypeScript
- **API / ML:** FastAPI + PyTorch
- **Model:** EfficientNet-B0 transfer learning (7-class setup by default)
- **Dataset target:** HAM10000 / ISIC-style image folders + metadata

## Project structure

```
frontend/        Next.js app
backend/         FastAPI inference service
ml/              training + evaluation utilities
models/          local trained weights (gitignored)
```

## Run locally

### 1. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:3000.

The frontend expects the API at `http://localhost:8000`. Override with:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Model weights

Place trained weights at:

```
models/dermalens_efficientnet_b0.pt
```

If no weights are present, the backend starts in **demo mode** and returns a clearly labeled placeholder response. Demo mode is intentionally not presented as a real prediction.

## Training

Prepare a CSV with at least:

- `image_path`
- `label`

Then:

```bash
python ml/train.py --csv data/train.csv --epochs 10
```

Evaluation:

```bash
python ml/evaluate.py --csv data/val.csv --weights models/dermalens_efficientnet_b0.pt
```

## Classes

The default 7 HAM10000-style classes are:

- akiec
- bcc
- bkl
- df
- mel
- nv
- vasc

These are research labels, not end-user diagnoses.

## CAC direction

For the Congressional App Challenge, the strongest version is not merely “AI says cancer/no cancer.” The more defensible demo is:

1. image analysis,
2. transparent probabilities,
3. explanation heatmap,
4. uncertainty,
5. model limitations,
6. robustness testing across image conditions.

That demonstrates both ML engineering and responsible product design.

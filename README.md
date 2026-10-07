# DermaLens

DermaLens is an educational/research web app for experimenting with machine-learning analysis of skin-lesion images.

> **Important:** DermaLens is not a medical device and does not diagnose cancer. Its outputs are for education and model research only.

## At a glance

- **What it does:** upload a dermatoscopic image → see all seven class probabilities, a Grad-CAM attention overlay, and whether the answer survives darker, brighter, lower-contrast and blurred versions of the same photo.
- **Model:** EfficientNet-B0 fine-tuned on HAM10000 with a lesion-level train / validation / test split.
- **Held-out test results** (1,494 images, each scored once):
  - v1: accuracy 0.801 · balanced accuracy 0.741 · **melanoma sensitivity 0.64** · calibration error 0.05
  - v2: accuracy 0.785 · balanced accuracy 0.733 · **melanoma sensitivity 0.78** · calibration error 0.32 — catches 26 more melanomas, at the cost of more false alarms, lower sensitivity for two other cancers, and poorly calibrated percentages. The trade-off is analysed in [`MODEL_CARD.md`](MODEL_CARD.md).
- **Honesty by design:** metrics are only shown for the exact model file being served (SHA-256 checked); illustrative sections are labelled; every result says it is not a diagnosis.
- **Quality:** 35 automated tests; Lighthouse performance 98, accessibility 100, best practices 100, SEO 100; API load-tested inside a 512 MB container.
- **Docs:** [`DEPLOY.md`](DEPLOY.md) (cloud setup) · [`SECURITY.md`](SECURITY.md) · [`docs/DEMO.md`](docs/DEMO.md) (demo script) · [`docs/ROADMAP.md`](docs/ROADMAP.md) · [`docs/STUDY_GUIDE.md`](docs/STUDY_GUIDE.md).

### Run it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt -r ml/requirements.txt

# API (port 8765)
cd backend && CORS_ORIGINS=http://localhost:3210 uvicorn app.main:app --port 8765

# Website (another terminal, port 3210)
cd frontend && npm install && NEXT_PUBLIC_API_URL=http://127.0.0.1:8765 npm run dev -- -p 3210
```

Each trained model lives in its own folder (`models/v1/`, `models/v2/`) and the API serves all of them: the sandbox can switch versions or **compare them on the same image**, and the Evidence page shows them side by side. Install a new one with `python ml/install_model.py --from ~/Downloads --version v3`. With no models the API runs in a clearly labelled demo mode.

## What is implemented

- Next.js site with an interactive scope hero, a four-step "how it works" walkthrough, a 3D "attention landscape" (three.js, loaded only when scrolled into view), a robustness lab, and an evidence page that only shows real held-out metrics, including per-class sensitivity once an evaluation exists
- sandbox wired to the API: upload an image (or a clearly labelled synthetic sample), read all seven probabilities, toggle the Grad-CAM overlay, and run the stress test
- FastAPI + PyTorch inference API
- EfficientNet-B0 transfer-learning pipeline
- HAM10000 metadata preparation
- lesion-level train/validation/test splitting to reduce leakage
- class-weighted training for class imbalance
- evaluation with accuracy, balanced accuracy, macro/weighted F1, confusion matrix, classification report and multiclass ROC-AUC
- normalized predictive entropy
- Grad-CAM attention heatmap returned by the API and shown in the UI, colored with the same blue → red scale as the site legend
- automatic device selection for training and inference: CUDA, then Apple Silicon (MPS), then CPU
- demo mode when trained weights are absent

## Project structure

```
frontend/        Next.js app
backend/         FastAPI inference service + Grad-CAM
ml/              dataset prep, training, evaluation
models/          local weights/metrics (gitignored)
data/            local dataset/splits (gitignored)
```

## 1. Get HAM10000

Download the HAM10000 images and `HAM10000_metadata.csv` from an authorized source such as the ISIC Archive / dataset distribution.

Do not commit the dataset to GitHub.

Example local layout:

```
data/
  raw/
    HAM10000_metadata.csv
    images/
      ISIC_0024306.jpg
      ...
```

## 2. Prepare lesion-level splits

One virtual environment at the repository root covers training, evaluation and the API:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt -r backend/requirements.txt
```

Then build the splits:

```bash
python ml/prepare_ham10000.py \
  --metadata data/raw/HAM10000_metadata.csv \
  --images-dir data/raw/images \
  --output-dir data/splits
```

This creates:

```
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

The split is grouped by `lesion_id`, so images of the same physical lesion are kept in one split.

## 3. Train

```bash
python ml/train.py \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv \
  --epochs 12 \
  --output models/dermalens_efficientnet_b0.pt
```

The first run downloads the ImageNet EfficientNet-B0 starting weights (about 20 MB) from `download.pytorch.org`. With the python.org installer on macOS this can fail with `CERTIFICATE_VERIFY_FAILED`, because that Python ships without root certificates. Use the certificates bundled in the environment:

```bash
export SSL_CERT_FILE=$(python -m certifi)
```

The training code uses ImageNet-pretrained EfficientNet-B0, augmentation, AdamW, cosine learning-rate scheduling, and class-weighted cross-entropy. It trains on CUDA when available, otherwise on the Apple Silicon GPU (MPS), otherwise on the CPU.

## 4. Evaluate on the held-out test set

```bash
python ml/evaluate.py \
  --csv data/splits/test.csv \
  --weights models/dermalens_efficientnet_b0.pt \
  --output models/evaluation.json
```

Do not use the test set to tune the model. Keep it for final evaluation.

## 5. Run the API

With the root `.venv` active:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

The API only accepts browser requests from origins listed in `CORS_ORIGINS` (default `http://localhost:3000`). If the frontend runs on another port, list it:

```bash
CORS_ORIGINS=http://localhost:3000,http://localhost:3100 uvicorn app.main:app --reload --port 8000
```

The backend looks for:

```
models/dermalens_efficientnet_b0.pt
```

If weights are not present, it runs in clearly labeled demo mode and returns no medical prediction.

## 6. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000.

Set another API URL with:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Model output

The API returns:

- seven class probabilities
- highest-scoring class
- max-probability confidence
- `1 - max probability` uncertainty proxy
- normalized predictive entropy
- Grad-CAM attention heatmap: a 448 × 448 PNG overlay using the site's attention scale (blue = low contribution, red = high)
- explicit research-use disclaimer

The uncertainty values are model-output summaries, not calibrated probabilities of clinical correctness.

## Research limitations

HAM10000 is useful for education and model research, but performance on a held-out dataset does not establish clinical validity. Important limitations include image/device distribution shift, demographic representation, class imbalance, label quality, acquisition differences, and the fact that dermatoscopic images are not equivalent to ordinary phone photos.

Grad-CAM visualizations show which model features influenced a score; they do not prove that those features are medically meaningful.

## Demo story

A strong demo is:

```
upload image
→ model produces class distribution
→ uncertainty is shown
→ Grad-CAM visualizes attention
→ limitations explain when the model can fail
```

This positions DermaLens as an explainable medical-ML research tool rather than claiming to diagnose cancer.


## Quality checks

GitHub Actions builds the Next.js frontend and runs backend API tests on pushes and pull requests.

Run the backend tests locally:

```bash
cd backend
python -m pytest -q
```

`tests/test_inference.py` exercises the loaded-model path (probabilities, Grad-CAM overlay, stress test) with a randomly initialised network, so it checks mechanics without needing trained weights.

To verify dataset splits locally:

```bash
python ml/check_splits.py \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --test data/splits/test.csv
```

This checks that no `lesion_id` appears in more than one split.

## Robustness experiment

After training:

```bash
python ml/robustness.py \
  --image path/to/example.jpg \
  --weights models/dermalens_efficientnet_b0.pt
```

This compares model output after brightness, contrast, and blur perturbations.

## Deployment

Step-by-step cloud setup (Colab training, GitHub Release for weights, Render API, Vercel site) is in [`DEPLOY.md`](DEPLOY.md). To train without downloading the dataset locally, use [`ml/train_in_colab.ipynb`](ml/train_in_colab.ipynb).

### Frontend

The `frontend/` folder is Vercel-ready.

Set:

```
NEXT_PUBLIC_API_URL=https://your-api-host.example
```

### Backend

The repository includes a Dockerfile and Render service definition. The API reads allowed frontend origins from:

```
CORS_ORIGINS=https://your-frontend.example
```

Multiple origins can be comma-separated.

## Privacy note

The default API processes image bytes in memory and does not intentionally write uploads to disk. Production deployments should use HTTPS and avoid request logging that captures medical images.

To explain the project in your own words (methods, metrics, limitations, likely interview questions), see [`docs/STUDY_GUIDE.md`](docs/STUDY_GUIDE.md).

See `MODEL_CARD.md` for intended use and evaluation requirements, and `SECURITY.md` for deployment/privacy considerations.


## One-command training pipeline

After placing HAM10000 metadata and images locally, the complete leakage check, training, and held-out evaluation can be run with:

```bash
python ml/run_pipeline.py \
  --metadata data/raw/HAM10000_metadata.csv \
  --images-dir data/raw/images \
  --epochs 12
```

The final evaluation includes imbalance-aware and calibration-aware metrics, including balanced accuracy, macro F1, per-class sensitivity/specificity, multiclass ROC-AUC, expected calibration error, and multiclass Brier score.

## Presentation

See [`docs/DEMO.md`](docs/DEMO.md) for a three-minute demonstration structure and [`docs/ROADMAP.md`](docs/ROADMAP.md) for where the project goes next.

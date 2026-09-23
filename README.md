# DermaLens

DermaLens is an educational/research web app for experimenting with machine-learning analysis of skin-lesion images.

> **Important:** DermaLens is not a medical device and does not diagnose cancer. Its outputs are for education and model research only.

## What is implemented

- Next.js image-upload interface
- FastAPI + PyTorch inference API
- EfficientNet-B0 transfer-learning pipeline
- HAM10000 metadata preparation
- lesion-level train/validation/test splitting to reduce leakage
- class-weighted training for class imbalance
- evaluation with accuracy, balanced accuracy, macro/weighted F1, confusion matrix, classification report and multiclass ROC-AUC
- normalized predictive entropy
- Grad-CAM attention heatmap returned by the API and shown in the UI
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

```bash
python -m pip install -r ml/requirements.txt

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

The training code uses ImageNet-pretrained EfficientNet-B0, augmentation, AdamW, cosine learning-rate scheduling, and class-weighted cross-entropy.

## 4. Evaluate on the held-out test set

```bash
python ml/evaluate.py \
  --csv data/splits/test.csv \
  --weights models/dermalens_efficientnet_b0.pt \
  --output models/evaluation.json
```

Do not use the test set to tune the model. Keep it for final evaluation.

## 5. Run the API

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
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
- Grad-CAM attention heatmap
- explicit research-use disclaimer

The uncertainty values are model-output summaries, not calibrated probabilities of clinical correctness.

## Research limitations

HAM10000 is useful for education and model research, but performance on a held-out dataset does not establish clinical validity. Important limitations include image/device distribution shift, demographic representation, class imbalance, label quality, acquisition differences, and the fact that dermatoscopic images are not equivalent to ordinary phone photos.

Grad-CAM visualizations show which model features influenced a score; they do not prove that those features are medically meaningful.

## CAC demo story

A strong demo is:

```
upload image
→ model produces class distribution
→ uncertainty is shown
→ Grad-CAM visualizes attention
→ limitations explain when the model can fail
```

This positions DermaLens as an explainable medical-ML research tool rather than claiming to diagnose cancer.

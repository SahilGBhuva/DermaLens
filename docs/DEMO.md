# DermaLens — Demo and presentation plan

A three-minute walkthrough for portfolio videos, interviews and presentations. It is built to show three things a reviewer looks for: an original idea, a well-built and honest product, and real understanding of the technical work.

## Core story

DermaLens is not a replacement for a dermatologist. It is an explainable medical-image machine-learning research app that asks a harder question:

> When an image classifier gives a medical-image prediction, can a user see why it responded that way, and whether the answer is stable?

## Three-minute demo

Record in one take if you can; keep the browser full screen so text is readable.

### 0:00–0:25 — Problem (hero gallery)

Most image classifiers return one confident-looking answer. That hides what influenced the network and whether a slightly worse photo would change the result.

Point at the two attention maps in the gallery: "These are real outputs of my two model versions on the same public-domain melanoma image. Version 2 called it melanoma; version 1 called it a mole — and they looked at different parts of the lesion."

### 0:25–0:55 — How it works (scroll story)

Scroll through the four steps: prediction, attention, robustness, evidence.

### 0:55–1:50 — Real analysis (sandbox)

1. Load the sample image (or one you are allowed to use).
2. **Prediction** tab: read the full distribution, not just the top class, and the response time.
3. **Attention** tab: the real Grad-CAM overlay.
4. **Robustness** tab: run the stress test — darker, brighter, lower contrast, blurred. If the top class changes, say: "This is the kind of fragility the app is built to expose."
5. **Second opinion:** click "Compare with v1". On the sample, v1 disagrees (mole, 54%). Say: "When two trained models disagree, that's a signal this image is hard, which is exactly when a person should look closer."
6. **Export result** to show the research record (it includes the second opinion).
7. Say clearly: "This is not a diagnosis."

### 1:50–2:40 — Evidence and what you improved

Scroll to **Evidence** and give the honest story:

- v1 caught 64% of test melanomas. v2 added stronger augmentation, a second training recipe and validation-only tuning with a melanoma target, then scored the test set once: **78% of melanomas caught**.
- The cost: more false alarms, lower detection of two other cancers, and much less trustworthy percentages (calibration error 0.05 → 0.32).
- "No version wins everywhere. Improving one number moved errors elsewhere — and that's the lesson."

### 2:40–3:00 — Responsible ML and close

"DermaLens refuses to show made-up numbers, ties every metric to the exact model file it serves, and says what it hasn't proven: other cameras, clinics and skin tones."

End with:

> Good medical AI should not only make a prediction. It should help us understand when that prediction may be fragile.

## Technical depth (for Q&A or a second video)

- EfficientNet-B0 transfer learning; lesion-level train / validation / test split with a leakage check
- Class imbalance: weighted loss, focal loss, balanced sampling — chosen on validation
- Validation-only calibration (temperature scaling, per-class offsets with a melanoma floor); test scored once
- Balanced accuracy, macro F1, per-class sensitivity/specificity, ROC-AUC, calibration error
- Grad-CAM with PyTorch hooks; flip-averaged predictions
- FastAPI backend serving several model versions, with upload hardening, rate limiting, SHA-256 model integrity checks, memory-tested for a 512 MB server
- Automated tests and code scanning in GitHub Actions

## Before sharing it

- [ ] Deploy (see `DEPLOY.md`) and test the live link from a clean browser and a phone.
- [ ] Open the live site a minute before a demo or recording so the free API is awake.
- [ ] Record with an image you are allowed to use.
- [ ] Practise the questions in `docs/STUDY_GUIDE.md` out loud.
- [ ] Clearly disclose any AI assistance used while building the project.
- [ ] Credit the HAM10000 dataset (Tschandl, Rosendahl & Kittler, 2018) and the ISIC Archive.

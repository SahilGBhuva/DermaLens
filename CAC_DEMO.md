# DermaLens — CAC Demo Plan

This plan is designed around the 2026 Congressional App Challenge's stated judging areas: quality/originality of the idea, implementation and user experience, and demonstrated coding/programming skill.

## Core story

DermaLens is not presented as a replacement for a dermatologist. It is an explainable medical-image machine-learning research app that asks a harder question:

> When an image classifier gives a medical-image prediction, can a user see why it responded that way and whether the answer is stable?

## Three-minute demo structure

### 0:00–0:25 — Problem

Most image classifiers return one confident-looking answer. That hides two important questions: what influenced the network, and would a small change in the image change the result?

Introduce DermaLens as a tool for making those failure modes visible.

### 0:25–1:05 — Main analysis

1. Upload a skin-lesion image.
2. Show the seven-class probability distribution.
3. Point out normalized predictive entropy and the top model score.
4. Show the Grad-CAM visualization.
5. State clearly that the output is educational/research-only and not a diagnosis.

### 1:05–1:50 — Robustness Lab

1. Run the built-in stress test.
2. Explain that DermaLens creates controlled darker, brighter, lower-contrast, and blurred versions.
3. Show whether the top class remains stable.
4. If a class changes, highlight that this is exactly the kind of model weakness the app is meant to expose.

### 1:50–2:25 — Technical depth

Briefly show the research-status panel and source code.

Explain:
- EfficientNet-B0 transfer learning
- lesion-level train/validation/test splitting to reduce leakage
- class-weighted loss for imbalance
- held-out test evaluation
- balanced accuracy and macro F1
- calibration error and Brier score
- Grad-CAM hooks
- FastAPI backend and Next.js frontend
- automated GitHub Actions tests

### 2:25–2:50 — Responsible ML

Show that DermaLens refuses to display made-up performance numbers. Metrics only appear when a real evaluation file exists.

Mention dataset shift, image quality, class imbalance, demographic representation, and the difference between dermatoscopic and ordinary phone images.

### 2:50–3:00 — Close

End with the central idea:

> Good medical AI should not only make a prediction. It should help us understand when that prediction may be fragile.

## Before submission

- Train the final model.
- Evaluate exactly once on the untouched test split after tuning is finished.
- Save the real evaluation JSON.
- Test the live deployment from a clean browser.
- Record the demo using an image you are allowed to use.
- Make sure every team member can explain the main code paths and ML decisions.
- Clearly disclose any AI assistance used while building the project.

# DermaLens Model Card

## Intended use

Educational and research exploration of skin-lesion image classification.

## Not intended for

- diagnosis,
- treatment decisions,
- triage,
- screening claims,
- or use as a medical device.

## Model

EfficientNet-B0 initialized with ImageNet weights and fine-tuned for seven HAM10000-style labels:

- akiec
- bcc
- bkl
- df
- mel
- nv
- vasc

## Evaluation policy

Report results only after training on the training split, tuning on the validation split, and evaluating once on the untouched test split.

Recommended metrics:
- balanced accuracy,
- macro F1,
- per-class precision/recall,
- confusion matrix,
- multiclass ROC-AUC.

Accuracy alone is insufficient because the dataset is class-imbalanced.

## Known limitations

Performance can change with:
- skin tone representation,
- camera/device type,
- dermatoscopic vs phone images,
- lighting,
- blur,
- crop,
- class imbalance,
- and dataset shift.

Grad-CAM indicates model attention, not medical causality or clinical correctness.

## Explainability output

Grad-CAM is computed on the final convolutional block for the highest-scoring class. The API returns it as a 448 × 448 overlay colored from blue (low contribution) through cyan and yellow to red (high contribution), matching the legend in the web interface.

## Illustrative content in the interface

The hero, the "how it works" walkthrough, the 3D attention landscape and the robustness lab use a synthetic, hand-drawn lesion and fixed example numbers so the site can explain itself without patient images. They are labelled "illustrative" or "simulated" and are not model output. Only the sandbox and the evidence section show real API results.

## Current status

No performance claims should be made until trained weights and a held-out evaluation are produced.

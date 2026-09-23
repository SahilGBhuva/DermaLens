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

## Current status

No performance claims should be made until trained weights and a held-out evaluation are produced.

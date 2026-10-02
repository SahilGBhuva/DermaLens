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

How the pipeline (`ml/run_pipeline.py`) enforces this:

1. **Split** HAM10000 by `lesion_id` (70 / 15 / 15) so no physical lesion appears in two splits; `check_splits.py` verifies it.
2. **Train** each candidate recipe (weighted cross-entropy; focal loss with class-balanced sampling). Each keeps the epoch with the best **validation** balanced accuracy.
3. **Choose** the recipe with the best validation balanced accuracy.
4. **Calibrate on validation only** (`ml/calibrate.py`): temperature scaling, then per-class log-odds offsets that maximise validation balanced accuracy subject to a minimum melanoma sensitivity (default 0.80). Saved to `model_config.json` with the weights' SHA-256.
5. **Score the test split once** (`ml/evaluate.py`) with exactly those settings. The test split is not used for any choice.

The API reads the same `model_config.json` (image size, normalisation, flip-averaging, temperature, offsets), refuses weights whose SHA-256 does not match it, and only publishes metrics measured on the weights it is serving. A test (`backend/tests/test_model_config.py`) checks that API preprocessing equals the training pipeline's evaluation transform.

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

## Results — v1 (2026-10-01)

Original recipe: EfficientNet-B0, weighted cross-entropy, 12 epochs (weights from the epoch with the lowest validation loss, epoch 6), no flip-averaging, no calibration. Trained on a Colab T4 in ~17 minutes.

Split: 7,002 train / 1,519 validation / 1,494 test images, no lesion leakage. Scored once on the test split.

| Metric | Test |
|---|---|
| Accuracy | 0.801 |
| Balanced accuracy | 0.741 |
| Macro F1 | 0.640 |
| Macro one-vs-rest ROC-AUC | 0.944 |
| Expected calibration error | 0.052 |

| Class | Test images | Sensitivity | Specificity |
|---|---|---|---|
| akiec | 63 | 0.62 | 0.98 |
| bcc | 68 | 0.82 | 0.98 |
| bkl | 152 | 0.68 | 0.96 |
| df | 7 | 0.57 | 0.98 |
| **mel** | **187** | **0.64** | **0.92** |
| nv | 996 | 0.86 | 0.91 |
| vasc | 21 | 1.00 | 0.99 |

Reading these honestly:

- **Melanoma sensitivity is 0.64: about 1 in 3 test melanomas was missed** (most were called nevus or benign keratosis). This is the main weakness and the target of v2.
- Accuracy is inflated by class imbalance (two-thirds of images are nevi); balanced accuracy and macro F1 matter more.
- The dermatofibroma (7 images) and vascular (21 images) results rest on very few examples and are highly uncertain.
- Training accuracy reached 0.92 against 0.80 on validation by epoch 12, i.e. the model overfits; v2 adds stronger augmentation and early stopping.

## v2 plan (not yet run)

Stronger augmentation (crops, rotation, lighting and blur), label smoothing, up to 25 epochs with early stopping, recipe choice between weighted CE and focal loss with balanced sampling on validation, flip-averaged predictions, and validation-only calibration with a melanoma-sensitivity floor. Raising melanoma sensitivity will lower specificity (more false alarms); that trade-off is deliberate and will be reported.

## Not clinically ready

DermaLens is a research and education project. Before any clinical use it would still need, at minimum: external validation on independent datasets and devices, review by dermatologists, prospective testing, evaluation across skin tones, privacy and security work, and regulatory review. None of these has been done.

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

**It always answers with one of seven classes.** The model has no "this is not a skin lesion" option: random noise, a phone photo, or a picture of something else still gets a top class. In testing, random-noise images received low scores (27–41%), which the interface labels "inconclusive", but low scores are not a reliable detector of invalid input. Only dermatoscopic lesion images are in scope.

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

## Results — v2 (2026-10-05)

Recipe: stronger augmentation, label smoothing 0.1, warm-up + cosine schedule, up to 25 epochs with early stopping. Two candidates were trained; **weighted cross-entropy** was chosen on validation balanced accuracy (0.776 vs 0.732 for focal loss + balanced sampling). Then, on validation only: flip-averaged predictions, temperature 0.7, and per-class offsets `[-1.3, -1.2, -0.5, -0.4, +0.3, 0, -0.2]` (akiec … vasc) chosen to maximise balanced accuracy with a melanoma-sensitivity floor of 0.80. Validation: balanced accuracy 0.776 → 0.790, melanoma sensitivity 0.724 → 0.801.

Test split scored once (1,494 images). Same split as v1.

| Metric | v1 | v2 |
|---|---|---|
| Accuracy | 0.801 | 0.785 |
| Balanced accuracy | 0.741 | 0.733 |
| Macro F1 | 0.640 | 0.615 |
| Macro one-vs-rest ROC-AUC | 0.944 | 0.902 |
| Expected calibration error | **0.052** | **0.315** |
| **Melanoma sensitivity** | **0.642** | **0.781** |
| Melanoma specificity | 0.919 | 0.871 |

| Class | Test images | Sensitivity v1 → v2 | Specificity v1 → v2 |
|---|---|---|---|
| akiec | 63 | 0.62 → **0.46** | 0.98 → 0.99 |
| bcc | 68 | 0.82 → 0.76 | 0.98 → 0.99 |
| bkl | 152 | 0.68 → 0.63 | 0.96 → 0.98 |
| df | 7 | 0.57 → 0.71 | 0.98 → 0.98 |
| **mel** | **187** | **0.64 → 0.78** | **0.92 → 0.87** |
| nv | 996 | 0.86 → 0.83 | 0.91 → 0.93 |
| vasc | 21 | 1.00 → 0.95 | 0.99 → 0.98 |

What changed, honestly:

- **Melanoma: 146 of 187 caught (was 120).** The main goal was met: sensitivity rose 14 points.
- **More false alarms.** v2 labelled 315 test images melanoma (146 correct) versus v1's 226 (120 correct).
- **Two other cancers got worse.** Actinic keratosis / intraepithelial carcinoma sensitivity fell from 0.62 to 0.46 and basal cell carcinoma from 0.82 to 0.76. The negative offsets on those classes moved decisions toward melanoma partly at their expense.
- **Probabilities became poorly calibrated** (ECE 0.05 → 0.32). The offsets were tuned for decisions, not for probability accuracy, and label smoothing with class weighting already distorts the probabilities. The interface now warns when a served model's calibration error exceeds 0.10.
- Overall balanced accuracy is essentially unchanged (−0.8 points), and ROC-AUC fell, partly because the offsets and temperature are applied before scoring.
- **Reproducibility:** the pipeline was run twice (the first run's weights and test results were lost to a Colab disconnect before anyone saw them). Both runs produced byte-identical weights (SHA-256 `a9033a0c…`), so the reported test result is the only one ever viewed.

### Lessons for a v3 (not run)

1. Separate probability calibration (fit by validation NLL, e.g. vector scaling) from decision thresholds, so percentages stay honest while decisions favour sensitivity.
2. Constrain sensitivity for all malignant or pre-malignant classes (mel, bcc, akiec), not just melanoma.
3. Report a confidence interval for every per-class number; df and vasc have very few test images.
4. Because v3 would be designed after seeing v2's test results, its test score should be reported as such, or validated on an external dataset (e.g. ISIC 2019/2020).

## Not clinically ready

DermaLens is a research and education project. Before any clinical use it would still need, at minimum: external validation on independent datasets and devices, review by dermatologists, prospective testing, evaluation across skin tones, privacy and security work, and regulatory review. None of these has been done.

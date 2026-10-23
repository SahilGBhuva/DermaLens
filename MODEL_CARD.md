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

**Regular photos are accepted but flagged.** Users can choose "Regular photo" instead of "Dermoscopy" (`image_type=photo` in the API). The image goes through the same model, and the result carries a reliability note with the measured accuracy on smartphone photos (see "External check" below): much lower than on dermoscopy, so photo results are a demonstration only.

## Explainability output

Grad-CAM is computed on the final convolutional block for the highest-scoring class. The API returns it as a 448 × 448 overlay colored from blue (low contribution) through cyan and yellow to red (high contribution), matching the legend in the web interface.

## Illustrative content in the interface

The hero gallery and the scroll story use one public-domain dermoscopy image (ISIC_0016128, CC0) and fixed example numbers so the site can explain itself; the "87%" score card is labelled "example interface". Two gallery cards are real Grad-CAM outputs of v2 and v1 on that image, generated offline (v2 called it melanoma at 0.64, v1 called it a nevus at 0.54; the ISIC record lists it as malignant). Only the sandbox and the evidence section show live API results.

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

## External check — smartphone photos, PAD-UFES-20 (2026-10-08)

Both versions were run **unchanged** (same preprocessing, flip-averaging and calibration as the API) on [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1): 2,298 clinical photos taken with smartphones at a free skin clinic in Brazil (Pacheco et al., 2020, CC BY 4.0). Nothing was tuned on it. The run happened on a GitHub Actions runner with every download checked by SHA-256 (`.github/workflows/external-eval.yml`); the full output is [`docs/results/pad_ufes_20.json`](docs/results/pad_ufes_20.json), tied to the weight hashes.

Five of its six diagnoses map onto DermaLens classes (ACK→akiec, BCC→bcc, MEL→mel, NEV→nv, SEK→bkl): 2,106 images. Invasive squamous cell carcinoma (192 images) has no DermaLens class and is scored only on whether it was flagged as akiec, bcc or mel.

| | v1 | v2 | v2 on dermoscopy (HAM10000 test) |
|---|---|---|---|
| Accuracy | 0.302 | 0.272 | 0.785 |
| Balanced accuracy (5 classes; chance = 0.20) | 0.292 | 0.312 | 0.733 |
| Melanoma sensitivity | **4/52 = 8%** (95% CI 3–18%) | **20/52 = 38%** (95% CI 26–52%) | 0.781 |
| Calibration error (ECE) | 0.372 | 0.281 | 0.315 |
| Concerning lesions flagged (ACK, BCC, MEL, SCC) | 46% | 44% | — |
| Benign lesions not flagged (NEV, SEK) | 89% | 84% | — |
| SCC flagged as concerning | 47% | 46% | — |

Per class (sensitivity, v1 / v2): actinic keratosis 8% / 2% (730 images), basal cell carcinoma 43% / 43% (845), melanoma 8% / 38% (52), nevus 71% / 46% (244), seborrheic keratosis 17% / 27% (235).

**What this shows.** Performance falls from well above chance to only a little above it when the camera changes from a dermatoscope to a phone. That is the domain shift the roadmap warned about, now measured. v2's melanoma tuning still helps relatively (5× more melanomas caught than v1), but 38% is far too low to rely on. Actinic keratoses, the largest group here, are almost never recognised. Most missed melanomas were called nevi.

**Skin type.** Accuracy by Fitzpatrick type (mapped classes, v1 / v2): I–II 36% / 36% (887 images), III–IV 35% / 33% (406), V–VI 22% / 11% (only 9 images: too few to conclude anything), not recorded 21% / 16% (804). This dataset cannot answer the fairness question for darker skin.

**Caveats.** The class mix is very different from HAM10000 (mostly BCC and ACK), labels are partly clinical rather than biopsy-proven, and the class mapping is approximate (HAM10000's "benign keratosis-like" is broader than seborrheic keratosis). The numbers describe these two models on this dataset only.

## Results — v3 (2026-10-09): dermoscopy + smartphone photos

**What changed:** v2's weights, fine-tuned for up to 8 epochs (learning rate 1e-4, balanced sampling, cross-entropy) on PAD-UFES-20 smartphone photos plus a class-stratified replay of 2,999 HAM10000 training images, so dermoscopy is not forgotten. PAD-UFES-20 was split **by patient** (60/20/20, stratified, seed 42; 1,397 / 441 / 460 images from 823 / 277 / 273 patients; the split is in [`docs/results/pad_ufes_20_split/`](docs/results/pad_ufes_20_split/)). Invasive SCC was trained as akiec, its nearest DermaLens class. The epoch (7 of 8) was chosen on **validation only**, by the mean balanced accuracy of the dermoscopy and phone validation splits; calibration used the same method as v2 on both validation splits together. Each test split was then scored once. Run on a GitHub Actions CPU runner (`.github/workflows/train-v3.yml`, run 38000061783) with every download hash-checked.

**Weights SHA-256:** `e0cf8e628dd81a5780e0bc5338c56908de47beec2a877fbbab02d0782fe8ecb2`

**Smartphone photos — PAD-UFES-20 test patients** (423 mapped images + 37 SCC; v1 and v2 scored on the same images; full output in [`docs/results/v3_pad_ufes_20_test.json`](docs/results/v3_pad_ufes_20_test.json)):

| | v1 | v2 | v3 |
|---|---|---|---|
| Accuracy | 0.314 | 0.298 | **0.714** |
| Balanced accuracy (5 classes; chance = 0.20) | 0.322 | 0.363 | **0.654** |
| Melanoma sensitivity | 1/13 | 7/13 | **11/13** (95% CI 58–96%) |
| Calibration error (ECE) | 0.341 | 0.237 | **0.050** |
| Concerning lesions flagged (ACK, BCC, MEL, SCC) | 47% | 46% | **97%** |
| Benign lesions not flagged (NEV, SEK) | 87% | 85% | 66% |
| SCC flagged as concerning | 38% | 41% | 97% |

Per class on phone photos (v3 sensitivity): actinic keratosis 78%, basal cell carcinoma 80%, melanoma 85%, seborrheic keratosis 73%, **nevus 11% (5/45)**: of 45 moles, 20 were called benign keratosis and 19 were flagged as concerning (8 melanoma, 8 basal cell carcinoma, 3 actinic keratosis).

**Dermoscopy — HAM10000 test split** (the same 1,494 images as v1 and v2):

| | v2 | v3 |
|---|---|---|
| Accuracy | **0.785** | 0.718 |
| Balanced accuracy | **0.733** | 0.662 |
| Melanoma sensitivity | 0.781 | 0.797 |
| Macro one-vs-rest ROC-AUC | 0.902 | **0.919** |
| Calibration error (ECE) | 0.315 | **0.111** |

Per class (sensitivity, v2 → v3): akiec 0.46 → 0.32, bcc 0.76 → 0.68, bkl 0.63 → 0.68, df 0.71 → 0.57, mel 0.78 → 0.80, nv 0.83 → 0.73, vasc 0.95 → 0.86.

**Trade-offs.** v3 turns smartphone photos from near-chance into useful screening-style output: almost every concerning lesion is flagged and its scores are well calibrated. The cost is false alarms on moles, and a 7-point drop in dermoscopy balanced accuracy (its ranking, ROC-AUC, slightly improved and its calibration improved a lot). The phone test patients come from the **same clinic** as the training photos, so this is not an external test: other phones, clinics and populations may do worse. Only one test image was Fitzpatrick V–VI, so skin-tone fairness remains untested. Calibration again raised validation NLL (0.58 → 0.99) while improving balanced accuracy, the probabilities-versus-decisions problem the roadmap describes.

**How it is served.** The API sends **regular photos to v3** and **dermoscopy to v2** (`PHOTO_MODEL`, default `v3`); the second opinion compares the two. v1 is no longer served (memory on the free instance) but stays published as `model-v1`.

## Uncertainty: every headline number with a 95% interval (2026-10-17)

The frozen v1, v2 and v3 were scored again on the fixed test splits (`.github/workflows/evaluate-models.yml`, run 39050967706; nothing trained or tuned). Every point estimate reproduced the published one exactly, and the rebuilt PAD-UFES-20 split matched the published split image for image. Intervals are percentile bootstraps that resample **whole lesions** (HAM10000) or **whole patients** (PAD-UFES-20), because several images of one lesion are not independent evidence. Files: [`docs/results/rescored/`](docs/results/rescored/) (per-image probabilities included, so later analyses never re-score a test split).

**Dermoscopy — HAM10000 test (1,494 images):**

| | Balanced accuracy | Melanoma sensitivity | Calibration error |
|---|---|---|---|
| v1 | 0.741 (0.668–0.816) | 0.642 (0.542–0.727) | 0.052 (0.037–0.075) |
| v2 | 0.733 (0.674–0.797) | 0.781 (0.696–0.857) | 0.315 (0.292–0.338) |
| v3 | 0.662 (0.615–0.740) | 0.797 (0.720–0.861) | 0.111 (0.091–0.139) |

**Paired differences** (same lesions resampled for both models, 2,000 resamples):

| | Difference | 95% interval | Verdict |
|---|---|---|---|
| Melanoma sensitivity, v2 − v1 | +0.139 | +0.052 to +0.233 | real gain |
| Balanced accuracy, v2 − v1 | −0.008 | −0.087 to +0.105 | no detectable difference |
| Balanced accuracy, v3 − v2 | −0.071 | −0.124 to −0.027 | real loss |
| Melanoma sensitivity, v3 − v2 | +0.016 | −0.046 to +0.078 | no detectable difference |

**Smartphone photos — PAD-UFES-20 test patients (423 mapped images, 273 patients):**

| | Balanced accuracy | Melanoma sensitivity (13 images) | Concerning flagged | Benign not flagged |
|---|---|---|---|---|
| v1 | 0.322 (0.272–0.381) | 0.077 (0.000–0.300) | 0.466 (0.409–0.523) | 0.871 (0.793–0.939) |
| v2 | 0.363 (0.297–0.433) | 0.538 (0.286–0.778) | 0.463 (0.407–0.519) | 0.849 (0.765–0.925) |
| v3 | 0.654 (0.601–0.706) | 0.846 (0.636–1.000) | 0.970 (0.949–0.986) | 0.656 (0.559–0.748) |

**Reading these.** v2's extra melanoma catches are real; its overall score is indistinguishable from v1's. v3's dermoscopy loss is real but modest, and its phone-photo gain is large: the intervals do not overlap. With only 13 test melanomas on phone photos, v3's melanoma figure could be anywhere from about 64% to 100%.

## Outside dermoscopy test: BCN20000, Hospital Clinic Barcelona (2026-10-21)

The published v1, v2 and v3 were scored **unchanged** on BCN20000 (ISIC Archive collection 249, CC BY-NC): dermoscopy from a different hospital and country than HAM10000, 76% confirmed by histopathology, never used for training. Of 18,946 images, 16,843 (4,752 lesions) map onto DermaLens classes; 1,156 without a diagnosis, 633 melanoma metastases and 314 scars were left out. Intervals resample whole lesions (no patient ids are published). Run: `.github/workflows/external-bcn.yml`, run 39658009522; files in [`docs/results/bcn20000/`](docs/results/bcn20000/), per-image probabilities included.

| | v1 | v2 | v3 | for comparison: v2 on HAM10000 test |
|---|---|---|---|---|
| Balanced accuracy (7 classes) | 0.417 (0.39–0.45) | 0.486 (0.46–0.51) | 0.468 (0.44–0.50) | 0.733 |
| Melanoma sensitivity (4,003 images) | 0.328 (0.30–0.36) | 0.498 (0.47–0.53) | 0.566 (0.54–0.60) | 0.781 |
| Concerning lesions flagged | 0.471 | 0.447 | 0.573 | — |
| Benign lesions not flagged | 0.871 | 0.835 | 0.665 | — |
| Calibration error (ECE) | 0.222 | 0.150 | 0.182 | 0.315 |

**Paired v3 − v2 on BCN20000** (16,284 mapped images, 4,591 lesions, `ml/compare_models.py`): balanced accuracy −0.018 (95% interval −0.037 to −0.002, a real but small loss); melanoma sensitivity +0.068 (+0.047 to +0.088, a real gain). File: [`docs/results/bcn20000/paired_v3_vs_v2.json`](docs/results/bcn20000/paired_v3_vs_v2.json).

**What this shows.** Moving to another hospital's dermoscopy cuts balanced accuracy from about 0.73 to below 0.50 and melanoma sensitivity from 0.78 to 0.50 for v2. The HAM10000 numbers describe HAM10000-like images, not dermoscopy in general. Actinic keratoses (8–20% recognised) and basal cell carcinomas (32–44%) suffer most. This is the most important limitation found so far.

## v4: pre-registered result (2026-10-21)

v4 followed the pre-registration in `docs/ROADMAP.md` exactly: v2 fine-tuned for up to 15 epochs (epoch 8 chosen on validation) on PAD-UFES-20 train plus a 5,000-image HAM10000 replay; temperature-only probabilities (T = 0.9, no per-class offsets); a "needs a closer look" flag when P(akiec) + P(bcc) + P(mel) ≥ 0.148, the highest threshold keeping ≥ 90% concerning-lesion sensitivity on both validation domains. Run 39656257825; files in [`docs/results/v4/`](docs/results/v4/). Not published as a release and not served.

| PAD-UFES-20 test patients (phone) | v3 | v4 | ablation: v3 weights + v4 decisions |
|---|---|---|---|
| Balanced accuracy | 0.654 (0.60–0.71) | **0.738** (0.68–0.79) | 0.706 (0.64–0.77) |
| Nevus recognised | 0.11 | **0.51** | 0.42 |
| Calibration error | 0.050 | **0.029** | 0.056 |
| Flag: concerning caught | 0.970 | **1.000** | 0.995 |
| Flag: benign not flagged | **0.656** | 0.430 (0.32–0.54) | 0.527 |

| HAM10000 test (dermoscopy) | v2 | v4 |
|---|---|---|
| Balanced accuracy | 0.733 (0.67–0.80) | 0.695 (0.63–0.79) |
| Accuracy | 0.785 | **0.828** |
| Melanoma as top class | **0.781** | 0.642 |
| Flag: concerning caught | — | 0.877 (0.83–0.92) |
| Flag: benign not flagged | — | 0.812 (0.78–0.84) |
| Calibration error | 0.315 | **0.037** |
| ROC-AUC | 0.902 | **0.940** |

**Verdict against the pre-registered criteria.**
- *Phone photos:* criterion 1 met (flag catches 100%), **criterion 2 failed**: v4 leaves 43% of benign lesions unflagged against v3's 65.6%. **v3 stays** for regular photos.
- *Dermoscopy:* the flag catches 87.7% (criterion met); the balanced-accuracy criterion needs a **paired** comparison with v2 that this run did not save per-image predictions for, so it is undecided. **v2 stays** until it is run.

**Why it failed, and what is learned.** One flag threshold had to reach 90% sensitivity on dermoscopy validation, which on phone validation flagged 99.7% of concerning lesions but only 38% of benign ones were left alone. The ablation shows the two changes separately: the decision layer alone raised phone balanced accuracy (0.654 → 0.706) and fixed calibration on dermoscopy; longer training added more (→ 0.738). Removing per-class offsets also means the *top class* says "melanoma" less often (0.64 on dermoscopy), and the flag, not the top class, is now what catches concerning lesions. The next step (v4.1) is pre-registered in `docs/ROADMAP.md` before any v4.1 result.

## v4.1: pre-registered result, and a safety veto (2026-10-22)

v4.1 = v4's weights and temperature with **one flag threshold per photo type** (dermoscopy 0.148, phone 0.882), each the highest keeping ≥ 90% concerning-lesion sensitivity on that domain's validation split. No retraining. Run 39689803022 (`.github/workflows/v4-1.yml`; v4's weights checked against the committed SHA-256). Files: [`docs/results/v4.1/`](docs/results/v4.1/).

| Pre-registered criterion | Needed | v4.1 | |
|---|---|---|---|
| Phone: flag catches concerning lesions | ≥ 0.85 | 0.891 (0.85–0.92) | met |
| Phone: benign left unflagged | > 0.656 (v3) | **0.903** (0.84–0.96) | met |
| Dermoscopy: paired balanced accuracy v4.1 − v2 | interval includes 0 or above | −0.038 (−0.073 to +0.013) | met |
| Dermoscopy: flag catches concerning lesions | ≥ 0.85 | 0.877 (0.83–0.92) | met |

Also reported, as pre-registered: paired balanced accuracy on BCN20000 v4.1 − v2 = −0.041 (−0.062 to −0.022, a real loss); on PAD-UFES-20 v4.1 − v3 = +0.084 (+0.023 to +0.151, a real gain).

**Melanoma check (not in the criteria; added after seeing the results, and labelled as such).** How often each model's own rule would send a **melanoma** for a closer look:

| | HAM10000 (187) | BCN20000 (4,003) | PAD-UFES-20 phone (13) |
|---|---|---|---|
| v2 (top class) | 0.781 (0.70–0.86) | 0.498 (0.47–0.53) | 7/13 |
| v3 (top class) | 0.797 | 0.566 (0.54–0.60) | **11/13** |
| v4.1 (flag) | **0.856** (0.78–0.92) | **0.656** (0.63–0.68) | **5/13** (0.11–0.64) |

**Decision.**
- **Dermoscopy: v4.1 replaces v2** as the pre-registration says. Its flag also catches more melanomas than v2 at both hospitals, and its percentages are calibrated (ECE 0.037 vs 0.315). The cost, stated plainly: lower balanced accuracy at the Barcelona hospital (−0.041) and fewer melanomas named as the top class; the flag, not the top class, is now what sends a lesion for a closer look.
- **Regular photos: v3 stays, overriding the pre-registered criteria.** v4.1 met them by catching the many basal cell carcinomas and actinic keratoses in the phone test set, while its phone threshold sent only 5 of 13 melanomas for a closer look, against 11 of 13 for v3. The sample is small (interval 11–64%), but the risk is one-sided. This is a deviation from the pre-registration, made for safety and recorded here.
- **Lesson for every future pre-registration:** gate on each malignant class separately (melanoma, basal cell carcinoma, actinic keratosis / SCC), not only on "concerning lesions" pooled.

## v4.2: pre-registered result (2026-10-23): failed

v4.2 = v4's weights with a threshold **per concerning class and photo type** (a lesion is flagged if P(mel), P(bcc) or P(akiec) reaches its own threshold, each keeping ≥ 90% of that class on validation). Run 39912394996; files in [`docs/results/v4.2/`](docs/results/v4.2/).

| | Phone (PAD-UFES-20) | Dermoscopy (HAM10000) | Dermoscopy (BCN20000, reported) |
|---|---|---|---|
| Melanomas flagged | **13/13** (gate: ≥ 11/13, met) | 0.941 (0.89–0.98) | 0.935 |
| Basal cell carcinomas flagged | 1.000 | 0.985 | 0.986 |
| Actinic keratoses (+ SCC) flagged | 1.000 / 0.973 | 0.984 | 0.990 / 0.987 |
| Benign left unflagged | **0.301** (gate: > 0.656, **failed**) | **0.468** (gate: ≥ 0.782, **failed**) | 0.243 |

**Verdict:** fails on both photo types. Requiring each of three classes to keep 90% sensitivity, and flagging when *any* fires, piles up false alarms: on phone photos 70% of benign lesions would be sent for a closer look.

**What v4 → v4.2 shows together.** With these weights, moving thresholds only slides along one trade-off curve: v4.1 (pooled, per photo type) gives few false alarms but misses phone melanomas; v4.2 (per class) catches every cancer but flags most benign lesions; v3's top-class rule sits between. No threshold choice met both goals, so the next gain has to come from the model itself (more varied training data, especially phone photos and other hospitals), not from decision rules. **Serving is unchanged by v4.2:** v3 for regular photos; v4.1 for dermoscopy once approved, otherwise v2.

## Not clinically ready

DermaLens is a research and education project. Before any clinical use it would still need, at minimum: external validation on independent datasets and devices, review by dermatologists, prospective testing, evaluation across skin tones, privacy and security work, and regulatory review. None of these has been done.

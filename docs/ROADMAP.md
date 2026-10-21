# DermaLens roadmap

DermaLens is a four-year research project (2026–2030). Each stage below is meant to produce a result you can measure and explain, including results that turn out worse than hoped.

## Four-year plan

Each year ends with something finished and checkable: a released model, a measured result, or a written report.

| Year | Theme | Finished by the end of the year |
|---|---|---|
| **1 (2026–27)** | Build it and test it honestly | Live site; v1 → v3; phone-photo test; confidence intervals on every number; outside dermoscopy test at a second hospital (BCN20000: balanced accuracy falls below 0.50) — all done. v4 trained and failed its pre-registered photo criterion; **v4.1** (per-photo-type flag) is next. |
| **2 (2027–28)** | Make it trustworthy | An **"I don't know" answer**: the model declines non-skin images and low-confidence cases, with a measured trade-off between how often it answers and how accurate it is. A **photo-quality check** (blur, lighting, framing) before analysis: first version live 2026-10-21 (in-browser brightness, glare and sharpness heuristics; next, thresholds set from the training-data distributions and checked against model errors). A test of whether the heatmap actually lands on the lesion, using expert lesion outlines. |
| **3 (2028–29)** | Fairness and experts | **Skin-tone results** on datasets that cover darker skin, the gap reported and, if possible, reduced. **Feedback from dermatologists** on the wording, the failure cases and the second opinion, recorded with what changed. |
| **4 (2029–30)** | Publish | A **research paper** (methods, every result with its uncertainty, limitations) submitted to a student journal or science fair; a frozen, reproducible release of code, models and splits. |

Rules for every year: datasets only with a licence that allows it; any study with people goes through a mentor and an ethics review first; DermaLens stays a research tool, never a diagnosis.

## Done

- **Explainable research app:** probability distribution, Grad-CAM, stress test, research export, honest evidence page. Security-hardened API.
- **v1:** EfficientNet-B0 baseline. Balanced accuracy 0.741, melanoma sensitivity 0.64, well calibrated (ECE 0.05).
- **v2:** validation-tuned for melanoma. Melanoma sensitivity 0.78, but more false alarms, lower sensitivity for two other cancers, and poor calibration (ECE 0.32). Reproducible: two runs gave byte-identical weights.
- **Both versions served by the API** (`?model=v1` / `v2`); the website uses v2. A version switch or side-by-side comparison in the interface is a possible next step.

## Next stages, in order

### 1. v3: phone photos — done (2026-10-09); v4: fix the trade-offs

v3 (v2 fine-tuned on PAD-UFES-20 + dermoscopy replay) made phone photos usable: balanced accuracy 0.36 → 0.65 on held-out patients, 97% of concerning lesions flagged. It also raised false alarms on moles and lowered dermoscopy balanced accuracy to 0.66. The items below carry over to v4:

- Keep **probabilities** and **decisions** separate. Calibrate probabilities by validation likelihood (temperature or vector scaling), then choose decision thresholds separately.
- Protect **every malignant or pre-malignant class** (melanoma, basal cell carcinoma, actinic keratosis / intraepithelial carcinoma) with sensitivity floors, not just melanoma.
- ~~Report **bootstrap confidence intervals** for every test metric.~~ Done 2026-10-17 (lesion/patient-grouped, with paired model comparisons; see `MODEL_CARD.md`). Rare classes (dermatofibroma, vascular) still have very few test images.
- Be explicit that v3 was designed after seeing v2's test results. That makes stage 2 essential.

### 2. External validation (the most important credibility step) — first run done

Test v1, v2 and v3, unchanged, on a dataset the models never saw, from different clinics and cameras. Candidates: ISIC challenge datasets, or PAD-UFES-20 (smartphone images, a harder domain shift). **Check each dataset's licence and terms before downloading.** Expect performance to drop, and report by how much. That drop is the honest answer to "would this work elsewhere?"

**Done for PAD-UFES-20 (2026-10-08):** balanced accuracy fell from 0.73 on dermoscopy to 0.29 (v1) and 0.31 (v2) on smartphone photos (chance is 0.20); v2 caught 38% of melanomas. Details in `MODEL_CARD.md`. Still to do: a second dermoscopy dataset from another clinic (for example an ISIC challenge set), to separate "different camera" from "different clinic", and bootstrap intervals for every metric.

### 3. Does the attention look at the lesion?

HAM10000 also publishes lesion segmentation masks (`HAM10000_segmentations_lesion_tschandl`). Measure how much Grad-CAM attention falls inside the lesion versus on skin, rulers or ink markings, overall and for correct versus wrong predictions. This turns the app's central idea ("see what the model sees") into a measured result.

### 4. Knowing when to say "I don't know"

- **Selective prediction:** let the model abstain when uncertainty is high. Report coverage against accuracy (for example, "answering 70% of images, balanced accuracy rises to X").
- **Out-of-scope detection:** flag inputs that aren't dermatoscopic images at all. Today the model always picks one of seven classes.

### 5. Skin-tone fairness

HAM10000 has no skin-tone labels and comes mostly from lighter-skinned populations. Evaluate on a dataset with Fitzpatrick skin-type labels (for example Fitzpatrick17k or Stanford's Diverse Dermatology Images; both have access terms) and report performance by skin type. This is a known, serious gap in dermatology AI and worth stating plainly even if the results are uncomfortable.

### 6. Expert feedback

Ask a dermatologist or a medical-imaging researcher to review the interface wording, the failure cases and the limitations. Record what they said and what changed.

### 7. Write it up

A short research report in paper format (abstract, methods, results with confidence intervals, limitations), plus a poster and the demo video. Possible venues: a preprint, a student research journal, or a science fair.

## v4 pre-registration (written 2026-10-21, before any v4 result)

**Change.** v4 separates probabilities from decisions. Probabilities: temperature scaling only, no per-class offsets. Decision: "needs a closer look" when P(akiec) + P(bcc) + P(mel) reaches a threshold, set on validation as the highest value that keeps concerning-lesion sensitivity at or above 90% on the dermoscopy and the phone validation splits separately. Training: v2's weights fine-tuned on PAD-UFES-20 train plus a 5,000-image HAM10000 replay, up to 15 epochs (patience 4), epoch chosen on the mean validation balanced accuracy of the two domains (`.github/workflows/train-v4.yml`).

**Ablation, also fixed in advance.** v3's weights with v4's decision layer, to separate the effect of the decision layer from the effect of longer training.

**v4 replaces v3 for regular photos only if, on the PAD-UFES-20 test patients:**
1. its flag still catches at least 85% of concerning lesions (lower 95% bound reported, not used as a gate), and
2. its flag leaves more benign lesions unflagged than v3 did (v3: 65.6%).

**v4 replaces v2 for dermoscopy only if, on the HAM10000 test split,** its balanced accuracy is not clearly worse than v2's (the paired 95% interval of v4 − v2 includes 0 or is above it) **and** its flag catches at least 85% of concerning lesions. Otherwise v2 stays. Whatever the outcome, every number is reported.

**Outcome (2026-10-21).** Phone criterion 2 failed (benign not flagged 43.0% vs v3's 65.6%); dermoscopy undecided (paired comparison not yet run). Neither v2 nor v3 is replaced. Details in `MODEL_CARD.md`.

## v4.1 pre-registration (written 2026-10-21, before any v4.1 result)

**Change, and only this change:** v4's weights and temperature, with **one flag threshold per photo type**, each the highest that keeps ≥ 90% concerning-lesion sensitivity on that domain's validation split. The API applies the threshold for the photo type the visitor chose. No retraining.

**v4.1 replaces v3 for regular photos only if,** on the PAD-UFES-20 test patients, its flag catches ≥ 85% of concerning lesions **and** leaves more benign lesions unflagged than v3 (65.6%).

**v4.1 replaces v2 for dermoscopy only if,** on the HAM10000 test split, the paired 95% interval of balanced accuracy v4.1 − v2 includes 0 or lies above it, **and** its flag catches ≥ 85% of concerning lesions. The same comparison is reported on BCN20000.

## Maintenance (held upgrades)

These are planned by hand rather than merged from Dependabot:

- **Next.js 16 and TypeScript 7** (major releases). Upgrade on a branch, run the full check list, and compare the live site before merging.
- **Python 3.12 for the API image.** numpy 2.3 and later need it; until then the backend stays on numpy 2.2.
- **ml/ package versions** stay at the exact versions v1 and v2 were trained with. Move them together when training v3, and record the new versions in that model's card.

## Principles that stay fixed

- The test split is scored once per model; every choice is made on validation data.
- Every published number is tied to the exact model file (SHA-256).
- Results that get worse are reported, not hidden.
- DermaLens stays an education and research tool, not a diagnostic device.

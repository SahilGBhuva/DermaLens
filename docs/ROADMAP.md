# DermaLens roadmap

DermaLens is a long-term research project. Each stage below is meant to produce a result you can measure and explain, including results that turn out worse than hoped.

## Done

- **Explainable research app:** probability distribution, Grad-CAM, stress test, research export, honest evidence page. Security-hardened API.
- **v1:** EfficientNet-B0 baseline. Balanced accuracy 0.741, melanoma sensitivity 0.64, well calibrated (ECE 0.05).
- **v2:** validation-tuned for melanoma. Melanoma sensitivity 0.78, but more false alarms, lower sensitivity for two other cancers, and poor calibration (ECE 0.32). Reproducible: two runs gave byte-identical weights.
- **Both versions served by the API** (`?model=v1` / `v2`); the website uses v2. A version switch or side-by-side comparison in the interface is a possible next step.

## Next stages, in order

### 1. v3: fix v2's trade-offs

- Keep **probabilities** and **decisions** separate. Calibrate probabilities by validation likelihood (temperature or vector scaling), then choose decision thresholds separately.
- Protect **every malignant or pre-malignant class** (melanoma, basal cell carcinoma, actinic keratosis / intraepithelial carcinoma) with sensitivity floors, not just melanoma.
- Report **bootstrap confidence intervals** for every test metric. Rare classes (dermatofibroma, vascular) have very few test images.
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

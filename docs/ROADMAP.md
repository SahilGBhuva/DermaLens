# DermaLens roadmap

DermaLens is a long-term research project. Each stage below is meant to produce a result you can measure and explain, including results that turn out worse than hoped.

## Done

- **Explainable research app:** probability distribution, Grad-CAM, stress test, illustrative 3D attention view, honest evidence page. Security-hardened and accessibility-audited.
- **v1:** EfficientNet-B0 baseline. Balanced accuracy 0.741, melanoma sensitivity 0.64, well calibrated (ECE 0.05).
- **v2:** validation-tuned for melanoma. Melanoma sensitivity 0.78, but more false alarms, lower sensitivity for two other cancers, and poor calibration (ECE 0.32). Reproducible: two runs gave byte-identical weights.
- **Both versions served side by side**, with in-app comparison on the same image.

## Next stages, in order

### 1. v3: fix v2's trade-offs

- Keep **probabilities** and **decisions** separate. Calibrate probabilities by validation likelihood (temperature or vector scaling), then choose decision thresholds separately.
- Protect **every malignant or pre-malignant class** (melanoma, basal cell carcinoma, actinic keratosis / intraepithelial carcinoma) with sensitivity floors, not just melanoma.
- Report **bootstrap confidence intervals** for every test metric. Rare classes (dermatofibroma, vascular) have very few test images.
- Be explicit that v3 was designed after seeing v2's test results. That makes stage 2 essential.

### 2. External validation (the most important credibility step)

Test v1, v2 and v3, unchanged, on a dataset the models never saw, from different clinics and cameras. Candidates: ISIC challenge datasets, or PAD-UFES-20 (smartphone images, a harder domain shift). **Check each dataset's licence and terms before downloading.** Expect performance to drop, and report by how much. That drop is the honest answer to "would this work elsewhere?"

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

## Principles that stay fixed

- The test split is scored once per model; every choice is made on validation data.
- Every published number is tied to the exact model file (SHA-256).
- Results that get worse are reported, not hidden.
- DermaLens stays an education and research tool, not a diagnostic device.

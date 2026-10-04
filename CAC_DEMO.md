# DermaLens — CAC Demo Plan

This plan is designed around the 2026 Congressional App Challenge's stated judging areas: quality/originality of the idea, implementation and user experience, and demonstrated coding/programming skill.

## Core story

DermaLens is not presented as a replacement for a dermatologist. It is an explainable medical-image machine-learning research app that asks a harder question:

> When an image classifier gives a medical-image prediction, can a user see why it responded that way, and whether the answer is stable?

## Three-minute demo structure

Record in one take if you can; keep the browser at full screen and zoomed so text is readable.

### 0:00–0:20 — Problem (hero)

Most image classifiers return one confident-looking answer. That hides what influenced the network and whether a slightly worse photo would change the result.

On screen: the hero scope. Click **Original → Attention → Contour** while you talk. Point out the "Illustrative output" label — this part explains the idea; real results come later.

### 0:20–0:45 — The idea in 3D

Scroll to **Explore attention in 3D**. Drag to rotate, switch **Attention colors / Skin**, move the **Relief** slider.

Say: "Grad-CAM shows which regions pushed the model's score. Here it's raised into terrain — the peak is where the model looked hardest." Mention it's an illustrative lesion.

### 0:45–1:35 — Real analysis (sandbox)

1. Upload an image you are allowed to use (or the synthetic sample — say so).
2. Read the **full distribution**, not just the top class.
3. Toggle **Original / Grad-CAM** on the real overlay.
4. Read the **How to read this** box aloud: score band, close calls, and the model's real melanoma miss rate.
5. Say clearly: "This is not a diagnosis."

### 1:35–2:05 — Robustness

Click **Run stress test**. The same image is re-scored darker, brighter, lower contrast and blurred. If the top class changes, say: "This is the kind of fragility the app is built to expose."

(The separate **Robustness lab** section uses simulated numbers to explain the idea — use it only to explain, not as evidence.)

### 2:05–2:45 — Evidence and what you improved

Scroll to **Evidence**:

- The headline scores and the **held-out test set size**.
- **Where the mistakes go**: point at the melanoma row — "most missed melanomas were called ordinary moles."
- **How training went**: point at the gap between training and validation — "that's overfitting; I keep the epoch that did best on validation."
- Your improvement story: v1 caught 64% of test melanomas. v2 added stronger augmentation, a second training recipe, and validation-only tuning with a melanoma-sensitivity target — then scored the test set once. State v2's real numbers and the trade-off (more false alarms on benign lesions).

### 2:45–3:00 — Responsible ML and close

"DermaLens refuses to show made-up numbers, ties every metric to the exact model file it serves, and says what it hasn't proven: other cameras, clinics and skin tones."

End with:

> Good medical AI should not only make a prediction. It should help us understand when that prediction may be fragile.

## Technical depth (for Q&A or a second video)

- EfficientNet-B0 transfer learning; lesion-level train / validation / test split with a leakage check
- Class imbalance: weighted loss, focal loss, balanced sampling — chosen on validation
- Validation-only calibration (temperature scaling, per-class offsets with a melanoma floor); test scored once
- Balanced accuracy, macro F1, per-class sensitivity/specificity, ROC-AUC, calibration error
- Grad-CAM with PyTorch hooks; flip-averaged predictions
- FastAPI backend with upload hardening, SHA-256 model integrity checks, memory-tested for a 512 MB server
- Next.js + three.js frontend; Lighthouse 98 / 100 / 100 / 100
- Automated tests in GitHub Actions

## Before submission

- [ ] Train v2 in Colab and install it (`python ml/install_model.py --from ~/Downloads --version v2`).
- [ ] Do not re-run training to chase a better test score — report what the single test evaluation says.
- [ ] Deploy (see `DEPLOY.md`) and test the live link from a clean browser and a phone.
- [ ] Open the live site a minute before judges/recording so the free API is awake.
- [ ] Record the demo with an image you are allowed to use.
- [ ] Practise the questions in `docs/STUDY_GUIDE.md` out loud.
- [ ] Make sure every team member can explain the main code paths and ML decisions.
- [ ] Clearly disclose any AI assistance used while building the project.
- [ ] Credit the HAM10000 dataset (Tschandl, Rosendahl & Kittler, 2018).

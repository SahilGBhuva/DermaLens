# DermaLens study guide

Read this until you can answer every question **in your own words**, without looking. Judges and interviewers care far more about what you understand than about who typed the code.

---

## 1. The big picture

**What is DermaLens?**
A research website that shows how an AI model looks at photos of skin spots (dermatoscopic images) and how far you can trust its answer. You upload a photo; it shows the model's score for all 7 lesion types, where in the image the model "looked", and whether the answer changes if the photo is darker, brighter or blurrier.

**What problem does it address?**
Most AI demos show one confident answer. That hides two important things: *why* the model said it, and *how fragile* the answer is. DermaLens makes both visible.

**Is it a diagnosis tool?**
No. It's an education and research project. It misses about 1 in 3 melanomas in testing, has never been tested on other cameras or clinics, and hasn't been reviewed by doctors or regulators.

> One-sentence pitch: *"DermaLens shows not just what a skin-lesion AI predicts, but why, and when that prediction breaks."*

---

## 2. The data

**What data did you use?**
HAM10000: 10,015 dermatoscope images of 7 lesion types, labelled by experts, published by Tschandl, Rosendahl and Kittler (2018). It's free for non-commercial use with credit.

**The 7 classes:** melanocytic nevus (common mole), melanoma, benign keratosis, basal cell carcinoma, actinic keratosis, vascular lesion, dermatofibroma.

**What's tricky about it?**
- **Imbalance:** about two-thirds of the images are ordinary moles. A model that always said "mole" would be about 67% "accurate" and completely useless.
- **Same lesion, several photos:** some lesions were photographed more than once.

**How did you split it, and why that way?**
70% training, 15% validation, 15% test, **split by lesion, not by photo**. If two photos of the same spot ended up in training and test, the model could "recognise" the spot instead of learning, and the test score would be inflated. The pipeline checks that no lesion appears in two splits.

**What is each split for?**
- **Training (7,002 images):** the model learns from these.
- **Validation (1,519):** used to make choices: which recipe, which epoch, how to tune.
- **Test (1,494):** locked away and scored **once**, at the end. If you made choices using the test set, its score would stop being an honest estimate.

---

## 3. The model

**What model is it?**
EfficientNet-B0, a convolutional neural network. It started from weights pre-trained on ImageNet (millions of everyday photos), then was fine-tuned on skin images. This is called **transfer learning**: it reuses general visual features like edges and textures.

**Why not train from scratch?**
10,000 images isn't enough to learn vision from zero. Transfer learning gets much better results with less data and compute.

**How did you handle the imbalance?**
- v1: **class-weighted loss**. Mistakes on rare classes cost more.
- v2 also tries **focal loss** (focus on hard examples) with **balanced sampling** (rare classes shown as often as common ones), and keeps whichever does better on validation.

**What is data augmentation?**
Randomly changing training images (crops, rotations, flips, lighting, slight blur) so the model learns the lesion, not the exact photo. It also prepares it for the darker or blurrier photos in the stress test.

---

## 4. Results (v1)

| Metric | Score | What it means |
|---|---|---|
| Accuracy | 80.1% | Share of test images classified correctly. Inflated by all the moles |
| **Balanced accuracy** | **74.1%** | Average accuracy *per class*, so every class counts equally. The fairer number |
| Macro F1 | 64.0% | Balances false alarms and misses, averaged over classes |
| ROC-AUC | 94.4% | How well it *ranks* the right class above the others |
| **Melanoma sensitivity** | **64%** | Of real melanomas, the share it caught. **It missed 36%** |
| Melanoma specificity | 92% | Of non-melanomas, the share it correctly did *not* call melanoma |

**Sensitivity vs. specificity, simply:**
- **Sensitivity** = "of the sick ones, how many did we catch?"
- **Specificity** = "of the healthy ones, how many did we correctly leave alone?"

For something dangerous like melanoma, missing it (low sensitivity) is worse than a false alarm. That's why v2 pushes sensitivity up, knowing specificity will drop.

**What happened in v2? (know this cold)**
v2 tuned the model on the validation set to catch at least 80% of melanomas. On the test set it caught **78%** (up from 64%: 146 of 187 instead of 120). The costs:
- more false alarms (315 images called melanoma, only 146 really were);
- two other cancers were caught less often (actinic keratosis 62% → 46%, basal cell carcinoma 82% → 76%);
- its percentages became unreliable: calibration error rose from 0.05 to 0.32, meaning they were off by about 31 points on average.

The lesson: optimising one number (melanoma sensitivity) moved errors elsewhere. A better approach would protect all the cancer classes, and keep "which answer to give" separate from "how confident to say it is". Saying this clearly is a strength, not a weakness.

**Why can't you trust the dermatofibroma number?**
Only 7 test images. One more right or wrong answer moves it by 14 points.

---

## 5. Explainability

**What is Grad-CAM?**
A technique that highlights which parts of the image pushed the model toward its answer. It looks at the last convolutional layer and how much each region would change the score. Red means high influence, blue means low.

**Does a heatmap on the lesion mean the model is right?**
No. It shows *where the model looked*, not *whether it reasoned correctly*. A model can look at the right place and still be wrong, or look at a ruler or skin marking and be "right" for the wrong reason.

**What does "uncertainty" and "entropy" mean on the site?**
- **Uncertainty** = 1 minus the top score.
- **Entropy** = how spread out the 7 scores are (0 means all on one class, 1 means evenly spread).

Both are model outputs, not the probability that the model is correct.

**What is calibration?**
Making the model's percentages honest. If it says 80% on many images, about 80% of them should be right. v2 fits a "temperature" on the validation set to fix over- or under-confidence.

**What is the stress test?**
The same image is re-scored darker, brighter, with lower contrast, and blurred. If the top class changes, the answer is fragile and you shouldn't trust it, because a slightly worse photo would give a different result.

---

## 6. Honesty and limits (expect these questions)

**Could someone use this to check their own mole?**
They shouldn't, and the site says so on every result. It's trained only on dermatoscope images, not phone photos, and it misses about a third of melanomas.

**What would it take to make it clinically usable?**
Testing on independent datasets and devices (external validation), review by dermatologists, prospective testing on new patients, checking performance across skin tones, privacy and security work, and regulatory approval (e.g. FDA). None of these has been done.

**What about skin tone?**
HAM10000 comes mostly from lighter-skinned populations in Europe and Australia. Performance on darker skin is unknown and could be worse. That's a known, serious gap in dermatology AI.

**What happens if you upload a photo of a cat?**
It still picks one of the 7 classes, because it has no "not a lesion" option. The score is usually low, so the site says "inconclusive", but that isn't a guarantee.

**Why does the site refuse to show made-up numbers?**
Because invented metrics would mislead people. Scores only appear when a real test evaluation exists, and only for the exact model file being served (checked by its SHA-256 fingerprint).

---

## 7. How the app is built

- **Frontend:** Next.js and React website, with a three.js 3D view.
- **Backend:** a FastAPI (Python) server that loads the PyTorch model and returns probabilities, a Grad-CAM image and stress-test results.
- **Training:** Python scripts run in Google Colab on a free GPU.
- **Safety checks:** the model's settings file carries a fingerprint (SHA-256) of the model; the server refuses mismatched files.
- **Tests:** automated tests run on GitHub for every change.

---

## 8. Your part, and AI help (fill this in yourself, honestly)

Write 3–5 sentences about:
- What *you* decided: the idea, the goals, what to prioritise, what to cut.
- What you used AI tools for (writing code, design, explanations).
- What you checked or learned yourself.
- One thing that surprised you or went wrong, and how you handled it. (Ideas: Colab disconnecting; the model missing a third of melanomas.)

Being open about AI assistance is expected. Being vague or misleading about it is what hurts.

---

## 9. Practice questions

Say these out loud, in under 30 seconds each:

1. What does DermaLens do, in one sentence?
2. Why split the data by lesion instead of by photo?
3. Why is balanced accuracy more honest than accuracy here?
4. What is the difference between sensitivity and specificity?
5. v1 missed 36% of melanomas. What did you change in v2, what improved, and what got worse?
6. What does a Grad-CAM heatmap show, and what does it *not* prove?
7. Why did you only use the test set once?
8. Why isn't this ready for doctors to use?
9. What would you do next with more time?
10. What part of this did you do, and what did AI help with?

# Gunshot Audio Classification — Weapon Type Identification from Acoustic Signatures

A neural network pipeline that classifies firearm type from a single audio recording of a gunshot, using MFCC and spectral feature extraction fed into a dense neural network.

**Final leak-free test accuracy: 88.89%** across 8 weapon classes (merged from an original 9), trained on 510 recordings with augmentation applied only inside the training split.

---

## Table of Contents
- [Dataset](#dataset)
- [Methodology Evolution](#methodology-evolution)
- [Comparative Analysis: v1 vs v2 vs v3](#comparative-analysis-v1-vs-v2-vs-v3)
- [Final Model — Detailed Results](#final-model--detailed-results)
- [Key Lesson: The Data Leakage Bug](#key-lesson-the-data-leakage-bug)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Further Scope of Improvement](#further-scope-of-improvement)

---

## Dataset

- **Source:** [Gunshot Audio Dataset](https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset) (Kaggle, emrahaydemr)
- **Original classes (9):** AK-12, AK-47, IMI Desert Eagle, M16, M249, M4, MG-42, MP5, Zastava M92
- **Final classes (8):** M4 and M16 merged into `M4_M16` — both are 5.56×45mm NATO platforms with near-identical gas systems and barrel lengths, and were the single largest confusion pair in early testing.
- **Total original recordings:** 851 `.wav` files (48kHz, stereo)
- **Split:** 510 train / 170 validation / 171 test — split on **original file identity**, before any augmentation, to prevent data leakage (see below).

---

## Methodology Evolution

The model went through three distinct iterations. Each is documented here because the *jump between v2 and v3* is itself the most important finding in this project.

### v1 — Baseline (flat MFCC-mean features, no merge)
- Features: 40 MFCC coefficients, mean-pooled across time (`np.mean(mfccs.T, axis=0)`)
- 9 classes, no augmentation
- Simple 3-layer dense network (128 → 64 → 9), 14.8K params
- **Result: 70.2% test accuracy**, with M4/M16/MP5 as a heavily confused cluster (M4 F1 = 0.17, M16 F1 = 0.29)

### v2 — Richer features + class merge + augmentation (⚠️ leaked)
- Features expanded to 264-dim: MFCC mean/std, delta, delta-delta, spectral contrast, centroid, rolloff, bandwidth, ZCR, RMS (mean+std each)
- M4 + M16 merged → 8 classes
- 3x augmentation (pitch shift, time stretch, additive noise) applied to **all** samples before the train/val/test split
- Larger network (256 → 128 → 64 → 8), 111K params, L2 regularization, label smoothing
- **Result: 98.24% test accuracy** — but this number is **invalid**. Augmentation ran before splitting, so near-duplicate copies of the same source recording ended up in both train and test. The model was partly recognizing perturbed copies of clips it had already seen, not generalizing to new gunshots.
- Confirmed via sample-count math: 851 files × 4 (1 original + 3 augmented) = 3404 total feature vectors, matching the reported total — proof the augmentation loop ran before any split logic.

### v3 — Same pipeline, leak-free split (final model)
- Identical feature set and architecture to v2
- **Fix:** file paths and labels are split into train/val/test *first*. Augmentation is then applied only to the training split; validation and test sets consist exclusively of original, unaugmented recordings the model has never seen any derivative of.
- **Result: 88.89% test accuracy** — this is the trustworthy, reportable number.

---

## Comparative Analysis: v1 vs v2 vs v3

| Metric | v1 (baseline) | v2 (leaked) | v3 (leak-free, final) |
|---|---|---|---|
| Features | 40 (MFCC mean only) | 264 (MFCC+delta+spectral+ZCR+RMS) | 264 (same as v2) |
| Classes | 9 | 8 (M4+M16 merged) | 8 (M4+M16 merged) |
| Augmentation | None | 3x, **pre-split (leaked)** | 3x, **post-split (clean)** |
| Train samples | 510 | 2040 | 2040 |
| Test samples | 171 | 171 (contaminated) | 171 (clean, original only) |
| Model size | 14.8K params | 111K params | 111K params |
| Test Accuracy | 70.2% | 98.2% *(invalid)* | **88.9%** |
| Test Precision | — | 0.9925 | 0.9241 |
| Test Recall | — | 0.9750 | 0.8538 |
| Weakest class | M4 (F1 0.17) | MP5 (F1 0.95, still highest error count) | MP5 (F1 0.73) |
| Train-Val gap | 0.11 (overfitting) | ~0.01 (looked great, was fake) | ~0.09 (real, mild, acceptable) |
| Trustworthy? | Yes, but weak | **No — data leakage** | **Yes** |

**Reading this table honestly:** the jump from v1 to v3 (70.2% → 88.9%) reflects genuine improvement from better features, the M4/M16 merge, and legitimate augmentation. The jump from v2 to v3 (98.2% → 88.9%) is not a regression — it's the removal of an artifact. v2's number was never real; v3's is the first trustworthy measurement of what the pipeline can actually do on unseen audio.

---

## Final Model — Detailed Results (v3)

```
Test Accuracy:  0.8889
Test Precision: 0.9241
Test Recall:    0.8538
Top-2 Accuracy: 0.9649

                  precision    recall  f1-score   support
           AK-12     1.0000    1.0000    1.0000        20
           AK-47     1.0000    0.9286    0.9630        14
IMI Desert Eagle     0.8000    0.8000    0.8000        20
            M249     0.8421    0.8000    0.8205        20
          M4_M16     0.9000    0.9000    0.9000        40
           MG-42     1.0000    0.9000    0.9474        20
             MP5     0.6667    0.8000    0.7273        20
     Zastava M92     1.0000    1.0000    1.0000        17

        accuracy                         0.8889       171
       macro avg     0.9011    0.8911    0.8948       171
```

### Confusion analysis
- **AK-12, Zastava M92:** perfect classification — highly distinctive acoustic signatures.
- **AK-47, MG-42:** near-perfect (F1 ≥ 0.94).
- **M4_M16:** 90% F1 on the largest class (40 test samples) — validates the merge decision.
- **MP5 is the weakest class** (F1 0.73, precision 0.67). False positives into MP5 came from three different weapon families — IMI Desert Eagle (2), M249 (3), M4_M16 (3) — several at *high* confidence (0.84–0.92), which rules out simple model uncertainty as the sole cause. This scatter across acoustically dissimilar classes suggests either recording-condition inconsistency in the original MP5 clips or a genuinely broader/less distinctive acoustic profile for that weapon in this dataset — worth further investigation with a larger MP5 sample size before drawing firm conclusions.

Full plots (confusion matrices, ROC/PR curves, training curves, confidence distributions) are in [`eval_outputs/`](./eval_outputs).

---

## Key Lesson: The Data Leakage Bug

Worth calling out explicitly since it's the most instructive part of this project.

**What happened:** Audio augmentation (pitch shift, time stretch, noise injection) was applied to every recording *before* the train/val/test split. This meant several near-duplicate versions of the same underlying gunshot could land in different splits — the model would then be tested on a clip whose sibling it had already trained on.

**How it was caught:** The test accuracy (98.2%) was suspiciously high for a small, real-world audio dataset. A sample-count sanity check (`851 original files × 4 = 3404`, matching the total feature count exactly) confirmed augmentation had run across the whole dataset as one block, prior to any split.

**The fix:** Split file *paths* (not features) into train/val/test first, stratified by class. Only then load and augment — and only the training split gets augmented. Validation and test always consist of untouched, original recordings.

This is a common and easy-to-miss pitfall in any audio/image pipeline that augments before splitting — worth checking for in any project with unexpectedly high accuracy.

---

## Repository Structure

```
.
├── README.md
├── notebook.ipynb                  # full pipeline: extraction → split → train → eval
├── best_gunshot_model_v3.keras     # final trained model (leak-free)
├── eval_outputs/
│   ├── classification_report.txt
│   ├── classification_report.json
│   ├── metrics_summary.json
│   ├── misclassified_samples.json
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── confidence_analysis.png
│   └── README_metrics_section.md
└── requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
# requirements.txt: tensorflow, librosa, scikit-learn, pandas, numpy, matplotlib, seaborn, kagglehub

jupyter notebook notebook.ipynb
```

Dataset auto-downloads via `kagglehub.dataset_download("emrahaydemr/gunshot-audio-dataset")` on first run.

---

## Further Scope of Improvement

1. **Sequence-aware architecture.** The current pipeline mean/std-pools MFCCs across time, discarding the temporal shape of the acoustic report (attack transient → decay → echo tail). A 1D-CNN or CNN+LSTM operating directly on the MFCC time-series (40 × ~188 frames) would likely recover meaningful signal, particularly for the M4_M16/MP5/M249 confusion cluster, where the transient envelope shape differs even when spectral content overlaps.

2. **Targeted oversampling for MP5.** Rather than uniform 3x augmentation across all classes, apply heavier augmentation specifically to MP5 and IMI Desert Eagle (the two weakest classes) to give the model more varied examples of their decision boundary.

3. **Larger, more diverse recordings per class.** MP5's high-confidence false-positive spread across dissimilar weapon types hints that its ~95 original recordings may have limited acoustic diversity (e.g., single recording environment/distance). Sourcing additional MP5 clips from different ranges/conditions would directly test this hypothesis.

4. **Two-stage hierarchical classification.** Train a coarse classifier (e.g., handgun vs. rifle vs. SMG vs. machine-gun family) followed by a fine-grained classifier within each cluster. This could isolate MP5's cross-family confusion rather than forcing one flat 8-way boundary to separate it from both handguns and rifles simultaneously.

5. **Cross-validation instead of a single split.** With only 510–681 samples, a single train/val/test split (however leak-free) is still subject to sampling variance. K-fold cross-validation would give a more statistically robust accuracy estimate and confidence interval.

6. **Negative/background-noise class.** For real-world deployment (e.g., as part of a wearable acoustic detection system), the model needs a "not a gunshot" rejection class — this dataset only contains positive gunshot examples across weapon types, with no non-gunshot audio to reject.

7. **Raw-waveform or spectrogram-based deep learning.** Moving from hand-engineered MFCC/spectral features to a spectrogram-input CNN (or raw 1D-waveform CNN) could let the model learn discriminative features directly, potentially surfacing acoustic cues that manual feature engineering misses — at the cost of needing more data to avoid overfitting a larger model.

8. **Confidence calibration.** Several misclassifications in this run were high-confidence errors (0.84–0.92) rather than low-confidence uncertain guesses. Applying temperature scaling or other calibration techniques post-training would make the model's confidence scores more trustworthy for downstream decision-making (e.g., flagging low-confidence predictions for human review in a deployed system).

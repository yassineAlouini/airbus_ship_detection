# Airbus Ship Detection: what went wrong in 2018, and a modern solution

This note looks back at the original attempt in this repository (`asd/`). It explains why the attempt scored badly,
then describes the new pipeline in `modern/` that fixes those problems.

## 1. The score: worse than submitting nothing

Scores as reported by the Kaggle CLI. The competition was ranked on the **"public"** column: the old team shows up at
rank 816/879 with 0.51638 on the final leaderboard.

| Submission | Public (ranked) | Private |
|---|---|---|
| Old best: pretrained ResNet34 U-Net (`drop_no_ships_asd_10-11-2018.csv`) | 0.51638 | 0.75929 |
| Old: Mask R-CNN kernel | 0.51299 | 0.75707 |
| **All images predicted empty** (calibration submission, 2026-09-27) | **0.52090** | **0.76566** |
| Leaderboard reference: #1 / top-10% cutoff / median | 0.76444 / ~0.7305 / ~0.700 | |
| **New solution (`modern/`)**, late submission 2026-09-27 | **0.73128** | **0.84920** |

Both old submissions score **below the trivial "no ships anywhere" submission** on both splits. In other words, the old
models added negative value. Five of the eight old submissions did not score at all (status `ERROR`).

## 2. Why the old pipeline failed

The issues are ranked by roughly how much score they cost.

### 2.1 No image-level gate while ~78% of images are empty (the killer)

* 78% of the 192,556 training images contain no ship.
* The metric is computed **per image**. For an empty image, any predicted blob gives an F2 of 0 instead of 1.
* `get_data()` in `asd/preprocessing.py` keeps only images with ships (`ships_df.loc[lambda df: df.has_ship == 1]`).
  The segmentation model therefore never saw an empty sea, cloud or harbour tile, and it learned to find a ship in
  every image.
* The ship / no-ship classifier that was meant to filter these was never wired in. Both submissions that tried it
  (11-11-2018) ended in `ERROR`.

A few false-positive pixels on empty images are enough to push the score below the all-empty baseline, and that is
exactly what the leaderboard shows.

### 2.2 Train and inference resolution did not match

* Training images were downsampled 4x with `c_img[::4, ::4]` (`IMG_SCALING = (4, 4)`), giving 192x192.
* The pretrained `segmentation_models.Unet` used at submission time has no rescaling wrapper, and
  `prepare_submission()` fed it the full **768x768** image.
* The network therefore saw ships four times larger than anything it was trained on.

### 2.3 Small ships were destroyed twice

* Ship areas at full resolution: the 5th percentile is 35 px, the 25th percentile 111 px and the median 408 px.
* After 4x downsampling a 111 px ship becomes about 7 px, and a 35 px ship about 2 px.
* At inference, `binary_opening(pred > 0.5, disk(2))` removes every component narrower than 5 px.
* A large share of real ships were therefore impossible to detect.

### 2.4 Instance handling ignored how the metric works

* The metric matches **individual ships** at IoU thresholds 0.5 to 0.95 and computes F2 per image.
* The old code built instances with plain connected components. Ships moored side by side (common in harbours)
  merged into one instance, which counts as 1 false positive plus several false negatives.

### 2.5 The competition metric was never used for model selection

* Training monitored `val_loss` (BCE + Dice) plus pixel-level IoU, Dice and TPR, not the instance-level F2.
* The `FBetaMetricCallback` was disabled ("doesn't yet work").
* As a result, the mask threshold (0.5), the opening kernel and the minimum size were all hard-coded, with no way to
  know whether they helped.

### 2.6 Pipeline bugs

* **The best checkpoint was never used.** `ModelCheckpoint` saves the best weights to `best_weights.h5`, but
  `run.py` saves the *last* model as `best_model.h5` and predicts with it. With `EARLY_STOPPING_PATIENCE = 30`, that
  model can be up to 30 epochs past the best one.
* **`--debug` defaults to `True`.** Without the flag, only 10 test images are predicted.
* **`ImageId` lost its `.jpg` suffix** (`img_path.split('/')[-1].replace('.jpg', '')`). This caused at least one
  `ERROR` submission ("Add missing .jpg in ImageId column").
* **`IMGS_TO_IGNORE` never matched.** It holds names *with* `.jpg` but is compared to ids *without* it. Had it
  matched, those images would have been missing from the submission, which is another format error.
* **Validation was noisy:**
  * The validation generator applied random augmentation.
  * It ran `validation_steps = steps_per_epoch` (about 1,300 batches) over a 5% split, so every epoch re-validated
    the same few images many times, under different augmentations.
  * `train_test_split` had no `random_state`, and `SEED` is defined twice in `conf.py` (42, then 31415).
* **Mask augmentation used bilinear interpolation.** Masks went through `ImageDataGenerator` rotation, shear and
  zoom and were never re-thresholded, which produces soft, noisy labels.
* **`dice_metric` had a dead guard.** `if union == 0` compares a symbolic tensor in graph mode, so it is always
  False.
* **Prediction was slow.** It ran one image at a time, which made every submission slow to iterate on.
* **README typo:** `unzip test_v2.zip -d train_v2`.

### 2.7 Data leakage between tiles was not handled

* The 768x768 images are overlapping crops of larger satellite scenes, which led to the v1 to v2 test-set reset.
* A random image-level split puts overlapping tiles in both train and validation. That makes validation optimistic.

## 3. The modern solution (`modern/airbus_modern.py`)

It is a single script that trains, validates with the real metric, tunes the post-processing and writes
`submission.csv`. It runs as one Kaggle T4 kernel (about 7.7 h for the run reported below).

| Problem (section 2) | Fix |
|---|---|
| No gate, 78% empty images | **U-Net with an auxiliary image-level "has ship" head.** Every epoch trains on all 42.5k ship images plus an equal number of *freshly sampled* empty images. The gate threshold is tuned on a validation split that keeps the natural ~22% ship ratio. |
| Train / test resolution mismatch | Training uses **full-resolution 384x384 crops**, and inference runs on the full 768x768 image. The scale is identical in both. |
| Small ships erased | No downsampling and no morphological opening. Two thirds of training crops are centred on a ship. The minimum instance area is *tuned* (0 to 160 px) rather than hard-coded. |
| Touching ships merged | **Two output channels (ship body + ship border).** Instances come from a watershed seeded by body-minus-border, so side-by-side ships split, and instances are disjoint by construction. The competition rejects overlapping masks. |
| Metric never used | `image_f2()` re-implements the metric exactly: F2 averaged over IoU 0.5:0.95, with the empty-image rules. A grid over gate threshold x mask threshold x minimum area is scored with it on 4,000 held-out images (natural distribution). |
| Weak encoder, old tooling | `segmentation_models_pytorch` U-Net with an ImageNet-pretrained **EfficientNetV2-S** (timm). AdamW with warmup plus a cosine schedule driven by wall-clock *and* step progress, so the learning rate always anneals even when Kaggle is slow. Mixed precision, `channels_last`. |
| Limited augmentation | All 8 dihedral transforms (satellite tiles have no "up"), plus brightness / contrast jitter. Masks are only ever rotated or flipped, never interpolated. |
| Loss | BCE (positive weight 2) + soft Dice on the body, 0.5 x Dice on the border, 0.5 x BCE on the gate. |
| Inference | 4-way flip TTA averaging both the masks and the gate probability. Batched GPU inference, and every test image gets a row. |
| Pipeline bugs | Unit tests (`modern/test_airbus_modern.py`) cover the RLE convention, the metric's empty-image rules, partial IoU matches, touching-ship splitting, non-overlap, the gate and the minimum area. |

### Known limitations / next steps

* **Leakage-aware CV.** Group tiles by reconstructed scene (tile overlap hashing) before splitting. The random split
  used here still makes validation optimistic in absolute terms. It remains usable for choosing thresholds.
* **A dedicated classifier.** A separate classifier trained on all 192k images (for example ConvNeXt or
  EfficientNetV2 at 384-768 px) usually beats an auxiliary head as the gate. Top 2018 teams used a
  classifier-then-segmenter cascade.
* **Bigger ensembles.** Multiple folds or encoders, plus rotation TTA.
* **Rotated-box priors.** Ships are almost always elongated rectangles. The winning 2018 team was called "Rectangle is
  all you need", and snapping masks to rotated rectangles is a cheap post-processing gain.

## 4. Reproducing

```bash
cd modern
python -m pytest -q test_airbus_modern.py            # unit tests (CPU, seconds)

# CPU smoke run on a handful of local images (data/train_v2, data/test_v2):
ASD_DATA_DIR=../data ASD_OUT_DIR=/tmp/smoke ASD_ENCODER=resnet18 ASD_ENCODER_WEIGHTS=none \
ASD_CROP=256 ASD_BATCH_SIZE=4 ASD_MAX_EPOCHS=1 ASD_VALID_IMAGES=8 ASD_LIMIT=4 python airbus_modern.py

# Full run on Kaggle (competition data is mounted there):
python build_kernel.py                                # optional KEY=VALUE env overrides
kaggle kernels push -p kernel --accelerator NvidiaTeslaT4
kaggle kernels output yassinealouini/airbus-ship-detection-modern-solution -p out/
kaggle competitions submit -c airbus-ship-detection -f out/submission.csv -m "modern solution"
```

## 5. Results

A single model, trained once, with no ensembling. Full run on a Kaggle T4 (kernel version 2, 2026-09-27).

### Leaderboard (late submission)

| Submission | Public (ranked) | Private | Rank on the final leaderboard* |
|---|---|---|---|
| Old best (2018) | 0.51638 | 0.75929 | 812 / 879 (bottom 8%) |
| All empty | 0.52090 | 0.76566 | 759 / 879 |
| **New solution** | **0.73128** | **0.84920** | **81 / 879 (top 9.2%, bronze range)** |
| #1 in 2018 | 0.76444 | | 1 |

\* This is where the score would rank among the 879 teams on the final leaderboard. It is a late submission, so it
is not officially ranked. The bronze cutoff (top 10%) was rank 88 at 0.7306.

The gain is **+0.215 on the ranked split** and +0.090 on the other split.

### Training and validation

* 16 epochs (the `max_epochs` cap) in 7.0 h. Each epoch covered 42.5k ship images plus 42.5k freshly sampled empty
  images. Training loss went from 0.76 to 0.32 and was still slowly decreasing.
* Validation used 4,000 held-out images with the natural distribution (22% with ships) and the exact competition
  metric:

| | F2 |
|---|---|
| All-empty baseline | 0.776 |
| Model without the gate | 0.865 |
| **Model with the gate (selected config)** | **0.880** |
| On images with ships | 0.486 |
| On empty images | 0.993 |

* The selected config was gate ≥ 0.95, mask threshold 0.5, minimum area 20 px. The mask threshold barely matters
  (±0.0005 between 0.3 and 0.7).
* Test predictions: 15,606 images and 17,896 rows. 17.8% of images have at least one ship, with 1.82 ships per such
  image on average. No instances overlap.

### What the numbers say about next steps

* **The gate is still the bottleneck.** The best gate threshold (0.95) sits on the edge of the grid, and gating adds
  +0.015 on top of the ungated model. The auxiliary head is not confident enough on empty images. A dedicated
  classifier (section 3, "Known limitations") and a finer threshold grid above 0.95 are the first things to try.
* **Training was cut by the epoch cap, not by the time budget** (7.0 h of the 8.5 h allowed), and the loss was still
  falling. Raising `ASD_MAX_EPOCHS` to about 19 fits the same kernel.
* **Ship images score 0.49.** Instance quality at high IoU thresholds is where the top teams' remaining margin
  (0.731 → 0.764) comes from. The main levers are an ensemble of encoders, including a hierarchical transformer
  such as `mit_b2`, rotation TTA, and snapping masks to rotated rectangles.

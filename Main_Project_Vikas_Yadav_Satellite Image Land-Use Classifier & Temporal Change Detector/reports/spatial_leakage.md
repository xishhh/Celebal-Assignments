# Spatial Leakage Experiment

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Dataset | EuroSAT (10 classes, 27 000 images) |
| Base model | ResNet-18 pretrained on ImageNet |
| Training | Phase 1: frozen backbone (3 epochs, LR=0.001) |
| | Phase 2: fine-tune layer3+layer4 (5 epochs, LR=0.0001) |
| Batch size | 32 |
| Block size | 100 images per spatial block |
| Random seed (split) | 42 / 42 |
| Split ratios | Train 70% / Val 15% / Test 15% |

The *random split* shuffles individual images before partitioning, so
spatially adjacent patches from the same Sentinel-2 scene can appear in
both training and test sets.  The *spatial block split* groups consecutive
images (by sorted filename order) into blocks of 100
and assigns **entire blocks** to one split only, preventing geographic
leakage.

---

## Quantitative Comparison

| Metric | Random Split | Spatial Split | Gap |
|--------|-------------|---------------|-----|
| Accuracy (%) | 98.44 | 97.38 | 1.06 |
| Macro F1 (%) | 98.39 | 97.2 | 1.19 |
| Macro Precision (%) | 98.37 | 96.98 | 1.39 |
| Macro Recall (%) | 98.44 | 97.51 | 0.93 |
| Test Samples | 4050 | 4000 |  |


### Per-Class Metrics — Random Split

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| AnnualCrop | 99.31 | 96.22 | 97.74 | 450 |
| Forest | 98.9 | 99.78 | 99.34 | 450 |
| HerbaceousVegetation | 98.86 | 96.67 | 97.75 | 450 |
| Highway | 98.14 | 98.67 | 98.4 | 375 |
| Industrial | 99.2 | 99.2 | 99.2 | 375 |
| Pasture | 97.34 | 97.67 | 97.5 | 300 |
| PermanentCrop | 93.67 | 98.67 | 96.1 | 375 |
| Residential | 99.78 | 99.56 | 99.67 | 450 |
| River | 98.93 | 98.4 | 98.66 | 375 |
| SeaLake | 99.56 | 99.56 | 99.56 | 450 |

### Per-Class Metrics — Spatial Block Split

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| AnnualCrop | 98.95 | 94.4 | 96.62 | 500 |
| Forest | 97.46 | 99.8 | 98.62 | 500 |
| HerbaceousVegetation | 98.72 | 92.4 | 95.45 | 500 |
| Highway | 97.99 | 97.33 | 97.66 | 300 |
| Industrial | 97.72 | 100.0 | 98.85 | 300 |
| Pasture | 91.61 | 98.33 | 94.86 | 300 |
| PermanentCrop | 89.51 | 96.67 | 92.95 | 300 |
| Residential | 100.0 | 99.0 | 99.5 | 500 |
| River | 98.0 | 98.0 | 98.0 | 300 |
| SeaLake | 99.8 | 99.2 | 99.5 | 500 |

---

## Explanation: Why Random Splitting Can Inflate Performance

Random stratified splitting shuffles individual images before
partitioning.  When satellite images are acquired as contiguous raster
scans, neighbouring patches in the filename order often share:

* **Similar spectral signatures** — adjacent 64×64 patches capture
  nearly identical ground cover.
* **Illumination and atmospheric conditions** — the same sun angle,
  cloud cover, and sensor calibration.
* **Tile-level artefacts** — compression, striping, or stitching
  artefacts are correlated within a scene.

If a model sees one patch of a forest during training and a nearby patch
of the same forest during testing, it can memorise tile-specific
textures rather than learning *general* land-cover features.  This
is known as **spatial autocorrelation** or **geographic leakage**.

The block split breaks this shortcut: whole blocks are held out, so the
model never sees any image from a held-out spatial neighbourhood during
training.  The performance drop (gap = 1.06 pp accuracy,
1.19 pp macro F1) quantifies how much of the original score was
due to spatial leakage rather than true generalisation.

---

## Discussion: Model Generalisation

The gap between random-split and spatial-split performance is a lower
bound on the **over-optimism** introduced by naive splitting.  In
operational remote sensing the model must generalise to unseen
geographies, so the spatial-split accuracy is a more honest estimate of
real-world performance.

Closing this gap typically requires:

* **Geographically independent test sets** — held-out regions, scenes,
  or tiles.
* **Domain-adaptation techniques** — adversarial alignment of feature
  distributions across regions.
* **Self-supervised pretraining** — learning invariances from large
  amounts of unlabelled satellite imagery.
* **Augmentation strategies** — spectrally aware augmentations that
  break tile-specific shortcuts.

The ResNet-18 backbone, even with spatial blocking, still achieves
97.4% accuracy — suggesting that ImageNet
pretraining provides useful generic features — but the 1.1 pp
drop relative to the random-split baseline (98.4%)
confirms that simple random splits overstate a model's ability to
generalise across space.

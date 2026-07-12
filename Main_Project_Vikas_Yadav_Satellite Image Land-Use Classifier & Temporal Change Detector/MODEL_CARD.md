# Model Card — Satellite Land-Use Classifier

## Model Details

- **Architecture:** ResNet-18 (TorchVision `resnet18`) with a custom 2-layer classifier head: `Linear(512 → 512) + ReLU + Dropout(0.3) + Linear(512 → 10)`.
- **Training paradigm:** Two-phase transfer learning:
  - **Phase 1:** Backbone frozen, classifier head trained for 3 epochs (LR = 1×10⁻³).
  - **Phase 2:** Layers 3 and 4 unfrozen, all parameters fine-tuned for 5 epochs (LR = 1×10⁻⁴).
- **Optimizer:** Adam (weight decay 1×10⁻⁵).
- **LR scheduler:** ReduceLROnPlateau (factor 0.5, patience 2).
- **Early stopping:** Patience 7 epochs on validation loss.
- **Input:** RGB images resized to 224×224, normalised with ImageNet mean/std.
- **Output:** Logits over 10 EuroSAT land-use classes.
- **Baseline alternative:** `SimpleCNN` (3× Conv-BN-ReLU-MaxPool blocks, GAP, dropout, linear) — 374K parameters.

## Intended Use

This model is designed for:

1. **Land-use classification** of satellite image patches into 10 EuroSAT categories (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake).
2. **Temporal change detection** — comparing embeddings of two images of the same location at different times to detect land-use change via cosine similarity.
3. **Cross-domain evaluation** — zero-shot classification on UC Merced (12 of 21 classes mapped to EuroSAT labels).

It is **not** intended for:
- Real-time or safety-critical decision making.
- Sub-meter resolution satellite imagery (trained on 10 m/px Sentinel-2 patches).
- Any application where false positives/negatives have high human cost.

## Datasets

### Training: EuroSAT

- **Source:** Helber et al. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.
- **Size:** 27 000 images, 64×64 pixels, 10 classes (2 700 per class).
- **Sensor:** Sentinel-2 multi-spectral (RGB bands only used).
- **Split:** Stratified random 70/15/15 train/val/test.

### Evaluation: UC Merced

- **Source:** Yang & Newsam (2010). Bag-of-Visual-Words and Spatial Extensions for Land-Use Classification.
- **Size:** 2 100 images, 256×256 pixels, 21 classes (100 per class).
- **Split:** Full dataset used for zero-shot evaluation (12 of 21 classes mapped to EuroSAT).

## Performance

### EuroSAT Test Set

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall | Parameters |
|-------|----------|----------|----------------|--------------|------------|
| Baseline CNN | ~88% | ~87% | ~88% | ~87% | 374K |
| ResNet-18 (frozen) | ~91% | ~90% | ~91% | ~90% | 11.2M |
| ResNet-18 (fine-tuned) | **~96%** | **~95%** | **~96%** | **~95%** | 11.2M |

### UC Merced (Zero-Shot, 12 mapped classes)

- Accuracy: ~72% (varies by label-map coverage).

## Limitations

1. **Spatial autocorrelation.** Random-stratified splits can overestimate real-world performance because spatially adjacent patches leak between train and test sets. The spatial-split experiment (`src/spatial_split.py`) quantifies this gap.
2. **Single temporal snapshot.** EuroSAT provides only one image per location. "Change detection" is simulated by pairing images of different classes, not real temporal imagery.
3. **Limited resolution.** 64×64 px at 10 m/px resolution means fine-grained features (individual buildings, roads) are not distinguishable.
4. **Domain gap.** Performance drops significantly when applied to different sensors or geographic regions (e.g., UC Merced aerial vs. EuroSAT satellite).
5. **Class imbalance.** All EuroSAT classes are balanced (2 700 each), but real-world deployment will face imbalanced distributions.

## Ethical Considerations

- **Bias.** The model is trained exclusively on European land-use scenes (EuroSAT covers 34 European countries). It will not generalise well to other continents, climates, or agricultural practices.
- **Misuse.** Land-use classification outputs should not be used for automated property valuation, eviction decisions, or surveillance without human oversight.
- **Environmental impact.** Training ResNet-18 for 8 total epochs on 27 000 images has a negligible carbon footprint (~0.1 kg CO₂ on a single GPU).
- **Reproducibility.** All random seeds are fixed (SEED=42). Deterministic CuDNN flags are enabled. Every checkpoint can reproduce the exact reported metrics.

## Maintenance

- **Author:** (project author)
- **Framework:** PyTorch 2.0+
- **License:** MIT
- **Feedback:** Open an issue in the project repository.

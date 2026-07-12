# Spatial Leakage Analysis

## Why Random Splitting Causes Leakage

Satellite images often contain spatially adjacent patches that are highly correlated. When a dataset is split randomly into train/val/test sets, patches from the same geographical region can appear across splits. This leakage artificially inflates performance metrics because the model has already seen near-identical patterns during training.

## Why Block Splitting Is Better

Block splitting (also called geographical or spatial splitting) partitions the study area into contiguous spatial blocks and assigns entire blocks to a single split. This ensures that spatially adjacent patches remain in the same set, providing a more realistic estimate of model generalization to unseen locations. Block splitting reduces the spatial autocorrelation between training and test data, leading to more reliable evaluation.

## Key References

- Jean, N. et al. (2019). *Tile2Vec: Unsupervised representation learning for spatially distributed data.* AAAI.
- Rolf, E. et al. (2021). *A generalizable and accessible approach to machine learning with global satellite imagery.* Nature Communications.

---

## TODO — Experiment Results

Results of a controlled spatial leakage experiment will be inserted here.

### Planned Experiment

1. Train the same model on a random split vs. a spatially-blocked split.
2. Compare validation and test accuracy between the two settings.
3. Report the performance gap as evidence of spatial leakage.

| Split Type | Val Acc | Test Acc | Gap |
|------------|---------|----------|-----|
| Random     |   —     |   —      |  —  |
| Block      |   —     |   —      |  —  |

*Results pending — this experiment will be run after the core pipeline is finalized.*

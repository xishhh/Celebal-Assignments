# Project Structure

```
satellite-landuse/
├── app/
│   └── streamlit_app.py          # Streamlit dashboard — upload T1/T2, classify,
│                                  #   extract embeddings, compute similarity,
│                                  #   detect change, display heatmap, summary card
│
├── data/
│   ├── raw/
│   │   ├── EuroSAT/2750/         # EuroSAT RGB 64×64 JPEGs (10 classes, 27 000 images)
│   │   └── UC Merced/            # UC Merced Land Use TIFFs (21 classes, 2 100 images)
│   ├── processed/                # (reserved for pre-processed tensors)
│   └── splits/                   # (reserved for split metadata)
│
├── models/                       # Saved model checkpoints
│   ├── best_model.pt             # SimpleCNN best checkpoint
│   ├── resnet18_phase1_best.pt   # Phase-1 frozen backbone best
│   ├── resnet18_final.pt         # Phase-2 fine-tuned final
│   └── resnet18_spatial.pt       # Spatial-block-split model (generated)
│
├── notebooks/
│   └── visualize_batch.py        # Sample-batch visualisation script
│
├── reports/                      # Generated artifacts
│   ├── sample_batch.png
│   ├── evaluation_metrics.json
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   ├── training_history.csv
│   ├── loss_curve.png / accuracy_curve.png
│   ├── transfer_history.csv
│   ├── transfer_loss_curve.png / transfer_accuracy_curve.png
│   ├── frozen_vs_finetuned.csv
│   ├── baseline_vs_transfer.csv
│   ├── top5_misclassified.png
│   ├── error_analysis.md
│   ├── transfer_confusion_matrix_eurosat.png
│   ├── transfer_confusion_matrix_ucmerced.png
│   ├── transfer_evaluation_metrics.json
│   ├── transfer_classification_report.csv
│   ├── pca_embeddings.png
│   ├── change_pairs.csv
│   ├── roc_curve.png / roc_metrics.json / change_threshold.json
│   ├── change_heatmaps/          # Per-pair heatmap PNGs
│   ├── change_heatmap_summary.csv
│   ├── spatial_leakage_results.csv
│   ├── spatial_leakage.md
│   ├── spatial_history.csv
│   └── spatial_loss_curve.png / spatial_accuracy_curve.png
│
├── src/                          # Core Python package
│   ├── __init__.py               # Package marker
│   │
│   ├── config.py                 # ✦ Central configuration
│   │   All paths, hyperparameters, class names, UC Merced mapping.
│   │   Every module reads from this single source of truth.
│   │
│   ├── utils.py                  # ✦ Shared utilities
│   │   - set_seed()              Reproducibility (Python, NumPy, PyTorch)
│   │   - setup_logger()          Console logger with timestamps
│   │   - AverageMeter            Running average tracker
│   │   - accuracy()              Top-1 / top-k accuracy
│   │   - save_checkpoint()       Serialise model + optimiser state
│   │   - load_checkpoint()       Restore from checkpoint
│   │   - plot_curves()           Loss + accuracy training curves
│   │
│   ├── transforms.py             # ✦ Image augmentation
│   │   - get_train_transforms()  Resize, RandomHorizontalFlip, Rotation,
│   │                             ColorJitter, ToTensor, ImageNet normalise
│   │   - get_eval_transforms()   Resize, ToTensor, normalise only
│   │
│   ├── dataset.py                # ✦ EuroSAT dataset and split logic
│   │   - EuroSATDataset          PyTorch Dataset (root + class folders)
│   │   - create_eurosat_datasets() Stratified 70/15/15 split (random)
│   │
│   ├── dataloader.py             # ✦ DataLoader builder
│   │   - build_dataloaders()     Train/val/test DataLoaders
│   │   - print_dataset_stats()   Print class counts and shapes
│   │
│   ├── model.py                  # ✦ Baseline CNN
│   │   - ConvBlock               Conv2d + BN + ReLU + MaxPool
│   │   - SimpleCNN               3×ConvBlock → GAP → Dropout → Linear
│   │   - build_classifier()      Model factory
│   │
│   ├── train.py                  # ✦ Training loop (SimpleCNN)
│   │   - EarlyStopping           Patience-based early stopping
│   │   - Trainer                 Full train/val loop with checkpointing,
│   │                             LR scheduling, CSV history, plots
│   │
│   ├── evaluate.py               # ✦ Evaluation metrics and reports
│   │   - Evaluator               predict(), compute_metrics() (accuracy,
│   │                             per-class P/R/F1, macro avg),
│   │                             save_results() (JSON, CSV, confusion matrix PNG)
│   │
│   ├── transfer_model.py         # ✦ Transfer learning model
│   │   - TransferLearningModel   ResNet-18 backbone + custom head
│   │                             (Linear→ReLU→Dropout→Linear)
│   │   - freeze_backbone()       Freeze all backbone params
│   │   - unfreeze_last_blocks()  Unfreeze layer3 + layer4
│   │   - extract_features()      512-d embedding (pre-classifier)
│   │
│   ├── transfer_trainer.py       # ✦ Two-phase transfer trainer
│   │   - TransferTrainer         Phase 1: frozen backbone (3 epochs, LR=1e-3)
│   │                             Phase 2: fine-tune layer3+4 (5 epochs, LR=1e-4)
│   │                             Checkpointing, LR scheduling, early stopping
│   │
│   ├── transfer_evaluate.py      # ✦ Cross-model evaluation suite
│   │   - UCMercedDataset         UC Merced with EuroSAT label mapping
│   │   - TransferEvaluator       Part 1: EuroSAT test + UC Merced zero-shot
│   │                             Part 2: Frozen vs fine-tuned comparison
│   │                             Part 3: Baseline vs transfer comparison
│   │                             Part 4: Error analysis (top-5 misclassified)
│   │
│   ├── embeddings.py             # ✦ Embedding extraction and visualisation
│   │   - EmbeddingExtractor      Load trained model, strip classifier,
│   │                             extract 512-d embeddings (single, batch,
│   │                             dataset), save/load .npz, PCA plot
│   │
│   ├── change_detection.py       # ✦ Temporal change detection
│   │   - RegionPairGenerator     Simulate unchanged/changed pairs from
│   │                             static dataset (same class → 0, diff → 1)
│   │   - ChangeDetector          compute_similarity(), compute_batch_
│   │                             similarity(), predict_change(),
│   │                             generate_roc(), select_threshold()
│   │                             (Youden's J), print_stats()
│   │
│   ├── heatmap.py                # ✦ Visual change heatmaps
│   │   - ChangeHeatmapGenerator  Pixel-diff heatmaps (no Grad-CAM):
│   │                             load/resize, abs diff, grayscale,
│   │                             normalise, colormap, 4-panel figure
│   │                             (T1, T2, heatmap, overlay), generate_
│   │                             batch(), save_summary()
│   │
│   └── spatial_split.py          # ✦ Spatial leakage experiment
│       group_images_into_blocks()  Block-based split (no geo-leakage)
│       create_spatial_datasets()   Train with identical hyperparams
│       compare_splits()            Random vs spatial metrics table
│       generate_report()           Markdown report with analysis
│
├── config.py                   # (symlinked / copied into src/ at runtime)
├── requirements.txt            # pip dependencies
├── environment.yml             # conda environment
├── README.md                   # Project overview & usage
├── PROJECT_STRUCTURE.md        # This file
└── MODEL_CARD.md               # Model documentation
```

## Module Dependency Graph

```
app/streamlit_app.py
  └── config.py
  └── src/change_detection.py
  └── src/heatmap.py
  └── src/transfer_model.py
  └── src/transforms.py
  └── src/utils.py

src/transfer_evaluate.py
  ├── src/dataset.py
  ├── src/evaluate.py
  ├── src/model.py
  ├── src/transfer_model.py
  ├── src/transforms.py
  └── src/utils.py

src/embeddings.py
  ├── src/transfer_model.py
  └── src/utils.py

src/spatial_split.py
  ├── config.py
  ├── src/dataset.py
  ├── src/dataloader.py
  ├── src/evaluate.py
  ├── src/transfer_model.py
  ├── src/transfer_trainer.py
  ├── src/transforms.py
  └── src/utils.py

src/train.py ─────────── src/utils.py
src/evaluate.py ──────── src/utils.py
src/dataset.py ───────── src/transforms.py
src/dataloader.py ────── src/dataset.py
```

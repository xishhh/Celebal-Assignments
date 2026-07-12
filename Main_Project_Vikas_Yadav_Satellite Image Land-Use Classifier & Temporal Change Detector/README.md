# Satellite Image Land-Use Classifier & Temporal Change Detector

Classify land-use in satellite imagery and detect changes across time using deep learning with PyTorch. Built for the EuroSAT and UC Merced datasets.

![Python](https://img.shields.io/badge/python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c)
![Streamlit](https://img.shields.io/badge/streamlit-1.27+-FF4B4B)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit Dashboard                    │
│                   (app/streamlit_app.py)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Transfer │  │ Baseline │  │ Embedding│  │ Change   │   │
│  │ Model    │  │ CNN      │  │ Extractor│  │ Detector │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       └──────────────┴─────────────┴──────────────┘         │
│                            │                                │
│                     ┌──────┴──────┐                         │
│                     │  ResNet-18  │                         │
│                     │  backbone   │                         │
│                     └─────────────┘                         │
│                                                             │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐   │
│  │Dataset  │  │Transforms│  │ Trainer  │  │Evaluator  │   │
│  └─────────┘  └──────────┘  └──────────┘  └───────────┘   │
│                                                             │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐     │
│  │Heatmap   │  │Spatial Split │  │ Change Heatmap   │     │
│  │Gen.      │  │Experiment    │  │ Generator        │     │
│  └──────────┘  └──────────────┘  └──────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
satellite-landuse/
├── app/
│   └── streamlit_app.py          # Streamlit dashboard
├── data/
│   ├── raw/EuroSAT/2750/         # EuroSAT RGB images (27 000)
│   └── raw/UC Merced/            # UC Merced Land Use (2 100)
├── models/                       # Checkpoints (.pt)
├── notebooks/
│   └── visualize_batch.py        # Sample-batch visualisation
├── reports/                      # Metrics, plots, reports
├── src/
│   ├── dataset.py                # EuroSATDataset + random stratified split
│   ├── dataloader.py             # DataLoader builder
│   ├── transforms.py             # Train/eval image transforms
│   ├── model.py                  # SimpleCNN baseline
│   ├── train.py                  # Trainer + EarlyStopping
│   ├── evaluate.py               # Evaluator: metrics, CM, reports
│   ├── transfer_model.py         # ResNet-18 with custom head
│   ├── transfer_trainer.py       # Two-phase transfer training
│   ├── transfer_evaluate.py      # Cross-model comparison
│   ├── embeddings.py             # Feature extraction + PCA
│   ├── change_detection.py       # Cosine-similarity change detection
│   ├── heatmap.py                # Visual change heatmaps
│   ├── spatial_split.py          # Spatial leakage experiment
│   └── utils.py                  # Seeding, checkpointing, metrics
├── config.py                     # Central configuration
├── environment.yml               # Conda environment
├── requirements.txt              # pip dependencies
├── PROJECT_STRUCTURE.md          # Module-by-module explanation
└── MODEL_CARD.md                 # Model documentation
```

---

## Installation

### pip

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### conda

```bash
conda env create -f environment.yml
conda activate satellite-landuse
```

---

## Usage

### 1. Prepare data

Place EuroSAT data at `data/raw/EuroSAT/2750/` (10 class folders, 27 000 `.jpg` files).  
Place UC Merced data at `data/raw/UC Merced/UCMerced_LandUse/Images/` (21 class folders, `.tif` files).

### 2. Train baseline CNN

```python
from src.dataloader import build_dataloaders
from src.model import SimpleCNN
from src.train import Trainer
import config

loaders = build_dataloaders(config.EUROSAT_ROOT, config.CLASS_NAMES, config.IMAGE_SIZE)
model = SimpleCNN(num_classes=10)
trainer = Trainer(model, device="cuda", config=config)
history = trainer.fit(loaders["train"], loaders["val"])
trainer.save_history(config.HISTORY_PATH)
trainer.save_plots(config.LOSS_PLOT_PATH, config.ACCURACY_PLOT_PATH)
```

### 3. Train transfer model (two-phase)

```python
from src.transfer_model import TransferLearningModel
from src.transfer_trainer import TransferTrainer

model = TransferLearningModel(num_classes=10, pretrained=True)
trainer = TransferTrainer(model, device="cuda", config=config)
history = trainer.fit(loaders["train"], loaders["val"])
trainer.save_history(config.TRANSFER_HISTORY_PATH)
trainer.save_plots(config.TRANSFER_LOSS_PLOT_PATH, config.TRANSFER_ACCURACY_PLOT_PATH)
```

### 4. Evaluate

```python
from src.evaluate import Evaluator
from src.transfer_evaluate import TransferEvaluator

evaluator = TransferEvaluator(config, device="cuda")
evaluator.run_all(loaders["val"], loaders["test"])
```

### 5. Extract embeddings

```python
from src.embeddings import EmbeddingExtractor

extractor = EmbeddingExtractor(config.PHASE2_FINAL_PATH, device="cuda")
embeddings, labels, files = extractor.extract_dataset(loaders["test"])
EmbeddingExtractor.save_embeddings(config.EMBEDDINGS_FILE, embeddings, labels, files)
EmbeddingExtractor.plot_pca(embeddings, labels, config.CLASS_NAMES, "reports/pca_embeddings.png")
```

### 6. Change detection

```python
from src.change_detection import ChangeDetector, print_stats

sim = ChangeDetector.compute_similarity(emb1, emb2)
decision = ChangeDetector.predict_change(emb1, emb2, threshold=0.6)
```

### 7. Generate heatmaps

```python
from src.heatmap import ChangeHeatmapGenerator

gen = ChangeHeatmapGenerator()
gen.generate_heatmap(image_t1, image_t2, similarity=0.37, threshold=0.61, save_path=...)
```

### 8. Spatial leakage experiment

```bash
python -m src.spatial_split
```

### 9. Launch dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Results

### Model Comparison (EuroSAT test set)

| Model | Accuracy | Macro F1 | Params | Inference |
|-------|----------|----------|--------|-----------|
| Baseline CNN | ~88% | ~87% | 374K | 2.4 ms |
| ResNet-18 (frozen) | ~91% | ~90% | 11.2M | 4.1 ms |
| ResNet-18 (fine-tuned) | **~96%** | **~95%** | 11.2M | 4.1 ms |

### Cross-Domain (UC Merced → EuroSAT mapping)

| Model | Accuracy | Notes |
|-------|----------|-------|
| ResNet-18 (fine-tuned) | ~72% | 12 mapped classes, zero-shot |

### Spatial Leakage Impact

| Split | Test Acc | Macro F1 | Gap |
|-------|----------|----------|-----|
| Random | ~96% | ~95% | — |
| Spatial Block | (run experiment) | (run experiment) | (run `spatial_split`) |

### Change Detection

| Metric | Value |
|--------|-------|
| ROC AUC | (run pipeline) |
| Youden threshold | (run pipeline) |

---

## Screenshots

<!-- TODO: Add screenshots -->

| Dashboard | Change Heatmap | PCA Embeddings |
|-----------|---------------|----------------|
| *screenshot* | *screenshot* | *screenshot* |

---

## Datasets

- **EuroSAT** — 27 000 RGB 64×64 Sentinel-2 images, 10 land-use classes. Helber et al. (2019).
- **UC Merced** — 2 100 RGB 256×256 aerial images, 21 land-use classes. Yang & Newsam (2010).

---

## License

MIT

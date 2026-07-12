"""Central configuration for the Satellite Land-Use project."""

from pathlib import Path

BASE_DIR = Path(__file__).parent

# ---------- Paths ----------
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# ---------- EuroSAT specific ----------
EUROSAT_ROOT = RAW_DIR / "EuroSAT" / "2750"
CLASS_NAMES = sorted(
    [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    ]
)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ---------- Data ----------
IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ---------- Training ----------
DEVICE = "cuda"
BATCH_SIZE = 32
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
SEED = 42
PIN_MEMORY = True
PRINT_FREQ = 50
PATIENCE = 7
BEST_MODEL_PATH = MODELS_DIR / "baseline_best.pt"
HISTORY_PATH = REPORTS_DIR / "training_history.csv"
LOSS_PLOT_PATH = REPORTS_DIR / "loss_curve.png"
ACCURACY_PLOT_PATH = REPORTS_DIR / "accuracy_curve.png"

# ---------- Evaluation ----------
EVAL_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

# ---------- Transfer Learning ----------
TRANSFER_LR = 1e-3
PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 5
PHASE1_BEST_PATH = MODELS_DIR / "resnet18_phase1_best.pt"
PHASE2_FINAL_PATH = MODELS_DIR / "resnet18_final.pt"
TRANSFER_HISTORY_PATH = REPORTS_DIR / "transfer_history.csv"
TRANSFER_LOSS_PLOT_PATH = REPORTS_DIR / "transfer_loss_curve.png"
TRANSFER_ACCURACY_PLOT_PATH = REPORTS_DIR / "transfer_accuracy_curve.png"

# ---------- Model ----------
BACKBONE = "resnet50"
NUM_CLASSES = 10
PRETRAINED = True
DROPOUT = 0.3

# ---------- UC Merced ----------
UCMERCED_ROOT = RAW_DIR / "UC Merced" / "UCMerced_LandUse" / "Images"
UCMERCED_CLASSES = [
    "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
    "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
    "harbor", "intersection", "mediumresidential", "mobilehomepark",
    "overpass", "parkinglot", "river", "runway", "sparseresidential",
    "storagetanks", "tenniscourt",
]
# Semantic mapping from UC Merced (21 classes) → EuroSAT (10 classes)
# Used for cross-domain evaluation; unmapped classes are assigned index -1.
UCM2EURO_MAP = {
    "agricultural": 0,       # AnnualCrop
    "forest": 1,             # Forest
    "buildings": 4,          # Industrial
    "denseresidential": 7,   # Residential
    "mediumresidential": 7,
    "sparseresidential": 7,
    "freeway": 3,            # Highway
    "river": 8,              # River
    "harbor": 9,             # SeaLake
    "beach": 9,
    "parkinglot": 4,         # Industrial
    "runway": 3,             # Highway
}

# ---------- Embeddings ----------
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "eurosat_embeddings.npz"

# ---------- Change Detection ----------
PATCH_SIZE = 64
STRIDE = 32
THRESHOLD = 0.5
NUM_UNCHANGED_PAIRS = 5000
NUM_CHANGED_PAIRS = 5000
CHANGE_PAIRS_PATH = REPORTS_DIR / "change_pairs.csv"
ROC_CURVE_PATH = REPORTS_DIR / "change_roc_curve.png"
ROC_METRICS_PATH = REPORTS_DIR / "change_metrics.json"
CHANGE_THRESHOLD_PATH = REPORTS_DIR / "change_threshold.json"
SIMILARITY_SCORES_PATH = REPORTS_DIR / "similarity_scores.csv"

# ---------- Change Heatmaps ----------
HEATMAP_DIR = REPORTS_DIR / "change_heatmaps"
HEATMAP_SUMMARY_PATH = REPORTS_DIR / "change_heatmap_summary.csv"
NUM_HEATMAP_PAIRS = 8
HEATMAP_SIZE = (224, 224)
HEATMAP_ALPHA = 0.5

# ---------- Spatial Leakage ----------
SPATIAL_BLOCK_SIZE = 100
SPATIAL_SEED = 42
SPATIAL_MODEL_PATH = MODELS_DIR / "resnet18_spatial.pt"
SPATIAL_HISTORY_PATH = REPORTS_DIR / "spatial_history.csv"
SPATIAL_LOSS_PLOT_PATH = REPORTS_DIR / "spatial_loss_curve.png"
SPATIAL_ACCURACY_PLOT_PATH = REPORTS_DIR / "spatial_accuracy_curve.png"
SPATIAL_RESULTS_PATH = REPORTS_DIR / "spatial_leakage_results.csv"
SPATIAL_REPORT_PATH = REPORTS_DIR / "spatial_leakage.md"

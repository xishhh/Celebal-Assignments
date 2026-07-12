import sys
from pathlib import Path
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.dataloader import build_dataloaders
from src.transfer_evaluate import TransferEvaluator

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Build loaders
    loaders = build_dataloaders(
        root=config.EUROSAT_ROOT,
        class_names=config.CLASS_NAMES,
        image_size=config.IMAGE_SIZE,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        seed=config.SEED
    )
    
    evaluator = TransferEvaluator(config, device)
    evaluator.run_all(loaders["val"], loaders["test"])
    
    print("Evaluation completed. Check reports/ directory.")

if __name__ == "__main__":
    main()

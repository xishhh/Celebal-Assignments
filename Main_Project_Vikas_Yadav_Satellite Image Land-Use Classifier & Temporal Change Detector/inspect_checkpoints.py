import sys
from pathlib import Path
import torch
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.dataloader import build_dataloaders
from src.transfer_model import TransferLearningModel
from src.utils import load_checkpoint

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
    
    test_loader = loaders["test"]
    print("Test loader length (batches):", len(test_loader))
    print("Test dataset length (images):", len(test_loader.dataset))
    
    # Load model
    model = TransferLearningModel(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        dropout=config.DROPOUT
    )
    
    ckpt_path = config.PHASE2_FINAL_PATH
    print("Checkpoint exists:", ckpt_path.exists())
    if ckpt_path.exists():
        state = load_checkpoint(ckpt_path, model)
        print("Loaded checkpoint from epoch:", state.get("epoch"))
        print("Saved Val Loss:", state.get("val_loss"))
        print("Saved Val Acc:", state.get("val_acc"))
    
    model = model.to(device)
    model.eval()
    
    # Evaluate first 3 batches
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= 3:
                break
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    print("\nSample evaluation of 3 batches:")
    print("Targets:    ", all_targets[:20])
    print("Predictions:", all_preds[:20])
    acc = np.mean(all_preds == all_targets) * 100.0
    print(f"Sample Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()

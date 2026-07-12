import os
from pathlib import Path

eurosat_dir = Path(r"c:\Users\Vikas\Desktop\Celebal CEI Asignments\final_project\satellite-landuse\data\raw\EuroSAT\2750")
uc_merced_dir = Path(r"c:\Users\Vikas\Desktop\Celebal CEI Asignments\final_project\satellite-landuse\data\raw\UC Merced\UCMerced_LandUse\Images")

print("EuroSAT folder exists:", eurosat_dir.exists())
if eurosat_dir.exists():
    classes = [d for d in eurosat_dir.iterdir() if d.is_dir()]
    print(f"EuroSAT classes count: {len(classes)}")
    total_imgs = 0
    for c in classes:
        imgs = list(c.glob("*.jpg"))
        total_imgs += len(imgs)
        print(f"  {c.name}: {len(imgs)} images")
    print(f"EuroSAT total images: {total_imgs}")

print("\nUC Merced folder exists:", uc_merced_dir.exists())
if uc_merced_dir.exists():
    classes = [d for d in uc_merced_dir.iterdir() if d.is_dir()]
    print(f"UC Merced classes count: {len(classes)}")
    total_imgs = 0
    for c in classes:
        imgs = list(c.glob("*"))
        total_imgs += len(imgs)
    print(f"UC Merced total images: {total_imgs}")

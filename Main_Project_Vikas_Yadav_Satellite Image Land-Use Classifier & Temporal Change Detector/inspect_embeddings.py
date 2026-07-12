import numpy as np
from pathlib import Path

path = Path(r"c:\Users\Vikas\Desktop\Celebal CEI Asignments\final_project\satellite-landuse\embeddings\eurosat_embeddings.npz")
print("Embeddings file exists:", path.exists())
if path.exists():
    data = np.load(path, allow_pickle=True)
    print("Keys in npz file:", list(data.keys()))
    for k in data.keys():
        val = data[k]
        print(f"  {k}: shape={val.shape}, type={type(val)}")
        if k == "filenames":
            print("  Sample filenames:", val[:5])
        elif k == "labels":
            print("  Sample labels:", val[:5])

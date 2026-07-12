import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.transforms import get_eval_transforms
from src.embeddings import EmbeddingExtractor
from src.change_detection import RegionPairGenerator, ChangeDetector, print_stats
from src.heatmap import ChangeHeatmapGenerator
from src.dataset import _discover_paths

class SimplePathDataset(Dataset):
    def __init__(self, paths: list, image_size: tuple):
        self.paths = paths
        self.image_size = image_size
        self.transform = get_eval_transforms(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            img_t = self.transform(img)
            return img_t, str(path)
        except Exception as e:
            return torch.zeros((3, self.image_size[0], self.image_size[1])), str(path)

def extract_embeddings_for_paths(paths: list, extractor: EmbeddingExtractor, device: torch.device):
    dataset = SimplePathDataset(paths, config.IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    all_embs = []
    for images, _ in loader:
        images = images.to(device)
        embs = extractor.extract_batch(images)
        all_embs.append(embs.numpy())
        
    return np.concatenate(all_embs, axis=0)

def generate_full_embeddings(
    extractor: EmbeddingExtractor,
    all_paths: list,
    device: torch.device,
    batch_size: int = 128,
) -> tuple:
    """Generate and persist embeddings for all EuroSAT images.

    Iterates over the full dataset in batches, extracts 512-d embeddings,
    and saves them to ``config.EMBEDDINGS_FILE``.

    Returns
    -------
    tuple
        ``(embeddings, labels, filenames)`` where ``embeddings`` has shape
        ``(N, 512)``, ``labels`` is ``(N,)``, and ``filenames`` is a list
        of ``N`` path strings.
    """
    print(f"Generating embeddings for all {len(all_paths)} EuroSAT images...")
    dataset = SimplePathDataset(all_paths, config.IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embs, all_labels, all_files = [], [], []
    class_to_idx = {name: i for i, name in enumerate(config.CLASS_NAMES)}

    extractor.backbone.eval()
    with torch.no_grad():
        for images, paths in tqdm(loader, desc="Extracting embeddings"):
            images = images.to(device)
            embs = extractor.extract_batch(images).cpu().numpy()
            all_embs.append(embs)
            for p in paths:
                cls_name = Path(p).parent.name
                all_labels.append(class_to_idx[cls_name])
                all_files.append(p)

    embeddings = np.concatenate(all_embs, axis=0)
    labels = np.array(all_labels)

    # Save to disk
    EmbeddingExtractor.save_embeddings(config.EMBEDDINGS_FILE, embeddings, labels, all_files)
    print(f"Embeddings saved: shape={embeddings.shape}")
    return embeddings, labels, all_files


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    t_start = time.time()

    # ── 1. Discover all EuroSAT paths ─────────────────────────────────
    print("Discovering EuroSAT image paths...")
    all_paths = _discover_paths(config.EUROSAT_ROOT)
    class_to_idx = {name: i for i, name in enumerate(config.CLASS_NAMES)}
    print(f"Found {len(all_paths)} images across {len(class_to_idx)} classes")

    # ── 2. Generate & save full embeddings (if not already cached) ────
    extractor = EmbeddingExtractor(config.PHASE2_FINAL_PATH, device)
    if config.EMBEDDINGS_FILE.exists():
        data = EmbeddingExtractor.load_embeddings(config.EMBEDDINGS_FILE)
        embeddings_full, labels_full, filenames_full = (
            data["embeddings"], data["labels"], data["filenames"]
        )
        print(f"Loaded cached embeddings: shape={embeddings_full.shape}")
    else:
        embeddings_full, labels_full, filenames_full = generate_full_embeddings(
            extractor, all_paths, device
        )

    # ── 3. Simulate temporal pairs via offset-based regions ────────────
    print("Simulating temporal pairs using spatial-offset regions...")
    generator = RegionPairGenerator(seed=config.SEED, block_size=config.SPATIAL_BLOCK_SIZE)
    paths_t1, paths_t2, labels, class_t1, class_t2, region_ids = generator.generate(
        all_paths,
        class_to_idx,
        config.CLASS_NAMES,
        num_unchanged=config.NUM_UNCHANGED_PAIRS,
        num_changed=config.NUM_CHANGED_PAIRS,
    )
    generator.save_metadata(
        config.CHANGE_PAIRS_PATH, paths_t1, paths_t2, labels, class_t1, class_t2
    )

    # ── 4. Build embedding lookup and assemble pair arrays ────────────
    embs_dict = dict(zip(filenames_full, embeddings_full))
    # Verify all pair paths are present
    missing = [p for p in paths_t1 + paths_t2 if p not in embs_dict]
    if missing:
        print(f"WARNING: {len(missing)} image paths not found in embeddings dict, extracting on the fly")
        unique_missing = list(set(missing))
        dataset = SimplePathDataset(unique_missing, config.IMAGE_SIZE)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        with torch.no_grad():
            for images, paths in loader:
                images = images.to(device)
                embs = extractor.extract_batch(images).cpu().numpy()
                for p, emb in zip(paths, embs):
                    embs_dict[p] = emb

    embeddings_t1 = np.stack([embs_dict[p] for p in paths_t1])
    embeddings_t2 = np.stack([embs_dict[p] for p in paths_t2])
    ground_truth = np.array(labels)

    # ── 5. Compute cosine similarities ─────────────────────────────────
    print("Computing cosine similarities...")
    similarities = ChangeDetector.compute_batch_similarity(embeddings_t1, embeddings_t2)

    # Save similarity scores
    RegionPairGenerator.save_similarity_scores(
        config.SIMILARITY_SCORES_PATH,
        paths_t1, paths_t2, labels, similarities, class_t1, class_t2,
        region_ids=region_ids,
    )

    # ── 6. ROC curve and threshold selection ───────────────────────────
    print("Generating ROC curve and selecting threshold...")
    roc_results = ChangeDetector.generate_roc(
        similarities,
        ground_truth,
        save_curve=config.ROC_CURVE_PATH,
        save_metrics=config.ROC_METRICS_PATH,
    )

    threshold_results = ChangeDetector.select_threshold(
        roc_results["fpr"],
        roc_results["tpr"],
        roc_results["thresholds"],
        roc_results["roc_auc"],
        save_path=config.CHANGE_THRESHOLD_PATH,
    )

    threshold = threshold_results["selected_threshold"]
    print(f"Optimal threshold (Youden J): {threshold}")

    # ── 7. Summary statistics ──────────────────────────────────────────
    print_stats(ground_truth, similarities, threshold, roc_results["roc_auc"])

    # ── 8. Generate change heatmaps for sample pairs ───────────────────
    print("Generating visual change heatmaps...")
    heatmap_gen = ChangeHeatmapGenerator(
        target_size=config.HEATMAP_SIZE, alpha=config.HEATMAP_ALPHA
    )

    unchanged_indices = np.where(ground_truth == 0)[0][:4]
    changed_indices = np.where(ground_truth == 1)[0][:4]
    sample_indices = np.concatenate([unchanged_indices, changed_indices])

    sample_pairs = []
    for idx in sample_indices:
        sample_pairs.append({
            "image_t1": paths_t1[idx],
            "image_t2": paths_t2[idx],
            "similarity": float(similarities[idx]),
            "threshold": threshold,
        })

    heatmap_gen.generate_batch(
        sample_pairs,
        output_dir=config.HEATMAP_DIR,
        summary_path=config.HEATMAP_SUMMARY_PATH,
    )

    elapsed = time.time() - t_start
    print(f"\nChange detection pipeline complete  |  elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()

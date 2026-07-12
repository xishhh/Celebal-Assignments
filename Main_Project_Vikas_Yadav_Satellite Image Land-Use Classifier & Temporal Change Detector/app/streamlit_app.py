"""Streamlit front-end integrating all backend modules.

Usage
-----
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# ── Ensure project root is on sys.path ──────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import config
from src.change_detection import ChangeDetector
from src.heatmap import ChangeHeatmapGenerator
from src.transfer_model import TransferLearningModel
from src.transforms import get_eval_transforms
from src.utils import load_checkpoint

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite Land-Use Classifier & Temporal Change Detector",
    layout="wide",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════
#  CACHED RESOURCES
# ═════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading classification model ...")
def load_classification_model() -> torch.nn.Module:
    """Load the trained TransferLearningModel from the final checkpoint."""
    ckpt = config.PHASE2_FINAL_PATH
    if not ckpt.exists():
        st.error(f"Checkpoint not found: {ckpt}")
        st.stop()

    model = TransferLearningModel(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        dropout=config.DROPOUT,
    )
    load_checkpoint(ckpt, model)
    model = model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading change threshold ...")
def load_threshold() -> float:
    """Load the Youden threshold from ``change_threshold.json``."""
    path = config.CHANGE_THRESHOLD_PATH
    if not path.exists():
        st.warning(f"Threshold file not found: {path} — using default {config.THRESHOLD}")
        return config.THRESHOLD

    with open(path) as f:
        data = json.load(f)
    return float(data["selected_threshold"])


# ═════════════════════════════════════════════════════════════════════
#  INFERENCE HELPERS
# ═════════════════════════════════════════════════════════════════════

def predict_class(
    image: Image.Image,
    model: torch.nn.Module,
) -> tuple[str, float]:
    """Return ``(class_name, confidence)`` for a PIL image."""
    transform = get_eval_transforms(config.IMAGE_SIZE)
    tensor = transform(image).unsqueeze(0).to(DEVICE, non_blocking=True)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_idx = probs.max(dim=1)

    return config.CLASS_NAMES[top_idx.item()], float(top_prob.item())


def extract_embedding(
    image: Image.Image,
    model: torch.nn.Module,
) -> np.ndarray:
    """Return the 512-d backbone embedding for a PIL image."""
    transform = get_eval_transforms(config.IMAGE_SIZE)
    tensor = transform(image).unsqueeze(0).to(DEVICE, non_blocking=True)

    with torch.no_grad():
        emb = model.backbone(tensor)

    return emb.squeeze(0).cpu().numpy()


def save_uploaded(uploaded_file) -> Path:
    """Save a Streamlit ``UploadedFile`` to a temporary file and return its path."""
    suffix = Path(uploaded_file.name).suffix if uploaded_file.name else ".png"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(uploaded_file.read())
    tmp.close()
    return Path(tmp.name)


# ═════════════════════════════════════════════════════════════════════
#  UI — SIDEBAR
# ═════════════════════════════════════════════════════════════════════

st.title(
    "Satellite Image Land-Use Classifier  \n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &  \nTemporal Change Detector"
)

st.sidebar.header("Upload Satellite Images")
img_t1_file = st.sidebar.file_uploader("Image T1 (earlier)", type=["png", "jpg", "jpeg", "tif", "tiff"])
img_t2_file = st.sidebar.file_uploader("Image T2 (later)", type=["png", "jpg", "jpeg", "tif", "tiff"])

run_button = st.sidebar.button("Run Analysis", type="primary")

# ── Cancel gracefully if fewer than two images are provided ─────────
if run_button and (img_t1_file is None or img_t2_file is None):
    st.sidebar.error("Please upload both T1 and T2 images.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════

if run_button and img_t1_file is not None and img_t2_file is not None:
    # ── Load resources ────────────────────────────────────────────────
    model = load_classification_model()
    threshold = load_threshold()

    # ── Temp-persist uploaded images ──────────────────────────────────
    t1_path = save_uploaded(img_t1_file)
    t2_path = save_uploaded(img_t2_file)

    # Reload as PIL for display and classification
    pil_t1 = Image.open(t1_path).convert("RGB")
    pil_t2 = Image.open(t2_path).convert("RGB")

    # ══════════════════════════════════════════════════════════════════
    #  STEP 1 — Display loaded images
    # ══════════════════════════════════════════════════════════════════
    st.subheader("Step 1 — Loaded Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_t1, caption="T1 — Earlier", use_container_width=True)
    with col2:
        st.image(pil_t2, caption="T2 — Later", use_container_width=True)
    st.divider()

    # ══════════════════════════════════════════════════════════════════
    #  STEP 2 — Land-use classification
    # ══════════════════════════════════════════════════════════════════
    st.subheader("Step 2 — Land-Use Classification")

    cls1, conf1 = predict_class(pil_t1, model)
    cls2, conf2 = predict_class(pil_t2, model)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("T1 Predicted Class", cls1, f"{conf1:.1%}")
    with col2:
        st.metric("T2 Predicted Class", cls2, f"{conf2:.1%}")
    st.divider()

    # ══════════════════════════════════════════════════════════════════
    #  STEP 3 — Embedding extraction
    # ══════════════════════════════════════════════════════════════════
    st.subheader("Step 3 — Embedding Extraction")

    emb1 = extract_embedding(pil_t1, model)
    emb2 = extract_embedding(pil_t2, model)

    st.info(f"Embedding dimension: **{emb1.shape[0]}** (ResNet-18 backbone, GAP → 512-d)")
    st.divider()

    # ══════════════════════════════════════════════════════════════════
    #  STEP 4 — Cosine similarity
    # ══════════════════════════════════════════════════════════════════
    st.subheader("Step 4 — Cosine Similarity")

    similarity = float(ChangeDetector.compute_similarity(emb1, emb2))

    st.metric("Cosine Similarity", f"{similarity:.4f}")
    st.divider()

    # ══════════════════════════════════════════════════════════════════
    #  STEP 5 — Change decision
    # ══════════════════════════════════════════════════════════════════
    st.subheader("Step 5 — Change Decision")

    decision = ChangeDetector.predict_change(emb1, emb2, threshold)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Similarity", f"{decision['similarity']:.4f}")
    with col2:
        st.metric("Threshold", f"{decision['threshold']:.4f}")
    with col3:
        status = "CHANGE DETECTED" if decision["changed"] else "NO CHANGE"
        st.metric("Status", status)
    st.divider()

    # ══════════════════════════════════════════════════════════════════
    #  STEP 6 — Visual change heatmap
    # ══════════════════════════════════════════════════════════════════
    st.subheader("Step 6 — Visual Change Heatmap")

    generator = ChangeHeatmapGenerator(
        target_size=config.HEATMAP_SIZE,
        alpha=config.HEATMAP_ALPHA,
    )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        heatmap_out = Path(tmp.name)

    generator.generate_heatmap(
        image_t1=t1_path,
        image_t2=t2_path,
        similarity=decision["similarity"],
        threshold=decision["threshold"],
        save_path=heatmap_out,
    )
    st.image(str(heatmap_out), caption="Change Heatmap (T1 / T2 / Difference / Overlay)", use_container_width=True)
    st.divider()

    # ══════════════════════════════════════════════════════════════════
    #  STEP 7 — Summary card
    # ══════════════════════════════════════════════════════════════════
    st.subheader("Step 7 — Overall Summary")

    with st.container(border=True):
        row1, row2, row3, row4 = st.columns(4)

        with row1:
            st.html(
                "<div style='text-align:center'>"
                "<p style='font-size:0.85rem;margin-bottom:0'><b>Image 1</b></p>"
                f"<p style='font-size:1.3rem;margin:0'>{cls1}</p>"
                f"<p style='font-size:0.95rem'>{conf1:.1%}</p>"
                "</div>"
            )

        with row2:
            st.html(
                "<div style='text-align:center'>"
                "<p style='font-size:0.85rem;margin-bottom:0'><b>Image 2</b></p>"
                f"<p style='font-size:1.3rem;margin:0'>{cls2}</p>"
                f"<p style='font-size:0.95rem'>{conf2:.1%}</p>"
                "</div>"
            )

        with row3:
            st.html(
                "<div style='text-align:center'>"
                "<p style='font-size:0.85rem;margin-bottom:0'><b>Similarity</b></p>"
                f"<p style='font-size:1.3rem;margin:0'>{decision['similarity']:.4f}</p>"
                f"<p style='font-size:0.95rem'>Threshold: {decision['threshold']:.4f}</p>"
                "</div>"
            )

        with row4:
            st.html(
                "<div style='text-align:center'>"
                "<p style='font-size:0.85rem;margin-bottom:0'><b>Status</b></p>"
                f"<p style='font-size:1.3rem;margin:0'>{status}</p>"
                "</div>"
            )

    # ── Clean up temp files ─────────────────────────────────────────
    t1_path.unlink(missing_ok=True)
    t2_path.unlink(missing_ok=True)
    heatmap_out.unlink(missing_ok=True)

else:
    st.info("Upload two satellite images in the sidebar and click **Run Analysis**.")

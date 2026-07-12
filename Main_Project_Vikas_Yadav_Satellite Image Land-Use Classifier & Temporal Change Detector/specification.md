Satellite Image Land-Use Classifier & Temporal Change Detector
1. Overview
Build a computer vision system that classifies land-use types from satellite imagery and detects land-cover changes between two time periods. The project covers transfer learning, embedding-based change detection, and a working interactive dashboard.

Primary dataset
EuroSAT — 27,000 satellite tiles, 10 land-use classes (provided)
Holdout test set
UC Merced Land Use — 2,100 images, 21 classes (provided)
Base model
ResNet-18 or EfficientNet-B0 via torchvision (your choice)
Deliverable
GitHub repo + Streamlit/Gradio app + PDF report (max 8 pages)

2. What to Build
Module 1 — Land-Use Classifier
Train a CNN to classify satellite tiles into 10 land-use categories using transfer learning from a pretrained backbone.
You must implement a two-phase fine-tuning strategy:
Phase 1 — freeze the backbone, train only the classifier head for 3 epochs
Phase 2 — unfreeze the last 2 convolutional blocks, reduce learning rate by 10×, train for 5 more epochs
Report per-class F1, macro-F1, and a confusion matrix on both EuroSAT validation and UC Merced holdout.
Module 2 — Temporal Change Detector
Reuse Module 1's backbone as a feature extractor. Strip the classifier head and extract 512-dimensional embeddings for every tile.
Simulate a time series by partitioning EuroSAT into geographic regions and assigning them to T1 (before) and T2 (after) splits. Compute cosine similarity between T1 and T2 embeddings per region. Tile pairs below a similarity threshold are flagged as changed.
You must produce a ROC curve, select a threshold, and justify your operating point. Output a visual change heatmap for at least 5 sample region pairs.
Module 3 — Geo-Dashboard
Build a Streamlit or Gradio app that accepts two satellite tile images (before and after) and displays:
Predicted land-use class and confidence score for each tile
Cosine similarity score between their embeddings
Side-by-side heatmap with change flag if similarity falls below threshold
The app must run locally with no internet dependency after setup.

3. Suggested Timeline

Phase
Tasks
Data & setup
Download EuroSAT. Visualise 5 samples per class. Plot class distribution. Build data pipeline with spatial block train/val/test split.
Baseline CNN
Train a 3-layer scratch CNN as baseline. Log loss curves. Record per-class F1. This is the floor all future results are compared against.
Transfer learning
Fine-tune ResNet-18 or EfficientNet-B0 using the two-phase strategy. Ablation: frozen vs unfrozen results. Run on UC Merced holdout.
Change detector
Extract embeddings. Build cosine-similarity change detector. ROC curve and threshold selection. Generate change heatmaps.
Dashboard
Build Streamlit or Gradio app. Integrate classifier, similarity scorer, and heatmap. Test with unseen tile pairs.
Evaluation
Full evaluation on UC Merced holdout. Per-class metrics table. Error analysis on top-5 misclassified pairs. Spatial leakage experiment.
Report & polish
Write PDF report (max 8 pages). Clean up repo. Record 3-minute demo video. Final submission.

4. Deliverables
All eight items below are required for full marks.

#
Deliverable
What is expected
1
Data pipeline
Reproducible notebook. Spatial block split documented. Class distribution plot.
2
Baseline CNN
Scratch-trained 3-layer CNN. Train/val loss curves. Per-class F1.
3
Fine-tuned model
Saved .pt checkpoint. Frozen vs unfrozen comparison table. Confusion matrix on UC Merced.
4
Change detection module
Embedding extractor. Cosine similarity diff. ROC curve. Change heatmaps for 5 pairs.
5
Geo-dashboard
Working app. Correct outputs for uploaded tile pairs. Runs locally.
6
Spatial leakage write-up
Quantified experiment: random-split accuracy vs block-split accuracy with written explanation.
7
Error analysis
Top-5 misclassified pairs shown visually. Hypothesis for each failure.
8
PDF report + demo video
Max 8 pages covering problem, method, results, and limitations. 3-minute screen recording.

5. Grading

Bonus Tasks


Bonus
Requirement
A
GradCAM visualisation
Implement GradCAM on the fine-tuned model. Overlay heatmap showing which pixels drove each classification. Interpret at least 3 examples.
B
Multi-threshold toggle
Implement three operating points (high recall / balanced / high precision) in the dashboard. User can toggle and see how the change map shifts.
C
Embedding visualisation
Project all 27,000 EuroSAT embeddings to 2D using t-SNE or UMAP. Colour by class. Compare scratch CNN vs fine-tuned embeddings side by side.
D
Imbalance experiment
Downsample two classes to 20% of their size, retrain, compare F1. Apply one mitigation (weighted loss, oversampling, or Mixup). 1-page analysis.


6. Submission Checklist
GitHub repository — clean, with README and requirements.txt
All notebooks runnable top-to-bottom with no errors
Saved model checkpoint (.pt file) committed or linked via Git LFS
Streamlit or Gradio app tested locally before submission
PDF report — maximum 8 pages, including figures
3-minute demo video — screen recording of the live dashboard
If attempting bonuses — clearly labelled in the repo and flagged in the report
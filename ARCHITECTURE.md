# Arche — Brain Tumor Analysis Pipeline

This project is a multi-phase deep learning pipeline for analyzing brain MRI scans. It starts from raw images, detects whether a tumor exists, classifies what kind it is, segments it at voxel level, and then builds a 3D graph of supervoxels to refine and explain the segmentation. There are two graph-based research plans, each tackling the refinement differently.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Base Pipeline (Phases 1–3)](#base-pipeline-phases-13)
- [Supervoxel Preprocessing](#supervoxel-preprocessing)
- [Plan 1: Binary Refinement + Explainability](#plan-1-binary-refinement--explainability)
- [Plan 2: Multi-Task Learning + Explainability](#plan-2-multi-task-learning--explainability)
- [How the Pieces Connect](#how-the-pieces-connect)

---

## Project Structure

```
arche/
├── config.py              # shared config for Phases 1–3
├── prediction.py          # Phase 1: binary tumor detection
├── classification.py      # Phase 2: tumor type classification
├── segmentation.py        # Phase 3: voxel-level segmentation (DeepLabV3+)
├── pipeline.py            # orchestrator: chains Phases 1–3
│
├── plan1/                 # GNN Plan 1 (self-contained)
│   ├── config.py
│   ├── supervoxel.py      # SVGFormer preprocessing
│   ├── model.py           # TumorRefiner (binary + edge)
│   └── explainability.py  # 3-level traces
│
├── plan2/                 # GNN Plan 2 (self-contained)
│   ├── config.py
│   ├── supervoxel.py      # re-exports from plan1
│   ├── model.py           # MultiTaskRefiner (regression + edge + uncertainty)
│   └── explainability.py  # 5-level traces
│
├── BraTS/                 # BraTS 2023 GLI dataset (NIfTI volumes)
└── Brain MRI ND-5 Dataset/# 2D MRI dataset (tumor classification)
```

---

## Base Pipeline (Phases 1–3)

These three phases work on 2D MRI slices and use standard CNNs. They run in sequence.

### Phase 1 — Binary Tumor Detection (`prediction.py`)

**What it does:** Takes a single brain MRI image and answers "is there a tumor or not?"

- **Model:** ConvNeXt-Base (pretrained on ImageNet), with the final layer swapped to output 1 value instead of 1000
- **Dataset:** Brain MRI ND-5 — contains 2D MRI images in four folders (glioma, meningioma, pituitary, no_tumor)
- **Training:** First freezes the backbone for 4 epochs (only trains the new classification head), then unfreezes everything for 16 more epochs with a lower learning rate
- **Output:** A probability between 0 and 1 — above 0.5 means tumor detected

### Phase 2 — Tumor Type Classification (`classification.py`)

**What it does:** If Phase 1 says tumor exists, this tells you what kind — glioma, meningioma, or pituitary.

- **Model:** ConvNeXt-Base with dual pooling (average + max) — captures both broad patterns and sharp features
- **Loss:** Focal Loss — pays more attention to hard-to-classify cases and handles class imbalance
- **GradCAM:** After training, generates heatmaps showing which parts of the image the model focused on. These heatmaps are saved as pseudo segmentation masks for potential downstream use
- **Output:** One of three tumor types, plus confidence scores

### Phase 3 — Voxel-Level Segmentation (`segmentation.py`)

**What it does:** Takes a full 3D brain MRI scan (4 modalities: T1, T1-contrast, T2, T2-FLAIR) and labels every voxel as one of four classes: background (BG), necrotic core (NCR), peritumoral edema (ED), or enhancing tumor (ET).

- **Model:** DeepLabV3+ with EfficientNet-B4 encoder — processes 2D axial slices
- **Input:** 4-channel input (one channel per MRI modality), each slice resized to 224×224
- **Loss:** Combined Dice + Cross-Entropy — Dice handles class imbalance, CE provides stable gradients
- **Evaluation:** Per-class Dice scores plus official BraTS region metrics (Whole Tumor, Tumor Core, Enhancing Tumor)
- **Export:** After training, exports two things consumed by the GNN:
  1. Predicted segmentation masks
  2. Full 3D softmax probability volumes (4-class probability at every voxel) — these become input features for the graph pipeline

---

## Supervoxel Preprocessing

Both Plan 1 and Plan 2 share the same preprocessing pipeline. This is the bridge between voxel-level segmentation and graph-level analysis. The code lives in `plan1/supervoxel.py`.

### What is a supervoxel?

Instead of working with individual voxels (millions per scan), we group nearby similar voxels into clusters called supervoxels. A typical brain scan produces ~800 supervoxels after pruning. Each supervoxel becomes one node in the graph.

### The Pipeline (7 Steps)

**Step 1 — 3D SLIC Clustering.** The T1-native modality volume is partitioned into ~1000 supervoxels using SLIC (Simple Linear Iterative Clustering). SLIC groups voxels that are spatially close and have similar intensity values. A low compactness (0.1) means intensity similarity matters more than spatial regularity, so supervoxels follow tissue boundaries more closely.

**Step 2 — Background Pruning.** Many supervoxels are just empty background (air, skull). These are removed using the largest-gap heuristic: compute the mean T1 intensity of each supervoxel, sort them, find the biggest gap between consecutive values, and cut there. Everything below the gap is discarded. Supervoxels with fewer than 20 voxels are also removed. This typically reduces ~1000 SVs down to ~800.

**Step 3 — Ground Truth Computation.** For each remaining supervoxel, compute labels from the voxel-level ground truth segmentation:

- `y_reg`: the fraction of voxels inside this SV that are tumor (0.0 = all background, 1.0 = all tumor)
- `y_cls`: binary label — is this SV tumor? (1 if y_reg > 0.15, else 0)
- `y_dominant`: which tissue class appears most often inside this SV (0=BG, 1=NCR, 2=ED, 3=ET)
- `centroid`: the average (x, y, z) coordinate of all voxels in this SV

**Step 4 — Segmentation Prior Features.** If available, load the 3D softmax probability volume exported by Phase 3's DeepLabV3+. For each supervoxel, average the probabilities across its voxels to produce a 5-dimensional feature vector: [P(BG), P(NCR), P(ED), P(ET), entropy]. This gives the GNN access to what the segmentation model "thought" about each region. Entropy measures how uncertain the seg model was.

**Step 5 — Patch Extraction.** For each supervoxel, we need a compact representation of its raw MRI content. Following the SVGFormer paper:

1. Use k-means++ to find 4 representative centroids within the SV
2. For each centroid, find its 16 nearest voxels
3. For each modality (T1n, T1c, T2w, T2f), collect the intensity values at those 16 neighbors
4. Append the centroid's (x, y, z) coordinates

This produces a patch tensor of shape (16, 19) per supervoxel — 16 rows (4 patches × 4 modalities), each row having 19 values (16 intensity neighbors + 3 spatial coordinates).

**Step 6 — kNN Graph Construction.** Connect supervoxels into a graph. Each SV is linked to its 8 nearest neighbors based on Euclidean distance between centroids. Edge attributes encode the spatial relationship: [dx, dy, dz, distance]. The graph is symmetrized (if A connects to B, then B connects to A).

**Step 7 — Edge Ground Truth.** For each edge connecting two supervoxels, compute what kind of tissue boundary it crosses. With 4 tissue classes, there are 10 possible symmetric boundary types (BG↔BG, BG↔NCR, BG↔ED, BG↔ET, NCR↔NCR, NCR↔ED, NCR↔ET, ED↔ED, ED↔ET, ET↔ET). Also compute the transition gradient — the absolute difference in tumor proportion between the two connected SVs.

---

## Plan 1: Binary Refinement + Explainability

**Research Question:** Can a GNN, given the segmentation model's output as a prior, correct segmentation errors by reasoning over the graph neighborhood?

### Architecture (~885K parameters, ~3.5 MB)

```
                  Patch Tensor (16×19 per SV)
                           │
                    ┌──────┴──────┐
                    │ PatchEmbedder│  ← Transformer (3 layers, 4 heads)
                    │  + seg prior │     Converts patches → 128-dim embedding
                    └──────┬──────┘
                           │
                    128-dim node embedding
                           │
                     + Laplacian PE (8-dim positional encoding)
                           │
                    ┌──────┴──────┐
                    │ GraphEncoder │  ← GATv2 (3 layers, 4 heads)
                    │  + residuals │     Message passing over kNN graph
                    └──────┬──────┘
                           │
                    128-dim graph-aware embedding
                           │
                    ┌──────┴──────┐
                    │              │
              ┌─────┴─────┐ ┌─────┴─────┐
              │  NodeHead  │ │  EdgeHead  │
              │  Binary    │ │  10-class  │
              │  tumor?    │ │  boundary  │
              └───────────┘ └───────────┘
```

**PatchEmbedder:** Takes the 16×19 patch tensor, projects each row to 128 dimensions, adds modality embeddings (so the model knows which MRI type each row comes from) and patch position embeddings, prepends a learnable [CLS] token, and runs everything through a 3-layer Transformer. The [CLS] token's output is the supervoxel's embedding. If seg priors are available, they're concatenated and projected through an MLP.

**GraphEncoder:** Takes all node embeddings and passes them through 3 layers of GATv2 (Graph Attention Network v2). GATv2 learns which neighbors are most important for each node — a tumor SV might attend strongly to other tumor neighbors. Laplacian Positional Encoding (8-dim) gives each node a sense of its position in the graph topology. Residual connections and multi-scale fusion (concatenating outputs of all 3 layers) prevent information loss.

**NodeHead (Task 1):** A small MLP (128→64→1) that predicts whether each supervoxel is tumor or not. This is the refinement task — it can correct mistakes the seg model made by using graph neighborhood context.

**EdgeHead (Task 2):** For each edge, concatenates the source embedding, destination embedding, their absolute difference, and the spatial edge attributes. Feeds this through an MLP to predict one of 10 boundary types. This tells us what kind of tissue transition occurs at each graph edge.

**Loss:** Weighted sum of BCE (for nodes) and cross-entropy (for edges).

### Explainability (3 Levels)

**Level 1 — Graph Attention.** For any node, we can look at which neighbors the GATv2 layers attended to most strongly. This answers: "This SV was classified as tumor because its neighbors X, Y, Z had high attention scores, and they are also tumor regions."

**Level 2 — Patch Attention.** Within each supervoxel, we extract the Transformer's self-attention weights from [CLS] to each patch row. This answers: "The model focused most on the T1c and T2f patches near centroid (x,y,z)."

**Level 3 — Refinement Comparison.** Compare the GNN's prediction against the seg model's prediction for each SV. Find places where the GNN corrected the seg model (true positive corrections) and places where it made things worse (degradations). For corrections, trace back through graph attention to find which neighbors drove the fix.

---

## Plan 2: Multi-Task Learning + Explainability

**Research Question:** Can a single supervoxel encoder simultaneously learn tumor proportion regression, boundary classification, and segmentation uncertainty — three tasks that a standard segmentation model cannot individually address?

### Architecture (~919K parameters, ~3.7 MB)

The shared backbone (PatchEmbedder + GraphEncoder) is identical to Plan 1. The difference is in the task heads:

```
                  Shared Backbone
                  (same as Plan 1)
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
    ┌──────────────┐ ┌────────┐ ┌─────────────┐
    │ RegressionHead│ │EdgeHead│ │UncertaintyHead│
    │  Ensemble ×4  │ │10-class│ │   Binary     │
    │  y_reg ∈[0,1] │ │boundary│ │  seg error?  │
    └──────────────┘ └────────┘ └─────────────┘
```

**RegressionHead (Task 1):** Instead of binary classification, predicts a continuous tumor proportion between 0 and 1. Uses an ensemble of 4 parallel MLPs — each independently predicts y_reg, then a learned attention mechanism weights their predictions. The ensemble provides two benefits: (1) the averaged prediction is more robust, and (2) the variance across the 4 predictions serves as a built-in uncertainty estimate.

**EdgeHead (Task 2):** Same as Plan 1 — 10-class boundary type prediction.

**UncertaintyHead (Task 3):** Predicts whether the segmentation model was wrong at each supervoxel. The target is binary: was the seg model's predicted class different from the ground truth? The model learns to detect where the seg model fails by cross-referencing the seg probability vector (which is an input feature) with graph neighborhood context. If neighbors all agree on one class but the seg model predicted something else for this node, the graph attention can catch that inconsistency.

**Multi-Task Loss (Kendall et al., 2018):** Instead of manually tuning the weight of each task's loss, the model learns a noise parameter σ for each task. Tasks with higher inherent noise get automatically down-weighted:

```
L = (1/2σ₁²)·L_reg + (1/2σ₂²)·L_edge + (1/2σ₃²)·L_unc + log(σ₁σ₂σ₃)
```

The three σ values are learnable parameters that adjust during training.

### Explainability (5 Levels)

Levels 1 and 2 are the same as Plan 1 (graph attention, patch attention). The remaining three:

**Level 3 — Regression Refinement.** Adapted from Plan 1's refinement trace, but uses continuous regression instead of binary classification. Computes MAE, R², and zero-shot Dice (threshold the continuous prediction to recover a binary mask, then compute Dice against ground truth). Also reports ensemble variance as a proxy for model confidence.

**Level 4 — Task Divergence.** Compares the outputs of all 3 task heads for each node and categorizes the agreement pattern:

- **AGREEMENT_CONFIDENT:** High tumor regression + correct boundary type + low uncertainty → the model is very sure about this region
- **AGREEMENT_NEGATIVE:** Low regression + BG boundary + low uncertainty → confidently not tumor
- **DISAGREEMENT_REG_UNC:** High regression (sees tumor) but also high uncertainty (doesn't trust the seg model) → interesting boundary region
- **DISAGREEMENT_EDGE:** Regression says tumor but all edges say BG↔BG → possible isolated misclassification
- **UNCERTAIN_BOUNDARY:** Non-trivial edge types + high uncertainty → ambiguous tissue boundary (clinically interesting)

**Level 5 — Uncertainty-Driven Explanation.** For supervoxels where the uncertainty head predicts high seg error:

- Was the seg model actually wrong? (ground truth check)
- Does the regression head compensate? (if seg says BG but regression says tumor)
- What do the neighbors look like? (are they also uncertain?)
- AUROC: how well does the uncertainty head rank actual seg errors?
- Correlation between ensemble variance and uncertainty probability

---

## How the Pieces Connect

```
Brain MRI ND-5 Dataset           BraTS 2023 GLI Dataset
        │                                │
   Phase 1: Binary Detection        Phase 3: Segmentation
   Phase 2: Type Classification          │
        │                          exports probability volumes
        │                                │
        └────────── pipeline.py ─────────┘
                                         │
                              ┌──────────┴──────────┐
                              │  Supervoxel Pipeline │
                              │   (plan1/supervoxel) │
                              │   SLIC → prune →     │
                              │   GT → patches →     │
                              │   kNN → edge GT      │
                              └──────────┬──────────┘
                                         │
                           supervoxel graph data (.pt)
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                         plan1/model             plan2/model
                        TumorRefiner          MultiTaskRefiner
                        (binary + edge)     (reg + edge + unc)
                              │                     │
                     plan1/explainability    plan2/explainability
                       (3 levels)             (5 levels)
```

The segmentation model (Phase 3) runs first to produce probability volumes. These probabilities become input features for the GNN — the graph model can see what the seg model predicted and learn to correct it. The two plans offer different ways to do this correction: Plan 1 uses binary classification, Plan 2 uses continuous regression with multi-task learning.

Both plans are designed to run on an RTX 3060 12GB. The models are kept small (~900K parameters) and process 2 graphs per batch with 4-step gradient accumulation (effective batch size of 8).

---

## Hardware and Dependencies

- **GPU:** RTX 3060 12GB (all models fit comfortably)
- **Key libraries:** PyTorch, PyTorch Geometric (GATv2), nibabel (NIfTI loading), scikit-image (SLIC), segmentation-models-pytorch (DeepLabV3+)
- **Dataset formats:**
  - Brain MRI ND-5: 2D PNG/JPG images in class folders
  - BraTS 2023 GLI: 3D NIfTI volumes (.nii.gz), 4 modalities + ground truth segmentation per patient

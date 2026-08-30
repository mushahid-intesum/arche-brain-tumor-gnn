# BraTS 3D GNN Edge Prediction Pipeline

## Overview

This pipeline (`04_brats_gnn.py`) builds **3D volumetric tumor graphs** from BraTS segmentation outputs and performs edge prediction using an NCN (Neural Common Neighbor) architecture. Each patient's tumor is represented as a single graph where nodes are tissue sub-regions across multiple axial slices, and edges capture both within-slice and cross-slice spatial relationships.

---

## Why a GNN on Tumor Regions?

A segmentation model answers: *"what is this pixel?"* — but it doesn't answer: *"how are tumor regions related to each other?"*

The GNN addresses structural questions that segmentation cannot:
- **Do two enhancing regions on adjacent slices belong to the same tumor mass?**
- **Is this edema region infiltrative (connected to enhancing tumor) or reactive (isolated)?**
- **Which tissue transitions (e.g., ET→ED, NCR→ET) are most structurally significant?**

By operating on a graph of regions rather than a grid of pixels, the GNN reasons about tumor **topology** — the spatial and biological relationships between tissue compartments.

---

## Data Flow

```
BraTS NIfTI volumes
        │
        ▼
[03_brats_segmentation.py]
  DeepLabV3+ (4-channel → 4-class)
        │
        ▼
  brats_outputs/
  ├── masks/        (predicted multi-class masks)
  ├── raw_slices/   (4-channel MRI slices)
  └── metadata.pt
        │
        ▼
[04_brats_gnn.py]   ← this pipeline
  3D Graph Construction → NCN Training → Reasoning Traces
```

---

## Graph Construction

### Nodes

Each node represents a **connected component** of a single tissue type within a single axial slice.

For each slice:
1. Separate the predicted mask into 3 binary masks (NCR=1, ED=2, ET=3)
2. Run `cv2.connectedComponentsWithStats` on each binary mask
3. Each blob with area ≥ 10 pixels → 1 graph node

A patient with 40 tumor slices and an average of 3-5 regions per slice will produce a graph with ~120-200 nodes.

### Node Features (35 dimensions)

| Dims | Source | Features | Rationale |
|------|--------|----------|-----------|
| 0-2 | **3D Position** | cx/img, cy/img, slice_idx/total | Where in the volume is this region? |
| 3-5 | Morphology | Area, width, height (normalized) | How large/shaped is the region? |
| 6-7 | Shape | Aspect ratio, solidity | Compact vs. irregular region? |
| 8-10 | **Tissue type** | One-hot [NCR, ED, ET] | What tissue is this node? |
| 11-14 | T1n intensity | Mean, std, range, skewness | Anatomical tissue properties |
| 15-18 | **T1c intensity** | Mean, std, range, skewness | Enhancement = active tumor vasculature |
| 19-22 | T2w intensity | Mean, std, range, skewness | Fluid/edema signal |
| 23-26 | T2f intensity | Mean, std, range, skewness | Edema boundary clarity |
| 27-28 | Boundary | Gradient magnitude, texture contrast | Sharp boundary = well-circumscribed; weak = infiltrative |
| 29-32 | **Cross-modal** | T1c/T1n ratio, T2w/T2f ratio, T1c-T2w diff, FLAIR-T2w diff | Multi-modal tissue signatures |
| 33-34 | Slice context | Relative z-position, tumor area ratio | Where in the tumor's axial extent? |

### Edges

**Two types of edges**:

1. **Intra-slice edges** (within a single slice):
   - KNN graph (k=5) based on 2D centroid distance
   - Connects spatially nearby regions on the same slice
   - Example: an enhancing rim region connects to the adjacent edema region

2. **Inter-slice edges** (across consecutive slices):
   - Connect nodes from slice `s` to nodes in slices `s±1` (or `s±2`)
   - Conditions: 2D centroid distance < 50 pixels AND tissue types are compatible
   - Compatible pairs: same tissue, adjacent labels (NCR↔ED, ED↔ET), or NCR↔ET
   - Captures 3D tumor continuity — how regions persist or transform across the axial dimension

### Edge Features (4 dimensions)

| Dim | Feature | What it captures |
|-----|---------|-----------------|
| 0 | Distance (normalized) | Spatial proximity |
| 1 | Angle (normalized) | Directional relationship |
| 2 | **Slice gap** | 0 for intra-slice, >0 for inter-slice — encodes 3D separation |
| 3 | **Same tissue flag** | 1 if same tissue type, 0 otherwise — encodes tissue homogeneity |

---

## Model Architecture

### NCN (Neural Common Neighbor) Framework

The model follows the **MPNN-then-SF** paradigm from ICLR 2024:
1. Learn node embeddings via message passing (GATv2)
2. Predict edges using both learned embeddings AND structural heuristics

### NCNEncoder (GATv2)

```
Input (35-dim) → Linear(35→128)
    → GATv2Conv(128→128, 4 heads) + LayerNorm + Residual
    → GATv2Conv(128→128, 4 heads) + LayerNorm + Residual
    → GATv2Conv(128→128, 4 heads) + LayerNorm + Residual
    → Linear(128→64) → Node embeddings (64-dim)
```

- **GATv2** (not GATv1): dynamic attention — computes attention *after* combining query and key, allowing the model to attend to different features for different edges
- **Residual connections**: prevent gradient degradation in deeper layers
- **Edge-attr-aware**: each GATv2 layer receives the 4-dim edge features, so attention is conditioned on spatial distance, angle, slice gap, and tissue compatibility

### NCNEdgeDecoder

For each candidate edge (i, j), the decoder aggregates **6 information signals**:

| Signal | Dim | How it works |
|--------|-----|-------------|
| **Hadamard** `h_i ⊙ h_j` | 64 | Element-wise product — captures feature agreement |
| **Concat** `[h_i; h_j]` | 128 | Full pairwise information — captures asymmetric relationships |
| **CN Pool** | 64 | Mean-pooled embeddings of common neighbors — structural context |
| **Structural Features** | 64 | Projected CN count + Jaccard + Adamic-Adar — classical heuristics |
| **Tissue-pair embedding** | 64 | `Embedding(9, 64)` — one learned vector per (src_type, dst_type) pair |
| **Edge-type embedding** | 32 | `Embedding(2, 32)` — intra-slice vs. inter-slice |
| **Total** | **416** | |

```
416-dim → LayerNorm → Linear(416→128) → GELU → Dropout(0.3)
       → Linear(128→64) → GELU → Dropout(0.2)
       → Linear(64→1) → Score
```

### Why Tissue-Pair Embedding?

The biological meaning of an edge depends on **which tissues it connects**:
- **ET→ED**: enhancing tumor invading surrounding edema — clinically significant, indicates active infiltration
- **NCR→ET**: necrotic core surrounded by enhancing rim — classic GBM morphology
- **ED→ED**: edema region continuity — expected, less informative
- **NCR→NCR**: multiple necrotic foci — suggests multifocal necrosis

The tissue-pair embedding lets the model learn these biologically distinct relationships as separate 64-dim vectors, rather than forcing a single decoder to handle all 9 pair types identically.

### Why Edge-Type Embedding?

Intra-slice and inter-slice edges have fundamentally different semantics:
- **Intra-slice**: 2D spatial adjacency (same plane)
- **Inter-slice**: 3D continuity (across planes, 1mm+ physical distance)

The 32-dim edge-type embedding lets the model learn separate representations for these two connectivity modes.

---

## Training

### Negative Sampling

**Degree-biased negative sampling**: high-degree nodes are sampled more frequently as negative endpoints, producing "harder" negatives that are more informative for training. Pure uniform sampling produces easy negatives (disconnected regions that are obviously unrelated).

### Loss

Binary cross-entropy on positive (existing edges) and negative (sampled non-edges):
```
L = BCE(sigmoid(score_pos), 1) + BCE(sigmoid(score_neg), 0)
```

### Optimization
- AdamW (lr=5e-4, weight_decay=1e-4)
- OneCycleLR scheduler
- Gradient clipping (max_norm=1.0)
- 80 epochs, per-graph training (one forward/backward per graph)

---

## Evaluation

### Metrics

| Metric | What it measures |
|--------|-----------------|
| **AUC-ROC** (overall) | Edge prediction discrimination across all edge types |
| **AP** (overall) | Precision-recall tradeoff for edge prediction |
| **Intra-slice AUC** | How well the model discriminates within-slice edges |
| **Inter-slice AUC** | How well the model discriminates cross-slice edges |
| **Per-tissue-pair confidence** | Mean predicted link strength for each of 9 (src, dst) tissue pairs |

### Reasoning Traces

For each patient graph, the top-k highest-confidence edges are annotated with human-readable explanations:

```
Edge (3→7): strong link (conf=0.94). [INTER-SLICE: slice 70→71]
  Source: ET (T1c_mean=0.82), Target: ED (T1c_mean=0.71)
  Reasoning: ET→ED tissue pair; centroid_dist=12px; CN=4, Jaccard=0.42, AA=1.23
  3D context: cross-slice gap=1 slice(s)
```

This tells a clinician:
- The model is highly confident these two regions are connected
- It's an enhancing-tumor-to-edema connection across adjacent slices
- They're spatially close (12px apart) with strong structural support (4 common neighbors)
- The cross-slice proximity suggests this is a single tumor mass extending through the axial plane

### Visualizations

1. **Per-slice graph overlay**: nodes colored by tissue type (red=NCR, green=ED, yellow=ET), edges colored by prediction confidence (RdYlGn colormap)
2. **Tissue-pair bar chart**: mean link confidence per tissue pair — reveals which tissue transitions the model considers strongest
3. **ROC curve**: overall test set discrimination performance
4. **3D scatter plot** (Phase 1): all nodes in 3D space with inter-slice edges highlighted in blue

---

## Key Design Decisions

1. **Predicted masks (not ground truth)**: The GNN operates on segmentation model output, testing the full end-to-end pipeline. Segmentation errors propagate to the graph — this is realistic for deployment.

2. **3D graphs (not 2D)**: One graph per patient volume with inter-slice edges. This captures tumor topology that is invisible in 2D slice-level analysis — critical for understanding 3D tumor extent and infiltration patterns.

3. **Tissue-type conditioning**: The decoder explicitly models tissue pair semantics via learned embeddings, rather than treating all edges identically. This is the key architectural choice that enables biologically meaningful reasoning.

4. **NCN over vanilla GNN**: Common neighbor pooling and structural features provide a strong inductive bias for graph topology, complementing the learned node embeddings from GATv2.

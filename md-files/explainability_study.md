# Intrinsic Hierarchical GNN Explainability Study

## 1. Research Motivation

Post-hoc explainability methods such as GNNExplainer and Grad-CAM are the standard approach for interpreting GNN predictions. However, these methods operate on opaque learned representations and produce explanations that lack domain grounding — they highlight important nodes or edges without connecting those highlights to clinically meaningful structures.

This study investigates an alternative: **intrinsic explainability through hierarchical graph structure**. By constructing graphs from supervoxels grouped into tissue components, our architecture generates explanations at three granularity levels by design:

1. **Structural level** — which inter-node features (common neighbors, Jaccard, Adamic-Adar, OCN residual) matter most for an edge prediction
2. **Supervoxel attention level** — which sub-regions within a tissue component are most relevant, via learned Transformer attention weights
3. **Spatial heatmap level** — voxel-level importance derived from supervoxel attention, projectable back onto the original MRI volume

The central question is whether this built-in multi-level explanation system produces explanations that are more faithful, stable, and interpretable than post-hoc baselines applied to the same model.

---

## 2. Architecture Changes from Baseline

The following changes were made to the original GNN pipeline to support the explainability study:

### 2.1 Delaunay Triangulation for Intra-SV Edges

**File**: `supervoxel.py` — `build_intra_sv_edges_delaunay()`

The original pipeline used KNN (k=3) to connect supervoxels within each node. This introduced a free parameter (k) that affected explanation stability. We replaced it with **Delaunay triangulation** (Barber et al., 1996), which is:
- **Parameter-free**: the triangulation is uniquely determined by SV centroid positions
- **Spatially principled**: edges connect naturally adjacent regions in 3D space
- **Reproducible**: same input always produces the same topology

Falls back to fully connected graphs for degenerate cases (<4 SVs or coplanar centroids).

### 2.2 Compatibility-Only Edge Strategy

**File**: `gnn.py` — `build_inter_edges()`

The original pipeline used KNN (k=5) + distance threshold (0.3) + tissue compatibility filtering. We decoupled this into two configurable strategies:

| Strategy | Edges connected by | Parameters |
|---|---|---|
| `compatibility_only` | All tissue-compatible node pairs | None |
| `knn_filtered` | k-nearest neighbors + tissue filter | k ∈ {2, 3, 4, 5} |

Tissue compatibility rules encode known glioblastoma anatomy:
- Same type (NCR↔NCR, ED↔ED, ET↔ET)
- Adjacent types (NCR↔ED, ED↔ET)
- Necrotic-enhancing (NCR↔ET)

The distance threshold was removed entirely. The `compatibility_only` strategy is the default for the explainability study.

### 2.3 Gradient Saliency Method

**File**: `gnn.py` — `EdgePredictor.gradient_saliency()`

A new method on the model that backpropagates through a target edge's prediction logit to produce importance scores at all three hierarchy levels:
- **Node saliency**: (N,) tensor — gradient magnitude aggregated from SV features per node
- **SV saliency**: dict mapping node_id → (K,) tensor — per-supervoxel gradient magnitude
- **Edge saliency**: (edge_attr_dim,) tensor — importance per edge attribute dimension

This enables the attention faithfulness metric (comparing learned attention to gradient-based ground truth).

### 2.4 Configurable GNN Depth

**File**: `config.py`

Default encoder depth reduced from 3 to **2 GATv2 layers** to mitigate oversmoothing on the relatively small hierarchical graphs (5-30 nodes). Both 2 and 3 layers are tested in the KNN sensitivity experiments.

---

## 3. Experiment Design

### 3.1 Configuration Matrix

The study evaluates **15 configurations** across two experiment groups, each trained with **5 seeds** (42, 123, 256, 512, 1024) for a total of 75 training runs.

#### Group 1: KNN Sensitivity (10 configs)

All components active (SV ✓, Topo ✓, OCN ✓). Tests how edge construction strategy and encoder depth affect performance and explanation quality.

| Config | Edge Strategy | k | Layers |
|---|---|---|---|
| K0_compat_L2 | compatibility_only | — | 2 |
| K2_L2 | knn_filtered | 2 | 2 |
| K3_L2 | knn_filtered | 3 | 2 |
| K4_L2 | knn_filtered | 4 | 2 |
| K5_L2 | knn_filtered | 5 | 2 |
| K0_compat_L3 | compatibility_only | — | 3 |
| K2_L3 | knn_filtered | 2 | 3 |
| K3_L3 | knn_filtered | 3 | 3 |
| K4_L3 | knn_filtered | 4 | 3 |
| K5_L3 | knn_filtered | 5 | 3 |

#### Group 2: Component Ablation (5 configs)

All use compatibility_only edges with 2 layers. Isolates the contribution of each architectural component.

| Config | SV Aggregation | Intra Topology | OCN Features |
|---|---|---|---|
| A_baseline | ✗ | ✗ | ✗ |
| B_prime_sv | ✓ | ✗ | ✗ |
| B_sv_topo | ✓ | ✓ | ✗ |
| C_ocn_only | ✗ | ✗ | ✓ |
| D_full | ✓ | ✓ | ✓ |

**D_full ≡ K0_compat_L2** — they are the same model configuration. The runner automatically deduplicates: trains once, reports under both names.

### 3.2 Training Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 1e-4 |
| Scheduler | OneCycleLR |
| Epochs | 80 |
| Gradient clipping | 1.0 |
| Loss | BCE (pos edges) + BCE (neg edges) |
| Negative sampling | Degree-biased |
| Validation | Every 10 epochs, best model restored |

---

## 4. Explainability Metrics

Five metrics are computed for every edge in the test set, then aggregated (mean ± std across edges, then across seeds).

| Metric | What It Measures | Computation |
|---|---|---|
| **Fidelity+ (Necessity)** | How much does the prediction drop when the explanation is removed? | Original logit − logit with top-k SVs/SFs zeroed out |
| **Fidelity− (Sufficiency)** | How much prediction is retained using only the explanation? | Logit with only top-k SVs/SFs − random baseline logit |
| **Sparsity** | How concise is the explanation? | Fraction of total SVs/SFs used (lower = more focused) |
| **Stability** | Are explanations consistent across runs? | SV set Jaccard similarity + SF ranking Spearman correlation between seed pairs |
| **Complexity** | Total number of distinct elements in the explanation | Count of top-k SVs + non-zero SF dimensions |

Metrics are reported at two levels:
- **Level 1 (Structural)**: over the 5-dim structural feature vector per edge
- **Level 2 (Supervoxel)**: over the attention-weighted SV set per node

A **combined** fidelity score averages both levels.

---

## 5. Post-hoc Baselines

Three post-hoc methods are applied to the trained D_full model and compared against the intrinsic explanations:

| Method | How It Works |
|---|---|
| **GNNExplainer** | Learns a soft edge mask and feature mask that maximize mutual information with the prediction (50 optimization epochs per edge) |
| **Grad-CAM** | Computes gradient-weighted activation maps from the last GATv2 layer, producing per-node importance scores |
| **Attention-only** | Uses raw GATv2 attention weights as importance scores, without structural or SV context |

Each post-hoc method produces a standardized explanation (top-k SV indices + SF ranking), which is then compared to the intrinsic explanation via:
- **SV Jaccard**: overlap of top-k SV sets
- **SF Spearman correlation**: rank correlation of structural feature importance

---

## 6. Codebase Map

### Core Pipeline

| File | Lines | Purpose |
|---|---|---|
| `config.py` | 172 | Global hyperparameters: GNN, SUPERVOXEL, SHARED |
| `supervoxel.py` | 403 | 3D SLIC supervoxel extraction + Delaunay topology |
| `gnn.py` | 1533 | Graph construction, GATv2 encoder, MultiSignal decoder, EdgePredictor, training/eval |
| `pipeline.py` | 285 | End-to-end orchestration (segmentation → graph → train → predict) |

### Ablation Module (`ablation/`)

| File | Lines | Purpose |
|---|---|---|
| `config.py` | 109 | AblationConfig dataclass + 5 presets (A-D) |
| `model.py` | 202 | AblationModel wrapper with SV/Topo/OCN switches |
| `runner.py` | 336 | Multi-config ablation orchestration |

### Explainability Module (`explainability/`)

| File | Lines | Purpose |
|---|---|---|
| `config.py` | 130 | 15 ExplainabilityConfigs + seeds/hyperparams |
| `metrics.py` | 439 | Fidelity, sparsity, stability, faithfulness, complexity |
| `posthoc.py` | 309 | GNNExplainer, Grad-CAM, attention-only wrappers |
| `runner.py` | 576 | Full study orchestration (train + XAI + post-hoc) |
| `report.py` | 1136 | 3 tables + 6 plots + smoke test |

### Dependencies

- PyTorch, PyTorch Geometric (GATv2Conv)
- scikit-image (3D SLIC)
- scipy (Delaunay triangulation)
- matplotlib (report plots)
- numpy, scikit-learn (metrics)

---

## 7. How to Run

All parameters are controlled via constants — there are no command-line arguments.

### Step 1: Configure

Edit constants in `explainability/config.py`:

```python
SEEDS = [42, 123, 256, 512, 1024]   # random seeds
EPOCHS = 80                          # training epochs per run
LR = 5e-4                           # learning rate
CONFIG_ORDER = [...]                 # which configs to run (edit to subset)
```

### Step 2: Run the Study

```bash
python -m explainability.runner
```

This trains all configs × seeds, evaluates link prediction + XAI metrics, runs post-hoc baselines on D_full, and saves results to JSON.

### Step 3: Generate Report

Edit constants at the bottom of `explainability/report.py`:

```python
RUN_SMOKE_TEST = False     # True to validate with dummy data
GENERATE_TABLES = True     # False to skip tables
GENERATE_PLOTS = True      # False to skip plots
SAVE_FORMAT = "csv"        # "console", "csv", or "latex"
```

Then run:

```bash
python -m explainability.report
```

### Smoke Test

To validate the report pipeline without training:

```python
# In explainability/report.py
RUN_SMOKE_TEST = True
```

```bash
python -m explainability.report
```

This generates dummy data (5 configs × 2 seeds) and runs the full table + plot pipeline.

---

## 8. Output Artifacts

All outputs are saved to `brats_outputs/explainability/`:

```
brats_outputs/explainability/
├── results.json               # Per-run metrics (config × seed)
├── posthoc_results.json       # Post-hoc baseline comparisons
└── report/
    ├── table1_knn_sensitivity.csv
    ├── table2_component_contribution.csv
    ├── table3_posthoc_comparison.csv
    ├── plot1_fidelity_sparsity.png
    ├── plot2_faithfulness_heatmap.png
    ├── plot3_stability_boxplots.png
    ├── plot4_component_delta.png
    ├── plot5_posthoc_radar.png
    └── plot6_example_explanations.png
```

### Report Contents

**Tables:**
1. KNN Sensitivity — AUC, AP, Fid+, Fid−, Sparsity, Complexity for each k × layer combination
2. Component Contribution — same metrics for A-D ablation, plus Δ vs A_baseline
3. Post-hoc Comparison — SV Jaccard and SF Spearman correlation for each method vs intrinsic

**Plots:**
1. Fidelity-Sparsity tradeoff scatter (ideal: top-left)
2. Faithfulness heatmap (k × layers grid)
3. Stability boxplots (per-seed Fid+ variance)
4. Component delta chart (Δ Fid+/Fid− vs baseline)
5. Post-hoc radar chart (intrinsic vs baselines)
6. Example edge explanation panels (SV attention bars)

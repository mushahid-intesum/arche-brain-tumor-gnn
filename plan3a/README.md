# Plan 3a — Explainable Survival GNN for Brain Tumor Analysis

**Hypergraph Concept Bottleneck Network with Multimodal Fusion and Faithfulness Auditing**

Plan 3a is a research pipeline that predicts survival outcomes for Glioblastoma (GBM) patients from multi-modal MRI scans using a Graph Neural Network. Unlike black-box deep learning models, every prediction is traceable through a set of interpretable imaging concepts — the model *must* explain itself to make a prediction.

---

## Motivation

Standard GNN explainability methods (GNNExplainer, PGExplainer, etc.) suffer from a fundamental flaw identified in recent work: they can produce **degenerate explanations** — subgraphs that appear faithful but actually exploit structural shortcuts (anchor sets) rather than capturing the model's true reasoning. Plan 3a addresses this by:

1. **Forcing interpretability by construction** — a concept bottleneck ensures the classifier only sees human-readable concepts, never raw embeddings
2. **Auditing faithfulness post-hoc** — the EST (Extension Sufficiency Test) verifies that explanations are genuinely faithful, not degenerate
3. **Multi-scale reasoning** — hierarchical graph coarsening lets the model (and the user) reason at patch, region, and whole-tumor levels

---

## Research Foundations

Plan 3a synthesizes ideas from four papers:

| Paper | Venue | Contribution to Plan 3a |
|-------|-------|------------------------|
| **HyperCBM** | NeurIPS 2026 | Concept bottleneck architecture with HECRL inter-concept attention |
| **MRePath** | IJCAI 2025 | Sheaf hypergraph neural network + dynamic modality rebalancing for multimodal fusion |
| **Degenerate GNN Explanations** | ICLR 2026 | EST faithfulness metric and rejection ratios for auditing explanations |
| **TIF** | arXiv 2505.00364 | Multi-granular tree with adaptive routing for hierarchical interpretability |

---

## Architecture

```
MRI Patches (N, 6, 16×16)              Clinical Features (18-dim)
    │                                           │
    ▼                                           │
┌──────────────────────┐                        │
│  PatchEncoder (MLP)  │                        │
│  1536 → 128 dim      │                        │
└──────────┬───────────┘                        │
           ▼                                    │
┌──────────────────────┐                        │
│  SheafHGNN (3 layers)│                        │
│  Topological + Feature                        │
│  Hyperedges          │                        │
└──────────┬───────────┘                        │
           ▼                                    │
┌──────────────────────┐                        │
│  ConceptBottleneck   │                        │
│  8 disentangled heads│                        │
│  + HECRL attention   │                        │
└──────────┬───────────┘                        │
           ▼ (optional)                         │
┌──────────────────────┐                        │
│  MultiGranularTree   │                        │
│  L0→L1→L2→L3        │                        │
│  + AdaptiveRouter    │                        │
└──────────┬───────────┘                        │
           ▼                                    ▼
┌─────────────────────────────────────────────────┐
│  MultiModalFusion (MRePath-style)               │
│  DynamicWeighting (mono+holo confidence)        │
│  InteractiveAlignmentFusion (cross-attention)   │
└──────────────────────┬──────────────────────────┘
                       ▼
              ┌────────────────┐
              │  SurvivalHead  │
              │  → 4 time bins │
              └────────────────┘
```

### Key Design Decisions

- **No segmentation ground truth required.** All 8 concepts are derived directly from multi-modal MRI intensity patterns, not from tumor masks.
- **Ante-hoc interpretability.** The concept bottleneck is a hard information barrier — the survival head can only see concept activations, never raw node embeddings.
- **Survival-first.** The primary task is predicting time-to-event (overall survival), not classification. The loss is a discrete-time negative log-likelihood that properly handles right-censored data.
- **Memory-conscious.** The full model is ~5.6 MB (1.4M parameters) and processes a patient in <0.5s on CPU. Designed for an RTX 3060 12GB budget.

---

## Concepts

Each MRI patch (16×16 pixels across 6 modalities) is assigned 8 interpretable concept values:

| # | Concept | How it's computed | Clinical meaning |
|---|---------|-------------------|-----------------|
| c1 | Enhancement ratio | `log(1 + T1-post / T1-pre)` | Contrast uptake → blood-brain barrier breakdown |
| c2 | FLAIR z-score | `FLAIR_mean / FLAIR_std` | Perilesional edema extent |
| c3 | T2 abnormality | `T2 × FLAIR` interaction | Non-enhancing tumor / infiltration |
| c4 | DTI mean diffusivity | Mean DTI signal | White matter tract disruption |
| c5 | DTI FA proxy | DTI coefficient of variation | Fiber coherence loss |
| c6 | Intensity heterogeneity | Cross-modality std | Intra-tumoral heterogeneity |
| c7 | Boundary complexity | Graph-learned (SHGNN) | Tumor margin irregularity |
| c8 | Spatial location | Normalized z-coordinate | Anatomical depth |

These concepts are self-supervised: the bottleneck is trained to reconstruct them (MSE loss) while simultaneously learning to predict survival. This ensures the concepts stay grounded in imaging reality rather than drifting to arbitrary latent codes.

---

## Modules

### Data Pipeline (`data/`)

| File | Purpose |
|------|---------|
| `dicom_loader.py` | Load 6-modality DICOM volumes with missing-modality handling |
| `patch_extraction.py` | Extract multi-modal patches + compute 8 concept features |
| `clinical.py` | Parse clinical CSV with median imputation, one-hot encoding, missingness flags |
| `hypergraph.py` | Build dual-space hypergraph (topological δ-ball + feature top-k similarity) |
| `dataset.py` | PyTorch Dataset with on-the-fly hypergraph construction + K-fold splits |

### Model (`model/`)

| File | Purpose |
|------|---------|
| `sheaf_hgnn.py` | PatchEncoder + 3-layer Sheaf Hypergraph Neural Network with learned sheaf maps |
| `concept_bottleneck.py` | 8 disentangled concept heads + HECRL inter-concept multi-head attention |
| `fusion.py` | Clinical encoder + dynamic weighting (mono/holo confidence) + cross-attention fusion |
| `tree.py` | SoftAssignmentPool (DiffPool-style) + LevelEncoder + AdaptiveRouter |
| `full_model.py` | End-to-end composition of all stages + NLL survival loss + Kendall task weighting |

### Evaluation (`eval/`)

| File | Purpose |
|------|---------|
| `task_metrics.py` | Harrell's C-Index, per-concept MSE/Pearson-r, hazard→risk, adaptive time bins |
| `faithfulness.py` | EST, Fid⁻, RFid⁻, Sufficiency metrics + ExplanationExtractor + rejection ratios |

### Experiment Management

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters, paths, and run-control constants (single source of truth) |
| `train.py` | K-fold training loop with gradient accumulation, cosine LR, checkpointing |
| `runner.py` | Unified E1–E6 ablation experiment runner |
| `explain/report.py` | Auto-generates markdown report from ablation results JSON |

---

## Ablation Experiments

| Exp | Configuration | What it tests |
|-----|--------------|---------------|
| **E1** | kNN graph + simple GNN (no concepts) | Baseline — is a GNN useful at all? |
| **E2** | Sheaf hypergraph (no concepts, no fusion) | Does hypergraph structure help vs kNN? |
| **E3** | Hypergraph + Concept Bottleneck | Core contribution — does ante-hoc interpretability hurt performance? |
| **E4** | E3 + Clinical Fusion (MRePath) | Does adding clinical/molecular data improve survival prediction? |
| **E5** | E4 + Multi-Granular Tree (TIF) | Does hierarchical coarsening provide better multi-scale reasoning? |
| **E6** | E4 + EST Regularizer | Does training with faithfulness pressure improve explanation quality? |

---

## How to Run

All execution is controlled through constants in `config.py` — no command-line arguments.

### 1. Dataset Setup

```bash
# Create the upenn-filtered symlink structure from raw TCIA data
python setup_data.py --source /path/to/upenn-gbm
```

### 2. Preprocessing

Edit `config.py`:
```python
PREPROCESS_LIMIT = 10  # or None for all 420 patients
```

```bash
python -m plan3a.data.preprocess
```

### 3. Training

Edit `config.py`:
```python
TRAIN_LIMIT = None   # None = all patients
TRAIN_FOLD = None    # None = all 5 folds, or 0–4 for a single fold
EPOCHS = 30
```

```bash
python -m plan3a.train
```

### 4. Ablation Experiments

Edit `config.py`:
```python
RUN_EXPERIMENT = "E4"  # or "all" for full ablation
RUN_LIMIT = None
RUN_AUDIT = True       # run faithfulness audit after training
```

```bash
python -m plan3a.runner
```

### 5. Generate Report

```bash
python -m plan3a.explain.report
# → writes plan3a/RESULTS.md
```

---

## Dataset

**UPenn-GBM** (University of Pennsylvania Glioblastoma collection from TCIA)

- **630 patients** total in TCIA, **420 patients** in the filtered cohort (all 6 modalities present)
- **6 MRI modalities**: T1-pre, T1-post, T2, FLAIR, DTI, Perfusion
- **Clinical features**: Age, Gender, IDH1 mutation, MGMT methylation, KPS, GTR status
- **Survival labels**: Time-to-event in days + censoring status

---

## Directory Structure

```
plan3a/
├── config.py                   Central configuration (all constants)
├── train.py                    K-fold training loop
├── runner.py                   Ablation experiment runner (E1–E6)
├── README.md                   This file
├── RESULTS.md                  Auto-generated results report
│
├── data/
│   ├── dicom_loader.py         DICOM → numpy volumes
│   ├── patch_extraction.py     Patches + 8 concept features
│   ├── clinical.py             Clinical CSV parsing + encoding
│   ├── hypergraph.py           Dual-space hypergraph construction
│   └── dataset.py              PyTorch Dataset + K-fold splits
│
├── model/
│   ├── sheaf_hgnn.py           Sheaf Hypergraph Neural Network
│   ├── concept_bottleneck.py   Concept predictor + HECRL
│   ├── fusion.py               MRePath-style multimodal fusion
│   ├── tree.py                 TIF multi-granular tree
│   └── full_model.py           End-to-end model + survival loss
│
├── eval/
│   ├── task_metrics.py         C-Index, concept metrics
│   └── faithfulness.py         EST, Fid⁻, RFid⁻, Sufficiency
│
├── explain/
│   └── report.py               Markdown report generator
│
├── processed/                  Cached .pt files (per-patient tensors)
└── checkpoints/                Saved model weights + results JSON
```

---

## Requirements

- Python 3.8+
- PyTorch ≥ 1.12
- torch-geometric (for `HypergraphConv`)
- pydicom
- scipy
- numpy

---

## Citation Context

This pipeline is a research testbed, not a clinical tool. It is designed to answer the question: *Can we build survival models for brain tumors that are both accurate AND faithfully interpretable — with formal guarantees that the explanations are not degenerate?*

The answer Plan 3a proposes is: **yes, via concept bottlenecks (ante-hoc) + EST auditing (post-hoc)**, combining structural guarantees with empirical verification.

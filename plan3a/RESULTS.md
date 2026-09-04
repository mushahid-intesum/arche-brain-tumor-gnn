# Plan 3a: Hypergraph Concept Bottleneck GNN — Results Report

*Generated: 2026-09-04 15:18*

## 1. Architecture Overview

```
MRI Patches (N, 6, 16×16)           Clinical Features (18-dim)
    │                                        │
    ▼                                        │
PatchEncoder (MLP)                           │
    │                                        │
    ▼                                        │
SheafHGNN (3 layers)                         │
    │ ←── Topological + Feature              │
    │     Hyperedges                         │
    ▼                                        │
ConceptBottleneck (8 concepts)               │
    │ ←── HECRL inter-concept attention      │
    │                                        │
    ▼ (optional)                             │
MultiGranularTree                            │
    │ L0→L1→L2→L3 + AdaptiveRouter          │
    │                                        │
    ▼                                        ▼
MultiModalFusion ◄────────── ClinicalEncoder
    │ ←── DynamicWeighting                   
    │     (mono+holo confidence)             
    ▼                                        
SurvivalHead → Hazard Logits (4 bins)       
```

## 2. Ablation Results

### 2.1 C-Index Comparison

| Exp | Configuration | C-Index | Params |
|-----|--------------|---------|--------|
| **E3** | Hypergraph + Concept Bottleneck | 0.8000 ± 0.4000 | 1,221,521 |
| **E4** | E3 + Clinical Fusion (MRePath-style) | 0.0000 ± 0.0000 | 1,397,459 |

> **Best**: E3 (E3: Hypergraph + Concept Bottleneck) with C-Index = 0.8000

### 2.2 Per-Fold C-Index

| Exp | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|-----|--------|--------|--------|--------|--------|------|
| E3 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.8000 |
| E4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### 2.4 Training Dynamics

**E4** (Fold 1):

| Epoch | Train Loss | Val C-Index | Concept r |
|-------|-----------|-------------|-----------|
| 1 | 1.0956 | 0.0000 | 0.143 |
| 2 | 1.0530 | 0.0000 | -0.143 |
| 3 | 1.0343 | 0.0000 | -0.143 |

## 3. Ablation Insights

### Key Findings

3. **Multimodal fusion effect**: Adding clinical data (E4) decreases C-Index by 0.8000

## 4. Methodology

### Dataset
- **Patients**: 10
- **Folds**: 5-fold CV
- **Epochs**: 3
- **Modalities**: T1-pre, T1-post, T2, FLAIR, DTI, Perfusion
- **Task**: Survival prediction (discrete-time NLL loss)
- **Primary metric**: Harrell's C-Index

### Concepts (self-supervised, no segmentation GT)
| # | Concept | Source |
|---|---------|--------|
| c1 | Enhancement ratio | log(1 + T1-post/T1-pre) |
| c2 | FLAIR z-score | FLAIR_mean / FLAIR_std |
| c3 | T2 abnormality | T2 × FLAIR interaction |
| c4 | DTI mean diffusivity | Mean DTI signal |
| c5 | DTI FA proxy | DTI coefficient of variation |
| c6 | Intensity heterogeneity | Cross-modality std |
| c7 | Boundary complexity | Graph-learned (SHGNN) |
| c8 | Spatial location | Normalized z-coordinate |

### References
- **HyperCBM** (NeurIPS 2026): Concept bottleneck + HECRL
- **MRePath** (IJCAI 2025): Sheaf hypergraph + dynamic modality rebalancing
- **SE-GNN Audit** (ICLR 2026): EST faithfulness metric
- **TIF** (arXiv 2505.00364): Multi-granular tree interpretability

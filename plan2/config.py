import torch
import numpy as np
import random
from pathlib import Path


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


SHARED = {
    "seed": SEED,
    "device": DEVICE,
    "img_size": 224,
    "checkpoint_dir": Path("checkpoints"),
}


GRAPH = {
    # ── Data ──
    "data_root": Path("BraTS"),
    "modalities": ["t1n", "t1c", "t2w", "t2f"],
    "slic_modality": "t1n",
    "num_classes": 4,
    "class_names": {0: "BG", 1: "NCR", 2: "ED", 3: "ET"},

    # ── Step 1: 3D SLIC Supervoxel Generation ──
    "n_segments": 1000,
    "compactness": 0.1,
    "min_sv_volume": 20,

    # ── Step 3: Ground Truth ──
    "tau": 0.15,

    # ── Step 5: Patch Extraction ──
    "n_patch": 4,
    "patch_neighbors": 16,

    # ── Step 6: Graph Construction ──
    "knn_k": 8,

    # ── Encoder (scaled for RTX 3060 12GB) ──
    "embed_dim": 128,
    "transformer_layers": 3,
    "transformer_heads": 4,
    "gat_layers": 3,
    "gat_heads": 4,
    "laplacian_pe_dim": 8,

    # ── Task Heads ──
    "regression_ensemble_size": 4,          # parallel MLPs in regression head
    "n_boundary_types": 10,                 # edge classification classes

    # ── Training (Plan 2 specifics) ──
    "epochs": 120,                          # more epochs for multi-task convergence
    "lr": 3e-4,                             # slightly lower LR than Plan 1
    "weight_decay": 0.01,
    "batch_size": 2,
    "accum_steps": 4,                       # effective batch = 8
    "eval_every": 5,
    "num_folds": 5,

    # Multi-task loss: uncertainty-weighted (Kendall et al., 2018)
    # Learnable log(σ²) per task, initialized to 0 → σ=1 → equal weighting
    "init_log_var_reg": 0.0,
    "init_log_var_edge": 0.0,
    "init_log_var_unc": 0.0,

    "checkpoint": Path("checkpoints/graph_plan2.pth"),

    # ── Caching ──
    "cache_dir": Path("brats_outputs/graph_cache"),
}

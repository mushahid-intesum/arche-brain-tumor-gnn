"""
Unified configuration for all pipeline modules.
Prediction → Classification → Segmentation → GNN
"""

import torch
import numpy as np
import random
import os
from pathlib import Path


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# === Shared ===

SHARED = {
    "seed": SEED,
    "device": DEVICE,
    "img_size": IMG_SIZE,
    "checkpoint_dir": Path("checkpoints"),
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],
}


# === Phase 1: Binary Prediction (Brain MRI ND-5) ===

PREDICTION = {
    "data_root": Path("Brain MRI ND-5 Dataset/tumordata"),
    "batch_size": 16,
    "num_workers": 4,
    "val_split": 0.15,
    "lr": 5e-5,
    "weight_decay": 1e-4,
    "epochs": 20,
    "freeze_epochs": 4,
    "accum_steps": 2,
    "checkpoint": Path("checkpoints/prediction_binary.pth"),
}


# === Phase 2: Multi-class Classification (Brain MRI ND-5) ===

CLASSIFICATION = {
    "data_root": Path("Brain MRI ND-5 Dataset/tumordata"),
    "batch_size": 16,
    "num_workers": 4,
    "val_split": 0.15,
    "lr": 5e-5,
    "weight_decay": 1e-4,
    "epochs": 20,
    "freeze_epochs": 4,
    "accum_steps": 2,
    "focal_gamma": 2.0,
    "checkpoint": Path("checkpoints/classification_multiclass.pth"),
    "pseudo_mask_dir": Path("pseudo_masks"),
    "class_to_idx": {
        "glioma_tumor": 0,
        "meningioma_tumor": 1,
        "pituitary_tumor": 2,
        "no_tumor": 3,
    },
    "idx_to_class": {
        0: "glioma_tumor",
        1: "meningioma_tumor",
        2: "pituitary_tumor",
        3: "no_tumor",
    },
    "tumor_classes": ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"],
}


# === Phase 3: Multi-class Segmentation (BraTS 2023) ===

SEGMENTATION = {
    "data_root": Path("BraTS"),
    "output_dir": Path("brats_outputs"),
    "modalities": ["t1n", "t1c", "t2w", "t2f"],
    "num_classes": 4,
    "class_names": {0: "BG", 1: "NCR", 2: "ED", 3: "ET"},
    "batch_size": 8,
    "num_workers": 2,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 60,
    "accum_steps": 4,
    "eval_every": 5,
    "min_tumor_pixels": 50,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "checkpoint": Path("checkpoints/segmentation_brats.pth"),
}


# === Phase 4: 3D GNN Edge Prediction (BraTS outputs) ===

GNN = {
    "brats_output_dir": Path("brats_outputs"),
    "node_feat_dim": 35,
    "hidden_dim": 128,
    "embed_dim": 64,
    "num_heads": 4,
    "num_layers": 3,
    "edge_attr_dim": 4,
    "structural_feat_dim": 3,
    "min_region_area": 10,
    "k_neighbors": 5,
    "inter_slice_dist_thresh": 50.0,
    "epochs": 80,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "modalities": ["t1n", "t1c", "t2w", "t2f"],
    "tissue_labels": {1: "NCR", 2: "ED", 3: "ET"},
    "checkpoint": Path("checkpoints/gnn_3d.pth"),
}


# === Pipeline (orchestrator) ===

PIPELINE = {
    "output_dir": Path("pipeline_outputs"),
    "short_circuit": True,  # skip downstream if prediction says no tumor
}


def ensure_dirs():
    """Create all necessary output directories."""
    dirs = [
        SHARED["checkpoint_dir"],
        SEGMENTATION["output_dir"],
        CLASSIFICATION["pseudo_mask_dir"],
        PIPELINE["output_dir"],
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Image size: {IMG_SIZE}")
    print(f"\nPrediction config:    {PREDICTION['data_root']} | {PREDICTION['epochs']} epochs")
    print(f"Classification config: {CLASSIFICATION['data_root']} | {CLASSIFICATION['epochs']} epochs")
    print(f"Segmentation config:  {SEGMENTATION['data_root']} | {SEGMENTATION['epochs']} epochs")
    print(f"GNN config:           {GNN['brats_output_dir']} | {GNN['epochs']} epochs")
    print(f"\nAll directories created.")

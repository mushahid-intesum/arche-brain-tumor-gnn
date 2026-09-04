"""
Plan 3a Configuration — Hypergraph Concept Bottleneck GNN (MRI Patch Graph)
All hyperparameters and paths are centralized here.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "upenn-filtered"
CLINICAL_CSV = DATA_ROOT / "clinical_info.csv"
PROCESSED_DIR = PROJECT_ROOT / "plan3a" / "processed"  # cached tensors go here

# ── DICOM → Volume ─────────────────────────────────────────────────────────
# Core structural modalities (always present)
CORE_MODALITIES = ["T1-pre", "T1-post", "T2", "FLAIR"]
# Advanced modalities (present in this sample, may be absent in other cohorts)
ADVANCED_MODALITIES = ["DTI", "Perfusion"]
ALL_MODALITIES = CORE_MODALITIES + ADVANCED_MODALITIES

# ── Patch Extraction ───────────────────────────────────────────────────────
PATCH_SIZE = 16                   # pixels per patch side
SLICE_STRIDE = 2                  # take every Nth axial slice
TARGET_SLICE_SIZE = (192, 192)    # resize each axial slice before patching
MIN_PATCH_INTENSITY = 0.02        # discard background patches below this mean

# ── Concepts ───────────────────────────────────────────────────────────────
NUM_CONCEPTS = 8
# c1: enhancement ratio (T1-post / T1-pre)
# c2: FLAIR z-score (edema)
# c3: T2 abnormality (deviation from contralateral hemisphere)
# c4: DTI mean diffusivity
# c5: DTI fractional anisotropy
# c6: intensity heterogeneity (std across modalities within patch)
# c7: boundary complexity (entropy of neighbor embeddings — computed at graph time)
# c8: spatial location (normalized x, y, z)

# ── Clinical Features ─────────────────────────────────────────────────────
CLINICAL_CATEGORICAL = {
    "Gender": ["M", "F"],
    "IDH1": ["Wildtype", "Mutated"],     # NOS/NEC treated as missing
    "MGMT": ["Methylated", "Unmethylated"],  # Indeterminate treated as missing
    "GTR_over90percent": ["Y", "N"],
}
CLINICAL_CONTINUOUS = ["Age_at_scan_years"]
# High-missingness features — included with missingness indicator
CLINICAL_SPARSE = {
    "KPS": "continuous",        # 88% missing — normalize when available
    "PsP_TP_score": "ordinal",  # 87% missing — treat as ordinal 1-6
}

# ── Survival Labels ───────────────────────────────────────────────────────
SURVIVAL_TIME_COL = "Survival_from_surgery_days_UPDATED"
SURVIVAL_STATUS_COL = "Survival_Status"
# Map status to censoring indicator: 1 = event observed, 0 = censored
SURVIVAL_STATUS_MAP = {
    "Deceased": 1,
    "Deceased - uncertain date of death": 1,
    "Alive": 0,
    "Lost to Follow-up": 0,
}

# ── Hypergraph ─────────────────────────────────────────────────────────────
TOPO_HYPEREDGE_RADIUS = 2.5       # spatial δ for topological hyperedges (in patch-grid units)
FEATURE_HYPEREDGE_K = 9           # top-k for feature-based hyperedges (from MRePath ablation)
SHEAF_HGNN_LAYERS = 3
SHEAF_HGNN_DIM = 128

# ── Model ──────────────────────────────────────────────────────────────────
EMBED_DIM = 128
PATCH_ENCODER_CHANNELS = [32, 64, 128]  # small CNN for patch encoding
NUM_CLINICAL_GROUPS = 5

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 30
NUM_FOLDS = 5

# ── Device ─────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Run Control ────────────────────────────────────────────────────────────
# These constants replace all CLI arguments across every script.
# Change them here to control what runs — no command-line flags needed.

# Preprocessing (preprocess.py)
PREPROCESS_LIMIT = None            # int or None — process only first N patients
PREPROCESS_PATIENT_FILTER = None   # str or None — process only this patient ID

# Training (train.py)
TRAIN_LIMIT = None                 # int or None — limit patients for testing
TRAIN_FOLD = None                  # int or None — run only this fold (0-indexed), None = all

# Runner (runner.py)
RUN_EXPERIMENT = "E4"              # str — "E1"–"E6" or "all"
RUN_LIMIT = None                   # int or None — limit patients
RUN_AUDIT = True                   # bool — run faithfulness audit post-training

# Report (report.py)
RESULTS_JSON = None                # str or None — path to ablation_results.json (None = auto)
REPORT_OUTPUT = None               # str or None — path for RESULTS.md (None = auto)

# Derived paths
CHECKPOINTS_DIR = PROJECT_ROOT / "plan3a" / "checkpoints"

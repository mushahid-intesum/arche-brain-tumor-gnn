# BraTS 2023 Multi-Class Segmentation Pipeline

## Overview

This pipeline (`03_brats_segmentation.py`) trains a multi-class tumor segmentation model on BraTS 2023 GLI data. It takes 4-modality 3D MRI volumes (625 patients), extracts 2D axial slices, and trains a DeepLabV3+ model to segment three tumor sub-regions from the four MRI channels.

---

## Data

### Source
- **Dataset**: BraTS 2023 GLI (Glioma)
- **Location**: `BraTS/` — 625 patient directories
- **Format**: Uncompressed NIfTI (`.nii`), each volume is 240×240×155 voxels

### Per-Patient Files

| File suffix | Modality | What it captures |
|-------------|----------|-----------------|
| `-t1n.nii` | T1 native | Anatomical structure, gray/white matter contrast |
| `-t1c.nii` | T1 contrast-enhanced | Active tumor vasculature (gadolinium enhancement) |
| `-t2w.nii` | T2-weighted | Edema, fluid, CSF |
| `-t2f.nii` | T2-FLAIR | Edema boundaries (CSF signal suppressed) |
| `-seg.nii` | Segmentation mask | Expert-annotated tumor sub-regions |

### Segmentation Labels

| Label | Abbreviation | Tissue | Clinical significance |
|-------|-------------|--------|----------------------|
| 0 | BG | Background | Non-tumor tissue |
| 1 | NCR | Necrotic / non-enhancing core | Dead tissue inside tumor — indicates aggressive tumor that outgrew its blood supply |
| 2 | ED | Peritumoral edema | Swelling around tumor — indicates tumor-associated inflammation and infiltration |
| 3 | ET | GD-enhancing tumor | Active, vascularized tumor — indicates areas with broken blood-brain barrier |

---

## Pipeline Architecture

### Phase 1 — Data Loading & Preprocessing

**NIfTI → 2D slices**:
1. Load all 4 modalities + segmentation mask per patient using `nibabel`
2. Z-score normalize each modality per-volume (mean/std of non-zero voxels only — standard BraTS practice to handle background zeros)
3. Extract axial slices where tumor pixels ≥ 50 (skip near-empty slices)
4. Resize to 224×224 (bilinear for images, nearest-neighbor for masks)
5. Stack 4 modalities → 4-channel image tensor per slice

**Patient-level split** (70/15/15):
- Split at the patient level, not slice level
- This prevents data leakage: different slices from the same patient never appear in both train and test
- Estimated yield: ~25,000-30,000 total slices across 625 patients

**Augmentation** (train only):
- Random horizontal flip
- Random vertical flip
- Random 90°/180°/270° rotation

### Phase 2 — Model Architecture

**DeepLabV3+ with EfficientNet-B4 encoder**:
- Pretrained on ImageNet (transfer learning)
- Input modified: 3 channels → **4 channels** (T1n, T1c, T2w, T2f)
  - First conv layer expanded by copying the red-channel weights to the 4th channel
  - Preserves pretrained features for RGB channels, gives 4th channel a reasonable initialization
- Output: **4 classes** (BG, NCR, ED, ET) with softmax activation

### Phase 3 — Loss & Training

**Combined Dice + CrossEntropy loss**:
- **Dice loss**: computed on classes 1-3 only (excludes background)
  - Background dominates (~95%+ of pixels), so including it in Dice would mask poor tumor segmentation
  - Per-class Dice averaged across NCR, ED, ET
- **CrossEntropy**: standard per-pixel classification loss on all 4 classes
- Total loss = Dice + CE (equal weight)

**Training details**:
- AdamW (lr=1e-4, weight_decay=1e-4)
- OneCycleLR scheduler
- Gradient accumulation: effective batch size 32 (batch=8, accum=4)
- Gradient clipping (max_norm=1.0)
- 60 epochs, validate every 5

### Phase 4 — Evaluation

**Per-class Dice** (NCR, ED, ET):
- Standard segmentation metric: 2×|P∩G| / (|P|+|G|)
- Reported as mean ± std across all test slices

**Official BraTS region Dice** (nested regions):

| Region | Labels included | Clinical meaning |
|--------|----------------|-----------------|
| **Whole Tumor (WT)** | NCR + ED + ET | Total tumor extent including edema |
| **Tumor Core (TC)** | NCR + ET | Solid tumor without surrounding edema |
| **Enhancing Tumor (ET)** | ET only | Active, contrast-enhancing tumor |

**Confusion matrix**: 4×4 matrix showing per-pixel classification accuracy across all classes.

**Visualization**: 6 random test slices showing T1c image → ground truth overlay → prediction overlay, with per-slice WT/TC Dice scores.

### Phase 5 — Export for GNN

Exports to `brats_outputs/`:

```
brats_outputs/
├── masks/           # Predicted multi-class masks (uint8 .npy)
├── gt_masks/        # Ground-truth masks (uint8 .npy)
├── raw_slices/      # 4-channel MRI slices (float32 .npy, shape [4,224,224])
├── metadata.pt      # Case IDs, slice indices, split assignments
└── best_model.pt    # Trained DeepLabV3+ weights
```

The GNN pipeline (`04_brats_gnn.py`) consumes `masks/`, `raw_slices/`, and `metadata.pt`.

---

## Key Design Decisions

1. **2D slices, not 3D volumes**: Training a 3D segmentation model (e.g., 3D U-Net) requires significantly more VRAM. 2D DeepLabV3+ fits comfortably on RTX 3060 12GB with batch=8.

2. **Z-score normalization per volume**: BraTS MRI intensities are not standardized across patients or scanners. Z-score on non-zero voxels normalizes each volume individually.

3. **Patient-level split**: Ensures no information leakage between train/val/test. Adjacent slices from the same patient share very similar appearance — splitting by slice would inflate metrics.

4. **Predicted masks for GNN**: The GNN receives predicted masks (not ground truth) to test the full end-to-end pipeline. Ground truth is exported separately for ablation studies.

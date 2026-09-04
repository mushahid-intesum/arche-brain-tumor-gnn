"""
MRI Patch Extraction & Concept Computation.

Converts multi-modal 3D MRI volumes into a set of 2D patches with
precomputed concept features. This is Track A of Plan 3a — analogous
to how MRePath extracts patches from Whole Slide Images.

Patch pipeline:
  1. Select axial slices (every SLICE_STRIDE-th slice)
  2. Resize slices to TARGET_SLICE_SIZE
  3. Partition into non-overlapping PATCH_SIZE × PATCH_SIZE patches
  4. Discard background patches (below MIN_PATCH_INTENSITY)
  5. Stack multi-modal channels per patch
  6. Compute 8 concept values per patch from raw intensities

Concepts (all derived from imaging — no segmentation needed):
  c1: Enhancement ratio  (T1-post / T1-pre)
  c2: FLAIR z-score       (edema signature)
  c3: T2 abnormality      (deviation from contralateral hemisphere)
  c4: DTI mean diffusivity proxy
  c5: DTI fractional anisotropy proxy (std/mean of DTI signal)
  c6: Intensity heterogeneity (std across modalities)
  c7: Boundary complexity   (placeholder — computed at graph construction time)
  c8: Spatial location      (normalized x, y, z coordinates)
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import zoom

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import (
    PATCH_SIZE, SLICE_STRIDE, TARGET_SLICE_SIZE, MIN_PATCH_INTENSITY
)
from plan3a.data.dicom_loader import normalize_volume


def resize_slice(slice_2d: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize a 2D slice to target_size using scipy zoom."""
    if slice_2d.shape == target_size:
        return slice_2d
    zoom_factors = (target_size[0] / slice_2d.shape[0],
                    target_size[1] / slice_2d.shape[1])
    return zoom(slice_2d, zoom_factors, order=1)


def extract_patches_from_slice(
    slice_2d: np.ndarray,
    patch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract non-overlapping patches from a 2D slice.

    Returns:
        patches: (num_patches, patch_size, patch_size)
        coords:  (num_patches, 2) — (row_center, col_center) in pixel coords
    """
    H, W = slice_2d.shape
    rows = H // patch_size
    cols = W // patch_size

    patches = []
    coords = []
    for r in range(rows):
        for c in range(cols):
            patch = slice_2d[
                r * patch_size : (r + 1) * patch_size,
                c * patch_size : (c + 1) * patch_size,
            ]
            patches.append(patch)
            coords.append([
                (r + 0.5) * patch_size,
                (c + 0.5) * patch_size,
            ])

    return np.array(patches), np.array(coords)


def _align_volume_to_reference(vol: np.ndarray, ref_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize each slice in a volume to match a reference (H, W).
    Used to align DTI/Perfusion (which may differ in resolution) to structural.
    """
    aligned = []
    for s in range(vol.shape[0]):
        aligned.append(resize_slice(vol[s], ref_shape))
    return np.stack(aligned, axis=0)


def extract_patient_patches(
    modality_volumes: Dict[str, Tuple[np.ndarray, bool]],
    patch_size: int = PATCH_SIZE,
    slice_stride: int = SLICE_STRIDE,
    target_size: Tuple[int, int] = TARGET_SLICE_SIZE,
    min_intensity: float = MIN_PATCH_INTENSITY,
) -> Dict:
    """
    Extract multi-modal patches and concept features for one patient.

    Args:
        modality_volumes: dict from load_all_modalities()
            maps modality_name → (volume, is_present)
        patch_size: side length of square patches
        slice_stride: take every Nth axial slice
        target_size: resize each slice to this (H, W)
        min_intensity: discard patches with mean below this

    Returns:
        dict with keys:
            "patches": np.ndarray (N, C, patch_size, patch_size) — multi-channel patches
            "coords": np.ndarray (N, 3) — (x, y, z_slice) normalized to [0, 1]
            "concepts": np.ndarray (N, 8) — precomputed concept values
            "modality_mask": dict modality → bool — which modalities were available
            "num_patches": int
    """
    # ── Step 1: Normalize and align volumes ──────────────────────────────
    # Use T1-pre as the spatial reference
    t1_pre_vol, t1_pre_ok = modality_volumes["T1-pre"]
    if not t1_pre_ok:
        raise ValueError("T1-pre must be present (reference modality)")

    t1_pre_vol = normalize_volume(t1_pre_vol)
    num_slices = t1_pre_vol.shape[0]
    ref_shape = (t1_pre_vol.shape[1], t1_pre_vol.shape[2])

    # Normalize and align all modalities
    volumes = {"T1-pre": t1_pre_vol}
    modality_mask = {"T1-pre": True}

    for mod in ["T1-post", "T2", "FLAIR"]:
        vol, ok = modality_volumes[mod]
        modality_mask[mod] = ok
        if ok:
            vol = normalize_volume(vol)
            # Align slice count to reference (take center slices or pad)
            vol = _match_slice_count(vol, num_slices)
            # Align spatial resolution
            vol = _align_volume_to_reference(vol, ref_shape)
        else:
            vol = np.zeros((num_slices,) + ref_shape, dtype=np.float32)
        volumes[mod] = vol

    for mod in ["DTI", "Perfusion"]:
        vol, ok = modality_volumes[mod]
        modality_mask[mod] = ok
        if ok:
            vol = normalize_volume(vol)
            # These are scalar maps — may be single-slice or few-slice
            # Broadcast to match structural slice count
            if vol.shape[0] == 1:
                vol = np.repeat(vol, num_slices, axis=0)
            else:
                vol = _match_slice_count(vol, num_slices)
            vol = _align_volume_to_reference(vol, ref_shape)
        else:
            vol = np.zeros((num_slices,) + ref_shape, dtype=np.float32)
        volumes[mod] = vol

    # ── Step 2: Extract patches slice by slice ───────────────────────────
    all_patches = []    # list of (C, ps, ps) arrays
    all_coords = []     # list of (3,) arrays [x, y, z_norm]
    all_concepts = []   # list of (8,) arrays

    selected_slices = list(range(0, num_slices, slice_stride))

    for s_idx in selected_slices:
        # Resize each modality's slice to target_size
        slices = {}
        for mod in volumes:
            slc = volumes[mod][s_idx]
            slices[mod] = resize_slice(slc, target_size)

        # Extract patches from T1-pre (reference grid)
        patches_ref, coords_2d = extract_patches_from_slice(slices["T1-pre"], patch_size)
        num_p = patches_ref.shape[0]

        for p_idx in range(num_p):
            # Background filter: skip patches with low mean intensity across core modalities
            core_mean = np.mean([
                slices[m][
                    int(coords_2d[p_idx, 0] - patch_size/2):int(coords_2d[p_idx, 0] + patch_size/2),
                    int(coords_2d[p_idx, 1] - patch_size/2):int(coords_2d[p_idx, 1] + patch_size/2),
                ].mean()
                for m in ["T1-pre", "T1-post", "T2", "FLAIR"]
            ])
            if core_mean < min_intensity:
                continue

            # Stack multi-modal patch: shape (C, ps, ps)
            r_start = int(coords_2d[p_idx, 0] - patch_size / 2)
            c_start = int(coords_2d[p_idx, 1] - patch_size / 2)
            r_end = r_start + patch_size
            c_end = c_start + patch_size

            channel_patches = []
            for mod in ["T1-pre", "T1-post", "T2", "FLAIR", "DTI", "Perfusion"]:
                channel_patches.append(slices[mod][r_start:r_end, c_start:c_end])
            multi_patch = np.stack(channel_patches, axis=0)  # (6, ps, ps)

            # Normalized spatial coordinates
            z_norm = s_idx / max(num_slices - 1, 1)
            x_norm = coords_2d[p_idx, 0] / target_size[0]
            y_norm = coords_2d[p_idx, 1] / target_size[1]

            # ── Compute concepts ─────────────────────────────────────
            concepts = compute_patch_concepts(
                multi_patch, modality_mask, x_norm, y_norm, z_norm
            )

            all_patches.append(multi_patch)
            all_coords.append([x_norm, y_norm, z_norm])
            all_concepts.append(concepts)

    if not all_patches:
        # Edge case: all patches were background
        return {
            "patches": np.zeros((0, 6, patch_size, patch_size), dtype=np.float32),
            "coords": np.zeros((0, 3), dtype=np.float32),
            "concepts": np.zeros((0, 8), dtype=np.float32),
            "modality_mask": modality_mask,
            "num_patches": 0,
        }

    return {
        "patches": np.stack(all_patches, axis=0).astype(np.float32),
        "coords": np.array(all_coords, dtype=np.float32),
        "concepts": np.stack(all_concepts, axis=0).astype(np.float32),
        "modality_mask": modality_mask,
        "num_patches": len(all_patches),
    }


def compute_patch_concepts(
    multi_patch: np.ndarray,
    modality_mask: Dict[str, bool],
    x_norm: float,
    y_norm: float,
    z_norm: float,
) -> np.ndarray:
    """
    Compute 8 concept values for a single multi-channel patch.

    Args:
        multi_patch: (6, ps, ps) — channels: T1-pre, T1-post, T2, FLAIR, DTI, Perfusion
        modality_mask: which modalities are present
        x_norm, y_norm, z_norm: normalized spatial coordinates

    Returns:
        concepts: (8,) array
    """
    t1_pre = multi_patch[0]   # channel 0
    t1_post = multi_patch[1]  # channel 1
    t2 = multi_patch[2]       # channel 2
    flair = multi_patch[3]    # channel 3
    dti = multi_patch[4]      # channel 4
    perf = multi_patch[5]     # channel 5

    eps = 1e-8

    # c1: Enhancement ratio — T1-post / T1-pre (log-scaled, clamped)
    # Higher values indicate contrast-enhancing tissue (active tumor)
    t1_pre_mean = t1_pre.mean() + eps
    t1_post_mean = t1_post.mean() + eps
    if modality_mask.get("T1-post", False) and t1_pre_mean > 0.01:
        ratio = t1_post_mean / t1_pre_mean
        c1 = np.clip(np.log1p(ratio), 0.0, 5.0)  # log(1+ratio), clamped to [0, 5]
    else:
        c1 = 0.0

    # c2: FLAIR z-score — edema signature
    # High FLAIR signal relative to patch mean indicates edema
    flair_mean = flair.mean()
    flair_std = flair.std() + eps
    c2 = flair_mean / flair_std if modality_mask.get("FLAIR", False) else 0.0

    # c3: T2 abnormality — deviation from expected normal tissue
    # High T2 + high FLAIR = edema; high T2 + low FLAIR = CSF
    t2_mean = t2.mean()
    c3 = t2_mean * flair_mean if modality_mask.get("T2", False) else 0.0

    # c4: DTI mean diffusivity proxy
    # Higher MD = more free water diffusion = edema or necrosis
    c4 = dti.mean() if modality_mask.get("DTI", False) else 0.0

    # c5: DTI fractional anisotropy proxy (coefficient of variation)
    # Low FA = disrupted white matter tracts (tumor infiltration)
    dti_mean = dti.mean() + eps
    dti_std = dti.std()
    c5 = dti_std / dti_mean if modality_mask.get("DTI", False) else 0.0

    # c6: Intensity heterogeneity — std across all modality means within patch
    # High heterogeneity = tissue boundary or mixed composition
    modality_means = [t1_pre_mean, t1_post_mean, t2_mean, flair_mean]
    if modality_mask.get("DTI", False):
        modality_means.append(dti.mean())
    if modality_mask.get("Perfusion", False):
        modality_means.append(perf.mean())
    c6 = np.std(modality_means)

    # c7: Boundary complexity — placeholder (computed later at graph level)
    c7 = 0.0

    # c8: Spatial location — normalized z-coordinate (axial depth)
    # Encodes superior-inferior position; tumor location matters for prognosis
    c8 = z_norm

    return np.array([c1, c2, c3, c4, c5, c6, c7, c8], dtype=np.float32)


def _match_slice_count(vol: np.ndarray, target_count: int) -> np.ndarray:
    """
    Match a volume's slice count to target_count by center-cropping or zero-padding.
    """
    current = vol.shape[0]
    if current == target_count:
        return vol
    elif current > target_count:
        # Center crop
        start = (current - target_count) // 2
        return vol[start : start + target_count]
    else:
        # Zero-pad symmetrically
        pad_before = (target_count - current) // 2
        pad_after = target_count - current - pad_before
        return np.pad(vol, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant")

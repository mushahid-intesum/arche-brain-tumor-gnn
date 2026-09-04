"""
DICOM → Volume loader for UPenn-GBM dataset.

Handles all 6 modalities (T1-pre, T1-post, T2, FLAIR, DTI, Perfusion).
For DTI: extracts a single representative scalar map (mean across directions).
For Perfusion: extracts a mean-over-time map from the DSC time-series.

Missing modality mitigation:
  - Returns a zero-volume + a boolean mask indicating absence.
  - Downstream modules use the mask to skip missing-modality concepts.
"""
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pydicom

warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")


def _sort_key_for_dcm(filename: str) -> Tuple:
    """Extract numeric sort key from DICOM filenames like '1-024.dcm' or '02-01.dcm'."""
    stem = filename.replace(".dcm", "")
    parts = stem.split("-")
    return tuple(int(p) for p in parts)


def load_dicom_volume(dicom_dir: str) -> np.ndarray:
    """
    Load a directory of DICOM slices into a 3D numpy array.
    Returns array of shape (num_slices, H, W) with float32 pixel values.
    """
    dcm_files = sorted(
        [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")],
        key=_sort_key_for_dcm,
    )
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files in {dicom_dir}")

    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(os.path.join(dicom_dir, f), force=True)
        if hasattr(ds, "pixel_array"):
            slices.append(ds.pixel_array.astype(np.float32))

    if not slices:
        raise ValueError(f"No readable pixel data in {dicom_dir}")

    return np.stack(slices, axis=0)


def load_structural_modality(patient_dir: str, modality: str) -> Tuple[np.ndarray, bool]:
    """
    Load a structural MRI modality (T1-pre, T1-post, T2, FLAIR).
    Returns (volume, is_present).
    """
    mod_dir = os.path.join(patient_dir, modality)
    if not os.path.isdir(mod_dir) or len(os.listdir(mod_dir)) == 0:
        return np.zeros((1, 1, 1), dtype=np.float32), False

    try:
        vol = load_dicom_volume(mod_dir)
        return vol, True
    except (FileNotFoundError, ValueError) as e:
        print(f"  Warning: {modality} failed for {patient_dir}: {e}")
        return np.zeros((1, 1, 1), dtype=np.float32), False


def load_dti_scalar(patient_dir: str) -> Tuple[np.ndarray, bool]:
    """
    Load DTI and collapse to a single scalar map.

    DTI DICOM files represent different diffusion directions/b-values.
    We compute the mean intensity across all directions per spatial location
    as a proxy for mean diffusivity (MD). A proper pipeline would use
    tensor fitting (dipy), but this gives a usable scalar map.

    Returns (scalar_volume, is_present).
    """
    dti_dir = os.path.join(patient_dir, "DTI")
    if not os.path.isdir(dti_dir) or len(os.listdir(dti_dir)) == 0:
        return np.zeros((1, 1, 1), dtype=np.float32), False

    try:
        vol = load_dicom_volume(dti_dir)
        # DTI files are multi-direction: average across the direction axis
        # Result: a single 2D map (or 3D if multi-slice DTI)
        # For single-slice-per-direction DTI, vol shape = (num_directions, H, W)
        scalar_map = vol.mean(axis=0, keepdims=True)  # (1, H, W) mean diffusivity proxy
        return scalar_map, True
    except (FileNotFoundError, ValueError) as e:
        print(f"  Warning: DTI failed for {patient_dir}: {e}")
        return np.zeros((1, 1, 1), dtype=np.float32), False


def load_perfusion_scalar(patient_dir: str) -> Tuple[np.ndarray, bool]:
    """
    Load DSC Perfusion and collapse to a single scalar map.

    Perfusion DICOM files are a time-series: (timepoints × slices).
    File naming convention: 'TT-SSS.dcm' where TT=timepoint, SSS=slice.
    We compute the mean-over-time for each spatial location as a proxy
    for relative cerebral blood volume (rCBV). A proper pipeline would
    fit a gamma-variate, but this gives a usable signal.

    Returns (scalar_volume, is_present).
    """
    perf_dir = os.path.join(patient_dir, "Perfusion")
    if not os.path.isdir(perf_dir) or len(os.listdir(perf_dir)) == 0:
        return np.zeros((1, 1, 1), dtype=np.float32), False

    try:
        dcm_files = sorted(
            [f for f in os.listdir(perf_dir) if f.endswith(".dcm")],
            key=_sort_key_for_dcm,
        )
        if not dcm_files:
            return np.zeros((1, 1, 1), dtype=np.float32), False

        # Parse timepoint-slice structure from filenames
        slices_by_tp = {}
        for f in dcm_files:
            stem = f.replace(".dcm", "")
            parts = stem.split("-")
            tp = int(parts[0])
            if tp not in slices_by_tp:
                slices_by_tp[tp] = []
            ds = pydicom.dcmread(os.path.join(perf_dir, f), force=True)
            if hasattr(ds, "pixel_array"):
                slices_by_tp[tp].append(ds.pixel_array.astype(np.float32))

        if not slices_by_tp:
            return np.zeros((1, 1, 1), dtype=np.float32), False

        # Stack timepoints: for each timepoint, take the mean slice
        # (not all timepoints may have the same number of slices)
        tp_means = []
        for tp in sorted(slices_by_tp.keys()):
            if slices_by_tp[tp]:
                tp_vol = np.stack(slices_by_tp[tp], axis=0)  # (num_slices_this_tp, H, W)
                tp_means.append(tp_vol.mean(axis=0))  # mean across slices → (H, W)

        if not tp_means:
            return np.zeros((1, 1, 1), dtype=np.float32), False

        # Mean over time → proxy for rCBV
        tp_stack = np.stack(tp_means, axis=0)  # (num_timepoints, H, W)
        rcbv_proxy = tp_stack.mean(axis=0, keepdims=True)  # (1, H, W)
        return rcbv_proxy, True

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"  Warning: Perfusion failed for {patient_dir}: {e}")
        return np.zeros((1, 1, 1), dtype=np.float32), False


def load_all_modalities(patient_dir: str) -> Dict[str, Tuple[np.ndarray, bool]]:
    """
    Load all 6 modalities for a single patient.

    Returns dict mapping modality name → (volume, is_present).
    Missing modalities get zero-volumes and is_present=False.
    """
    result = {}

    # Core structural modalities
    for mod in ["T1-pre", "T1-post", "T2", "FLAIR"]:
        result[mod] = load_structural_modality(patient_dir, mod)

    # Advanced modalities (collapsed to scalar maps)
    result["DTI"] = load_dti_scalar(patient_dir)
    result["Perfusion"] = load_perfusion_scalar(patient_dir)

    return result


def normalize_volume(vol: np.ndarray, clip_percentile: float = 99.5) -> np.ndarray:
    """
    Percentile-based intensity normalization to [0, 1].
    Clips extreme values and normalizes.
    """
    if vol.max() == vol.min():
        return np.zeros_like(vol)
    clip_val = np.percentile(vol, clip_percentile)
    vol = np.clip(vol, 0, clip_val)
    vol = vol / (clip_val + 1e-8)
    return vol

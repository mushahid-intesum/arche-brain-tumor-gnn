"""
Preprocessing Orchestrator for Plan 3a.

Processes all patients in the UPenn-GBM dataset:
  1. Load DICOM volumes for all 6 modalities
  2. Extract patches with concept features
  3. Parse clinical metadata
  4. Save as PyTorch-compatible .pt files for training

Configuration:
    All settings are in plan3a/config.py:
      PREPROCESS_LIMIT         — process only first N patients (None = all)
      PREPROCESS_PATIENT_FILTER — process only this patient ID (None = all)

Usage:
    python -m plan3a.data.preprocess
"""
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import (
    DATA_ROOT, PROCESSED_DIR,
    PREPROCESS_LIMIT, PREPROCESS_PATIENT_FILTER,
)
from plan3a.data.dicom_loader import load_all_modalities
from plan3a.data.patch_extraction import extract_patient_patches
from plan3a.data.clinical import parse_clinical_csv, get_feature_dim


def preprocess_patient(patient_id: str, patient_dir: str) -> dict:
    """
    Run full preprocessing for one patient.

    Returns dict with all tensors needed for graph construction + training.
    """
    # ── Load DICOM volumes ───────────────────────────────────────────
    modality_vols = load_all_modalities(patient_dir)

    # ── Extract patches + concepts ───────────────────────────────────
    patch_data = extract_patient_patches(modality_vols)

    return patch_data


def run_preprocessing(
    data_root: str = None,
    output_dir: str = None,
    limit: int = None,
    patient_filter: str = None,
    verbose: bool = True,
):
    """
    Process all patients and save results.

    Args:
        data_root: path to upenn-filtered/
        output_dir: where to save .pt files
        limit: process only first N patients
        patient_filter: process only this patient ID
        verbose: print progress
    """
    if data_root is None:
        data_root = str(DATA_ROOT)
    if output_dir is None:
        output_dir = str(PROCESSED_DIR)

    os.makedirs(output_dir, exist_ok=True)

    # ── Discover patients ────────────────────────────────────────────
    patient_dirs = sorted([
        d for d in os.listdir(data_root)
        if d.startswith("UPENN-GBM") and os.path.isdir(os.path.join(data_root, d))
    ])

    if patient_filter:
        patient_dirs = [d for d in patient_dirs if patient_filter in d]
    if limit:
        patient_dirs = patient_dirs[:limit]

    if verbose:
        print(f"Plan 3a Preprocessing")
        print(f"  Data root: {data_root}")
        print(f"  Output dir: {output_dir}")
        print(f"  Patients to process: {len(patient_dirs)}")
        print()

    # ── Parse clinical data ──────────────────────────────────────────
    clinical_csv = os.path.join(data_root, "clinical_info.csv")
    if os.path.exists(clinical_csv):
        clinical_data = parse_clinical_csv(clinical_csv)
        if verbose:
            print(f"  Clinical data loaded: {len(clinical_data)} entries")
            print(f"  Clinical feature dim: {get_feature_dim()}")
    else:
        clinical_data = {}
        if verbose:
            print("  Warning: clinical_info.csv not found, proceeding without")

    # ── Process patients ─────────────────────────────────────────────
    manifest = {}
    errors = []

    for i, pid in enumerate(patient_dirs):
        t0 = time.time()
        patient_dir = os.path.join(data_root, pid)
        out_path = os.path.join(output_dir, f"{pid}.pt")

        if verbose:
            print(f"  [{i+1}/{len(patient_dirs)}] {pid} ... ", end="", flush=True)

        try:
            # Preprocessing
            patch_data = preprocess_patient(pid, patient_dir)

            # Attach clinical features
            clin = clinical_data.get(pid, None)
            if clin is not None:
                clinical_features = clin["features"]
                survival_time = clin["survival_time"]
                event = clin["event"]
                feature_names = clin["feature_names"]
            else:
                clinical_features = np.zeros(get_feature_dim(), dtype=np.float32)
                survival_time = None
                event = None
                feature_names = []

            # Build save dict
            save_dict = {
                "patient_id": pid,
                # Patch data
                "patches": torch.from_numpy(patch_data["patches"]),        # (N, 6, ps, ps)
                "coords": torch.from_numpy(patch_data["coords"]),          # (N, 3)
                "concepts": torch.from_numpy(patch_data["concepts"]),      # (N, 8)
                "modality_mask": patch_data["modality_mask"],              # dict
                "num_patches": patch_data["num_patches"],
                # Clinical data
                "clinical_features": torch.from_numpy(clinical_features),  # (D,)
                "clinical_feature_names": feature_names,
                # Survival labels
                "survival_time": torch.tensor(survival_time if survival_time is not None else 0.0,
                                              dtype=torch.float32),
                "event": torch.tensor(event if event is not None else 0, dtype=torch.long),
                "has_survival": survival_time is not None and event is not None,
            }

            torch.save(save_dict, out_path)

            elapsed = time.time() - t0
            manifest[pid] = {
                "num_patches": patch_data["num_patches"],
                "has_survival": save_dict["has_survival"],
                "survival_time": survival_time,
                "event": event,
                "modality_mask": patch_data["modality_mask"],
                "time_seconds": round(elapsed, 2),
            }

            if verbose:
                print(f"{patch_data['num_patches']} patches, {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            errors.append((pid, str(e)))
            if verbose:
                print(f"ERROR: {e} ({elapsed:.1f}s)")

    # ── Save manifest ────────────────────────────────────────────────
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"Preprocessing complete.")
        print(f"  Processed: {len(manifest)}/{len(patient_dirs)} patients")
        print(f"  Errors: {len(errors)}")
        if errors:
            for pid, err in errors[:5]:
                print(f"    {pid}: {err}")
        if manifest:
            patch_counts = [m["num_patches"] for m in manifest.values()]
            print(f"  Patch counts: min={min(patch_counts)}, "
                  f"max={max(patch_counts)}, "
                  f"mean={np.mean(patch_counts):.0f}")
            surv_count = sum(1 for m in manifest.values() if m["has_survival"])
            print(f"  With survival data: {surv_count}/{len(manifest)}")
        print(f"  Manifest saved to: {manifest_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    run_preprocessing(
        limit=PREPROCESS_LIMIT,
        patient_filter=PREPROCESS_PATIENT_FILTER,
    )

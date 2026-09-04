#!/usr/bin/env python3
"""
UPenn-GBM Dataset Setup Script — Create upenn-filtered from upenn-gbm.

The UPenn-GBM dataset from TCIA comes as deeply nested DICOM directories
with long, inconsistent session and series names. This script creates a
clean, flat structure by symlinking DICOM files into modality-labeled
folders per patient.

Result structure:
    upenn-filtered/
    ├── clinical_info.csv
    ├── manifest.json
    ├── UPENN-GBM-00001/
    │   ├── T1-pre/
    │   │   ├── 1-001.dcm → (symlink to original)
    │   │   └── ...
    │   ├── T1-post/
    │   ├── T2/
    │   ├── FLAIR/
    │   ├── DTI/
    │   └── Perfusion/
    ├── UPENN-GBM-00002/
    │   └── ...
    └── ...

Prerequisites:
    - Download UPenn-GBM from TCIA (https://wiki.cancerimagingarchive.net/display/Public/UPENN-GBM)
    - Place the extracted data somewhere accessible
    - The script expects the TCIA MRI structure:
        <source>/mri/UPENN-GBM/<PatientID>/<SessionDir>/<SeriesDir>/*.dcm

Usage:
    python setup_data.py --source /path/to/upenn-gbm --target /path/to/upenn-filtered
    python setup_data.py --source ../upenn-gbm         # default target = ./upenn-filtered
    python setup_data.py --help

Modality detection:
    Series directories contain CaPTk-processed names like:
      "3.000000-t1 axial ProcessedCaPTk-92201"
    The script matches keywords in the series name to assign modalities:
      T1-pre:     "t1" and NOT "post"/"stealth-post"
      T1-post:    "t1" and ("post" or "stealth-post")
      T2:         "t2" and NOT "flair"
      FLAIR:      "flair"
      DTI:        "dti" or "diff" (diffusion)
      Perfusion:  "perf"
"""
import os
import sys
import json
import csv
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


# ── Modality Detection ───────────────────────────────────────────────────

MODALITY_RULES = [
    # (modality, match_fn) — order matters: more specific first
    ("T1-post",   lambda s: ("t1" in s) and ("post" in s or "stealth" in s)),
    ("T1-pre",    lambda s: ("t1" in s) and ("post" not in s) and ("stealth" not in s)),
    ("FLAIR",     lambda s: "flair" in s),
    ("T2",        lambda s: ("t2" in s or "T2" in s.split("-")[0]) and "flair" not in s),
    ("DTI",       lambda s: "dti" in s or "diff" in s),
    ("Perfusion", lambda s: "perf" in s),
]


def detect_modality(series_name: str) -> str:
    """
    Detect which MRI modality a series directory represents.

    Args:
        series_name: e.g. "3.000000-t1 axial ProcessedCaPTk-92201"

    Returns:
        modality name or "unknown"
    """
    s = series_name.lower()
    for modality, match_fn in MODALITY_RULES:
        if match_fn(s):
            return modality
    return "unknown"


# ── Source Discovery ─────────────────────────────────────────────────────

def discover_patients(source_mri_dir: str) -> dict:
    """
    Discover all patients and their series in the TCIA MRI directory.

    Args:
        source_mri_dir: path to <source>/mri/UPENN-GBM/

    Returns:
        dict: patient_id → {modality → {series, session_dir, dicom_count, full_path}}
    """
    patients = {}

    if not os.path.isdir(source_mri_dir):
        print(f"ERROR: Source MRI directory not found: {source_mri_dir}")
        sys.exit(1)

    patient_dirs = sorted([
        d for d in os.listdir(source_mri_dir)
        if d.startswith("UPENN-GBM-") and os.path.isdir(os.path.join(source_mri_dir, d))
    ])

    for pid in patient_dirs:
        patient_path = os.path.join(source_mri_dir, pid)
        modalities = {}

        # Walk through session directories
        session_dirs = sorted([
            d for d in os.listdir(patient_path)
            if os.path.isdir(os.path.join(patient_path, d))
        ])

        for session in session_dirs:
            session_path = os.path.join(patient_path, session)

            # Walk through series directories within each session
            series_dirs = sorted([
                d for d in os.listdir(session_path)
                if os.path.isdir(os.path.join(session_path, d))
            ])

            for series in series_dirs:
                series_path = os.path.join(session_path, series)
                modality = detect_modality(series)

                if modality == "unknown":
                    continue

                # Count DICOM files
                dcm_files = [f for f in os.listdir(series_path) if f.endswith(".dcm")]
                if not dcm_files:
                    continue

                # If modality already found, keep the one with more DICOMs
                if modality in modalities:
                    if len(dcm_files) <= modalities[modality]["dicom_count"]:
                        continue

                modalities[modality] = {
                    "series": series,
                    "session": session,
                    "dicom_count": len(dcm_files),
                    "full_path": series_path,
                }

        if modalities:
            patients[pid] = modalities

    return patients


# ── Clinical Data ────────────────────────────────────────────────────────

def find_clinical_csv(source_dir: str) -> str:
    """Find the clinical info CSV in the source directory."""
    candidates = [
        os.path.join(source_dir, "UPENN-GBM_clinical_info_v2.1.csv"),
        os.path.join(source_dir, "UPENN-GBM_clinical_info_v2.0.csv"),
        os.path.join(source_dir, "clinical_info.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Search recursively
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if "clinical" in f.lower() and f.endswith(".csv"):
                return os.path.join(root, f)
    return None


# ── Symlink Creation ─────────────────────────────────────────────────────

def create_filtered_dataset(
    patients: dict,
    target_dir: str,
    source_dir: str,
    use_symlinks: bool = True,
    verbose: bool = True,
):
    """
    Create the upenn-filtered directory structure with symlinks.

    Args:
        patients: output of discover_patients()
        target_dir: where to create upenn-filtered
        source_dir: root of upenn-gbm (for clinical CSV)
        use_symlinks: True=symlink, False=copy (for systems without symlink support)
        verbose: print progress
    """
    os.makedirs(target_dir, exist_ok=True)

    expected_modalities = ["T1-pre", "T1-post", "T2", "FLAIR", "DTI", "Perfusion"]
    manifest = {}
    stats = defaultdict(int)

    for i, (pid, mods) in enumerate(sorted(patients.items())):
        patient_target = os.path.join(target_dir, pid)
        os.makedirs(patient_target, exist_ok=True)

        patient_manifest = {"modalities": {}}
        missing = []

        for mod in expected_modalities:
            mod_target = os.path.join(patient_target, mod)
            os.makedirs(mod_target, exist_ok=True)

            if mod not in mods:
                missing.append(mod)
                continue

            info = mods[mod]
            source_series_path = info["full_path"]
            dcm_files = sorted([
                f for f in os.listdir(source_series_path) if f.endswith(".dcm")
            ])

            for dcm in dcm_files:
                src = os.path.join(source_series_path, dcm)
                dst = os.path.join(mod_target, dcm)

                # Remove existing link/file if present
                if os.path.exists(dst) or os.path.islink(dst):
                    os.remove(dst)

                if use_symlinks:
                    os.symlink(os.path.abspath(src), dst)
                else:
                    shutil.copy2(src, dst)

            patient_manifest["modalities"][mod] = {
                "dicom_count": info["dicom_count"],
                "series": info["series"],
            }
            stats[mod] += 1

        manifest[pid] = patient_manifest

        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(patients)}] {pid}")

    # ── Copy clinical CSV ────────────────────────────────────────────
    clinical_src = find_clinical_csv(source_dir)
    if clinical_src:
        clinical_dst = os.path.join(target_dir, "clinical_info.csv")
        if not os.path.exists(clinical_dst):
            shutil.copy2(clinical_src, clinical_dst)
        stats["clinical_csv"] = 1
        if verbose:
            print(f"  Clinical CSV: {clinical_src}")
    else:
        if verbose:
            print("  WARNING: Clinical CSV not found in source directory")

    # ── Save manifest ────────────────────────────────────────────────
    # Add clinical info to manifest
    clinical_dst = os.path.join(target_dir, "clinical_info.csv")
    if os.path.isfile(clinical_dst):
        with open(clinical_dst) as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_id = row.get("ID", "").strip()
                pid = raw_id.rsplit("_", 1)[0] if "_" in raw_id else raw_id
                if pid in manifest:
                    manifest[pid]["clinical"] = {
                        "age": row.get("Age_at_scan_years", ""),
                        "gender": row.get("Gender", ""),
                        "survival_days": row.get("Survival_from_surgery_days_UPDATED", ""),
                        "survival_status": row.get("Survival_Status", ""),
                        "IDH1": row.get("IDH1", ""),
                        "MGMT": row.get("MGMT", ""),
                    }

    manifest_path = os.path.join(target_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest, stats


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Setup UPenn-GBM filtered dataset with clean symlinked structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --source /path/to/upenn-gbm
  %(prog)s --source ../upenn-gbm --target ./upenn-filtered
  %(prog)s --source /data/upenn-gbm --copy  # copy instead of symlink (Kaggle/Windows)

Notes:
  - The source directory should contain: mri/, UPENN-GBM_clinical_info_v2.1.csv
  - Uses symlinks by default (fast, no disk duplication)
  - Use --copy for platforms without symlink support (Kaggle datasets, Windows)
        """,
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to the upenn-gbm root directory (containing mri/ and CSVs)",
    )
    parser.add_argument(
        "--target", type=str, default="./upenn-filtered",
        help="Where to create the filtered dataset (default: ./upenn-filtered)",
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of symlink (for Kaggle/Windows compatibility)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without creating anything",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    target = os.path.abspath(args.target)
    verbose = not args.quiet

    # ── Validate source ──────────────────────────────────────────────
    mri_dir = os.path.join(source, "mri", "UPENN-GBM")
    if not os.path.isdir(mri_dir):
        # Try alternate structures
        alt_mri = os.path.join(source, "UPENN-GBM")
        alt_mri2 = os.path.join(source, "mri")
        if os.path.isdir(alt_mri):
            mri_dir = alt_mri
        elif os.path.isdir(alt_mri2):
            mri_dir = alt_mri2
        else:
            print(f"ERROR: Cannot find MRI data directory.")
            print(f"  Tried: {mri_dir}")
            print(f"  Tried: {alt_mri}")
            print(f"  Expected structure: <source>/mri/UPENN-GBM/<PatientID>/...")
            sys.exit(1)

    if verbose:
        print("=" * 60)
        print("UPenn-GBM Dataset Setup")
        print("=" * 60)
        print(f"  Source:  {source}")
        print(f"  MRI dir: {mri_dir}")
        print(f"  Target:  {target}")
        print(f"  Mode:    {'copy' if args.copy else 'symlink'}")
        print()

    # ── Discover ─────────────────────────────────────────────────────
    if verbose:
        print("Discovering patients and series...")
    patients = discover_patients(mri_dir)

    if not patients:
        print("ERROR: No patients found. Check source directory structure.")
        sys.exit(1)

    # Stats
    modality_counts = defaultdict(int)
    for pid, mods in patients.items():
        for mod in mods:
            modality_counts[mod] += 1

    if verbose:
        print(f"  Found {len(patients)} patients")
        print(f"  Modality coverage:")
        for mod in ["T1-pre", "T1-post", "T2", "FLAIR", "DTI", "Perfusion"]:
            count = modality_counts.get(mod, 0)
            pct = 100 * count / len(patients) if patients else 0
            print(f"    {mod:12s}: {count:4d}/{len(patients)} ({pct:.0f}%)")
        print()

    if args.dry_run:
        print("Dry run — no files created.")
        return

    # ── Create filtered dataset ──────────────────────────────────────
    if verbose:
        print("Creating filtered dataset...")
    manifest, stats = create_filtered_dataset(
        patients, target, source,
        use_symlinks=not args.copy,
        verbose=verbose,
    )

    # ── Summary ──────────────────────────────────────────────────────
    if verbose:
        print()
        print("=" * 60)
        print("Setup complete!")
        print("=" * 60)
        print(f"  Patients linked:  {len(manifest)}")
        for mod in ["T1-pre", "T1-post", "T2", "FLAIR", "DTI", "Perfusion"]:
            print(f"    {mod:12s}: {stats.get(mod, 0)} patients")
        print(f"  Clinical CSV:     {'✓' if stats.get('clinical_csv') else '✗'}")
        print(f"  Manifest:         {os.path.join(target, 'manifest.json')}")
        print(f"  Target directory: {target}")
        print()
        print("To use with Plan 3a:")
        print(f"  python -m plan3a.data.preprocess --data-root {target}")


if __name__ == "__main__":
    main()

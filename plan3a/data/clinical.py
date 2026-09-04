"""
Clinical Feature Parser for UPenn-GBM.

Handles the clinical_info.csv with:
  - One-hot encoding for categorical features
  - Normalization for continuous features
  - Missingness indicator columns for high-missing features (KPS, MGMT, PsP_TP)
  - Median imputation for missing continuous values
  - Survival label extraction with censoring

Mitigation strategies for missing data:
  1. Categorical: "Not Available" / "NOS/NEC" / "Indeterminate" → dedicated
     missing token (one-hot all-zeros + missingness flag)
  2. Continuous: Missing → median imputation + boolean 'is_missing' column
  3. Survival: Missing time or "Lost to Follow-up" → censored at last known time
"""
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import (
    CLINICAL_CSV, CLINICAL_CATEGORICAL, CLINICAL_CONTINUOUS,
    CLINICAL_SPARSE, SURVIVAL_TIME_COL, SURVIVAL_STATUS_COL,
    SURVIVAL_STATUS_MAP,
)


def parse_clinical_csv(csv_path: Optional[str] = None) -> Dict[str, Dict]:
    """
    Parse clinical_info.csv into per-patient feature dictionaries.

    Returns:
        dict mapping patient_id (e.g. 'UPENN-GBM-00001') → {
            'features': np.ndarray of shape (D,) — all clinical features
            'feature_names': list of str — feature dimension labels
            'survival_time': float or None
            'event': int (1=deceased, 0=censored) or None
            'raw': dict of raw CSV values
        }
    """
    if csv_path is None:
        csv_path = str(CLINICAL_CSV)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # ── Step 1: Compute median values for imputation ─────────────────
    continuous_values = {col: [] for col in CLINICAL_CONTINUOUS}
    sparse_values = {col: [] for col in CLINICAL_SPARSE}

    for row in rows:
        for col in CLINICAL_CONTINUOUS:
            val = row.get(col, "").strip()
            try:
                continuous_values[col].append(float(val))
            except (ValueError, TypeError):
                pass
        for col in CLINICAL_SPARSE:
            val = row.get(col, "").strip()
            try:
                sparse_values[col].append(float(val))
            except (ValueError, TypeError):
                pass

    medians = {}
    for col in CLINICAL_CONTINUOUS:
        vals = continuous_values[col]
        medians[col] = np.median(vals) if vals else 0.0
    for col in CLINICAL_SPARSE:
        vals = sparse_values[col]
        medians[col] = np.median(vals) if vals else 0.0

    # Also compute std for normalization
    stds = {}
    means = {}
    for col in list(CLINICAL_CONTINUOUS) + list(CLINICAL_SPARSE.keys()):
        vals = continuous_values.get(col, sparse_values.get(col, []))
        if vals:
            means[col] = np.mean(vals)
            stds[col] = np.std(vals) + 1e-8
        else:
            means[col] = 0.0
            stds[col] = 1.0

    # ── Step 2: Build per-patient features ───────────────────────────
    result = {}

    for row in rows:
        raw_id = row["ID"].strip()
        # Normalize ID: 'UPENN-GBM-00001_11' → 'UPENN-GBM-00001'
        patient_id = raw_id.rsplit("_", 1)[0] if "_" in raw_id else raw_id

        features = []
        feature_names = []

        # Categorical features: one-hot + missingness flag
        for col, valid_values in CLINICAL_CATEGORICAL.items():
            val = row.get(col, "").strip()
            is_missing = val not in valid_values
            one_hot = [1.0 if val == v else 0.0 for v in valid_values]
            features.extend(one_hot)
            features.append(1.0 if is_missing else 0.0)
            for v in valid_values:
                feature_names.append(f"{col}_{v}")
            feature_names.append(f"{col}_missing")

        # Continuous features: z-normalized
        for col in CLINICAL_CONTINUOUS:
            val = row.get(col, "").strip()
            try:
                numeric = float(val)
            except (ValueError, TypeError):
                numeric = medians[col]
                features.append((numeric - means[col]) / stds[col])
                features.append(1.0)  # missing flag
            else:
                features.append((numeric - means[col]) / stds[col])
                features.append(0.0)  # not missing
            feature_names.append(f"{col}_norm")
            feature_names.append(f"{col}_missing")

        # Sparse features: z-normalized + missingness indicator
        for col, dtype in CLINICAL_SPARSE.items():
            val = row.get(col, "").strip()
            try:
                numeric = float(val)
            except (ValueError, TypeError):
                numeric = medians[col]
                features.append((numeric - means[col]) / stds[col])
                features.append(1.0)  # is_missing = True
            else:
                features.append((numeric - means[col]) / stds[col])
                features.append(0.0)
            feature_names.append(f"{col}_norm")
            feature_names.append(f"{col}_missing")

        # ── Survival labels ──────────────────────────────────────────
        surv_time_str = row.get(SURVIVAL_TIME_COL, "").strip()
        surv_status_str = row.get(SURVIVAL_STATUS_COL, "").strip()

        try:
            survival_time = float(surv_time_str)
        except (ValueError, TypeError):
            survival_time = None

        event = SURVIVAL_STATUS_MAP.get(surv_status_str, None)

        result[patient_id] = {
            "features": np.array(features, dtype=np.float32),
            "feature_names": feature_names,
            "survival_time": survival_time,
            "event": event,
            "raw": dict(row),
        }

    return result


def get_feature_dim() -> int:
    """Return the dimensionality of the clinical feature vector."""
    dim = 0
    for col, valid_values in CLINICAL_CATEGORICAL.items():
        dim += len(valid_values) + 1  # one-hot + missing flag
    for col in CLINICAL_CONTINUOUS:
        dim += 2  # value + missing flag
    for col in CLINICAL_SPARSE:
        dim += 2  # value + missing flag
    return dim


def get_survival_labels(
    clinical_data: Dict[str, Dict],
    patient_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract aligned survival labels for a list of patient IDs.

    Returns:
        times: (N,) survival times in days
        events: (N,) event indicators (1=deceased, 0=censored)
        valid_mask: (N,) boolean — True if survival data is available
    """
    times = []
    events = []
    valid = []

    for pid in patient_ids:
        info = clinical_data.get(pid, {})
        t = info.get("survival_time")
        e = info.get("event")
        if t is not None and e is not None:
            times.append(t)
            events.append(e)
            valid.append(True)
        else:
            times.append(0.0)
            events.append(0)
            valid.append(False)

    return (
        np.array(times, dtype=np.float32),
        np.array(events, dtype=np.int64),
        np.array(valid, dtype=bool),
    )

"""
Task and Concept Metrics for Plan 3a.

Survival: C-Index (concordance index)
Concepts: Per-concept MSE, correlation, accuracy
Risk Stratification: Kaplan-Meier log-rank test
"""
import numpy as np
from typing import Dict, List, Tuple


def concordance_index(
    predicted_risk: np.ndarray,
    survival_time: np.ndarray,
    event: np.ndarray,
) -> float:
    """
    Compute Harrell's concordance index (C-Index).

    For all comparable pairs (i, j) where patient i had an event
    before patient j, check if the model predicted higher risk for i.

    Args:
        predicted_risk: (N,) predicted risk scores (higher = worse prognosis)
        survival_time: (N,) survival times in days
        event: (N,) event indicators (1=deceased, 0=censored)

    Returns:
        C-Index ∈ [0, 1]. 0.5 = random, 1.0 = perfect concordance.
    """
    N = len(predicted_risk)
    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(N):
        for j in range(i + 1, N):
            # Only compare if at least one had an event
            # and the event patient had shorter survival
            if event[i] == 1 and survival_time[i] < survival_time[j]:
                if predicted_risk[i] > predicted_risk[j]:
                    concordant += 1
                elif predicted_risk[i] < predicted_risk[j]:
                    discordant += 1
                else:
                    tied_risk += 0.5

            elif event[j] == 1 and survival_time[j] < survival_time[i]:
                if predicted_risk[j] > predicted_risk[i]:
                    concordant += 1
                elif predicted_risk[j] < predicted_risk[i]:
                    discordant += 1
                else:
                    tied_risk += 0.5

            elif event[i] == 1 and event[j] == 1 and survival_time[i] == survival_time[j]:
                tied_risk += 1

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied_risk) / total


def concept_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    concept_names: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-concept evaluation metrics.

    Args:
        predicted: (N, C) predicted concept values
        target: (N, C) ground truth concept values
        concept_names: list of concept names

    Returns:
        dict mapping concept_name → {mse, pearson_r, mae}
    """
    N, C = predicted.shape
    if concept_names is None:
        concept_names = [f"c{i+1}" for i in range(C)]

    results = {}
    for i, name in enumerate(concept_names):
        pred_i = predicted[:, i]
        tgt_i = target[:, i]

        mse = float(np.mean((pred_i - tgt_i) ** 2))
        mae = float(np.mean(np.abs(pred_i - tgt_i)))

        # Pearson correlation
        if np.std(pred_i) > 1e-8 and np.std(tgt_i) > 1e-8:
            pearson_r = float(np.corrcoef(pred_i, tgt_i)[0, 1])
        else:
            pearson_r = 0.0

        results[name] = {
            "mse": mse,
            "mae": mae,
            "pearson_r": pearson_r,
        }

    return results


def hazard_to_risk(hazard_logits: np.ndarray) -> np.ndarray:
    """
    Convert hazard logits to cumulative risk scores for C-Index computation.

    Risk = 1 - S(t) where S(t) = Π(1 - h_k) is the survival function.
    Higher risk = worse prognosis.

    Args:
        hazard_logits: (B, K) raw hazard logits

    Returns:
        risk: (B,) cumulative risk scores
    """
    hazard = 1.0 / (1.0 + np.exp(-hazard_logits))  # sigmoid
    survival = np.prod(1.0 - hazard, axis=1)
    risk = 1.0 - survival
    return risk


def compute_time_bins(
    survival_times: np.ndarray,
    events: np.ndarray,
    num_bins: int = 4,
) -> np.ndarray:
    """
    Compute time bin boundaries from the training set.

    Uses quantiles of event times (not censored times) to ensure
    each bin has roughly equal numbers of events.

    Args:
        survival_times: (N,) all survival times
        events: (N,) event indicators
        num_bins: number of bins

    Returns:
        bins: (num_bins,) boundary values in days
    """
    event_times = survival_times[events == 1]
    if len(event_times) == 0:
        # Fallback: use all times
        event_times = survival_times

    quantiles = np.linspace(0, 100, num_bins + 1)[1:]  # skip 0%
    bins = np.percentile(event_times, quantiles)

    return bins.astype(np.float32)

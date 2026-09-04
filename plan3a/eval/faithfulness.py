"""
Faithfulness Evaluation for Plan 3a (ICLR 2026 — Degenerate GNN Explanations).

Implements the auditing framework to detect unfaithful/degenerate explanations:

Metrics:
  EST  — Extension Sufficiency Test: worst-case prediction shift when adding
         non-explanation nodes back. Detects anchor-set degeneracy.
  Fid⁻ — Fidelity-minus: prediction shift when removing the complement
         (feeding only the explanation subgraph).
  RFid⁻— Randomized Fid-minus: prediction shift under random complement
         perturbation (edge removal with probability p).
  Suf  — Sufficiency: prediction preservation when swapping the complement
         with another patient's complement.

Rejection Ratios:
  For each metric, the fraction of explanations that "fail" —
  where the prediction changes significantly (beyond a threshold).
  Low rejection = faithful explanations.

Usage:
  from plan3a.eval.faithfulness import FaithfulnessAuditor
  auditor = FaithfulnessAuditor(model)
  report = auditor.audit_patient(patient_data, explanation_mask)
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import SHEAF_HGNN_DIM


class ExplanationExtractor:
    """
    Extract explanations from a trained Plan3a model.

    Explanations are defined as the top-k most important nodes
    (patches) based on concept activation magnitude.

    This is the "explanation subgraph" R ⊆ G.
    """

    def __init__(self, top_k_ratio: float = 0.2):
        """
        Args:
            top_k_ratio: fraction of nodes to include in the explanation
                         (e.g., 0.2 = top 20% most activated nodes)
        """
        self.top_k_ratio = top_k_ratio

    @torch.no_grad()
    def extract(
        self,
        model,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        concepts_target: torch.Tensor = None,
        clinical_features: torch.Tensor = None,
    ) -> Dict:
        """
        Run model and extract explanation mask.

        Returns:
            dict with:
                "explanation_mask": (N,) boolean — True for explanation nodes
                "importance_scores": (N,) — concept-based node importance
                "full_output": model output dict
                "full_prediction": (K,) hazard logits for full graph
        """
        model.eval()

        outputs = model(
            node_features=node_features,
            hyperedge_index=hyperedge_index,
            num_nodes=num_nodes,
            num_edges=num_edges,
            concept_targets=concepts_target,
            clinical_features=clinical_features,
        )

        # Importance = L2 norm of concept activations per node
        concepts = outputs["concepts"]  # (N, C)
        importance = torch.norm(concepts, dim=-1)  # (N,)

        # Top-k mask
        k = max(1, int(num_nodes * self.top_k_ratio))
        _, top_indices = torch.topk(importance, k)
        explanation_mask = torch.zeros(num_nodes, dtype=torch.bool,
                                       device=node_features.device)
        explanation_mask[top_indices] = True

        return {
            "explanation_mask": explanation_mask,
            "importance_scores": importance,
            "full_output": outputs,
            "full_prediction": outputs["hazard_logits"].detach(),
        }


def _filter_hyperedges(
    hyperedge_index: torch.Tensor,
    node_mask: torch.Tensor,
    num_edges: int,
) -> Tuple[torch.Tensor, int]:
    """
    Filter hyperedge incidence to only include specified nodes.

    Removes connections to masked-out nodes and drops empty hyperedges.

    Returns:
        new_hyperedge_index: filtered incidence matrix
        new_num_edges: number of remaining hyperedges
    """
    if hyperedge_index.shape[1] == 0:
        return hyperedge_index, 0

    node_idx = hyperedge_index[0]
    edge_idx = hyperedge_index[1]

    # Keep only connections where the node is in the mask
    keep = node_mask[node_idx]
    new_node_idx = node_idx[keep]
    new_edge_idx = edge_idx[keep]

    if new_node_idx.shape[0] == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=hyperedge_index.device), 0

    # Reindex edges to be contiguous
    unique_edges = torch.unique(new_edge_idx)
    edge_mapping = torch.zeros(num_edges, dtype=torch.long, device=hyperedge_index.device)
    edge_mapping[unique_edges] = torch.arange(len(unique_edges), device=hyperedge_index.device)
    new_edge_idx = edge_mapping[new_edge_idx]

    return torch.stack([new_node_idx, new_edge_idx]), len(unique_edges)


class FaithfulnessAuditor:
    """
    Auditor for GNN explanation faithfulness.

    Implements EST, Fid⁻, RFid⁻, and Suf metrics from ICLR 2026.
    """

    def __init__(
        self,
        model,
        device: str = "cpu",
        est_samples: int = 50,
        rfid_p: float = 0.9,
        prediction_threshold: float = 0.1,
    ):
        """
        Args:
            model: trained Plan3aModel
            device: compute device
            est_samples: number of Monte Carlo supergraph samples for EST
            rfid_p: edge removal probability for RFid⁻
            prediction_threshold: max allowed prediction shift for "faithful"
        """
        self.model = model
        self.device = device
        self.est_samples = est_samples
        self.rfid_p = rfid_p
        self.threshold = prediction_threshold

    @torch.no_grad()
    def _get_prediction(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        clinical_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run forward pass and return hazard logits."""
        self.model.eval()
        outputs = self.model(
            node_features=node_features,
            hyperedge_index=hyperedge_index,
            num_nodes=num_nodes,
            num_edges=num_edges,
            clinical_features=clinical_features,
        )
        return outputs["hazard_logits"].detach()

    def _prediction_shift(
        self,
        pred_a: torch.Tensor,
        pred_b: torch.Tensor,
    ) -> float:
        """L1 distance between two hazard predictions (after sigmoid)."""
        p_a = torch.sigmoid(pred_a)
        p_b = torch.sigmoid(pred_b)
        return float((p_a - p_b).abs().mean())

    def compute_est(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        explanation_mask: torch.Tensor,
        full_prediction: torch.Tensor,
        clinical_features: torch.Tensor = None,
    ) -> Dict[str, float]:
        """
        Extension Sufficiency Test (EST).

        Tests: can the prediction change when we add non-explanation nodes
        back to the explanation subgraph?

        For a faithful explanation R, adding any subset of the complement
        G\\R should NOT change the prediction. EST finds the worst case.

        Procedure:
          1. Start with explanation subgraph R
          2. Sample random subsets S of complement nodes
          3. Build supergraph R' = R ∪ S
          4. Get prediction on R'
          5. EST = max shift across all samples

        Returns:
            dict with "est_score" and "est_pass" (bool)
        """
        complement_mask = ~explanation_mask
        complement_indices = torch.where(complement_mask)[0]
        n_complement = len(complement_indices)

        if n_complement == 0:
            return {"est_score": 0.0, "est_pass": True}

        max_shift = 0.0

        for _ in range(self.est_samples):
            # Sample random subset of complement
            sample_size = torch.randint(1, max(2, n_complement), (1,)).item()
            perm = torch.randperm(n_complement)[:sample_size]
            sampled_complement = complement_indices[perm]

            # Build supergraph mask: explanation + sampled complement
            supergraph_mask = explanation_mask.clone()
            supergraph_mask[sampled_complement] = True

            # Filter hyperedges for supergraph
            filtered_he, n_he = _filter_hyperedges(
                hyperedge_index, supergraph_mask, num_edges
            )

            # Get prediction on supergraph
            supergraph_pred = self._get_prediction(
                node_features,  # all node features (masking is via hyperedges)
                filtered_he,
                num_nodes,
                n_he,
                clinical_features,
            )

            shift = self._prediction_shift(full_prediction, supergraph_pred)
            max_shift = max(max_shift, shift)

        return {
            "est_score": max_shift,
            "est_pass": max_shift < self.threshold,
        }

    def compute_fid_minus(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        explanation_mask: torch.Tensor,
        full_prediction: torch.Tensor,
        clinical_features: torch.Tensor = None,
    ) -> Dict[str, float]:
        """
        Fidelity-minus (Fid⁻).

        Tests: does the prediction change when we feed ONLY the explanation
        subgraph (removing the complement entirely)?

        If the explanation captures the true reasoning, prediction on R alone
        should match prediction on the full graph G.

        Returns:
            dict with "fid_minus_score" and "fid_minus_pass"
        """
        # Filter to explanation-only subgraph
        filtered_he, n_he = _filter_hyperedges(
            hyperedge_index, explanation_mask, num_edges
        )

        # Predict on explanation subgraph
        expl_pred = self._get_prediction(
            node_features, filtered_he, num_nodes, n_he, clinical_features,
        )

        shift = self._prediction_shift(full_prediction, expl_pred)

        return {
            "fid_minus_score": shift,
            "fid_minus_pass": shift < self.threshold,
        }

    def compute_rfid_minus(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        explanation_mask: torch.Tensor,
        full_prediction: torch.Tensor,
        clinical_features: torch.Tensor = None,
    ) -> Dict[str, float]:
        """
        Randomized Fidelity-minus (RFid⁻).

        Tests: does randomly perturbing the complement change the prediction?

        Procedure:
          1. Randomly drop complement connections with probability p
          2. Keep all explanation connections
          3. Measure prediction shift

        Multiple runs are averaged.

        Returns:
            dict with "rfid_minus_score" and "rfid_minus_pass"
        """
        complement_mask = ~explanation_mask

        shifts = []
        for _ in range(min(self.est_samples, 20)):
            # Randomly drop complement connections
            node_idx = hyperedge_index[0]
            edge_idx = hyperedge_index[1]

            # For each connection, drop if node is in complement AND random < p
            is_complement = complement_mask[node_idx]
            drop = torch.rand(is_complement.shape, device=node_idx.device) < self.rfid_p
            keep = ~(is_complement & drop)

            perturbed_he = torch.stack([node_idx[keep], edge_idx[keep]])

            # Reindex edges
            if perturbed_he.shape[1] > 0:
                unique_edges = torch.unique(perturbed_he[1])
                edge_map = torch.zeros(num_edges, dtype=torch.long,
                                       device=perturbed_he.device)
                edge_map[unique_edges] = torch.arange(len(unique_edges),
                                                       device=perturbed_he.device)
                perturbed_he[1] = edge_map[perturbed_he[1]]
                n_he = len(unique_edges)
            else:
                n_he = 0

            pred = self._get_prediction(
                node_features, perturbed_he, num_nodes, n_he, clinical_features,
            )
            shifts.append(self._prediction_shift(full_prediction, pred))

        mean_shift = np.mean(shifts) if shifts else 0.0

        return {
            "rfid_minus_score": mean_shift,
            "rfid_minus_pass": mean_shift < self.threshold,
        }

    def compute_sufficiency(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        explanation_mask: torch.Tensor,
        full_prediction: torch.Tensor,
        other_node_features: torch.Tensor = None,
        clinical_features: torch.Tensor = None,
    ) -> Dict[str, float]:
        """
        Sufficiency (Suf).

        Tests: if we keep the explanation but replace the complement with
        random noise, does the prediction stay the same?

        If the explanation truly captures the model's reasoning,
        the complement should be irrelevant.

        Returns:
            dict with "sufficiency_score" and "sufficiency_pass"
        """
        complement_mask = ~explanation_mask

        # Replace complement node features with random noise
        perturbed_features = node_features.clone()
        n_complement = complement_mask.sum().item()
        if n_complement > 0:
            noise = torch.randn_like(perturbed_features[complement_mask])
            # Scale noise to match feature statistics
            feat_std = node_features.std()
            feat_mean = node_features.mean()
            noise = noise * feat_std + feat_mean
            perturbed_features[complement_mask] = noise

        pred = self._get_prediction(
            perturbed_features, hyperedge_index, num_nodes, num_edges,
            clinical_features,
        )

        shift = self._prediction_shift(full_prediction, pred)

        return {
            "sufficiency_score": shift,
            "sufficiency_pass": shift < self.threshold,
        }

    def audit_patient(
        self,
        data: Dict,
        explanation_mask: torch.Tensor = None,
        full_prediction: torch.Tensor = None,
        top_k_ratio: float = 0.2,
    ) -> Dict:
        """
        Run full faithfulness audit on one patient.

        If no explanation_mask is provided, extracts one automatically
        using concept activation magnitude.

        Returns comprehensive audit report.
        """
        self.model.eval()
        device = self.device

        node_features = data["node_features"].to(device)
        hyperedge_index = data["hyperedge_index"].to(device)
        num_nodes = data["num_nodes"]
        num_edges = data["num_hyperedges"]
        clinical = data["clinical_features"].to(device)

        # Extract explanation if not provided
        if explanation_mask is None:
            extractor = ExplanationExtractor(top_k_ratio)
            expl_data = extractor.extract(
                self.model, node_features, hyperedge_index,
                num_nodes, num_edges,
                data.get("concepts", None),
                clinical,
            )
            explanation_mask = expl_data["explanation_mask"]
            full_prediction = expl_data["full_prediction"]
        elif full_prediction is None:
            full_prediction = self._get_prediction(
                node_features, hyperedge_index, num_nodes, num_edges, clinical,
            )

        explanation_mask = explanation_mask.to(device)
        full_prediction = full_prediction.to(device)

        # Run all metrics
        est = self.compute_est(
            node_features, hyperedge_index, num_nodes, num_edges,
            explanation_mask, full_prediction, clinical,
        )
        fid = self.compute_fid_minus(
            node_features, hyperedge_index, num_nodes, num_edges,
            explanation_mask, full_prediction, clinical,
        )
        rfid = self.compute_rfid_minus(
            node_features, hyperedge_index, num_nodes, num_edges,
            explanation_mask, full_prediction, clinical,
        )
        suf = self.compute_sufficiency(
            node_features, hyperedge_index, num_nodes, num_edges,
            explanation_mask, full_prediction, clinical_features=clinical,
        )

        # Aggregate
        n_explanation = explanation_mask.sum().item()
        n_total = num_nodes

        return {
            "patient_id": data.get("patient_id", "unknown"),
            "num_nodes": n_total,
            "num_explanation_nodes": n_explanation,
            "explanation_ratio": n_explanation / max(n_total, 1),
            # Metrics
            "est": est,
            "fid_minus": fid,
            "rfid_minus": rfid,
            "sufficiency": suf,
            # Overall pass/fail
            "all_pass": all([
                est["est_pass"],
                fid["fid_minus_pass"],
                rfid["rfid_minus_pass"],
                suf["sufficiency_pass"],
            ]),
        }


def compute_rejection_ratios(
    audit_reports: List[Dict],
) -> Dict[str, float]:
    """
    Compute rejection ratios across multiple patients.

    Rejection ratio = fraction of explanations that FAIL each metric.
    Lower is better.

    Returns:
        dict with rejection ratios per metric and overall.
    """
    if not audit_reports:
        return {}

    n = len(audit_reports)

    est_fails = sum(1 for r in audit_reports if not r["est"]["est_pass"])
    fid_fails = sum(1 for r in audit_reports if not r["fid_minus"]["fid_minus_pass"])
    rfid_fails = sum(1 for r in audit_reports if not r["rfid_minus"]["rfid_minus_pass"])
    suf_fails = sum(1 for r in audit_reports if not r["sufficiency"]["sufficiency_pass"])
    all_fails = sum(1 for r in audit_reports if not r["all_pass"])

    return {
        "est_rejection": est_fails / n,
        "fid_minus_rejection": fid_fails / n,
        "rfid_minus_rejection": rfid_fails / n,
        "sufficiency_rejection": suf_fails / n,
        "overall_rejection": all_fails / n,
        "n_patients": n,
        # Mean scores
        "mean_est": np.mean([r["est"]["est_score"] for r in audit_reports]),
        "mean_fid_minus": np.mean([r["fid_minus"]["fid_minus_score"] for r in audit_reports]),
        "mean_rfid_minus": np.mean([r["rfid_minus"]["rfid_minus_score"] for r in audit_reports]),
        "mean_sufficiency": np.mean([r["sufficiency"]["sufficiency_score"] for r in audit_reports]),
    }

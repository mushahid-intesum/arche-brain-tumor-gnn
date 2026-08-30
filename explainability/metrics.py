"""
Explainability evaluation metrics for the hierarchical GNN.

Implements standard XAI metrics adapted for link prediction:
  - Fidelity (necessity and sufficiency)
  - Sparsity
  - Stability
  - Attention faithfulness
  - Complexity

References:
  - GraphFramEx (Amara et al., ICML 2022): fidelity/sufficiency framework
  - Robust Fidelity (Zheng et al., ICLR 2024): distribution-aware fidelity
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

from gnn import StructuralFeatureComputer, get_edge_metadata


class ExplainabilityMetrics:
    """Compute XAI metrics for edge explanations on a trained model."""

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.sf_computer = StructuralFeatureComputer()

    # ── Helper: get prediction logit for a single edge ────────────────

    def _predict_edge(self, data, z, edge_pair, structural_feats, cn_list,
                      src_tissue, dst_tissue, is_inter):
        """Get raw logit for a single edge."""
        ei = torch.tensor([[edge_pair[0]], [edge_pair[1]]], dtype=torch.long)
        logit = self.model.decode(z, ei, structural_feats, cn_list,
                                  src_tissue, dst_tissue, is_inter)
        return logit.item()

    # ══════════════════════════════════════════════════════════════════
    # 1. FIDELITY
    # ══════════════════════════════════════════════════════════════════

    def fidelity_necessity(self, data, z, sv_attentions, edge_idx,
                           structural_feats, cn_indices, src_tissue,
                           dst_tissue, is_inter, top_k_svs=3):
        """Fidelity+ (necessity): how much does prediction drop when explanation is removed?

        Higher = explanation is more necessary for the prediction.

        Tests two levels:
          - Level 1 (structural): zero out OCN features
          - Level 2 (supervoxel): mask top-k attended SVs and re-encode
        """
        self.model.eval()

        # Original prediction
        ei = data.edge_index[:, edge_idx:edge_idx+1]
        sf = structural_feats[edge_idx:edge_idx+1]
        cn = [cn_indices[edge_idx]]
        st = src_tissue[edge_idx:edge_idx+1]
        dt = dst_tissue[edge_idx:edge_idx+1]
        inter = is_inter[edge_idx:edge_idx+1]

        original_logit = self.model.decode(z, ei, sf, cn, st, dt, inter).item()
        original_prob = torch.sigmoid(torch.tensor(original_logit)).item()

        results = {}

        # Level 1: zero out structural features
        sf_zeroed = torch.zeros_like(sf)
        l1_logit = self.model.decode(z, ei, sf_zeroed, cn, st, dt, inter).item()
        l1_prob = torch.sigmoid(torch.tensor(l1_logit)).item()
        results["level1_drop"] = original_prob - l1_prob

        # Level 2: mask top-k SVs and re-encode
        src_node = ei[0, 0].item()
        dst_node = ei[1, 0].item()

        if sv_attentions and len(sv_attentions) > max(src_node, dst_node):
            z_masked = z.clone()

            for node in [src_node, dst_node]:
                attn = sv_attentions[node]
                if len(attn) <= top_k_svs:
                    continue

                # Find top-k SV indices
                top_k_idx = torch.topk(attn, min(top_k_svs, len(attn))).indices

                # Create masked SV features (zero out top-k)
                sv_feats_masked = data.sv_features[node].clone()
                sv_feats_masked[top_k_idx] = 0.0

                # Re-aggregate
                with torch.no_grad():
                    masked_embed, _ = self.model.aggregator([sv_feats_masked], device=self.device)
                    z_masked[node] = masked_embed[0]

            l2_logit = self.model.decode(z_masked, ei, sf, cn, st, dt, inter).item()
            l2_prob = torch.sigmoid(torch.tensor(l2_logit)).item()
            results["level2_drop"] = original_prob - l2_prob
        else:
            results["level2_drop"] = 0.0

        # Combined: remove both
        if sv_attentions and len(sv_attentions) > max(src_node, dst_node):
            combined_logit = self.model.decode(z_masked, ei, sf_zeroed, cn, st, dt, inter).item()
            combined_prob = torch.sigmoid(torch.tensor(combined_logit)).item()
            results["combined_drop"] = original_prob - combined_prob
        else:
            results["combined_drop"] = results["level1_drop"]

        results["original_prob"] = original_prob
        return results

    def fidelity_sufficiency(self, data, z, sv_attentions, edge_idx,
                             structural_feats, cn_indices, src_tissue,
                             dst_tissue, is_inter, top_k_svs=3):
        """Fidelity- (sufficiency): how much prediction is retained with ONLY the explanation?

        Higher = explanation alone is sufficient for the prediction.

        Tests two levels:
          - Level 1: keep only structural features, use zero embeddings
          - Level 2: keep only top-k SVs, mask all others
        """
        self.model.eval()

        ei = data.edge_index[:, edge_idx:edge_idx+1]
        sf = structural_feats[edge_idx:edge_idx+1]
        cn = [cn_indices[edge_idx]]
        st = src_tissue[edge_idx:edge_idx+1]
        dt = dst_tissue[edge_idx:edge_idx+1]
        inter = is_inter[edge_idx:edge_idx+1]

        original_logit = self.model.decode(z, ei, sf, cn, st, dt, inter).item()
        original_prob = torch.sigmoid(torch.tensor(original_logit)).item()

        results = {}

        # Level 1: keep only structural features, zero node embeddings
        z_zero = torch.zeros_like(z)
        l1_logit = self.model.decode(z_zero, ei, sf, cn, st, dt, inter).item()
        l1_prob = torch.sigmoid(torch.tensor(l1_logit)).item()
        results["level1_retention"] = l1_prob / max(original_prob, 1e-8)

        # Level 2: keep only top-k SVs
        src_node = ei[0, 0].item()
        dst_node = ei[1, 0].item()

        if sv_attentions and len(sv_attentions) > max(src_node, dst_node):
            z_topk = z.clone()

            for node in [src_node, dst_node]:
                attn = sv_attentions[node]
                if len(attn) == 0:
                    continue

                top_k_idx = torch.topk(attn, min(top_k_svs, len(attn))).indices

                # Keep only top-k SVs, zero everything else
                sv_feats_topk = torch.zeros_like(data.sv_features[node])
                sv_feats_topk[top_k_idx] = data.sv_features[node][top_k_idx]

                with torch.no_grad():
                    topk_embed, _ = self.model.aggregator([sv_feats_topk], device=self.device)
                    z_topk[node] = topk_embed[0]

            l2_logit = self.model.decode(z_topk, ei, sf, cn, st, dt, inter).item()
            l2_prob = torch.sigmoid(torch.tensor(l2_logit)).item()
            results["level2_retention"] = l2_prob / max(original_prob, 1e-8)
        else:
            results["level2_retention"] = 0.0

        results["original_prob"] = original_prob
        return results

    # ══════════════════════════════════════════════════════════════════
    # 2. SPARSITY
    # ══════════════════════════════════════════════════════════════════

    def sparsity(self, sv_attentions, structural_feats, edge_idx,
                 src_node, dst_node, top_k_svs=3, sf_threshold=0.1):
        """Sparsity: what fraction of total features does the explanation use?

        Lower = more concise explanation.

        Returns:
            sv_sparsity: fraction of SVs in the explanation
            sf_sparsity: fraction of non-trivial structural features
        """
        results = {}

        # SV sparsity: top-k / total per node
        sv_fractions = []
        for node in [src_node, dst_node]:
            if sv_attentions and node < len(sv_attentions):
                total_svs = len(sv_attentions[node])
                if total_svs > 0:
                    sv_fractions.append(min(top_k_svs, total_svs) / total_svs)
        results["sv_sparsity"] = float(np.mean(sv_fractions)) if sv_fractions else 1.0

        # Structural feature sparsity
        sf = structural_feats[edge_idx]
        active = (sf.abs() > sf_threshold).sum().item()
        results["sf_sparsity"] = active / max(sf.size(0), 1)

        return results

    # ══════════════════════════════════════════════════════════════════
    # 3. STABILITY
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def stability(explanations_run1, explanations_run2):
        """Stability: how consistent are explanations across different model runs?

        Args:
            explanations_run1: list of dicts with 'top_sv_indices' and 'sf_ranking'
            explanations_run2: same structure from a different seed run

        Returns:
            sv_jaccard: mean Jaccard similarity of top-k SV sets
            sf_correlation: mean Spearman correlation of structural feature rankings
        """
        sv_jaccards = []
        sf_correlations = []

        for e1, e2 in zip(explanations_run1, explanations_run2):
            # SV Jaccard
            set1 = set(e1.get("top_sv_indices", []))
            set2 = set(e2.get("top_sv_indices", []))
            if set1 or set2:
                jaccard = len(set1 & set2) / max(len(set1 | set2), 1)
                sv_jaccards.append(jaccard)

            # Structural feature ranking correlation
            r1 = e1.get("sf_ranking", [])
            r2 = e2.get("sf_ranking", [])
            if len(r1) >= 2 and len(r2) >= 2 and len(r1) == len(r2):
                corr, _ = spearmanr(r1, r2)
                if not np.isnan(corr):
                    sf_correlations.append(corr)

        return {
            "sv_jaccard": float(np.mean(sv_jaccards)) if sv_jaccards else 0.0,
            "sf_correlation": float(np.mean(sf_correlations)) if sf_correlations else 0.0,
        }

    # ══════════════════════════════════════════════════════════════════
    # 4. ATTENTION FAITHFULNESS
    # ══════════════════════════════════════════════════════════════════

    def attention_faithfulness(self, data, edge_idx, structural_feats,
                               cn_indices, src_tissue, dst_tissue, is_inter):
        """Compare GATv2 attention weights against gradient-based saliency.

        High Spearman correlation = attention is faithful to the model's
        actual decision process. Low correlation = attention may be misleading.
        """
        self.model.eval()
        data_device = data.x.to(self.device)

        # Enable gradients on input
        if hasattr(data, 'sv_features') and data.sv_features:
            for sv_f in data.sv_features:
                if sv_f.requires_grad is False and sv_f.size(0) > 0:
                    sv_f.requires_grad_(True)

        # Forward with attention
        z, alphas, sv_attns = self.model.encode(data, return_attention=True)

        # Compute prediction for the target edge
        ei = data.edge_index[:, edge_idx:edge_idx+1]
        sf = structural_feats[edge_idx:edge_idx+1]
        cn = [cn_indices[edge_idx]]
        st = src_tissue[edge_idx:edge_idx+1]
        dt = dst_tissue[edge_idx:edge_idx+1]
        inter = is_inter[edge_idx:edge_idx+1]

        logit = self.model.decode(z, ei, sf, cn, st, dt, inter)

        # Backward to get gradients
        logit.backward(retain_graph=True)

        # Compare attention vs gradients for SV features
        src_node = ei[0, 0].item()
        dst_node = ei[1, 0].item()

        correlations = []

        for node in [src_node, dst_node]:
            if (sv_attns and node < len(sv_attns)
                    and len(sv_attns[node]) >= 3
                    and node < len(data.sv_features)
                    and data.sv_features[node].grad is not None):

                attn_weights = sv_attns[node].detach().cpu().numpy()
                grad_magnitude = data.sv_features[node].grad.abs().sum(dim=-1).detach().cpu().numpy()

                if len(attn_weights) == len(grad_magnitude) and len(attn_weights) >= 3:
                    corr, _ = spearmanr(attn_weights, grad_magnitude)
                    if not np.isnan(corr):
                        correlations.append(corr)

        # Clean up gradients
        self.model.zero_grad()
        if hasattr(data, 'sv_features') and data.sv_features:
            for sv_f in data.sv_features:
                if sv_f.grad is not None:
                    sv_f.grad = None
                sv_f.requires_grad_(False)

        return {
            "sv_attention_faithfulness": float(np.mean(correlations)) if correlations else 0.0,
            "n_nodes_evaluated": len(correlations),
        }

    # ══════════════════════════════════════════════════════════════════
    # 5. COMPLEXITY
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def complexity(sv_attentions, structural_feats, edge_idx,
                   src_node, dst_node, top_k_svs=3, sf_threshold=0.1):
        """Complexity: total number of distinct elements in the explanation.

        Lower = simpler, more human-interpretable explanation.
        """
        count = 0

        # Count top-k SVs per endpoint
        for node in [src_node, dst_node]:
            if sv_attentions and node < len(sv_attentions):
                count += min(top_k_svs, len(sv_attentions[node]))

        # Count active structural features
        sf = structural_feats[edge_idx]
        count += (sf.abs() > sf_threshold).sum().item()

        # Tissue pair (always 1 element)
        count += 1

        # Edge type (intra/inter, always 1 element)
        count += 1

        return {"complexity": count}

    # ══════════════════════════════════════════════════════════════════
    # BATCH EVALUATION
    # ══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def evaluate_graph(self, data, top_k_svs=3):
        """Run all metrics on one graph. Returns per-edge metric dicts."""
        self.model.eval()
        data = data.to(self.device)

        # Encode
        z, alphas, sv_attns = self.model.encode(data, return_attention=True)

        # Compute structural features
        sf, cn = self.sf_computer.compute(
            data.edge_index, data.x.size(0), data.edge_index,
            node_embeddings=z, tissue_labels=data.tissue_labels,
            slice_ids=data.slice_ids,
        )
        sf = sf.to(self.device)
        src_tissue, dst_tissue, is_inter = get_edge_metadata(data, data.edge_index)

        n_edges = data.edge_index.size(1)
        edge_metrics = []

        for idx in range(n_edges):
            src_node = data.edge_index[0, idx].item()
            dst_node = data.edge_index[1, idx].item()

            # Fidelity
            fid_nec = self.fidelity_necessity(
                data, z, sv_attns, idx, sf, cn, src_tissue, dst_tissue, is_inter,
                top_k_svs=top_k_svs,
            )
            fid_suf = self.fidelity_sufficiency(
                data, z, sv_attns, idx, sf, cn, src_tissue, dst_tissue, is_inter,
                top_k_svs=top_k_svs,
            )

            # Sparsity
            spar = self.sparsity(sv_attns, sf, idx, src_node, dst_node, top_k_svs=top_k_svs)

            # Complexity
            comp = self.complexity(sv_attns, sf, idx, src_node, dst_node, top_k_svs=top_k_svs)

            # Collect top-k SV indices for stability comparison
            top_sv_indices = []
            for node in [src_node, dst_node]:
                if sv_attns and node < len(sv_attns) and len(sv_attns[node]) > 0:
                    k = min(top_k_svs, len(sv_attns[node]))
                    top_sv_indices.extend(
                        torch.topk(sv_attns[node], k).indices.tolist()
                    )

            sf_ranking = sf[idx].abs().tolist()

            edge_metrics.append({
                "edge_idx": idx,
                "src_node": src_node,
                "dst_node": dst_node,
                **fid_nec,
                **fid_suf,
                **spar,
                **comp,
                "top_sv_indices": top_sv_indices,
                "sf_ranking": sf_ranking,
            })

        return edge_metrics

    @staticmethod
    def aggregate_metrics(all_edge_metrics):
        """Aggregate per-edge metrics into summary statistics."""
        keys = [
            "level1_drop", "level2_drop", "combined_drop",
            "level1_retention", "level2_retention",
            "sv_sparsity", "sf_sparsity",
            "complexity",
        ]
        summary = {}
        for key in keys:
            values = [m[key] for m in all_edge_metrics if key in m]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))

        return summary

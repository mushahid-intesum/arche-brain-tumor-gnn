"""
Post-hoc explainability baselines for comparison against intrinsic explanations.

Methods:
  1. GNNExplainer (Ying et al., NeurIPS 2019): learns edge/feature importance masks
  2. Grad-CAM for GNN: gradient-weighted activation maps per node
  3. Attention-only: raw GATv2 attention weights (no structural/SV context)
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from gnn import StructuralFeatureComputer, get_edge_metadata


class GNNExplainerWrapper:
    """Wraps a trained EdgePredictor to produce edge importance masks.

    Learns a soft mask over edges that maximizes the mutual information
    between the masked subgraph and the model's prediction for a target edge.
    """

    def __init__(self, model, device=None, epochs=100, lr=0.01):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.epochs = epochs
        self.lr = lr
        self.sf_computer = StructuralFeatureComputer()

    def explain_edge(self, data, target_edge_idx, z=None, structural_feats=None,
                     cn_indices=None, src_tissue=None, dst_tissue=None, is_inter=None):
        """Generate an edge importance mask for a target edge prediction.

        Returns:
            edge_mask: (E,) tensor of importance scores in [0, 1]
            node_feat_mask: (F,) tensor of feature importance scores
        """
        self.model.eval()
        data = data.to(self.device)

        n_edges = data.edge_index.size(1)
        n_feats = data.x.size(1)

        # Learnable masks
        edge_mask_logits = torch.nn.Parameter(torch.randn(n_edges, device=self.device) * 0.1)
        feat_mask_logits = torch.nn.Parameter(torch.randn(n_feats, device=self.device) * 0.1)

        optimizer = torch.optim.Adam([edge_mask_logits, feat_mask_logits], lr=self.lr)

        # Get original prediction
        with torch.no_grad():
            if z is None:
                z, _, _ = self.model.encode(data, return_attention=False)

            ei = data.edge_index[:, target_edge_idx:target_edge_idx+1]
            sf = structural_feats[target_edge_idx:target_edge_idx+1]
            cn = [cn_indices[target_edge_idx]]
            st = src_tissue[target_edge_idx:target_edge_idx+1]
            dt = dst_tissue[target_edge_idx:target_edge_idx+1]
            inter = is_inter[target_edge_idx:target_edge_idx+1]
            original_logit = self.model.decode(z, ei, sf, cn, st, dt, inter).item()

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            edge_mask = torch.sigmoid(edge_mask_logits)
            feat_mask = torch.sigmoid(feat_mask_logits)

            # Mask the edge attributes
            masked_edge_attr = None
            if data.edge_attr is not None and data.edge_attr.size(0) > 0:
                masked_edge_attr = data.edge_attr * edge_mask.unsqueeze(-1)

            # Re-encode with masked features
            # Apply feature mask to SV features before aggregation
            masked_sv_features = []
            for sv_f in data.sv_features:
                if sv_f.size(0) > 0:
                    # Apply feature mask (broadcast across SVs)
                    masked_sv_features.append(sv_f * feat_mask[:sv_f.size(1)])
                else:
                    masked_sv_features.append(sv_f)

            node_embeds, _ = self.model.aggregator(masked_sv_features, device=self.device)

            sf_comp = StructuralFeatureComputer()
            sv_ei_list = data.sv_edge_indices if hasattr(data, 'sv_edge_indices') else []
            n_svs = data.n_svs_per_node if hasattr(data, 'n_svs_per_node') else []
            if sv_ei_list and n_svs:
                topo_feats = sf_comp.compute_intra_node_topology(sv_ei_list, n_svs).to(self.device)
            else:
                topo_feats = torch.zeros(node_embeds.size(0), 4, device=self.device)

            x_masked = torch.cat([node_embeds, topo_feats], dim=-1)
            z_masked, _ = self.model.encoder(x_masked, data.edge_index,
                                              edge_attr=masked_edge_attr)

            # Predict with masked embeddings
            masked_logit = self.model.decode(z_masked, ei, sf, cn, st, dt, inter)

            # Loss: prediction should stay close + masks should be sparse
            pred_loss = F.binary_cross_entropy_with_logits(
                masked_logit, torch.sigmoid(torch.tensor([original_logit], device=self.device))
            )
            edge_sparsity = edge_mask.mean()
            feat_sparsity = feat_mask.mean()
            entropy_loss = -edge_mask * torch.log(edge_mask + 1e-8) - (1 - edge_mask) * torch.log(1 - edge_mask + 1e-8)

            loss = pred_loss + 0.5 * edge_sparsity + 0.5 * feat_sparsity + 0.1 * entropy_loss.mean()
            loss.backward()
            optimizer.step()

        return {
            "edge_mask": torch.sigmoid(edge_mask_logits).detach().cpu(),
            "feat_mask": torch.sigmoid(feat_mask_logits).detach().cpu(),
        }

    def to_standard_explanation(self, explanation, data, sv_attentions, structural_feats, edge_idx, top_k_svs=3):
        """Convert GNNExplainer output to the standard format for metric comparison."""
        src_node = data.edge_index[0, edge_idx].item()
        dst_node = data.edge_index[1, edge_idx].item()

        # Use edge mask as proxy for SV importance (via connected edges)
        edge_mask = explanation["edge_mask"]

        top_sv_indices = []
        for node in [src_node, dst_node]:
            # Edges connected to this node
            connected = (data.edge_index[0] == node) | (data.edge_index[1] == node)
            if connected.any():
                node_importance = edge_mask[connected].mean().item()
            else:
                node_importance = 0.0

            if sv_attentions and node < len(sv_attentions) and len(sv_attentions[node]) > 0:
                k = min(top_k_svs, len(sv_attentions[node]))
                # Weight SV attention by node importance
                weighted_attn = sv_attentions[node] * node_importance
                top_sv_indices.extend(torch.topk(weighted_attn, k).indices.tolist())

        sf_ranking = explanation["feat_mask"][:structural_feats.size(1)].abs().tolist() if structural_feats.size(1) <= len(explanation["feat_mask"]) else structural_feats[edge_idx].abs().tolist()

        return {
            "top_sv_indices": top_sv_indices,
            "sf_ranking": sf_ranking,
        }


class GradCAMExplainer:
    """Gradient-weighted activation mapping for GNN nodes.

    Computes gradients of the prediction logit w.r.t. intermediate GATv2
    layer activations, weights by global average, produces per-node
    importance scores.
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.sf_computer = StructuralFeatureComputer()

    def explain_edge(self, data, target_edge_idx, structural_feats,
                     cn_indices, src_tissue, dst_tissue, is_inter):
        """Generate node importance scores via Grad-CAM for a target edge."""
        self.model.eval()
        data = data.to(self.device)

        # Hook to capture intermediate activations
        activations = {}
        gradients = {}

        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                activations["value"] = output[0]
            else:
                activations["value"] = output

        def backward_hook(module, grad_input, grad_output):
            gradients["value"] = grad_output[0]

        # Register hooks on the last GATv2 layer
        last_conv = self.model.encoder.convs[-1]
        fwd_handle = last_conv.register_forward_hook(forward_hook)
        bwd_handle = last_conv.register_full_backward_hook(backward_hook)

        try:
            # Forward
            z, _, sv_attns = self.model.encode(data, return_attention=False)

            ei = data.edge_index[:, target_edge_idx:target_edge_idx+1]
            sf = structural_feats[target_edge_idx:target_edge_idx+1]
            cn = [cn_indices[target_edge_idx]]
            st = src_tissue[target_edge_idx:target_edge_idx+1]
            dt = dst_tissue[target_edge_idx:target_edge_idx+1]
            inter_flag = is_inter[target_edge_idx:target_edge_idx+1]

            logit = self.model.decode(z, ei, sf, cn, st, dt, inter_flag)
            logit.backward()

            # Grad-CAM: global average of gradients * activations
            if "value" in activations and "value" in gradients:
                act = activations["value"]      # (N, hidden_dim)
                grad = gradients["value"]       # (N, hidden_dim)

                # Per-node importance
                weights = grad.mean(dim=-1, keepdim=True)  # (N, 1)
                cam = (weights * act).sum(dim=-1)           # (N,)
                cam = F.relu(cam)                           # only positive contributions
                cam = cam / (cam.max() + 1e-8)              # normalize to [0, 1]
            else:
                cam = torch.zeros(data.x.size(0), device=self.device)

        finally:
            fwd_handle.remove()
            bwd_handle.remove()
            self.model.zero_grad()

        return {
            "node_importance": cam.detach().cpu(),
            "sv_attentions": sv_attns,
        }

    def to_standard_explanation(self, explanation, data, structural_feats, edge_idx, top_k_svs=3):
        """Convert Grad-CAM output to standard format."""
        src_node = data.edge_index[0, edge_idx].item()
        dst_node = data.edge_index[1, edge_idx].item()
        sv_attns = explanation["sv_attentions"]
        node_imp = explanation["node_importance"]

        top_sv_indices = []
        for node in [src_node, dst_node]:
            if sv_attns and node < len(sv_attns) and len(sv_attns[node]) > 0:
                k = min(top_k_svs, len(sv_attns[node]))
                # Weight SV attention by node Grad-CAM importance
                weight = node_imp[node].item() if node < len(node_imp) else 1.0
                weighted_attn = sv_attns[node] * weight
                top_sv_indices.extend(torch.topk(weighted_attn, k).indices.tolist())

        sf_ranking = structural_feats[edge_idx].abs().tolist()

        return {
            "top_sv_indices": top_sv_indices,
            "sf_ranking": sf_ranking,
        }


class AttentionOnlyExplainer:
    """Baseline: use raw GATv2 attention weights as the explanation.

    No structural context, no SV-level attribution. Just the attention
    weights from the encoder layers.
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device

    def explain_edge(self, data, target_edge_idx):
        """Extract attention weights for a target edge."""
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            z, alphas, _ = self.model.encode(data, return_attention=True)

        src_node = data.edge_index[0, target_edge_idx].item()
        dst_node = data.edge_index[1, target_edge_idx].item()

        # Collect attention weights for the target edge across layers
        edge_attentions = []
        for layer_alpha in alphas:
            if layer_alpha is not None:
                # alpha shape: (E, heads) -- find attention for our target edge
                edge_attentions.append(layer_alpha[target_edge_idx].detach().cpu())

        return {
            "edge_attentions": edge_attentions,
            "src_node": src_node,
            "dst_node": dst_node,
        }

    def to_standard_explanation(self, explanation, data, sv_attentions,
                                structural_feats, edge_idx, top_k_svs=3):
        """Convert attention-only output to standard format."""
        src_node = explanation["src_node"]
        dst_node = explanation["dst_node"]

        # Use uniform SV importance (attention-only has no SV-level info)
        top_sv_indices = []
        for node in [src_node, dst_node]:
            if sv_attentions and node < len(sv_attentions) and len(sv_attentions[node]) > 0:
                k = min(top_k_svs, len(sv_attentions[node]))
                # Random top-k since we have no SV-level signal
                top_sv_indices.extend(list(range(k)))

        # Use mean attention across layers as proxy for structural importance
        edge_attn = explanation.get("edge_attentions", [])
        if edge_attn:
            mean_attn = torch.stack(edge_attn).mean().item()
            sf_ranking = [mean_attn] * structural_feats.size(1)
        else:
            sf_ranking = structural_feats[edge_idx].abs().tolist()

        return {
            "top_sv_indices": top_sv_indices,
            "sf_ranking": sf_ranking,
        }

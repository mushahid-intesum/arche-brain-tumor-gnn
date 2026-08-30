"""
Ablation Model Wrappers

Wraps gnn.EdgePredictor and gnn.StructuralFeatureComputer with
ablation-aware conditional logic. gnn.py is not modified.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Ensure project root is on path so we can import gnn
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import SHARED, GNN, SUPERVOXEL
from gnn import (
    EdgePredictor,
    GATv2Encoder,
    MultiSignalDecoder,
    IntraNodeAggregator,
    StructuralFeatureComputer,
)
from ablation.config import AblationConfig


class AblationStructuralComputer:
    """Wraps StructuralFeatureComputer with conditional OCN computation.

    When use_ocn=False, the OCN residual (dim 3) and path-normalized CN
    (dim 4) are set to zero. The output is still 5-dim to keep the
    decoder architecture identical across all ablation configs.
    """

    def __init__(self, use_ocn=True):
        self.use_ocn = use_ocn
        self._inner = StructuralFeatureComputer()

    def compute(self, edge_index, num_nodes, candidate_edges,
                node_embeddings=None, tissue_labels=None, slice_ids=None):
        """Compute structural features, optionally zeroing OCN dims."""
        if self.use_ocn:
            # Full OCN computation
            return self._inner.compute(
                edge_index, num_nodes, candidate_edges,
                node_embeddings=node_embeddings,
                tissue_labels=tissue_labels,
                slice_ids=slice_ids,
            )
        else:
            # Compute without OCN: pass node_embeddings=None so OCN residual is 0
            feats, cn_list = self._inner.compute(
                edge_index, num_nodes, candidate_edges,
                node_embeddings=None,  # disables OCN residual
                tissue_labels=tissue_labels,
                slice_ids=slice_ids,
            )
            # Also zero out path-normalized CN (dim 4) for clean ablation
            if feats.size(1) >= 5:
                feats[:, 3] = 0.0  # OCN residual
                feats[:, 4] = 0.0  # path-normalized CN
            return feats, cn_list

    def compute_intra_node_topology(self, sv_edge_indices_list, n_svs_per_node):
        """Delegate to inner computer (topology is controlled separately)."""
        return self._inner.compute_intra_node_topology(
            sv_edge_indices_list, n_svs_per_node,
        )


class AblationModel(nn.Module):
    """EdgePredictor wrapper with ablation-controlled encode path.

    Controls three independent switches:
      1. use_sv_aggregation: Transformer aggregation vs flat features
      2. use_intra_topology: 4-dim topology features vs zeros
      3. use_ocn_features: controlled via AblationStructuralComputer (external)

    The model architecture (encoder dims, decoder dims) is identical
    across all configs. Only the feature computation path changes.
    """

    def __init__(self, ablation_config=None, gnn_config=None):
        super().__init__()
        self.abl = ablation_config or AblationConfig()
        config = gnn_config or GNN

        # Always build all submodules so param count is comparable.
        # The aggregator is simply not used when use_sv_aggregation=False.
        self.aggregator = IntraNodeAggregator(
            sv_feat_dim=SUPERVOXEL["sv_feat_dim"],
            embed_dim=config["embed_dim"],
        )
        self.encoder = GATv2Encoder(
            in_dim=config["node_feat_dim"],  # always 68
            hidden_dim=config["hidden_dim"],
            out_dim=config["embed_dim"],
            num_layers=config["num_layers"],
            heads=config["num_heads"],
            edge_dim=config["edge_attr_dim"],
        )
        self.decoder = MultiSignalDecoder(
            embed_dim=config["embed_dim"],
            structural_feat_dim=config["structural_feat_dim"],
        )

        self._node_feat_dim = config["node_feat_dim"]  # 68
        self._embed_dim = config["embed_dim"]           # 64

    def encode(self, data, return_attention=False):
        """Encode node features, respecting ablation switches."""
        edge_attr = (
            data.edge_attr
            if hasattr(data, 'edge_attr') and data.edge_attr is not None
            and data.edge_attr.size(0) > 0
            else None
        )
        device = data.x.device

        # ── SV Aggregation switch ──
        if (self.abl.use_sv_aggregation
                and hasattr(data, 'sv_features')
                and len(data.sv_features) > 0):
            # Hierarchical path: Transformer aggregation
            node_embeds, sv_attns = self.aggregator(data.sv_features, device=device)
        else:
            # Flat path: use data.x directly, pad to embed_dim (64)
            node_embeds = data.x
            if node_embeds.size(1) < self._embed_dim:
                pad = torch.zeros(
                    node_embeds.size(0),
                    self._embed_dim - node_embeds.size(1),
                    device=device,
                )
                node_embeds = torch.cat([node_embeds, pad], dim=-1)
            elif node_embeds.size(1) > self._embed_dim:
                # Truncate if somehow wider (shouldn't happen)
                node_embeds = node_embeds[:, :self._embed_dim]
            sv_attns = []

        # ── Topology switch ──
        if (self.abl.use_intra_topology
                and hasattr(data, 'sv_edge_indices')
                and data.sv_edge_indices
                and hasattr(data, 'n_svs_per_node')
                and data.n_svs_per_node):
            sf_computer = StructuralFeatureComputer()
            topo_feats = sf_computer.compute_intra_node_topology(
                data.sv_edge_indices, data.n_svs_per_node,
            ).to(device)
        else:
            # Zero topology
            topo_feats = torch.zeros(node_embeds.size(0), 4, device=device)

        # Concatenate: (N, 64) + (N, 4) = (N, 68)
        x = torch.cat([node_embeds, topo_feats], dim=-1)

        # Pad or truncate to node_feat_dim if needed
        if x.size(1) < self._node_feat_dim:
            pad = torch.zeros(x.size(0), self._node_feat_dim - x.size(1), device=device)
            x = torch.cat([x, pad], dim=-1)

        z, alphas = self.encoder(
            x, data.edge_index, edge_attr=edge_attr,
            return_attention=return_attention,
        )
        return z, alphas, sv_attns

    def decode(self, z, edge_index, structural_feats, cn_indices_list,
               src_tissue, dst_tissue, is_inter_slice):
        """Decode edge predictions (identical across all configs)."""
        return self.decoder(
            z, edge_index, structural_feats, cn_indices_list,
            src_tissue, dst_tissue, is_inter_slice,
        )

    def forward(self, data, pos_ei, neg_ei, pos_sf, neg_sf, pos_cn, neg_cn,
                pos_src_t, pos_dst_t, neg_src_t, neg_dst_t,
                pos_inter, neg_inter, return_attention=False):
        """Full forward pass (train mode)."""
        z, alphas, sv_attns = self.encode(data, return_attention=return_attention)
        pos_pred = self.decode(z, pos_ei, pos_sf, pos_cn, pos_src_t, pos_dst_t, pos_inter)
        neg_pred = self.decode(z, neg_ei, neg_sf, neg_cn, neg_src_t, neg_dst_t, neg_inter)
        return pos_pred, neg_pred, z, alphas

    def describe(self):
        """Return a human-readable description of active components."""
        parts = []
        if self.abl.use_sv_aggregation:
            parts.append("SV-Aggregation")
        if self.abl.use_intra_topology:
            parts.append("Topology")
        if self.abl.use_ocn_features:
            parts.append("OCN")
        if not parts:
            parts.append("Baseline (flat features only)")
        total_params = sum(p.numel() for p in self.parameters())
        return f"{self.abl.name}: {' + '.join(parts)} | {total_params:,} params"

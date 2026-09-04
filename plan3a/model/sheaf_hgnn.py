"""
Sheaf Hypergraph Neural Network (SHGNN) for Plan 3a.

Implements the sheaf Laplacian-based message passing from MRePath (IJCAI-25),
adapted for brain MRI patch hypergraphs.

Architecture:
  1. PatchEncoder: Projects flattened multi-modal patches (N, 1536) → (N, d)
  2. SheafHGNNLayer: Sheaf Laplacian message passing on hypergraph
  3. Multi-layer SHGNN: Stack L layers with residuals

The sheaf Laplacian replaces the standard hypergraph Laplacian with
learned linear maps F_{v⊥e} that control information flow between
vertices and hyperedges, enabling differentiated message passing.

Reference: MRePath Eq. 4-6; Duta et al. 2024 "Sheaf Hypergraph Networks"
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import (
    EMBED_DIM, SHEAF_HGNN_LAYERS, SHEAF_HGNN_DIM,
    PATCH_ENCODER_CHANNELS, NUM_CONCEPTS,
)


class PatchEncoder(nn.Module):
    """
    Encode flattened multi-modal MRI patches into dense embeddings.

    Input: (N, 6 * 16 * 16) = (N, 1536) flattened patches
    Output: (N, embed_dim)

    Uses a simple MLP (not a heavy CNN) since patches are small (16×16)
    and we want the model to be lightweight for RTX 3060.
    """

    def __init__(self, in_dim: int = 1536, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1536) → (N, embed_dim)"""
        return self.encoder(x)


class SheafHGNNLayer(nn.Module):
    """
    Single layer of Sheaf Hypergraph Neural Network.

    Implements the sheaf Laplacian-based message passing:
      1. Vertex → Hyperedge aggregation: average node features within each hyperedge
      2. Apply learned sheaf maps F_{v⊥e} for structured information flow
      3. Hyperedge → Vertex propagation: weighted sum back to nodes
      4. Apply sheaf Laplacian normalization

    This replaces standard HGNN's isotropic aggregation with
    anisotropic (direction-aware) message passing.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Sheaf maps: learned projections for vertex-to-hyperedge flow
        # F_{v⊥e}: transforms vertex features before aggregation
        self.vertex_to_edge = nn.Linear(in_dim, out_dim, bias=False)
        # F_{e⊥v}: transforms hyperedge features before propagation back
        self.edge_to_vertex = nn.Linear(out_dim, out_dim, bias=False)

        # Learnable weight matrix Θ for the message passing
        self.weight = nn.Linear(out_dim, out_dim)

        # Normalization and activation
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            hyperedge_index: (2, E_connections) — [node_idx; hyperedge_idx]
            num_nodes: N
            num_edges: total number of hyperedges

        Returns:
            x_out: (N, out_dim) updated node features
        """
        if hyperedge_index.shape[1] == 0:
            # No hyperedges — just project
            return self.activation(self.norm(self.weight(self.vertex_to_edge(x))))

        node_idx = hyperedge_index[0]  # which nodes
        edge_idx = hyperedge_index[1]  # which hyperedges

        # ── Step 1: Vertex → Hyperedge (with sheaf map) ──────────────
        # Apply sheaf map F_{v⊥e} to transform node features
        x_transformed = self.vertex_to_edge(x)  # (N, out_dim)

        # Aggregate: average node features within each hyperedge
        # scatter_mean equivalent using index_add
        edge_features = torch.zeros(num_edges, self.out_dim, device=x.device)
        edge_counts = torch.zeros(num_edges, 1, device=x.device)

        edge_features.index_add_(0, edge_idx, x_transformed[node_idx])
        edge_counts.index_add_(0, edge_idx, torch.ones(node_idx.shape[0], 1, device=x.device))
        edge_counts = edge_counts.clamp(min=1)
        edge_features = edge_features / edge_counts  # (E, out_dim)

        # ── Step 2: Hyperedge → Vertex (with sheaf map) ──────────────
        # Apply sheaf map F_{e⊥v} to transform hyperedge features
        edge_transformed = self.edge_to_vertex(edge_features)  # (E, out_dim)

        # Propagate back: each node receives aggregated info from its hyperedges
        node_updates = torch.zeros(num_nodes, self.out_dim, device=x.device)
        node_degree = torch.zeros(num_nodes, 1, device=x.device)

        node_updates.index_add_(0, node_idx, edge_transformed[edge_idx])
        node_degree.index_add_(0, node_idx, torch.ones(node_idx.shape[0], 1, device=x.device))
        node_degree = node_degree.clamp(min=1)

        # ── Step 3: Sheaf Laplacian normalization ────────────────────
        # Δ_F = I - D_v^{-1/2} L_F D_v^{-1/2}
        # Simplified: normalize by degree
        inv_sqrt_degree = 1.0 / torch.sqrt(node_degree)
        node_updates = node_updates * inv_sqrt_degree

        # ── Step 4: Apply weight, normalize, activate ────────────────
        x_out = self.weight(node_updates)
        x_out = self.norm(x_out)
        x_out = self.activation(x_out)
        x_out = self.dropout(x_out)

        return x_out


class SheafHGNN(nn.Module):
    """
    Multi-layer Sheaf Hypergraph Neural Network.

    Stacks L SheafHGNNLayers with residual connections and
    multi-scale fusion (concatenate outputs from all layers).

    Architecture:
        PatchEncoder → SheafHGNNLayer ×L → LayerNorm → output

    Output dim = embed_dim (after multi-scale projection)
    """

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = SHEAF_HGNN_DIM,
        num_layers: int = SHEAF_HGNN_LAYERS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Patch encoder: raw patches → initial embeddings
        self.patch_encoder = PatchEncoder(in_dim, embed_dim)

        # SHGNN layers
        self.layers = nn.ModuleList([
            SheafHGNNLayer(embed_dim, embed_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Multi-scale fusion: concatenate all layer outputs → project back
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * (num_layers + 1), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (N, 1536) flattened patch features
            hyperedge_index: (2, E_conn) combined incidence matrix
            num_nodes: N
            num_edges: total hyperedges

        Returns:
            x: (N, embed_dim) refined node embeddings
        """
        # Initial encoding
        x = self.patch_encoder(node_features)  # (N, embed_dim)

        # Collect multi-scale representations
        layer_outputs = [x]

        for layer in self.layers:
            x_new = layer(x, hyperedge_index, num_nodes, num_edges)
            # Residual connection
            x = x + x_new
            layer_outputs.append(x)

        # Multi-scale fusion: concat all layer outputs
        x_multi = torch.cat(layer_outputs, dim=-1)  # (N, embed_dim * (L+1))
        x = self.fusion(x_multi)  # (N, embed_dim)

        return x


class GraphPooling(nn.Module):
    """
    Pool node-level embeddings to a single graph-level embedding.

    Uses attention-weighted mean pooling (learnable).
    """

    def __init__(self, embed_dim: int = SHEAF_HGNN_DIM):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, embed_dim) → (1, embed_dim) graph-level embedding
        """
        attn_scores = self.attention(x)  # (N, 1)
        attn_weights = F.softmax(attn_scores, dim=0)  # (N, 1)
        graph_embed = (attn_weights * x).sum(dim=0, keepdim=True)  # (1, embed_dim)
        return graph_embed

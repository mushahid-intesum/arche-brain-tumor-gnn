"""
Multi-Granular Tree Module for Plan 3a (TIF-inspired).

Implements hierarchical graph coarsening to produce explanations
at multiple granularity levels, from individual patches up to
whole-tumor-component level.

Architecture (from TIF — Tree-like Interpretable Framework):
  1. GraphCoarsener: Progressively pools nodes into clusters using
     soft assignment (DiffPool-style), creating L coarsened levels.
  2. LevelEncoder: Runs SHGNN-like message passing at each level.
  3. AdaptiveRouter: Selects the most informative root-to-leaf path
     through the tree, producing multi-granular explanations.

Tree structure:
  Level 0 (finest):  ~N nodes   — individual patch level
  Level 1:           ~N/4 nodes — local tissue neighborhood
  Level 2:           ~N/16 nodes — tissue region level
  Level 3 (coarsest): ~N/64 nodes — tumor component level

Each level provides a different granularity of explanation:
  L0: "Patches 342 and 387 had high enhancement (c1)"
  L1: "The peritumoral edema zone shows elevated FLAIR (c2)"
  L2: "The enhancing rim is structurally distinct from core"
  L3: "Overall tumor has high perfusion and heterogeneity"
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import SHEAF_HGNN_DIM, NUM_CONCEPTS


class SoftAssignmentPool(nn.Module):
    """
    Differentiable graph coarsening via soft cluster assignment.

    Learns a soft assignment matrix S ∈ ℝ^{N×K} that maps N nodes
    to K clusters. The coarsened graph has K nodes with:
      - Features: X' = S^T X  (aggregated features)
      - Adjacency reconstructed from cluster assignments

    Based on DiffPool (Ying et al. 2018) adapted for hypergraphs.
    """

    def __init__(self, in_dim: int, ratio: float = 0.25):
        """
        Args:
            in_dim: input node feature dimension
            ratio: coarsening ratio (0.25 = keep 25% of nodes)
        """
        super().__init__()
        self.ratio = ratio

        # Assignment network: predicts cluster membership
        self.assign_net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),  # output dim set dynamically
        )

        # Feature transform for coarsened nodes
        self.feat_transform = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_edges: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (N, D) node features
            hyperedge_index: (2, E_conn) hypergraph incidence
            num_edges: number of hyperedges

        Returns:
            dict with coarsened graph data
        """
        N, D = x.shape
        K = max(2, int(N * self.ratio))  # target number of clusters

        # Compute soft assignment: S ∈ (N, K)
        # Project to K dimensions dynamically
        assign_logits = self.assign_net(x)  # (N, D)
        # Reduce to K clusters via linear projection
        if D != K:
            proj = nn.Linear(D, K, bias=False).to(x.device)
            nn.init.xavier_uniform_(proj.weight)
            assign_logits = proj(assign_logits)
        S = F.softmax(assign_logits, dim=-1)  # (N, K) soft assignment

        # Coarsened features: X' = S^T @ X
        x_coarse = torch.mm(S.t(), x)  # (K, D)
        x_coarse = self.feat_transform(x_coarse)

        # Build coarsened hyperedge incidence
        # Map original hyperedges through assignment
        coarse_he, n_coarse_edges = self._coarsen_hyperedges(
            hyperedge_index, S, K, num_edges
        )

        # Cluster membership: hard assignment for interpretability
        cluster_assign = S.argmax(dim=-1)  # (N,) — which cluster each node belongs to

        return {
            "x_coarse": x_coarse,           # (K, D) coarsened features
            "hyperedge_index": coarse_he,    # (2, E') coarsened incidence
            "num_nodes": K,
            "num_edges": n_coarse_edges,
            "assignment": S,                 # (N, K) soft assignment
            "cluster_assign": cluster_assign,  # (N,) hard assignment
            "assignment_entropy": self._assignment_entropy(S),
        }

    def _coarsen_hyperedges(
        self,
        hyperedge_index: torch.Tensor,
        S: torch.Tensor,
        K: int,
        num_edges: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Map hyperedges from fine to coarse graph.

        For each original hyperedge, find the dominant clusters
        of its member nodes and create a coarsened hyperedge.
        """
        if hyperedge_index.shape[1] == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=S.device), 0

        node_idx = hyperedge_index[0]
        edge_idx = hyperedge_index[1]

        # Map nodes to their dominant cluster
        cluster_assign = S.argmax(dim=-1)  # (N,)
        coarse_nodes = cluster_assign[node_idx]  # mapped node indices

        # Build coarsened incidence (may have duplicates)
        coarse_he = torch.stack([coarse_nodes, edge_idx])

        # Deduplicate: remove duplicate (node, edge) pairs
        combined = coarse_he[0] * (num_edges + 1) + coarse_he[1]
        unique_combined, inverse = torch.unique(combined, return_inverse=True)
        coarse_nodes_dedup = unique_combined // (num_edges + 1)
        coarse_edges_dedup = unique_combined % (num_edges + 1)

        # Reindex edges to be contiguous
        unique_edges = torch.unique(coarse_edges_dedup)
        edge_map = torch.zeros(num_edges + 1, dtype=torch.long, device=S.device)
        edge_map[unique_edges] = torch.arange(len(unique_edges), device=S.device)
        coarse_edges_reindexed = edge_map[coarse_edges_dedup]

        return (
            torch.stack([coarse_nodes_dedup, coarse_edges_reindexed]),
            len(unique_edges),
        )

    def _assignment_entropy(self, S: torch.Tensor) -> torch.Tensor:
        """Compute mean entropy of assignment — lower = sharper clusters."""
        eps = 1e-8
        entropy = -(S * torch.log(S + eps)).sum(dim=-1).mean()
        return entropy


class LevelEncoder(nn.Module):
    """
    Lightweight message passing encoder for a single coarsened level.

    Uses a simplified version of SHGNN (single layer) since coarsened
    graphs are much smaller than the original.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.v2e = nn.Linear(dim, dim, bias=False)
        self.e2v = nn.Linear(dim, dim, bias=False)
        self.weight = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> torch.Tensor:
        """Single-layer sheaf message passing."""
        if hyperedge_index.shape[1] == 0 or num_edges == 0:
            return self.act(self.norm(self.weight(x)))

        node_idx = hyperedge_index[0]
        edge_idx = hyperedge_index[1]
        D = x.shape[-1]

        # V → E
        x_t = self.v2e(x)
        edge_feat = torch.zeros(num_edges, D, device=x.device)
        edge_count = torch.zeros(num_edges, 1, device=x.device)
        edge_feat.index_add_(0, edge_idx, x_t[node_idx])
        edge_count.index_add_(0, edge_idx, torch.ones(node_idx.shape[0], 1, device=x.device))
        edge_feat = edge_feat / edge_count.clamp(min=1)

        # E → V
        edge_out = self.e2v(edge_feat)
        node_upd = torch.zeros(num_nodes, D, device=x.device)
        node_deg = torch.zeros(num_nodes, 1, device=x.device)
        node_upd.index_add_(0, node_idx, edge_out[edge_idx])
        node_deg.index_add_(0, node_idx, torch.ones(node_idx.shape[0], 1, device=x.device))
        node_upd = node_upd / node_deg.clamp(min=1).sqrt()

        return self.act(self.norm(x + self.weight(node_upd)))


class AdaptiveRouter(nn.Module):
    """
    Adaptive routing module for path selection through the tree.

    At each level, decides whether to:
      a) Use this level's representation for the final prediction
      b) Continue to a finer/coarser level

    The routing produces a weighted combination of level representations,
    where the weights indicate which granularity is most informative.
    This is interpretable: "the model found level 2 (region-level)
    most useful for this patient's survival prediction."
    """

    def __init__(self, dim: int, num_levels: int):
        super().__init__()
        self.num_levels = num_levels

        # Per-level scoring networks
        self.level_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.Tanh(),
                nn.Linear(dim // 2, 1),
            )
            for _ in range(num_levels)
        ])

    def forward(self, level_embeds: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            level_embeds: list of (1, D) graph-level embeddings, one per level

        Returns:
            dict with:
                "fused": (1, D) weighted combination of level embeddings
                "routing_weights": (num_levels,) which levels were selected
        """
        scores = []
        for i, embed in enumerate(level_embeds):
            score = self.level_scorers[i](embed)  # (1, 1)
            scores.append(score)

        scores = torch.cat(scores, dim=-1)  # (1, L)
        weights = F.softmax(scores, dim=-1)  # (1, L)

        # Weighted sum of level embeddings
        stacked = torch.stack(level_embeds, dim=1)  # (1, L, D)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # (1, D)

        return {
            "fused": fused,
            "routing_weights": weights.squeeze(0),  # (L,)
        }


class MultiGranularTree(nn.Module):
    """
    Complete Multi-Granular Tree Module.

    Combines:
      1. Progressive coarsening: L0 → L1 → L2 → L3
      2. Per-level encoding: message passing at each granularity
      3. Per-level pooling: attention-weighted graph embedding
      4. Adaptive routing: select best granularity combination

    Provides multi-scale interpretability:
      - Which granularity level was most informative?
      - What are the cluster assignments at each level?
      - What concepts dominate at each scale?
    """

    def __init__(
        self,
        dim: int = SHEAF_HGNN_DIM,
        num_levels: int = 3,
        coarsen_ratio: float = 0.25,
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels

        # Coarsening modules (one per level transition)
        self.coarseners = nn.ModuleList([
            SoftAssignmentPool(dim, ratio=coarsen_ratio)
            for _ in range(num_levels)
        ])

        # Per-level encoders
        self.level_encoders = nn.ModuleList([
            LevelEncoder(dim) for _ in range(num_levels)
        ])

        # Per-level attention pooling
        self.level_poolers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.Tanh(),
                nn.Linear(dim // 2, 1),
            )
            for _ in range(num_levels + 1)  # +1 for level 0 (original)
        ])

        # Adaptive routing
        self.router = AdaptiveRouter(dim, num_levels + 1)

    def _attention_pool(self, x: torch.Tensor, pooler: nn.Module) -> torch.Tensor:
        """Attention-weighted mean pooling: (N, D) → (1, D)."""
        scores = pooler(x)  # (N, 1)
        weights = F.softmax(scores, dim=0)  # (N, 1)
        return (weights * x).sum(dim=0, keepdim=True)  # (1, D)

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> Dict:
        """
        Build multi-granular tree and route through it.

        Args:
            x: (N, D) node embeddings (from SHGNN)
            hyperedge_index: (2, E) hypergraph incidence
            num_nodes: N
            num_edges: number of hyperedges

        Returns:
            dict with tree structure and routing results
        """
        # ── Level 0: original graph ──────────────────────────────────
        level_embeds = []
        level_data = []

        l0_pool = self._attention_pool(x, self.level_poolers[0])
        level_embeds.append(l0_pool)
        level_data.append({
            "num_nodes": num_nodes,
            "features": x,
        })

        # ── Progressive coarsening: L1, L2, L3... ───────────────────
        current_x = x
        current_he = hyperedge_index
        current_n = num_nodes
        current_e = num_edges

        for i in range(self.num_levels):
            # Coarsen
            coarse = self.coarseners[i](current_x, current_he, current_e)

            # Encode at this level
            encoded = self.level_encoders[i](
                coarse["x_coarse"],
                coarse["hyperedge_index"],
                coarse["num_nodes"],
                coarse["num_edges"],
            )

            # Pool to graph level
            pool = self._attention_pool(encoded, self.level_poolers[i + 1])
            level_embeds.append(pool)

            level_data.append({
                "num_nodes": coarse["num_nodes"],
                "features": encoded,
                "assignment": coarse["assignment"],
                "cluster_assign": coarse["cluster_assign"],
                "entropy": coarse["assignment_entropy"],
            })

            # Update for next level
            current_x = encoded
            current_he = coarse["hyperedge_index"]
            current_n = coarse["num_nodes"]
            current_e = coarse["num_edges"]

        # ── Adaptive routing ─────────────────────────────────────────
        routing = self.router(level_embeds)

        return {
            "fused_embed": routing["fused"],             # (1, D) final embedding
            "routing_weights": routing["routing_weights"],  # (L+1,) level importance
            "level_embeds": level_embeds,                # list of (1, D)
            "level_data": level_data,                    # list of level info dicts
            "num_levels": self.num_levels + 1,
        }

"""
Dual-Space Hypergraph Construction for Plan 3a.

Constructs hypergraphs from preprocessed MRI patch data using two
complementary strategies (from MRePath):

1. Topological Hyperedges (E_T): Group patches by spatial proximity
   using their (x, y, z) coordinates with a distance threshold δ.
   Captures local tissue neighborhoods.

2. Feature Hyperedges (E_F): Group patches by feature similarity
   using top-k cosine similarity on patch embeddings.
   Captures non-local structural similarity.

The union E = E_T ∪ E_F forms the complete hypergraph.

Hypergraph representation:
  - Incidence matrix H ∈ {0,1}^{N×E} where H[i,e]=1 if node i ∈ hyperedge e
  - Node features X ∈ ℝ^{N×d}
  - Stored as PyG HypergraphData objects for compatibility with torch_geometric
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import (
    TOPO_HYPEREDGE_RADIUS, FEATURE_HYPEREDGE_K,
    EMBED_DIM, PATCH_SIZE,
)


def build_topological_hyperedges(
    coords: np.ndarray,
    radius: float = TOPO_HYPEREDGE_RADIUS,
    min_size: int = 2,
    max_size: int = 12,
) -> List[List[int]]:
    """
    Build topological hyperedges by spatial proximity (δ-ball).

    Each patch becomes the center of a hyperedge containing all patches
    within Euclidean distance ≤ radius in normalized coordinate space.

    Args:
        coords: (N, 3) normalized spatial coordinates
        radius: distance threshold δ (in normalized units, so ~2.5 patch-widths)
        min_size: minimum hyperedge size (discard trivial ones)
        max_size: cap hyperedge size to limit memory

    Returns:
        List of hyperedges, each a list of node indices
    """
    # Compute pairwise distances
    # Coords are normalized to [0,1], so distance scale is relative
    # A radius of 2.5 in patch-grid units maps to:
    #   2.5 * (PATCH_SIZE / TARGET_SLICE_SIZE[0]) ≈ 2.5 * (16/192) ≈ 0.208
    # in normalized coordinates
    patch_grid_scale = PATCH_SIZE / 192.0  # TARGET_SLICE_SIZE[0]
    norm_radius = radius * patch_grid_scale

    dist_matrix = cdist(coords, coords, metric="euclidean")

    hyperedges = []
    seen = set()

    for i in range(len(coords)):
        # Find all neighbors within radius
        neighbors = np.where(dist_matrix[i] <= norm_radius)[0].tolist()

        if len(neighbors) < min_size:
            continue
        if len(neighbors) > max_size:
            # Keep closest max_size neighbors
            dists = dist_matrix[i, neighbors]
            sorted_idx = np.argsort(dists)[:max_size]
            neighbors = [neighbors[j] for j in sorted_idx]

        # Deduplicate: use frozenset as key
        key = frozenset(neighbors)
        if key not in seen:
            seen.add(key)
            hyperedges.append(sorted(neighbors))

    return hyperedges


def build_feature_hyperedges(
    features: np.ndarray,
    k: int = FEATURE_HYPEREDGE_K,
    min_size: int = 2,
) -> List[List[int]]:
    """
    Build feature-based hyperedges by top-k cosine similarity.

    For each patch, find its k most similar patches by cosine similarity
    on the feature vector (flattened patch or concept vector).

    Args:
        features: (N, D) feature vectors for similarity computation
        k: number of nearest neighbors per patch
        min_size: minimum hyperedge size

    Returns:
        List of hyperedges, each a list of node indices
    """
    N = features.shape[0]

    # Normalize features for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features_norm = features / norms

    # Compute cosine similarity matrix
    sim_matrix = features_norm @ features_norm.T

    hyperedges = []
    seen = set()

    for i in range(N):
        # Top-k most similar (excluding self)
        sims = sim_matrix[i].copy()
        sims[i] = -1  # exclude self
        top_k_idx = np.argsort(sims)[-k:]
        neighbors = [i] + top_k_idx.tolist()

        if len(neighbors) < min_size:
            continue

        key = frozenset(neighbors)
        if key not in seen:
            seen.add(key)
            hyperedges.append(sorted(neighbors))

    return hyperedges


def hyperedges_to_incidence(
    hyperedges: List[List[int]],
    num_nodes: int,
) -> torch.Tensor:
    """
    Convert list of hyperedges to a sparse incidence matrix.

    Returns:
        hyperedge_index: (2, num_connections) — [node_indices; hyperedge_indices]
        Compatible with torch_geometric's HypergraphConv format.
    """
    node_indices = []
    hedge_indices = []

    for e_idx, hedge in enumerate(hyperedges):
        for n_idx in hedge:
            node_indices.append(n_idx)
            hedge_indices.append(e_idx)

    return torch.tensor([node_indices, hedge_indices], dtype=torch.long)


def build_patient_hypergraph(
    patient_data: Dict,
    use_concepts_for_features: bool = True,
    topo_radius: float = TOPO_HYPEREDGE_RADIUS,
    feature_k: int = FEATURE_HYPEREDGE_K,
) -> Dict:
    """
    Build complete dual-space hypergraph for one patient.

    Args:
        patient_data: dict loaded from preprocessed .pt file
        use_concepts_for_features: if True, use concept vectors for feature
            hyperedges. If False, use flattened patch intensities.
        topo_radius: spatial radius for topological hyperedges
        feature_k: k for feature-based hyperedges

    Returns:
        dict with:
            "node_features": (N, D) patch features for HGNN input
            "coords": (N, 3) spatial coordinates
            "concepts": (N, 8) concept values
            "hyperedge_index_topo": (2, E_t) topological incidence
            "hyperedge_index_feat": (2, E_f) feature incidence
            "hyperedge_index": (2, E_total) combined incidence
            "hyperedge_type": (E_total,) — 0=topological, 1=feature
            "num_nodes": int
            "num_hyperedges_topo": int
            "num_hyperedges_feat": int
            "num_hyperedges": int
            + all survival/clinical data from patient_data
    """
    patches = patient_data["patches"]      # (N, 6, ps, ps)
    coords = patient_data["coords"]        # (N, 3)
    concepts = patient_data["concepts"]    # (N, 8)
    N = patches.shape[0]

    if N == 0:
        return _empty_hypergraph(patient_data)

    coords_np = coords.numpy() if isinstance(coords, torch.Tensor) else coords
    concepts_np = concepts.numpy() if isinstance(concepts, torch.Tensor) else concepts

    # ── Topological hyperedges ───────────────────────────────────────
    topo_hedges = build_topological_hyperedges(coords_np, radius=topo_radius)
    num_topo = len(topo_hedges)

    # ── Feature hyperedges ───────────────────────────────────────────
    if use_concepts_for_features:
        feat_vectors = concepts_np
    else:
        # Flatten patches: (N, 6*ps*ps)
        patches_np = patches.numpy() if isinstance(patches, torch.Tensor) else patches
        feat_vectors = patches_np.reshape(N, -1)

    feat_hedges = build_feature_hyperedges(feat_vectors, k=feature_k)
    num_feat = len(feat_hedges)

    # ── Combine: E = E_T ∪ E_F ──────────────────────────────────────
    # Offset feature hyperedge indices
    combined_hedges = topo_hedges + feat_hedges

    # Build incidence matrices
    hyperedge_index_topo = hyperedges_to_incidence(topo_hedges, N)
    hyperedge_index_feat = hyperedges_to_incidence(feat_hedges, N)

    # Combined incidence (feature hedges offset by num_topo)
    if num_feat > 0 and hyperedge_index_feat.shape[1] > 0:
        feat_offset = hyperedge_index_feat.clone()
        feat_offset[1] += num_topo
        hyperedge_index = torch.cat([hyperedge_index_topo, feat_offset], dim=1)
    else:
        hyperedge_index = hyperedge_index_topo

    # Type labels
    hyperedge_type = torch.cat([
        torch.zeros(num_topo, dtype=torch.long),
        torch.ones(num_feat, dtype=torch.long),
    ])

    # ── Node features: flatten patches as initial features ───────────
    # (N, 6, 16, 16) → (N, 6*16*16) = (N, 1536) — will be projected by encoder
    patches_t = patches if isinstance(patches, torch.Tensor) else torch.from_numpy(patches)
    node_features = patches_t.reshape(N, -1).float()

    # ── Assemble result ──────────────────────────────────────────────
    result = {
        "node_features": node_features,
        "coords": coords if isinstance(coords, torch.Tensor) else torch.from_numpy(coords),
        "concepts": concepts if isinstance(concepts, torch.Tensor) else torch.from_numpy(concepts),
        "hyperedge_index_topo": hyperedge_index_topo,
        "hyperedge_index_feat": hyperedge_index_feat,
        "hyperedge_index": hyperedge_index,
        "hyperedge_type": hyperedge_type,
        "num_nodes": N,
        "num_hyperedges_topo": num_topo,
        "num_hyperedges_feat": num_feat,
        "num_hyperedges": num_topo + num_feat,
        # Pass through labels
        "patient_id": patient_data["patient_id"],
        "clinical_features": patient_data["clinical_features"],
        "survival_time": patient_data["survival_time"],
        "event": patient_data["event"],
        "has_survival": patient_data["has_survival"],
        "modality_mask": patient_data["modality_mask"],
    }

    return result


def _empty_hypergraph(patient_data: Dict) -> Dict:
    """Return an empty hypergraph structure for patients with no valid patches."""
    return {
        "node_features": torch.zeros(0, 1536),
        "coords": torch.zeros(0, 3),
        "concepts": torch.zeros(0, 8),
        "hyperedge_index_topo": torch.zeros(2, 0, dtype=torch.long),
        "hyperedge_index_feat": torch.zeros(2, 0, dtype=torch.long),
        "hyperedge_index": torch.zeros(2, 0, dtype=torch.long),
        "hyperedge_type": torch.zeros(0, dtype=torch.long),
        "num_nodes": 0,
        "num_hyperedges_topo": 0,
        "num_hyperedges_feat": 0,
        "num_hyperedges": 0,
        "patient_id": patient_data.get("patient_id", "unknown"),
        "clinical_features": patient_data.get("clinical_features", torch.zeros(18)),
        "survival_time": patient_data.get("survival_time", torch.tensor(0.0)),
        "event": patient_data.get("event", torch.tensor(0)),
        "has_survival": False,
        "modality_mask": patient_data.get("modality_mask", {}),
    }

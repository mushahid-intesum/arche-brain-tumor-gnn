"""
Supervoxel Generation Module (SVGFormer-inspired)
3D SLIC clustering on raw MRI volumes → supervoxel features, assignment matrices,
and intra-node edge construction for the hierarchical GNN pipeline.
"""

import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from config import SHARED, SUPERVOXEL, SEGMENTATION


# ══════════════════════════════════════════════════════════════════════
# 3D SLIC Supervoxel Generation
# ══════════════════════════════════════════════════════════════════════

def generate_3d_supervoxels(t1_volume, n_segments=None, compactness=None):
    """Run 3D SLIC on the T1 volume to produce supervoxel labels.

    Args:
        t1_volume: (H, W, D) float32 array, z-score normalized T1 volume.
        n_segments: target number of supervoxels.
        compactness: SLIC compactness (lower = more intensity-driven boundaries).

    Returns:
        sv_labels: (H, W, D) int array, supervoxel ID per voxel.
    """
    from skimage.segmentation import slic

    n_segments = n_segments or SUPERVOXEL["n_segments"]
    compactness = compactness or SUPERVOXEL["compactness"]

    sv_labels = slic(
        t1_volume,
        n_segments=n_segments,
        compactness=compactness,
        channel_axis=None,      # single-channel 3D volume
        enforce_connectivity=True,
        start_label=0,
    )
    return sv_labels.astype(np.int32)


def prune_background_svs(sv_labels, t1_volume, min_volume=None):
    """Remove background supervoxels using intensity gap detection.

    SVGFormer §3: sort SV mean intensities, find the largest gap in the
    sorted sequence, discard all SVs below the gap.

    Args:
        sv_labels: (H, W, D) int array from generate_3d_supervoxels.
        t1_volume: (H, W, D) float32 array.
        min_volume: minimum voxel count per SV.

    Returns:
        valid_sv_ids: list of int, IDs of retained supervoxels.
        sv_means: dict[sv_id] → mean T1 intensity.
    """
    min_volume = min_volume or SUPERVOXEL["min_sv_volume"]
    unique_ids = np.unique(sv_labels)

    # Compute per-SV stats
    sv_means = {}
    sv_volumes = {}
    for sv_id in unique_ids:
        mask = sv_labels == sv_id
        volume = int(mask.sum())
        if volume < min_volume:
            continue
        sv_means[sv_id] = float(np.mean(t1_volume[mask]))
        sv_volumes[sv_id] = volume

    if len(sv_means) < 2:
        return list(sv_means.keys()), sv_means

    # Sort by mean intensity and find largest gap
    sorted_ids = sorted(sv_means.keys(), key=lambda x: sv_means[x])
    sorted_vals = [sv_means[sid] for sid in sorted_ids]

    gaps = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
    max_gap_idx = int(np.argmax(gaps))

    # Only prune if the gap is significant (> 1 std of all gaps)
    gap_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0
    gap_mean = float(np.mean(gaps))

    if gaps[max_gap_idx] > gap_mean + gap_std:
        # Keep SVs above the gap (higher intensity = brain tissue)
        threshold_intensity = sorted_vals[max_gap_idx]
        valid_sv_ids = [sid for sid in sv_means if sv_means[sid] > threshold_intensity]
    else:
        # No clear gap — keep all
        valid_sv_ids = list(sv_means.keys())

    return valid_sv_ids, sv_means


# ══════════════════════════════════════════════════════════════════════
# Supervoxel Feature Extraction
# ══════════════════════════════════════════════════════════════════════

def compute_sv_features(sv_labels, raw_4ch_volume, valid_sv_ids):
    """Compute handcrafted features for each supervoxel (22-dim).

    Per-modality (4 modalities × 4 stats = 16):
        mean, std, range, skewness
    Spatial (3): centroid (x, y, z) normalized
    Morphology (3): volume, surface area (approx), compactness

    Args:
        sv_labels: (H, W, D) int array.
        raw_4ch_volume: (4, H, W, D) float32 array.
        valid_sv_ids: list of SV IDs to process.

    Returns:
        sv_feats: dict[sv_id] → numpy array of shape (22,)
        sv_centroids: dict[sv_id] → (x, y, z) normalized centroid
    """
    H, W, D = sv_labels.shape
    max_dim = max(H, W, D)
    sv_feats = {}
    sv_centroids = {}

    for sv_id in valid_sv_ids:
        mask = sv_labels == sv_id
        coords = np.argwhere(mask)  # (N_voxels, 3)

        if len(coords) == 0:
            continue

        # Centroid (normalized)
        centroid = coords.mean(axis=0)
        cx, cy, cz = centroid[0] / H, centroid[1] / W, centroid[2] / D
        sv_centroids[sv_id] = np.array([cx, cy, cz], dtype=np.float32)

        # Per-modality intensity features (4 × 4 = 16)
        intensity_feats = []
        for ch in range(4):
            pixels = raw_4ch_volume[ch][mask]
            m = float(np.mean(pixels))
            s = float(np.std(pixels))
            r = float(np.max(pixels) - np.min(pixels))
            sk = float(np.mean(((pixels - m) / max(s, 1e-8)) ** 3))
            intensity_feats.extend([m, s, r, sk])

        # Volume (normalized)
        volume = len(coords)
        vol_norm = volume / (H * W * D)

        # Surface area approximation: count boundary voxels
        dilated = ndimage.binary_dilation(mask, iterations=1)
        surface = int((dilated & ~mask).sum())
        sa_norm = surface / max(volume, 1)

        # Compactness: volume / (surface_area ^ 1.5)
        compactness = volume / max(surface ** 1.5, 1e-8)

        feat = np.array(
            intensity_feats + [cx, cy, cz, vol_norm, sa_norm, compactness],
            dtype=np.float32,
        )
        sv_feats[sv_id] = feat

    return sv_feats, sv_centroids


def compute_relative_pe(sv_centroids_in_node, seg_node_centroid):
    """Compute 3D relative positional encoding for SVs within a node.

    PE = (sv_centroid - seg_centroid) / seg_radius

    This distinguishes boundary SVs from interior SVs.

    Args:
        sv_centroids_in_node: (K, 3) array of SV centroids.
        seg_node_centroid: (3,) array, centroid of the parent seg node.

    Returns:
        relative_pe: (K, 3) array.
    """
    offsets = sv_centroids_in_node - seg_node_centroid[np.newaxis, :]

    # Radius = max distance of any SV from seg centroid
    dists = np.linalg.norm(offsets, axis=1)
    radius = max(float(np.max(dists)), 1e-8)

    return (offsets / radius).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# Assignment Matrix & Intra-Node Edges
# ══════════════════════════════════════════════════════════════════════

def extract_seg_components_3d(seg_mask):
    """Extract connected components from a 3D segmentation mask.

    Args:
        seg_mask: (H, W, D) int array with labels {0=BG, 1=NCR, 2=ED, 3=ET}.

    Returns:
        components: list of dicts with keys:
            - 'id': component index
            - 'tissue_label': int (1, 2, or 3)
            - 'mask': (H, W, D) bool array
            - 'centroid': (3,) normalized centroid
            - 'volume': int
    """
    H, W, D = seg_mask.shape
    components = []
    comp_id = 0

    for tissue_label in [1, 2, 3]:
        binary = (seg_mask == tissue_label).astype(np.uint8)
        if binary.sum() == 0:
            continue

        labeled, n_comps = ndimage.label(binary)

        for label_id in range(1, n_comps + 1):
            comp_mask = labeled == label_id
            volume = int(comp_mask.sum())
            if volume < SUPERVOXEL["min_sv_volume"]:
                continue

            coords = np.argwhere(comp_mask)
            centroid = coords.mean(axis=0)
            centroid_norm = np.array(
                [centroid[0] / H, centroid[1] / W, centroid[2] / D],
                dtype=np.float32,
            )

            components.append({
                "id": comp_id,
                "tissue_label": tissue_label,
                "mask": comp_mask,
                "centroid": centroid_norm,
                "volume": volume,
            })
            comp_id += 1

    return components


def compute_assignment_matrix(sv_labels, valid_sv_ids, components):
    """Compute spatial overlap between supervoxels and segmentation components.

    S[i, j] = |SV_i ∩ Seg_j| / |SV_i|

    Each row sums to ≤ 1.0 (< 1.0 if the SV partially overlaps background).

    Args:
        sv_labels: (H, W, D) int array.
        valid_sv_ids: list of retained SV IDs.
        components: list of seg component dicts from extract_seg_components_3d.

    Returns:
        S: (N_sv, N_seg) float32 numpy array.
        sv_to_seg: dict[sv_id] → seg_comp_id (majority assignment).
        tumor_sv_ids: list of SV IDs that overlap with any seg component.
    """
    n_sv = len(valid_sv_ids)
    n_seg = len(components)

    if n_sv == 0 or n_seg == 0:
        return np.zeros((n_sv, n_seg), dtype=np.float32), {}, []

    sv_id_to_idx = {sv_id: idx for idx, sv_id in enumerate(valid_sv_ids)}

    # Precompute SV volumes
    sv_volumes = {}
    for sv_id in valid_sv_ids:
        sv_volumes[sv_id] = int((sv_labels == sv_id).sum())

    S = np.zeros((n_sv, n_seg), dtype=np.float32)

    for j, comp in enumerate(components):
        comp_mask = comp["mask"]
        # Find which SVs overlap with this component
        sv_ids_in_comp = np.unique(sv_labels[comp_mask])

        for sv_id in sv_ids_in_comp:
            if sv_id not in sv_id_to_idx:
                continue
            i = sv_id_to_idx[sv_id]
            overlap = int(((sv_labels == sv_id) & comp_mask).sum())
            S[i, j] = overlap / max(sv_volumes[sv_id], 1)

    # Majority assignment
    sv_to_seg = {}
    tumor_sv_ids = []
    for i, sv_id in enumerate(valid_sv_ids):
        row_sum = S[i].sum()
        if row_sum > 0.05:  # at least 5% overlap with tumor
            sv_to_seg[sv_id] = int(np.argmax(S[i]))
            tumor_sv_ids.append(sv_id)

    return S, sv_to_seg, tumor_sv_ids


def build_intra_sv_edges(sv_centroids, tumor_sv_ids, sv_to_seg, k=None):
    """Build KNN edges among supervoxels within each segmentation component.

    Args:
        sv_centroids: dict[sv_id] → (3,) centroid array.
        tumor_sv_ids: list of SV IDs overlapping tumor.
        sv_to_seg: dict[sv_id] → seg component index.
        k: number of neighbors.

    Returns:
        edges_per_comp: dict[seg_comp_id] → (2, E) edge index tensor
                        (indices are LOCAL within the component's SV list).
        svs_per_comp: dict[seg_comp_id] → list of sv_ids in that component.
    """
    from torch_geometric.nn import knn_graph

    k = k or SUPERVOXEL["intra_k"]

    # Group SVs by their seg component
    comp_svs = defaultdict(list)
    for sv_id in tumor_sv_ids:
        if sv_id in sv_to_seg:
            comp_svs[sv_to_seg[sv_id]].append(sv_id)

    edges_per_comp = {}
    svs_per_comp = {}

    for comp_id, sv_list in comp_svs.items():
        svs_per_comp[comp_id] = sv_list

        if len(sv_list) < 2:
            edges_per_comp[comp_id] = torch.zeros(2, 0, dtype=torch.long)
            continue

        # Stack centroids for SVs in this component
        positions = np.stack([sv_centroids[sid] for sid in sv_list])
        pos_tensor = torch.tensor(positions, dtype=torch.float32)

        local_k = min(k, len(sv_list) - 1)
        if local_k < 1:
            edges_per_comp[comp_id] = torch.zeros(2, 0, dtype=torch.long)
            continue

        edge_index = knn_graph(pos_tensor, k=local_k, loop=False)
        edges_per_comp[comp_id] = edge_index

    return edges_per_comp, svs_per_comp


# ══════════════════════════════════════════════════════════════════════
# Caching & Full Pipeline
# ══════════════════════════════════════════════════════════════════════

def process_case_supervoxels(case_id, raw_4ch_volume, seg_mask, config=None):
    """Full supervoxel pipeline for one patient case.

    1. 3D SLIC on T1 → sv_labels
    2. Prune background
    3. Extract 3D seg components
    4. Compute assignment matrix
    5. Extract SV features + relative PE per seg node
    6. Build intra-SV edges per seg node

    Args:
        case_id: str, patient identifier.
        raw_4ch_volume: (4, H, W, D) float32.
        seg_mask: (H, W, D) int.
        config: optional config override.

    Returns:
        result dict with all supervoxel data for this case.
    """
    config = config or SUPERVOXEL

    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{case_id}_sv.npz"

    # Step 1: 3D SLIC (cached)
    if cache_path.exists():
        cached = np.load(str(cache_path), allow_pickle=True)
        sv_labels = cached["sv_labels"]
    else:
        t1_volume = raw_4ch_volume[0]  # SLIC on T1 only
        sv_labels = generate_3d_supervoxels(t1_volume, config["n_segments"], config["compactness"])
        np.savez_compressed(str(cache_path), sv_labels=sv_labels)

    # Step 2: Prune background
    t1_volume = raw_4ch_volume[0]
    valid_sv_ids, sv_means = prune_background_svs(sv_labels, t1_volume, config["min_sv_volume"])

    # Step 3: 3D seg components
    components = extract_seg_components_3d(seg_mask)

    if len(components) == 0 or len(valid_sv_ids) == 0:
        return {
            "case_id": case_id,
            "components": [],
            "sv_features_per_comp": {},
            "sv_edges_per_comp": {},
            "svs_per_comp": {},
            "n_tumor_svs": 0,
        }

    # Step 4: Assignment matrix
    S, sv_to_seg, tumor_sv_ids = compute_assignment_matrix(
        sv_labels, valid_sv_ids, components,
    )

    # Step 5: SV features
    sv_feats, sv_centroids = compute_sv_features(sv_labels, raw_4ch_volume, tumor_sv_ids)

    # Step 6: Intra-SV edges
    edges_per_comp, svs_per_comp = build_intra_sv_edges(
        sv_centroids, tumor_sv_ids, sv_to_seg, config["intra_k"],
    )

    # Step 7: Relative PE + final feature assembly per seg component
    sv_features_per_comp = {}  # comp_id → (K, 25) tensor

    for comp_id, sv_list in svs_per_comp.items():
        if len(sv_list) == 0:
            continue

        comp = components[comp_id]
        comp_centroid = comp["centroid"]

        # Stack base features (22-dim)
        base_feats = np.stack([sv_feats[sid] for sid in sv_list])  # (K, 22)

        # Relative PE (3-dim)
        centroids_arr = np.stack([sv_centroids[sid] for sid in sv_list])  # (K, 3)
        rel_pe = compute_relative_pe(centroids_arr, comp_centroid)  # (K, 3)

        # Concatenate: (K, 25)
        full_feats = np.concatenate([base_feats, rel_pe], axis=1)
        sv_features_per_comp[comp_id] = torch.tensor(full_feats, dtype=torch.float32)

    return {
        "case_id": case_id,
        "components": components,
        "sv_features_per_comp": sv_features_per_comp,
        "sv_edges_per_comp": edges_per_comp,
        "svs_per_comp": svs_per_comp,
        "sv_labels": sv_labels,
        "assignment_matrix": S,
        "tumor_sv_ids": tumor_sv_ids,
        "n_tumor_svs": len(tumor_sv_ids),
    }


# ── Main (standalone test) ───────────────────────────────────────────

if __name__ == "__main__":
    print("Supervoxel module loaded.")
    print(f"Config: {SUPERVOXEL}")
    print("Run with a BraTS case to test. Example:")
    print("  from supervoxel import process_case_supervoxels")
    print("  result = process_case_supervoxels(case_id, raw_4ch, seg_mask)")

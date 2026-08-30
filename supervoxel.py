import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from config import SHARED, SUPERVOXEL, SEGMENTATION
from skimage.segmentation import slic
from torch_geometric.nn import knn_graph
from scipy.spatial import Delaunay


def generate_3d_supervoxels(t1_volume, n_segments=None, compactness=None):
    n_segments = n_segments or SUPERVOXEL["n_segments"]
    compactness = compactness or SUPERVOXEL["compactness"]

    sv_labels = slic(
        t1_volume,
        n_segments=n_segments,
        compactness=compactness,
        channel_axis=None,
        enforce_connectivity=True,
        start_label=0,
    )
    return sv_labels.astype(np.int32)


def prune_background_svs(sv_labels, t1_volume, min_volume=None):
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

def compute_sv_features(sv_labels, raw_4ch_volume, valid_sv_ids):
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
    offsets = sv_centroids_in_node - seg_node_centroid[np.newaxis, :]

    # Radius = max distance of any SV from seg centroid
    dists = np.linalg.norm(offsets, axis=1)
    radius = max(float(np.max(dists)), 1e-8)

    return (offsets / radius).astype(np.float32)


def extract_seg_components_3d(seg_mask):
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


def build_intra_sv_edges_delaunay(sv_centroids, tumor_sv_ids, sv_to_seg):
    """Build edges among SVs within each seg component using Delaunay triangulation.

    Delaunay triangulation connects points whose Voronoi cells share a face,
    producing a spatially principled graph without requiring a k parameter
    (Barber et al., 1996, ACM Trans. on Mathematical Software).

    For fewer than 4 SVs (insufficient for 3D Delaunay), falls back to
    fully connected.
    """
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

        positions = np.stack([sv_centroids[sid] for sid in sv_list])

        if len(sv_list) < 4:
            # Fewer than 4 points: 3D Delaunay needs at least 4 non-coplanar points.
            # Fall back to fully connected.
            src, dst = [], []
            for i in range(len(sv_list)):
                for j in range(i + 1, len(sv_list)):
                    src.extend([i, j])
                    dst.extend([j, i])
            edges_per_comp[comp_id] = torch.tensor([src, dst], dtype=torch.long)
            continue

        try:
            tri = Delaunay(positions)
            edge_set = set()
            for simplex in tri.simplices:
                # Each simplex is a tetrahedron (4 vertices in 3D).
                # Extract all 6 edges from the tetrahedron.
                for i in range(4):
                    for j in range(i + 1, 4):
                        a, b = int(simplex[i]), int(simplex[j])
                        edge_set.add((min(a, b), max(a, b)))

            if len(edge_set) == 0:
                edges_per_comp[comp_id] = torch.zeros(2, 0, dtype=torch.long)
                continue

            src, dst = [], []
            for a, b in edge_set:
                src.extend([a, b])
                dst.extend([b, a])
            edges_per_comp[comp_id] = torch.tensor([src, dst], dtype=torch.long)

        except Exception:
            # Degenerate case (coplanar points etc.): fall back to fully connected
            src, dst = [], []
            for i in range(len(sv_list)):
                for j in range(i + 1, len(sv_list)):
                    src.extend([i, j])
                    dst.extend([j, i])
            edges_per_comp[comp_id] = torch.tensor([src, dst], dtype=torch.long)

    return edges_per_comp, svs_per_comp


def process_case_supervoxels(case_id, raw_4ch_volume, seg_mask, config=None):
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

    # Step 6: Intra-SV edges (dispatch based on config)
    sv_edge_method = config.get("sv_edge_method", "delaunay")
    if sv_edge_method == "delaunay":
        edges_per_comp, svs_per_comp = build_intra_sv_edges_delaunay(
            sv_centroids, tumor_sv_ids, sv_to_seg,
        )
    else:
        edges_per_comp, svs_per_comp = build_intra_sv_edges(
            sv_centroids, tumor_sv_ids, sv_to_seg, config.get("intra_k", 3),
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

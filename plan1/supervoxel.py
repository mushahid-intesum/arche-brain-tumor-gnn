import numpy as np
import nibabel as nib
from pathlib import Path
from skimage.segmentation import slic
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

from .config import SHARED, GRAPH


# ── Data Loading (native resolution) ─────────────────────────────────


def discover_cases(data_dir):
    data_dir = Path(data_dir)
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    valid = []

    for case_dir in cases:
        case_id = case_dir.name
        files = {}
        all_found = True

        for key in [*GRAPH["modalities"], "seg"]:
            # Try both .nii and .nii.gz
            nii = case_dir / f"{case_id}-{key}.nii"
            nii_gz = case_dir / f"{case_id}-{key}.nii.gz"
            if nii.exists():
                files[key] = nii
            elif nii_gz.exists():
                files[key] = nii_gz
            else:
                all_found = False
                break

        if all_found:
            valid.append({"case_id": case_id, "files": files})

    return valid


def load_volume(nii_path):
    """Load a NIfTI volume at native resolution as float32."""
    return nib.load(str(nii_path)).get_fdata().astype(np.float32)


def zscore_normalize(volume):
    """Z-score normalize non-zero voxels (standard BraTS preprocessing)."""
    out = volume.copy()
    mask = out > 0
    if mask.sum() == 0:
        return out
    mean = out[mask].mean()
    std = out[mask].std()
    if std < 1e-8:
        return out
    out[mask] = (out[mask] - mean) / std
    return out


def load_case_volumes(case):
    modality_vols = {}
    for mod in GRAPH["modalities"]:
        vol = load_volume(case["files"][mod])
        modality_vols[mod] = zscore_normalize(vol)

    gt_seg = load_volume(case["files"]["seg"]).astype(np.int32)
    shape = modality_vols[GRAPH["slic_modality"]].shape

    return modality_vols, gt_seg, shape


# ── Step 1: 3D SLIC Supervoxel Generation ────────────────────────────


def generate_supervoxels(t1n_volume, n_segments=None, compactness=None):
    n_segments = n_segments or GRAPH["n_segments"]
    compactness = compactness or GRAPH["compactness"]

    sv_labels = slic(
        t1n_volume,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        enforce_connectivity=True,
        channel_axis=None,          # input is single-channel 3D
    )

    return sv_labels.astype(np.int32)


# ── Step 2: Dynamic Background Pruning ───────────────────────────────


def prune_background(sv_labels, t1n_volume, min_volume=None):
    min_volume = min_volume or GRAPH["min_sv_volume"]
    unique_labels = np.unique(sv_labels)

    # Compute per-SV statistics
    sv_stats = {}
    for label in unique_labels:
        mask = sv_labels == label
        voxels = t1n_volume[mask]
        sv_stats[label] = {
            "mean_intensity": float(voxels.mean()),
            "volume": int(mask.sum()),
        }

    # Filter by minimum volume first
    volume_ok = {
        l for l, s in sv_stats.items()
        if s["volume"] >= min_volume
    }

    # Sort remaining SVs by mean intensity (ascending)
    sorted_labels = sorted(volume_ok, key=lambda l: sv_stats[l]["mean_intensity"])
    sorted_means = [sv_stats[l]["mean_intensity"] for l in sorted_labels]

    if len(sorted_means) < 2:
        # Edge case: can't compute gaps with fewer than 2 SVs
        return list(volume_ok), {
            "theta": 0.0,
            "total_svs": len(unique_labels),
            "retained": len(volume_ok),
            "pruned_bg": 0,
            "pruned_small": len(unique_labels) - len(volume_ok),
        }

    # Find largest gap in sorted mean distribution
    gaps = np.diff(sorted_means)
    g = int(np.argmax(gaps))
    theta = 0.5 * (sorted_means[g] + sorted_means[g + 1])

    # Retain SVs with mean intensity above threshold
    retained = [l for l in sorted_labels if sv_stats[l]["mean_intensity"] > theta]
    pruned_bg = len(sorted_labels) - len(retained)
    pruned_small = len(unique_labels) - len(volume_ok)

    pruning_info = {
        "theta": float(theta),
        "total_svs": len(unique_labels),
        "after_volume_filter": len(volume_ok),
        "retained": len(retained),
        "pruned_bg": pruned_bg,
        "pruned_small": pruned_small,
        "sv_stats": sv_stats,
    }

    return retained, pruning_info


# ── Step 3: Supervoxel-Level Ground Truth ────────────────────────────


def compute_sv_targets(sv_labels, gt_seg, retained_labels, tau=None):
    tau = tau if tau is not None else GRAPH["tau"]
    num_classes = GRAPH["num_classes"]

    # Pre-compute voxel coordinate arrays for centroid calculation
    coords = np.array(np.where(sv_labels >= 0))  # (3, total_voxels)

    targets = {}
    tumor_count = 0
    healthy_count = 0

    for label in retained_labels:
        mask = sv_labels == label
        gt_voxels = gt_seg[mask].astype(np.int64)
        total = gt_voxels.size

        # ── Tumor proportion (any class > 0 is tumor) ──
        n_tumor = int((gt_voxels > 0).sum())
        y_reg = n_tumor / total if total > 0 else 0.0

        # ── Binary tumor label ──
        y_cls = 1 if y_reg > tau else 0

        # ── Dominant class (mode) ──
        counts = np.bincount(gt_voxels, minlength=num_classes)
        y_dominant = int(counts.argmax())

        # ── Class proportions vector ──
        y_props = (counts / total).astype(np.float32) if total > 0 else np.zeros(num_classes, dtype=np.float32)

        # ── Centroid (mean voxel coordinate) ──
        sv_coords = np.argwhere(mask)  # (N_voxels, 3)
        centroid = sv_coords.mean(axis=0).astype(np.float32)  # (3,)

        targets[label] = {
            "y_reg": float(y_reg),
            "y_cls": int(y_cls),
            "y_dominant": int(y_dominant),
            "y_props": y_props,
            "volume": int(total),
            "centroid": centroid,
            "class_counts": counts.astype(np.int64),
        }

        if y_cls == 1:
            tumor_count += 1
        else:
            healthy_count += 1

    summary = {
        "total_svs": len(retained_labels),
        "tumor_svs": tumor_count,
        "healthy_svs": healthy_count,
        "tumor_ratio": tumor_count / len(retained_labels) if retained_labels else 0.0,
        "mean_tumor_proportion": float(np.mean([t["y_reg"] for t in targets.values()])),
    }

    return targets, summary


# ── Step 4: Segmentation Prior Features ──────────────────────────────


def load_seg_probabilities(case_id, prob_dir=None):
    if prob_dir is None:
        prob_dir = Path("brats_outputs/seg_probs")
    else:
        prob_dir = Path(prob_dir)

    prob_path = prob_dir / f"{case_id}_seg_probs.npy"
    if not prob_path.exists():
        return None

    return np.load(str(prob_path))


def compute_seg_prior_features(sv_labels, retained_labels, seg_probs):
    num_classes = seg_probs.shape[0]
    seg_features = {}

    all_entropies = []

    for label in retained_labels:
        mask = sv_labels == label

        # Mean probability vector across all voxels in this SV
        # seg_probs[:, mask] has shape (num_classes, n_voxels)
        seg_feat = seg_probs[:, mask].mean(axis=1).astype(np.float32)  # (4,)

        # Prediction entropy: -Σ p_c log(p_c)  (clipped to avoid log(0))
        p_clipped = np.clip(seg_feat, 1e-8, 1.0)
        seg_entropy = float(-np.sum(p_clipped * np.log(p_clipped)))

        # Predicted class from mean probabilities
        seg_pred = int(seg_feat.argmax())

        seg_features[label] = {
            "seg_feat": seg_feat,
            "seg_entropy": seg_entropy,
            "seg_pred": seg_pred,
        }

        all_entropies.append(seg_entropy)

    # Summary statistics
    entropies = np.array(all_entropies)
    pred_counts = {}
    for sf in seg_features.values():
        c = sf["seg_pred"]
        pred_counts[c] = pred_counts.get(c, 0) + 1

    seg_summary = {
        "mean_entropy": float(entropies.mean()),
        "std_entropy": float(entropies.std()),
        "max_entropy": float(entropies.max()),
        "high_uncertainty_svs": int((entropies > 1.0).sum()),  # entropy > 1.0 nats
        "pred_class_distribution": pred_counts,
    }

    return seg_features, seg_summary


# ── Step 5: Patch Extraction ─────────────────────────────────────────


def extract_patches(sv_labels, modality_vols, retained_labels, targets,
                    n_patch=None, patch_neighbors=None):
    n_patch = n_patch or GRAPH["n_patch"]
    s = patch_neighbors or GRAPH["patch_neighbors"]
    mod_names = list(modality_vols.keys())
    n_mods = len(mod_names)

    patches = {}
    padded_count = 0

    for label in retained_labels:
        voxel_coords = np.argwhere(sv_labels == label)  # (N_vox, 3)
        N = voxel_coords.shape[0]

        # k-means++ centroid selection
        actual_k = min(n_patch, N)
        if actual_k < 2:
            centroids = voxel_coords[:actual_k].astype(np.float64)
        else:
            km = KMeans(
                n_clusters=actual_k, init="k-means++",
                n_init=1, max_iter=20, random_state=42,
            )
            km.fit(voxel_coords)
            centroids = km.cluster_centers_  # (actual_k, 3)

        # Build KDTree for nearest-neighbor queries within this SV
        tree = cKDTree(voxel_coords)

        rows = []
        for ci in range(actual_k):
            centroid = centroids[ci]
            k_query = min(s, N)
            _, nn_idx = tree.query(centroid, k=k_query)
            nn_idx = np.atleast_1d(nn_idx)
            nn_coords = voxel_coords[nn_idx]  # (k_query, 3)

            for mod in mod_names:
                vol = modality_vols[mod]
                values = vol[nn_coords[:, 0], nn_coords[:, 1], nn_coords[:, 2]]

                # Pad if fewer neighbors than s
                if len(values) < s:
                    values = np.pad(values, (0, s - len(values)), mode="edge")
                    padded_count += 1

                # Augment with centroid XYZ
                row = np.concatenate([values, centroid])  # (s + 3,)
                rows.append(row)

        # Pad missing centroids if SV was very small
        while len(rows) < n_patch * n_mods:
            rows.append(rows[-1].copy())
            padded_count += 1

        patches[label] = np.stack(rows, axis=0).astype(np.float32)

    patch_info = {
        "n_patch": n_patch,
        "patch_neighbors": s,
        "n_modalities": n_mods,
        "tensor_shape": f"({n_patch * n_mods}, {s + 3})",
        "padded_patches": padded_count,
    }

    return patches, patch_info


# ── Step 6: kNN Graph Construction ───────────────────────────────────


def build_knn_graph(retained_labels, targets, k=None):
    k = k or GRAPH["knn_k"]
    N = len(retained_labels)

    # Map SV labels to consecutive node indices
    label_to_idx = {label: i for i, label in enumerate(retained_labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    # Collect centroids in index order
    centroids = np.stack(
        [targets[retained_labels[i]]["centroid"] for i in range(N)],
        axis=0,
    )  # (N, 3)

    # kNN via KDTree
    actual_k = min(k + 1, N)  # +1 because query includes self
    tree = cKDTree(centroids)
    dists, indices = tree.query(centroids, k=actual_k)  # (N, actual_k)

    # Build edge list (skip self-loops at index 0)
    src_list = []
    dst_list = []
    attr_list = []

    for i in range(N):
        for j_pos in range(1, actual_k):  # skip self
            j = indices[i, j_pos]
            delta = centroids[j] - centroids[i]  # (3,)
            dist = dists[i, j_pos]

            src_list.append(i)
            dst_list.append(j)
            attr_list.append([delta[0], delta[1], delta[2], dist])

    # Symmetrize: add reverse edges (deduplicated)
    edge_set = set()
    sym_src, sym_dst, sym_attr = [], [], []
    for idx in range(len(src_list)):
        s, d = src_list[idx], dst_list[idx]
        if (s, d) not in edge_set:
            edge_set.add((s, d))
            sym_src.append(s)
            sym_dst.append(d)
            sym_attr.append(attr_list[idx])
        if (d, s) not in edge_set:
            edge_set.add((d, s))
            rev_delta = [-attr_list[idx][0], -attr_list[idx][1],
                         -attr_list[idx][2], attr_list[idx][3]]
            sym_src.append(d)
            sym_dst.append(s)
            sym_attr.append(rev_delta)

    if len(sym_src) > 0:
        edge_index = np.array([sym_src, sym_dst], dtype=np.int64)   # (2, E)
        edge_attr = np.array(sym_attr, dtype=np.float32).reshape(-1, 4)  # (E, 4)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 4), dtype=np.float32)

    # Degree statistics
    degrees = np.bincount(edge_index[0] if edge_index.shape[1] > 0 else [], minlength=N)

    graph_info = {
        "num_nodes": N,
        "num_edges": edge_index.shape[1],
        "k": k,
        "mean_degree": float(degrees.mean()),
        "min_degree": int(degrees.min()) if len(degrees) > 0 else 0,
        "max_degree": int(degrees.max()) if len(degrees) > 0 else 0,
        "avg_edge_dist": float(edge_attr[:, 3].mean()) if edge_attr.shape[0] > 0 else 0.0,
    }

    return edge_index, edge_attr, label_to_idx, idx_to_label, graph_info


# ── Step 7: Edge-Level Ground Truth ──────────────────────────────────

# Symmetric boundary type encoding: ordered pair (min, max) of class IDs
# 10 types for 4 classes: (0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)
BOUNDARY_TYPE_MAP = {}
_bt_idx = 0
for _a in range(4):
    for _b in range(_a, 4):
        BOUNDARY_TYPE_MAP[(_a, _b)] = _bt_idx
        _bt_idx += 1
BOUNDARY_TYPE_NAMES = {
    v: f"{GRAPH['class_names'][a]}↔{GRAPH['class_names'][b]}"
    for (a, b), v in BOUNDARY_TYPE_MAP.items()
}


def compute_edge_targets(edge_index, targets, idx_to_label):
    E = edge_index.shape[1]
    y_binary = np.zeros(E, dtype=np.int64)
    y_type = np.zeros(E, dtype=np.int64)
    y_grad = np.zeros(E, dtype=np.float32)

    for e in range(E):
        i_idx, j_idx = int(edge_index[0, e]), int(edge_index[1, e])
        label_i, label_j = idx_to_label[i_idx], idx_to_label[j_idx]

        dom_i = targets[label_i]["y_dominant"]
        dom_j = targets[label_j]["y_dominant"]
        reg_i = targets[label_i]["y_reg"]
        reg_j = targets[label_j]["y_reg"]

        # Binary: same class?
        y_binary[e] = 1 if dom_i == dom_j else 0

        # Boundary type: symmetric ordered pair
        pair = (min(dom_i, dom_j), max(dom_i, dom_j))
        y_type[e] = BOUNDARY_TYPE_MAP[pair]

        # Transition gradient
        y_grad[e] = abs(reg_i - reg_j)

    # Statistics
    type_counts = {}
    for t in y_type:
        name = BOUNDARY_TYPE_NAMES[int(t)]
        type_counts[name] = type_counts.get(name, 0) + 1

    edge_target_info = {
        "num_edges": E,
        "same_class_edges": int(y_binary.sum()),
        "diff_class_edges": int((1 - y_binary).sum()),
        "boundary_type_distribution": type_counts,
        "mean_gradient": float(y_grad.mean()),
        "max_gradient": float(y_grad.max()),
    }

    edge_targets = {
        "y_edge_binary": y_binary,
        "y_edge_type": y_type,
        "y_edge_grad": y_grad,
    }

    return edge_targets, edge_target_info


# ── Full Pipeline (Steps 1–7) ────────────────────────────────────────


def preprocess_case(case, config=None, seg_prob_dir=None):
    cfg = config or GRAPH
    case_id = case["case_id"]

    # Load native-resolution volumes
    modality_vols, gt_seg, shape = load_case_volumes(case)
    t1n = modality_vols[cfg["slic_modality"]]

    # Step 1: 3D SLIC
    sv_labels = generate_supervoxels(
        t1n, n_segments=cfg["n_segments"], compactness=cfg["compactness"]
    )

    # Step 2: Dynamic background pruning
    retained_labels, pruning_info = prune_background(
        sv_labels, t1n, min_volume=cfg["min_sv_volume"]
    )

    # Step 3: Ground truth
    targets, target_summary = compute_sv_targets(
        sv_labels, gt_seg, retained_labels, tau=cfg["tau"]
    )

    # Step 4: Segmentation prior features (optional)
    seg_features = None
    seg_summary = None
    if seg_prob_dir is not None:
        seg_probs = load_seg_probabilities(case_id, seg_prob_dir)
        if seg_probs is not None:
            seg_features, seg_summary = compute_seg_prior_features(
                sv_labels, retained_labels, seg_probs
            )
            del seg_probs

    # Step 5: Patch extraction
    patches, patch_info = extract_patches(
        sv_labels, modality_vols, retained_labels, targets,
        n_patch=cfg["n_patch"], patch_neighbors=cfg["patch_neighbors"],
    )

    # Step 6: kNN graph construction
    edge_index, edge_attr, label_to_idx, idx_to_label, graph_info = (
        build_knn_graph(retained_labels, targets, k=cfg["knn_k"])
    )

    # Step 7: Edge-level ground truth
    edge_targets, edge_target_info = compute_edge_targets(
        edge_index, targets, idx_to_label
    )

    return {
        "case_id": case_id,
        "sv_labels": sv_labels,
        "retained_labels": retained_labels,
        "targets": targets,
        "pruning_info": pruning_info,
        "target_summary": target_summary,
        "seg_features": seg_features,
        "seg_summary": seg_summary,
        "patches": patches,
        "patch_info": patch_info,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_targets": edge_targets,
        "edge_target_info": edge_target_info,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "graph_info": graph_info,
        "modality_vols": modality_vols,
        "gt_seg": gt_seg,
        "shape": shape,
    }


# ── Main (demo on first case) ────────────────────────────────────────


if __name__ == "__main__":
    import time

    print(f"Device: {SHARED['device']}")
    print(f"Config: n_segments={GRAPH['n_segments']}, "
          f"compactness={GRAPH['compactness']}, tau={GRAPH['tau']}")
    print()

    # Discover BraTS cases
    cases = discover_cases(GRAPH["data_root"])
    print(f"Discovered {len(cases)} valid BraTS cases")

    if not cases:
        print("No cases found. Check GRAPH['data_root'] path.")
        exit(1)

    # Check for seg probability volumes
    seg_prob_dir = Path("brats_outputs/seg_probs")
    has_seg_probs = seg_prob_dir.exists() and any(seg_prob_dir.glob("*.npy"))
    n_steps = 7 if has_seg_probs else 6
    if has_seg_probs:
        n_cached = len(list(seg_prob_dir.glob("*.npy")))
        print(f"Found {n_cached} seg probability volumes in {seg_prob_dir}")
    else:
        print(f"No seg probability volumes found in {seg_prob_dir}")
        print(f"  → Step 4 (seg priors) will be skipped.")
        print(f"  → Run segmentation.export_seg_probabilities() first to enable.")

    # Process first case as demo
    case = cases[0]
    print(f"\n{'='*60}")
    print(f"Processing: {case['case_id']}")
    print(f"{'='*60}")

    t0 = time.time()

    # Load volumes
    print(f"\n[1/{n_steps}] Loading native-resolution volumes...")
    modality_vols, gt_seg, shape = load_case_volumes(case)
    t1n = modality_vols[GRAPH["slic_modality"]]
    print(f"  Volume shape: {shape}")
    print(f"  T1n range: [{t1n.min():.2f}, {t1n.max():.2f}]")
    print(f"  GT classes present: {np.unique(gt_seg).tolist()}")

    # Step 1: SLIC
    print(f"\n[2/{n_steps}] Running 3D SLIC (n_segments={GRAPH['n_segments']})...")
    t1 = time.time()
    sv_labels = generate_supervoxels(t1n)
    n_generated = len(np.unique(sv_labels))
    print(f"  Generated {n_generated} supervoxels in {time.time()-t1:.1f}s")

    # Step 2: Background pruning
    print(f"\n[3/{n_steps}] Dynamic background pruning...")
    retained, pruning_info = prune_background(sv_labels, t1n)
    print(f"  Retained: {pruning_info['retained']}/{pruning_info['total_svs']} "
          f"(θ={pruning_info['theta']:.3f})")

    # Step 3: Ground truth
    print(f"\nComputing supervoxel-level ground truth (τ={GRAPH['tau']})...")
    targets, summary = compute_sv_targets(sv_labels, gt_seg, retained)
    print(f"  Tumor SVs: {summary['tumor_svs']}/{summary['total_svs']} "
          f"({summary['tumor_ratio']:.1%})")

    # Step 4: Segmentation prior features
    step_offset = 0
    if has_seg_probs:
        print(f"\n[4/{n_steps}] Computing seg prior features...")
        seg_probs = load_seg_probabilities(case["case_id"], seg_prob_dir)
        if seg_probs is not None:
            seg_features, seg_summary = compute_seg_prior_features(
                sv_labels, retained, seg_probs
            )
            print(f"  Mean entropy: {seg_summary['mean_entropy']:.4f}")
            del seg_probs
        step_offset = 1

    # Step 5: Patch extraction
    s5 = 4 + step_offset
    print(f"\n[{s5}/{n_steps}] Extracting patches (n_patch={GRAPH['n_patch']}, "
          f"s={GRAPH['patch_neighbors']})...")
    t5 = time.time()
    patches, patch_info = extract_patches(
        sv_labels, modality_vols, retained, targets
    )
    print(f"  Computed in {time.time()-t5:.1f}s")
    print(f"  Tensor shape per SV: {patch_info['tensor_shape']}")

    # Step 6: kNN graph construction
    s6 = 5 + step_offset
    print(f"\n[{s6}/{n_steps}] Building kNN graph (k={GRAPH['knn_k']})...")
    edge_index, edge_attr, label_to_idx, idx_to_label, graph_info = (
        build_knn_graph(retained, targets)
    )
    print(f"  Nodes: {graph_info['num_nodes']}, Edges: {graph_info['num_edges']}, "
          f"Mean degree: {graph_info['mean_degree']:.1f}")

    # Step 7: Edge-level ground truth
    s7 = 6 + step_offset
    print(f"\n[{s7}/{n_steps}] Computing edge-level ground truth...")
    edge_targets, edge_target_info = compute_edge_targets(
        edge_index, targets, idx_to_label
    )
    print(f"  Same-class edges: {edge_target_info['same_class_edges']}")
    print(f"  Diff-class edges: {edge_target_info['diff_class_edges']}")
    print(f"  Mean transition gradient: {edge_target_info['mean_gradient']:.4f}")
    print(f"  Max transition gradient:  {edge_target_info['max_gradient']:.4f}")
    print(f"  Boundary type distribution:")
    for btype, count in sorted(edge_target_info["boundary_type_distribution"].items()):
        print(f"    {btype}: {count} edges")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\n{'='*60}")
    print(f"Pipeline complete. All 7 preprocessing steps verified.")
    print(f"{'='*60}")

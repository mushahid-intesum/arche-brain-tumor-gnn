"""
Flat Graph Construction (Ablation Baseline)

Builds graphs from 2D segmentation slices with 35-dim handcrafted features.
This is used only for ablation configs A (baseline) and C (OCN-only)
where supervoxel aggregation is disabled.

Extracted from gnn.py to keep the main pipeline focused on the
hierarchical architecture.
"""

import sys
from pathlib import Path

import torch
import numpy as np
import cv2
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import knn_graph

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import SHARED, GNN


# ── Feature Extractors ────────────────────────────────────────────────

def compute_region_raw_features(raw_4ch, component_mask):
    """Per-modality intensity stats: mean, std, range, skewness (4 x 4 = 16 dims)."""
    feats = []
    for ch in range(4):
        pixels = raw_4ch[ch][component_mask == 1]
        if len(pixels) == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue
        m = float(np.mean(pixels))
        s = float(np.std(pixels))
        r = float(np.max(pixels) - np.min(pixels))
        sk = float(np.mean(((pixels - m) / max(s, 1e-8)) ** 3))
        feats.extend([m, s, r, sk])
    return feats


def compute_boundary_features(raw_4ch, component_mask):
    """Gradient magnitude at boundary + texture contrast (2 dims)."""
    t1c = raw_4ch[1]
    dilated = cv2.dilate(component_mask, np.ones((3, 3), np.uint8), iterations=1)
    boundary = dilated - component_mask
    if boundary.sum() > 0 and component_mask.sum() > 0:
        inner = t1c[component_mask == 1].mean()
        outer = t1c[boundary == 1].mean()
        grad = float(abs(inner - outer))
    else:
        grad = 0.0
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    local_mean = cv2.filter2D(t1c, -1, kernel)
    local_var = cv2.filter2D((t1c - local_mean) ** 2, -1, kernel)
    texture = float(np.mean(local_var[component_mask == 1])) if component_mask.sum() > 0 else 0.0
    return [grad, texture]


def compute_crossmodal_features(raw_4ch, component_mask):
    """Cross-modal ratios: enhancement, edema signal, diffs (4 dims)."""
    if component_mask.sum() == 0:
        return [0.0, 0.0, 0.0, 0.0]
    means = []
    for ch in range(4):
        means.append(float(np.mean(raw_4ch[ch][component_mask == 1])))
    t1n_m, t1c_m, t2w_m, t2f_m = means
    enhancement = t1c_m / max(t1n_m, 1e-8)
    edema_sig = t2w_m / max(t2f_m, 1e-8)
    t1c_t2w_diff = t1c_m - t2w_m
    flair_t2w_diff = t2f_m - t2w_m
    return [enhancement, edema_sig, t1c_t2w_diff, flair_t2w_diff]


# ── Region Extraction ─────────────────────────────────────────────────

def extract_regions_multiclass(mask_np, raw_4ch, slice_idx, total_slices, config=None):
    """Extract connected components per tissue type with 35-dim node features."""
    config = config or GNN
    regions = []
    img_size = SHARED["img_size"]
    tumor_area = float((mask_np > 0).sum())
    tumor_ratio = tumor_area / (img_size * img_size)
    z_norm = slice_idx / max(total_slices, 1)

    for tissue_label in [1, 2, 3]:
        binary = (mask_np == tissue_label).astype(np.uint8)
        if binary.sum() < config["min_region_area"]:
            continue

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary * 255, connectivity=8
        )

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < config["min_region_area"]:
                continue

            cx, cy = centroids[label_id]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            component_mask = (labels == label_id).astype(np.uint8)
            bbox_area = max(w * h, 1)
            solidity = area / bbox_area
            aspect_ratio = w / max(h, 1)

            tissue_onehot = [0.0, 0.0, 0.0]
            tissue_onehot[tissue_label - 1] = 1.0

            raw_feats = compute_region_raw_features(raw_4ch, component_mask)
            boundary_feats = compute_boundary_features(raw_4ch, component_mask)
            crossmodal_feats = compute_crossmodal_features(raw_4ch, component_mask)

            feat = [
                cx / img_size, cy / img_size, z_norm,           # 3D position (3)
                area / (img_size * img_size), w / img_size,     # morphology (5)
                h / img_size, aspect_ratio, solidity,
            ] + tissue_onehot + raw_feats + boundary_feats + crossmodal_feats + [
                z_norm, tumor_ratio,                             # slice context (2)
            ]

            regions.append({
                "features": feat,
                "centroid_2d": (cx, cy),
                "slice_idx": slice_idx,
                "tissue_label": tissue_label,
            })

    return regions


# ── Flat 3D Graph Construction ────────────────────────────────────────

def build_3d_graph(case_id, slice_list, config=None):
    """Build a flat 3D graph from 2D segmentation slices.

    Each connected component in each slice becomes a node with 35-dim features.
    Edges are KNN within slices + tissue-compatible connections across slices.
    """
    config = config or GNN
    all_regions = []
    total_slices = max(s["slice_idx"] for s in slice_list) - min(s["slice_idx"] for s in slice_list) + 1

    for s_info in slice_list:
        mask = np.load(str(s_info["mask_file"]))
        raw = np.load(str(s_info["raw_file"]))
        regions = extract_regions_multiclass(mask, raw, s_info["slice_idx"], total_slices, config)
        all_regions.extend(regions)

    if len(all_regions) < 2:
        x = torch.randn(max(len(all_regions), 1), config["node_feat_dim"])
        return Data(
            x=x,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            pos=torch.tensor([[0.5, 0.5, 0.5]]),
            edge_attr=torch.zeros(0, config["edge_attr_dim"]),
        )

    x = torch.tensor([r["features"] for r in all_regions], dtype=torch.float32)
    pos_2d = torch.tensor([list(r["centroid_2d"]) for r in all_regions], dtype=torch.float32)
    slice_ids = torch.tensor([r["slice_idx"] for r in all_regions], dtype=torch.long)
    tissue_labels = torch.tensor([r["tissue_label"] for r in all_regions], dtype=torch.long)
    pos_3d = torch.cat([pos_2d, slice_ids.unsqueeze(1).float()], dim=1)

    slice_to_nodes = defaultdict(list)
    for i, r in enumerate(all_regions):
        slice_to_nodes[r["slice_idx"]].append(i)

    # Intra-slice edges: KNN within each slice
    intra_src, intra_dst = [], []
    for s_idx, node_ids in slice_to_nodes.items():
        if len(node_ids) < 2:
            continue
        local_pos = pos_2d[node_ids]
        k = min(config["k_neighbors"], len(node_ids) - 1)
        if k < 1:
            continue
        local_ei = knn_graph(local_pos, k=k, loop=False)
        for e in range(local_ei.size(1)):
            intra_src.append(node_ids[local_ei[0, e].item()])
            intra_dst.append(node_ids[local_ei[1, e].item()])

    # Inter-slice edges: spatially close + tissue compatible across adjacent slices
    inter_src, inter_dst = [], []
    sorted_slices = sorted(slice_to_nodes.keys())
    for idx in range(len(sorted_slices) - 1):
        s_curr = sorted_slices[idx]
        s_next = sorted_slices[idx + 1]
        if s_next - s_curr > 2:
            continue
        for ni in slice_to_nodes[s_curr]:
            for nj in slice_to_nodes[s_next]:
                dx = pos_2d[ni, 0] - pos_2d[nj, 0]
                dy = pos_2d[ni, 1] - pos_2d[nj, 1]
                dist = float(torch.sqrt(dx ** 2 + dy ** 2))
                if dist > config["inter_slice_dist_thresh"]:
                    continue
                ti = tissue_labels[ni].item()
                tj = tissue_labels[nj].item()
                compatible = (ti == tj) or (abs(ti - tj) <= 1) or ({ti, tj} == {1, 3})
                if compatible:
                    inter_src.extend([ni, nj])
                    inter_dst.extend([nj, ni])

    all_src = intra_src + inter_src
    all_dst = intra_dst + inter_dst

    if len(all_src) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, config["edge_attr_dim"])
    else:
        edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
        edge_index = to_undirected(edge_index)

        is_inter_set = set()
        for i in range(len(inter_src)):
            is_inter_set.add((inter_src[i], inter_dst[i]))

        edge_attr = []
        for e in range(edge_index.size(1)):
            si = edge_index[0, e].item()
            di = edge_index[1, e].item()
            dx = pos_2d[di, 0] - pos_2d[si, 0]
            dy = pos_2d[di, 1] - pos_2d[si, 1]
            dist = float(torch.sqrt(dx ** 2 + dy ** 2))
            angle = float(np.arctan2(dy.item(), dx.item()))
            slice_gap = abs(slice_ids[si].item() - slice_ids[di].item()) / max(total_slices, 1)
            same_tissue = 1.0 if tissue_labels[si].item() == tissue_labels[di].item() else 0.0
            edge_attr.append([
                dist / SHARED["img_size"],
                angle / np.pi,
                slice_gap,
                same_tissue,
            ])
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        pos=pos_3d,
        edge_attr=edge_attr,
        tissue_labels=tissue_labels,
        slice_ids=slice_ids,
    )
    return data

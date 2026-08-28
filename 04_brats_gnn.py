import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import networkx as nx
from pathlib import Path
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import knn_graph, GATv2Conv, LayerNorm
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


GNN_CONFIG = {
    "brats_output_dir": Path("brats_outputs"),
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 42,
    "img_size": 224,
    "node_feat_dim": 35,
    "hidden_dim": 128,
    "embed_dim": 64,
    "num_heads": 4,
    "num_layers": 3,
    "edge_attr_dim": 4,
    "structural_feat_dim": 3,
    "min_region_area": 10,
    "k_neighbors": 5,
    "inter_slice_dist_thresh": 50.0,
    "gnn_epochs": 80,
    "gnn_lr": 5e-4,
    "gnn_weight_decay": 1e-4,
    "modalities": ["t1n", "t1c", "t2w", "t2f"],
    "tissue_labels": {1: "NCR", 2: "ED", 3: "ET"},
}

torch.manual_seed(GNN_CONFIG["seed"])
np.random.seed(GNN_CONFIG["seed"])
random.seed(GNN_CONFIG["seed"])

print(f"Device: {GNN_CONFIG['device']}")



print("Loading BraTS pipeline outputs...")
metadata = torch.load(str(GNN_CONFIG["brats_output_dir"] / "metadata.pt"), weights_only=False)

masks_dir = GNN_CONFIG["brats_output_dir"] / "masks"
raw_dir = GNN_CONFIG["brats_output_dir"] / "raw_slices"

patient_slices = defaultdict(list)
for i in range(len(metadata["case_ids"])):
    case_id = metadata["case_ids"][i]
    patient_slices[case_id].append({
        "mask_file": masks_dir / metadata["mask_files"][i],
        "raw_file": raw_dir / metadata["raw_files"][i],
        "slice_idx": metadata["slice_indices"][i],
        "split": metadata["splits"][i],
    })

for case_id in patient_slices:
    patient_slices[case_id].sort(key=lambda s: s["slice_idx"])

print(f"Loaded {len(metadata['case_ids'])} slices from {len(patient_slices)} patients")
slice_counts = [len(v) for v in patient_slices.values()]
print(f"Slices per patient: min={min(slice_counts)}, max={max(slice_counts)}, mean={np.mean(slice_counts):.1f}")



def compute_region_raw_features(raw_4ch, component_mask):
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


def extract_regions_multiclass(mask_np, raw_4ch, slice_idx, total_slices, config):
    regions = []
    img_size = config["img_size"]
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
                cx / img_size,
                cy / img_size,
                z_norm,
                area / (img_size * img_size),
                w / img_size,
                h / img_size,
                aspect_ratio,
                solidity,
            ] + tissue_onehot + raw_feats + boundary_feats + crossmodal_feats + [
                z_norm,
                tumor_ratio,
            ]

            regions.append({
                "features": feat,
                "centroid_2d": (cx, cy),
                "slice_idx": slice_idx,
                "tissue_label": tissue_label,
            })

    return regions


def build_3d_graph(case_id, slice_list, config):
    all_regions = []
    total_slices = max(s["slice_idx"] for s in slice_list) - min(s["slice_idx"] for s in slice_list) + 1

    for s_info in slice_list:
        mask = np.load(str(s_info["mask_file"]))
        raw = np.load(str(s_info["raw_file"]))
        regions = extract_regions_multiclass(
            mask, raw, s_info["slice_idx"], total_slices, config
        )
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
                dist / config["img_size"],
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



print("\nBuilding 3D volumetric graphs...")
graphs = []
graph_case_ids = []
graph_splits = []

for case_id, slice_list in patient_slices.items():
    g = build_3d_graph(case_id, slice_list, GNN_CONFIG)
    g.case_id = case_id
    graphs.append(g)
    graph_case_ids.append(case_id)
    graph_splits.append(slice_list[0]["split"])

print(f"Built {len(graphs)} 3D volumetric graphs")

node_counts = [g.x.size(0) for g in graphs]
edge_counts = [g.edge_index.size(1) for g in graphs]
print(f"Nodes per graph: min={min(node_counts)}, max={max(node_counts)}, mean={np.mean(node_counts):.1f}")
print(f"Edges per graph: min={min(edge_counts)}, max={max(edge_counts)}, mean={np.mean(edge_counts):.1f}")

inter_count = 0
intra_count = 0
for g in graphs:
    if g.edge_index.size(1) == 0:
        continue
    for e in range(g.edge_index.size(1)):
        si = g.edge_index[0, e].item()
        di = g.edge_index[1, e].item()
        if g.slice_ids[si].item() != g.slice_ids[di].item():
            inter_count += 1
        else:
            intra_count += 1
print(f"Edge types: intra-slice={intra_count:,}, inter-slice={inter_count:,} ({100*inter_count/max(intra_count+inter_count,1):.1f}%)")

tissue_dist = defaultdict(int)
for g in graphs:
    for t in g.tissue_labels.numpy():
        tissue_dist[GNN_CONFIG["tissue_labels"][t]] += 1
print(f"Node tissue distribution: {dict(tissue_dist)}")

train_graphs = [g for g, s in zip(graphs, graph_splits) if s == "train"]
val_graphs = [g for g, s in zip(graphs, graph_splits) if s == "val"]
test_graphs = [g for g, s in zip(graphs, graph_splits) if s == "test"]
print(f"Split: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")



fig = plt.figure(figsize=(18, 6))

for plot_idx in range(min(3, len(graphs))):
    g = graphs[plot_idx]
    if g.edge_index.size(1) == 0:
        continue

    ax = fig.add_subplot(1, 3, plot_idx + 1, projection='3d')
    pos = g.pos.numpy()
    tl = g.tissue_labels.numpy()

    colors = {1: 'red', 2: 'green', 3: 'gold'}
    for tissue in [1, 2, 3]:
        mask = tl == tissue
        if mask.sum() > 0:
            ax.scatter(
                pos[mask, 0], pos[mask, 1], pos[mask, 2],
                c=colors[tissue], s=40, alpha=0.8,
                label=GNN_CONFIG["tissue_labels"][tissue],
            )

    ei = g.edge_index.numpy()
    for e in range(ei.shape[1]):
        si, di = ei[0, e], ei[1, e]
        is_inter = g.slice_ids[si].item() != g.slice_ids[di].item()
        color = 'blue' if is_inter else 'gray'
        alpha = 0.6 if is_inter else 0.15
        ax.plot(
            [pos[si, 0], pos[di, 0]],
            [pos[si, 1], pos[di, 1]],
            [pos[si, 2], pos[di, 2]],
            color=color, alpha=alpha, linewidth=0.5,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Slice")
    ax.set_title(f"{g.case_id[-7:]} | {g.x.size(0)} nodes", fontsize=9)
    ax.legend(fontsize=7)

plt.suptitle("3D Tumor Graphs (blue=inter-slice, gray=intra-slice)", fontsize=13)
plt.tight_layout()
plt.show()

print("\n=== Phase 1 Complete: 3D Graph Construction ===")



class StructuralFeatureComputer:
    def compute(self, edge_index, num_nodes, candidate_edges, tissue_labels=None, slice_ids=None):
        adj = defaultdict(set)
        for e in range(edge_index.size(1)):
            s, d = edge_index[0, e].item(), edge_index[1, e].item()
            adj[s].add(d)
            adj[d].add(s)

        max_deg = max((len(v) for v in adj.values()), default=1)
        feats = []
        cn_indices_list = []

        for e in range(candidate_edges.size(1)):
            i = candidate_edges[0, e].item()
            j = candidate_edges[1, e].item()

            cn = adj[i] & adj[j]
            cn_count = len(cn) / max(max_deg, 1)

            union = adj[i] | adj[j]
            jaccard = len(cn) / max(len(union), 1)

            aa = 0.0
            for w in cn:
                deg_w = len(adj[w])
                if deg_w > 1:
                    aa += 1.0 / np.log(deg_w)

            feats.append([cn_count, jaccard, aa])
            cn_indices_list.append(list(cn))

        return torch.tensor(feats, dtype=torch.float32), cn_indices_list

sf_computer = StructuralFeatureComputer()
print("StructuralFeatureComputer initialized")



class NCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, heads=4, edge_dim=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout, concat=True))
            self.norms.append(LayerNorm(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        x = self.input_proj(x)
        alphas = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            if return_attention:
                x, (_, alpha) = conv(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                alphas.append(alpha)
            else:
                x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = x + residual
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.out_proj(x)
        return out, alphas


class NCNEdgeDecoder(nn.Module):
    def __init__(self, embed_dim, structural_feat_dim=3, num_tissue_types=3):
        super().__init__()
        self.sf_proj = nn.Linear(structural_feat_dim, embed_dim)
        self.tissue_pair_embed = nn.Embedding(num_tissue_types * num_tissue_types, embed_dim)
        self.edge_type_embed = nn.Embedding(2, 32)
        self.num_tissue_types = num_tissue_types

        base_dim = embed_dim + embed_dim * 2 + embed_dim + embed_dim
        cat_dim = base_dim + embed_dim + 32

        self.mlp = nn.Sequential(
            nn.LayerNorm(cat_dim),
            nn.Linear(cat_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, z, edge_index, structural_feats, cn_indices_list,
                src_tissue, dst_tissue, is_inter_slice):
        src_z = z[edge_index[0]]
        dst_z = z[edge_index[1]]

        hadamard = src_z * dst_z
        pair_cat = torch.cat([src_z, dst_z], dim=1)

        cn_pool = torch.zeros(edge_index.size(1), z.size(1), device=z.device)
        for i, cn_list in enumerate(cn_indices_list):
            if len(cn_list) > 0:
                cn_embeds = z[cn_list]
                cn_pool[i] = cn_embeds.mean(dim=0)

        sf_embed = self.sf_proj(structural_feats.to(z.device))

        tissue_pair_idx = (src_tissue - 1) * self.num_tissue_types + (dst_tissue - 1)
        tissue_pair_idx = tissue_pair_idx.clamp(0, self.num_tissue_types ** 2 - 1).to(z.device)
        tp_embed = self.tissue_pair_embed(tissue_pair_idx)

        et_embed = self.edge_type_embed(is_inter_slice.long().to(z.device))

        combined = torch.cat([hadamard, pair_cat, cn_pool, sf_embed, tp_embed, et_embed], dim=1)
        return self.mlp(combined).squeeze(-1)


class NCNEdgePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = NCNEncoder(
            in_dim=config["node_feat_dim"],
            hidden_dim=config["hidden_dim"],
            out_dim=config["embed_dim"],
            num_layers=config["num_layers"],
            heads=config["num_heads"],
            edge_dim=config["edge_attr_dim"],
        )
        self.decoder = NCNEdgeDecoder(
            embed_dim=config["embed_dim"],
            structural_feat_dim=config["structural_feat_dim"],
        )

    def encode(self, data, return_attention=False):
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.size(0) > 0 else None
        return self.encoder(data.x, data.edge_index, edge_attr=edge_attr, return_attention=return_attention)

    def decode(self, z, edge_index, structural_feats, cn_indices_list,
               src_tissue, dst_tissue, is_inter_slice):
        return self.decoder(z, edge_index, structural_feats, cn_indices_list,
                            src_tissue, dst_tissue, is_inter_slice)

    def forward(self, data, pos_ei, neg_ei, pos_sf, neg_sf, pos_cn, neg_cn,
                pos_src_t, pos_dst_t, neg_src_t, neg_dst_t,
                pos_inter, neg_inter, return_attention=False):
        z, alphas = self.encode(data, return_attention=return_attention)
        pos_pred = self.decode(z, pos_ei, pos_sf, pos_cn, pos_src_t, pos_dst_t, pos_inter)
        neg_pred = self.decode(z, neg_ei, neg_sf, neg_cn, neg_src_t, neg_dst_t, neg_inter)
        return pos_pred, neg_pred, z, alphas


print("NCNEdgePredictor initialized")
model = NCNEdgePredictor(GNN_CONFIG).to(GNN_CONFIG["device"])
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

print("\n=== Phase 2 & 3 Complete: Structural Features + NCN Architecture ===")



def get_edge_metadata(data, edge_index):
    src_tissue = data.tissue_labels[edge_index[0]]
    dst_tissue = data.tissue_labels[edge_index[1]]
    is_inter = (data.slice_ids[edge_index[0]] != data.slice_ids[edge_index[1]]).long()
    return src_tissue, dst_tissue, is_inter


def degree_biased_negative_sampling(data, num_neg):
    num_nodes = data.x.size(0)
    if num_nodes < 2:
        return torch.zeros(2, 0, dtype=torch.long)
    deg = torch.zeros(num_nodes)
    for e in range(data.edge_index.size(1)):
        deg[data.edge_index[0, e]] += 1
    probs = deg + 1
    probs = probs / probs.sum()

    edge_set = set()
    for e in range(data.edge_index.size(1)):
        s, d = data.edge_index[0, e].item(), data.edge_index[1, e].item()
        edge_set.add((s, d))
        edge_set.add((d, s))

    neg_src, neg_dst = [], []
    attempts = 0
    while len(neg_src) < num_neg and attempts < num_neg * 10:
        s = torch.multinomial(probs, 1).item()
        d = torch.multinomial(probs, 1).item()
        if s != d and (s, d) not in edge_set:
            neg_src.append(s)
            neg_dst.append(d)
            edge_set.add((s, d))
        attempts += 1

    if len(neg_src) == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([neg_src, neg_dst], dtype=torch.long)


optimizer = torch.optim.AdamW(model.parameters(), lr=GNN_CONFIG["gnn_lr"], weight_decay=GNN_CONFIG["gnn_weight_decay"])
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=GNN_CONFIG["gnn_lr"],
    total_steps=GNN_CONFIG["gnn_epochs"] * len(train_graphs),
)

best_val_auc = 0.0
best_model_path = GNN_CONFIG["brats_output_dir"] / "best_gnn.pt"

print(f"\nStarting GNN training for {GNN_CONFIG['gnn_epochs']} epochs on {len(train_graphs)} train graphs...")

for epoch in range(GNN_CONFIG["gnn_epochs"]):
    model.train()
    epoch_loss = 0.0
    graphs_trained = 0

    for g in train_graphs:
        if g.edge_index.size(1) < 2 or g.x.size(0) < 3:
            continue

        data = g.to(GNN_CONFIG["device"])
        pos_ei = data.edge_index
        num_pos = pos_ei.size(1)

        neg_ei = degree_biased_negative_sampling(g, num_pos).to(GNN_CONFIG["device"])
        if neg_ei.size(1) == 0:
            continue

        pos_sf, pos_cn = sf_computer.compute(g.edge_index, g.x.size(0), pos_ei)
        neg_sf, neg_cn = sf_computer.compute(g.edge_index, g.x.size(0), neg_ei)

        pos_src_t, pos_dst_t, pos_inter = get_edge_metadata(g, pos_ei)
        neg_src_t, neg_dst_t, neg_inter = get_edge_metadata(g, neg_ei)

        pos_pred, neg_pred, z, _ = model(
            data, pos_ei, neg_ei,
            pos_sf, neg_sf, pos_cn, neg_cn,
            pos_src_t.to(GNN_CONFIG["device"]), pos_dst_t.to(GNN_CONFIG["device"]),
            neg_src_t.to(GNN_CONFIG["device"]), neg_dst_t.to(GNN_CONFIG["device"]),
            pos_inter.to(GNN_CONFIG["device"]), neg_inter.to(GNN_CONFIG["device"]),
        )

        pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
        loss = pos_loss + neg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        graphs_trained += 1

    avg_loss = epoch_loss / max(graphs_trained, 1)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        all_labels, all_scores = [], []
        inter_labels, inter_scores = [], []
        intra_labels, intra_scores = [], []

        with torch.no_grad():
            for g in val_graphs:
                if g.edge_index.size(1) < 2 or g.x.size(0) < 3:
                    continue

                data = g.to(GNN_CONFIG["device"])
                pos_ei = data.edge_index
                neg_ei = degree_biased_negative_sampling(g, pos_ei.size(1)).to(GNN_CONFIG["device"])
                if neg_ei.size(1) == 0:
                    continue

                pos_sf, pos_cn = sf_computer.compute(g.edge_index, g.x.size(0), pos_ei)
                neg_sf, neg_cn = sf_computer.compute(g.edge_index, g.x.size(0), neg_ei)
                pos_src_t, pos_dst_t, pos_inter = get_edge_metadata(g, pos_ei)
                neg_src_t, neg_dst_t, neg_inter = get_edge_metadata(g, neg_ei)

                pos_pred, neg_pred, _, _ = model(
                    data, pos_ei, neg_ei,
                    pos_sf, neg_sf, pos_cn, neg_cn,
                    pos_src_t.to(GNN_CONFIG["device"]), pos_dst_t.to(GNN_CONFIG["device"]),
                    neg_src_t.to(GNN_CONFIG["device"]), neg_dst_t.to(GNN_CONFIG["device"]),
                    pos_inter.to(GNN_CONFIG["device"]), neg_inter.to(GNN_CONFIG["device"]),
                )

                pos_s = torch.sigmoid(pos_pred).cpu().numpy()
                neg_s = torch.sigmoid(neg_pred).cpu().numpy()
                all_scores.extend(pos_s.tolist() + neg_s.tolist())
                all_labels.extend([1] * len(pos_s) + [0] * len(neg_s))

                for i, s in enumerate(pos_s):
                    if pos_inter[i].item() == 1:
                        inter_scores.append(s)
                        inter_labels.append(1)
                    else:
                        intra_scores.append(s)
                        intra_labels.append(1)
                for i, s in enumerate(neg_s):
                    if neg_inter[i].item() == 1:
                        inter_scores.append(s)
                        inter_labels.append(0)
                    else:
                        intra_scores.append(s)
                        intra_labels.append(0)

        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_scores)
            ap = average_precision_score(all_labels, all_scores)
            inter_auc = roc_auc_score(inter_labels, inter_scores) if len(set(inter_labels)) > 1 else 0.0
            intra_auc = roc_auc_score(intra_labels, intra_scores) if len(set(intra_labels)) > 1 else 0.0

            print(
                f"Epoch {epoch+1:3d}/{GNN_CONFIG['gnn_epochs']} | "
                f"Loss: {avg_loss:.4f} | AUC: {auc:.4f} | AP: {ap:.4f} | "
                f"Intra-AUC: {intra_auc:.4f} | Inter-AUC: {inter_auc:.4f}"
            )

            if auc > best_val_auc:
                best_val_auc = auc
                torch.save(model.state_dict(), str(best_model_path))
                print(f"  -> Best model saved (AUC: {auc:.4f})")

print(f"\nTraining complete. Best val AUC: {best_val_auc:.4f}")



print("\n=== Phase 5: Reasoning Traces & Visualization ===")

model.load_state_dict(torch.load(str(best_model_path), weights_only=True))
model.eval()

test_auc_all, test_ap_all = [], []
test_auc_inter, test_auc_intra = [], []
tissue_pair_scores = defaultdict(list)

all_test_labels, all_test_scores = [], []

with torch.no_grad():
    for g in test_graphs:
        if g.edge_index.size(1) < 2 or g.x.size(0) < 3:
            continue

        data = g.to(GNN_CONFIG["device"])
        pos_ei = data.edge_index
        neg_ei = degree_biased_negative_sampling(g, pos_ei.size(1)).to(GNN_CONFIG["device"])
        if neg_ei.size(1) == 0:
            continue

        pos_sf, pos_cn = sf_computer.compute(g.edge_index, g.x.size(0), pos_ei)
        neg_sf, neg_cn = sf_computer.compute(g.edge_index, g.x.size(0), neg_ei)
        pos_src_t, pos_dst_t, pos_inter = get_edge_metadata(g, pos_ei)
        neg_src_t, neg_dst_t, neg_inter = get_edge_metadata(g, neg_ei)

        pos_pred, neg_pred, z, _ = model(
            data, pos_ei, neg_ei,
            pos_sf, neg_sf, pos_cn, neg_cn,
            pos_src_t.to(GNN_CONFIG["device"]), pos_dst_t.to(GNN_CONFIG["device"]),
            neg_src_t.to(GNN_CONFIG["device"]), neg_dst_t.to(GNN_CONFIG["device"]),
            pos_inter.to(GNN_CONFIG["device"]), neg_inter.to(GNN_CONFIG["device"]),
        )

        pos_s = torch.sigmoid(pos_pred).cpu().numpy()
        neg_s = torch.sigmoid(neg_pred).cpu().numpy()
        labels = [1] * len(pos_s) + [0] * len(neg_s)
        scores = pos_s.tolist() + neg_s.tolist()
        all_test_labels.extend(labels)
        all_test_scores.extend(scores)

        if len(set(labels)) > 1:
            test_auc_all.append(roc_auc_score(labels, scores))
            test_ap_all.append(average_precision_score(labels, scores))

        for i in range(len(pos_s)):
            src_t = GNN_CONFIG["tissue_labels"][pos_src_t[i].item()]
            dst_t = GNN_CONFIG["tissue_labels"][pos_dst_t[i].item()]
            pair = f"{src_t}→{dst_t}"
            tissue_pair_scores[pair].append(float(pos_s[i]))

print(f"\nTest Results:")
print(f"  AUC-ROC: {np.mean(test_auc_all):.4f} ± {np.std(test_auc_all):.4f}")
print(f"  AP:      {np.mean(test_ap_all):.4f} ± {np.std(test_ap_all):.4f}")
print(f"\nTissue-pair mean confidence:")
for pair, scores in sorted(tissue_pair_scores.items(), key=lambda x: -np.mean(x[1])):
    print(f"  {pair}: {np.mean(scores):.3f} (n={len(scores)})")


class ReasoningTraceGenerator:
    def __init__(self, config, tissue_labels):
        self.config = config
        self.tissue_labels = tissue_labels

    def generate(self, data, z, pos_pred, sf, cn_list, src_tissue, dst_tissue, is_inter, top_k=10):
        scores = torch.sigmoid(pos_pred).cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]
        traces = []

        for idx in top_indices:
            i = data.edge_index[0, idx].item()
            j = data.edge_index[1, idx].item()
            conf = scores[idx]
            src_t = self.tissue_labels[src_tissue[idx].item()]
            dst_t = self.tissue_labels[dst_tissue[idx].item()]
            src_slice = data.slice_ids[i].item()
            dst_slice = data.slice_ids[j].item()
            is_cross = "INTER-SLICE" if is_inter[idx].item() == 1 else "INTRA-SLICE"

            dx = data.pos[j, 0] - data.pos[i, 0]
            dy = data.pos[j, 1] - data.pos[i, 1]
            centroid_dist = float(torch.sqrt(dx ** 2 + dy ** 2))

            cn_count = len(cn_list[idx])
            jac = sf[idx, 1].item()
            aa = sf[idx, 2].item()

            src_feats = data.x[i]
            dst_feats = data.x[j]
            src_t1c_mean = src_feats[15].item()
            dst_t1c_mean = dst_feats[15].item()
            src_t2f_mean = dst_feats[23].item()

            trace = (
                f"Edge ({i}→{j}): {'strong' if conf > 0.7 else 'moderate' if conf > 0.4 else 'weak'} "
                f"link (conf={conf:.3f}). [{is_cross}: slice {src_slice}→{dst_slice}]\n"
                f"  Source: {src_t} (T1c_mean={src_t1c_mean:.3f}), "
                f"Target: {dst_t} (T1c_mean={dst_t1c_mean:.3f})\n"
                f"  Reasoning: {src_t}→{dst_t} tissue pair; "
                f"centroid_dist={centroid_dist:.1f}px; "
                f"CN={cn_count}, Jaccard={jac:.3f}, AA={aa:.3f}"
            )

            if is_inter[idx].item() == 1:
                trace += f"\n  3D context: cross-slice gap={abs(src_slice-dst_slice)} slice(s)"

            traces.append(trace)

        return traces


trace_gen = ReasoningTraceGenerator(GNN_CONFIG, GNN_CONFIG["tissue_labels"])

demo_graph = None
for g in test_graphs:
    if g.edge_index.size(1) >= 10 and g.x.size(0) >= 5:
        demo_graph = g
        break
if demo_graph is None and len(test_graphs) > 0:
    demo_graph = test_graphs[0]

if demo_graph is not None and demo_graph.edge_index.size(1) >= 2:
    data = demo_graph.to(GNN_CONFIG["device"])
    pos_ei = data.edge_index
    pos_sf, pos_cn = sf_computer.compute(demo_graph.edge_index, demo_graph.x.size(0), pos_ei)
    pos_src_t, pos_dst_t, pos_inter = get_edge_metadata(demo_graph, pos_ei)

    z, alphas = model.encode(data, return_attention=True)
    pos_pred = model.decode(
        z, pos_ei, pos_sf, pos_cn,
        pos_src_t.to(GNN_CONFIG["device"]),
        pos_dst_t.to(GNN_CONFIG["device"]),
        pos_inter.to(GNN_CONFIG["device"]),
    )

    traces = trace_gen.generate(
        demo_graph, z, pos_pred, pos_sf, pos_cn,
        pos_src_t, pos_dst_t, pos_inter, top_k=8,
    )

    print(f"\n--- Reasoning Traces for {demo_graph.case_id} ---")
    for t in traces:
        print(t)
        print()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    unique_slices = sorted(demo_graph.slice_ids.unique().numpy())
    mid_slice = unique_slices[len(unique_slices) // 2]
    node_mask = demo_graph.slice_ids.numpy() == mid_slice
    pos = demo_graph.pos.numpy()
    tl = demo_graph.tissue_labels.numpy()
    colors = {1: 'red', 2: 'green', 3: 'gold'}
    for tissue in [1, 2, 3]:
        m = node_mask & (tl == tissue)
        if m.sum() > 0:
            ax.scatter(pos[m, 0], pos[m, 1], c=colors[tissue], s=80, alpha=0.9,
                       label=GNN_CONFIG["tissue_labels"][tissue], edgecolors='black', linewidths=0.5)
    pred_scores = torch.sigmoid(pos_pred).cpu().numpy()
    ei = demo_graph.edge_index.numpy()
    for e in range(ei.shape[1]):
        si, di = ei[0, e], ei[1, e]
        if node_mask[si] and node_mask[di]:
            conf = pred_scores[e]
            color = plt.cm.RdYlGn(conf)
            ax.plot([pos[si, 0], pos[di, 0]], [pos[si, 1], pos[di, 1]],
                    color=color, alpha=0.7, linewidth=max(conf * 3, 0.5))
    ax.set_title(f"Slice {mid_slice} | Graph Overlay", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

    ax2 = axes[1]
    pair_means = {}
    for pair, scores in tissue_pair_scores.items():
        pair_means[pair] = np.mean(scores)
    if pair_means:
        pairs = list(pair_means.keys())
        vals = [pair_means[p] for p in pairs]
        bar_colors = ['#e74c3c' if 'ET' in p else '#2ecc71' if 'ED' in p else '#3498db' for p in pairs]
        ax2.barh(pairs, vals, color=bar_colors, alpha=0.8)
        ax2.set_xlabel("Mean Confidence")
        ax2.set_title("Tissue-Pair Link Confidence", fontsize=10)
        ax2.set_xlim(0, 1)

    ax3 = axes[2]
    if len(all_test_labels) > 0:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(all_test_labels, all_test_scores)
        overall_auc = roc_auc_score(all_test_labels, all_test_scores)
        ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f"AUC = {overall_auc:.3f}")
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax3.set_xlabel("FPR")
        ax3.set_ylabel("TPR")
        ax3.set_title("Test ROC Curve", fontsize=10)
        ax3.legend()

    plt.suptitle(f"3D GNN Analysis — {demo_graph.case_id}", fontsize=13)
    plt.tight_layout()
    plt.show()

print("\n=== BraTS 3D GNN Pipeline Complete ===")

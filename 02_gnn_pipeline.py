import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, LayerNorm, knn_graph
from torch_geometric.utils import negative_sampling, to_undirected
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
import torch.optim as optim


GNN_CONFIG = {
    "pipeline_output_dir": Path("pipeline_outputs"),
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 42,
    "node_feat_dim": 8,
    "hidden_dim": 128,
    "embed_dim": 64,
    "num_heads": 4,
    "num_layers": 3,
    "k_neighbors": 5,
    "gnn_epochs": 80,
    "gnn_lr": 5e-4,
    "gnn_weight_decay": 1e-4,
    "neg_sampling_ratio": 1.0,
    "min_nodes_per_graph": 3,
    "grid_split": 4,
    "img_size": 224,
    "structural_feat_dim": 3,
    "edge_attr_dim": 2,
}

torch.manual_seed(GNN_CONFIG["seed"])
np.random.seed(GNN_CONFIG["seed"])
random.seed(GNN_CONFIG["seed"])

print(f"Device: {GNN_CONFIG['device']}")

metadata_path = GNN_CONFIG["pipeline_output_dir"] / "metadata.pt"

if metadata_path.exists():
    metadata = torch.load(str(metadata_path), weights_only=False)
    mask_dir = GNN_CONFIG["pipeline_output_dir"] / "masks"
    masks = []
    image_paths = metadata["image_paths"]
    tumor_types = metadata["tumor_types"]
    type_confidences = metadata["type_confidences"]
    idx_to_class = metadata["idx_to_class"]

    for mf in metadata["mask_files"]:
        m = torch.load(str(mask_dir / mf), weights_only=True)
        masks.append(m)

    print(f"Loaded {len(masks)} masks from pipeline outputs")
    print(f"Tumor type distribution: {dict(zip(*np.unique(tumor_types, return_counts=True)))}")

else:
    print("No pipeline outputs found. Generating synthetic masks for standalone testing.")
    num_synthetic = 200
    masks = []
    image_paths = [f"synthetic_{i}.png" for i in range(num_synthetic)]
    tumor_types = [random.randint(0, 2) for _ in range(num_synthetic)]
    type_confidences = [random.uniform(0.7, 0.99) for _ in range(num_synthetic)]
    idx_to_class = {0: "glioma_tumor", 1: "meningioma_tumor", 2: "pituitary_tumor", 3: "no_tumor"}

    for _ in range(num_synthetic):
        mask = torch.zeros(GNN_CONFIG["img_size"], GNN_CONFIG["img_size"])
        num_blobs = random.randint(1, 4)
        for _ in range(num_blobs):
            cx = random.randint(40, GNN_CONFIG["img_size"] - 40)
            cy = random.randint(40, GNN_CONFIG["img_size"] - 40)
            rx = random.randint(10, 35)
            ry = random.randint(10, 35)
            y_coords, x_coords = torch.meshgrid(
                torch.arange(GNN_CONFIG["img_size"]),
                torch.arange(GNN_CONFIG["img_size"]),
                indexing="ij",
            )
            ellipse = ((x_coords - cx).float() / rx) ** 2 + ((y_coords - cy).float() / ry) ** 2
            mask[ellipse <= 1.0] = 1.0
        masks.append(mask)

    print(f"Generated {len(masks)} synthetic masks")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(min(4, len(masks))):
    axes[i].imshow(masks[i].numpy(), cmap="gray")
    t = idx_to_class.get(tumor_types[i], "unknown")
    axes[i].set_title(f"{t[:8]} | conf={type_confidences[i]:.2f}", fontsize=9)
    axes[i].axis("off")
plt.suptitle("Sample Segmentation Masks", fontsize=14)
plt.tight_layout()
plt.show()



class MaskToGraph:
    def __init__(self, config):
        self.config = config

    def extract_regions(self, mask_np):
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8
        )

        regions = []
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < 10:
                continue
            cx, cy = centroids[label_id]
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            component_mask = (labels == label_id).astype(np.uint8)
            mean_intensity = mask_np[component_mask == 1].mean() if component_mask.sum() > 0 else 0.0
            bbox_area = max(w * h, 1)
            solidity = area / bbox_area
            aspect_ratio = w / max(h, 1)

            regions.append({
                "centroid": (cx, cy),
                "area": area,
                "width": w,
                "height": h,
                "mean_intensity": float(mean_intensity),
                "aspect_ratio": aspect_ratio,
                "solidity": solidity,
            })

        return regions

    def split_into_grid(self, mask_np, n_splits):
        h, w = mask_np.shape
        regions = []
        cell_h = h // n_splits
        cell_w = w // n_splits
        for i in range(n_splits):
            for j in range(n_splits):
                cell = mask_np[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                if cell.sum() < 5:
                    continue
                area = float(cell.sum())
                cy = i * cell_h + cell_h / 2
                cx = j * cell_w + cell_w / 2
                regions.append({
                    "centroid": (cx, cy),
                    "area": area,
                    "width": float(cell_w),
                    "height": float(cell_h),
                    "mean_intensity": float(cell.mean()),
                    "aspect_ratio": float(cell_w) / max(float(cell_h), 1),
                    "solidity": area / max(cell_w * cell_h, 1),
                })
        return regions

    def regions_to_features(self, regions):
        img_size = self.config["img_size"]
        features = []
        for r in regions:
            feat = [
                r["centroid"][0] / img_size,
                r["centroid"][1] / img_size,
                r["area"] / (img_size * img_size),
                r["width"] / img_size,
                r["height"] / img_size,
                r["mean_intensity"],
                r["aspect_ratio"],
                r["solidity"],
            ]
            features.append(feat)
        return torch.tensor(features, dtype=torch.float32)

    def convert(self, mask_tensor):
        mask_np = mask_tensor.numpy()

        if mask_np.sum() < 5:
            x = torch.randn(1, self.config["node_feat_dim"])
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            pos = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
            return Data(x=x, edge_index=edge_index, pos=pos, edge_attr=torch.zeros(0, 2))

        regions = self.extract_regions(mask_np)

        if len(regions) < self.config["min_nodes_per_graph"]:
            grid_regions = self.split_into_grid(mask_np, self.config["grid_split"])
            if len(grid_regions) >= self.config["min_nodes_per_graph"]:
                regions = grid_regions
            else:
                while len(regions) < self.config["min_nodes_per_graph"]:
                    base = regions[0].copy()
                    base["centroid"] = (
                        base["centroid"][0] + random.uniform(-10, 10),
                        base["centroid"][1] + random.uniform(-10, 10),
                    )
                    base["area"] = base["area"] * random.uniform(0.5, 1.5)
                    regions.append(base)

        x = self.regions_to_features(regions)
        pos = torch.tensor([[r["centroid"][0], r["centroid"][1]] for r in regions], dtype=torch.float32)

        k = min(self.config["k_neighbors"], len(regions) - 1)
        if k < 1:
            k = 1
        edge_index = knn_graph(pos, k=k, loop=False)
        edge_index = to_undirected(edge_index)

        edge_attr = []
        for e in range(edge_index.size(1)):
            src, dst = edge_index[0, e].item(), edge_index[1, e].item()
            dx = pos[dst, 0] - pos[src, 0]
            dy = pos[dst, 1] - pos[src, 1]
            dist = torch.sqrt(dx ** 2 + dy ** 2).item()
            angle = float(np.arctan2(dy.item(), dx.item()))
            edge_attr.append([dist / self.config["img_size"], angle / np.pi])
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.zeros(0, 2)

        return Data(x=x, edge_index=edge_index, pos=pos, edge_attr=edge_attr)

converter = MaskToGraph(GNN_CONFIG)
graphs = []
for i, mask in enumerate(masks):
    g = converter.convert(mask)
    g.tumor_type = tumor_types[i]
    g.confidence = type_confidences[i]
    g.img_path = image_paths[i]
    graphs.append(g)

node_counts = [g.x.size(0) for g in graphs]
edge_counts = [g.edge_index.size(1) for g in graphs]
print(f"Converted {len(graphs)} masks to graphs")
print(f"Nodes per graph: min={min(node_counts)}, max={max(node_counts)}, mean={np.mean(node_counts):.1f}")
print(f"Edges per graph: min={min(edge_counts)}, max={max(edge_counts)}, mean={np.mean(edge_counts):.1f}")


class StructuralFeatureComputer:
    @staticmethod
    def get_neighbors(edge_index, num_nodes):
        adj = [set() for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            s, t = edge_index[0, i].item(), edge_index[1, i].item()
            adj[s].add(t)
            adj[t].add(s)
        return adj

    @staticmethod
    def compute(edge_index, num_nodes, candidate_edges):
        adj = StructuralFeatureComputer.get_neighbors(edge_index, num_nodes)
        degrees = [len(adj[n]) for n in range(num_nodes)]
        max_degree = max(degrees) if degrees else 1

        cn_counts = []
        jaccards = []
        adamic_adars = []
        cn_indices_list = []

        for i in range(candidate_edges.size(1)):
            s = candidate_edges[0, i].item()
            t = candidate_edges[1, i].item()
            cn = adj[s] & adj[t]
            cn_list = list(cn)

            cn_count = len(cn) / max(max_degree, 1)
            union_size = len(adj[s] | adj[t])
            jaccard = len(cn) / max(union_size, 1)
            aa = sum(1.0 / max(np.log(degrees[w]), 1e-6) for w in cn_list) if cn_list else 0.0
            aa = aa / max(num_nodes, 1)

            cn_counts.append(cn_count)
            jaccards.append(jaccard)
            adamic_adars.append(aa)
            cn_indices_list.append(cn_list)

        feats = torch.tensor(
            list(zip(cn_counts, jaccards, adamic_adars)),
            dtype=torch.float32,
        )
        return feats, cn_indices_list

sf_computer = StructuralFeatureComputer()
print("StructuralFeatureComputer initialized")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
sample_indices = random.sample(range(len(graphs)), min(4, len(graphs)))
for col, idx in enumerate(sample_indices):
    g = graphs[idx]
    mask_np = masks[idx].numpy()

    axes[0, col].imshow(mask_np, cmap="gray")
    pos_np = g.pos.numpy()
    axes[0, col].scatter(pos_np[:, 0], pos_np[:, 1], c="red", s=40, zorder=5)
    for e in range(g.edge_index.size(1)):
        s, t = g.edge_index[0, e].item(), g.edge_index[1, e].item()
        axes[0, col].plot(
            [pos_np[s, 0], pos_np[t, 0]],
            [pos_np[s, 1], pos_np[t, 1]],
            "c-", alpha=0.5, linewidth=1,
        )
    axes[0, col].set_title(f"Mask + Graph ({g.x.size(0)} nodes)", fontsize=9)
    axes[0, col].axis("off")

    G = nx.Graph()
    for n in range(g.x.size(0)):
        G.add_node(n, pos=(pos_np[n, 0], pos_np[n, 1]))
    for e in range(g.edge_index.size(1)):
        s, t = g.edge_index[0, e].item(), g.edge_index[1, e].item()
        if s < t:
            G.add_edge(s, t)
    nx_pos = nx.get_node_attributes(G, "pos")
    nx.draw(G, nx_pos, ax=axes[1, col], node_size=100, node_color="red",
            edge_color="cyan", with_labels=True, font_size=7)
    axes[1, col].set_title(f"Graph Structure", fontsize=9)

plt.suptitle("Phase 4: Graph Construction from Segmentation Masks", fontsize=14)
plt.tight_layout()
plt.show()



class NCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, heads=4, edge_dim=2, dropout=0.2):
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
    def __init__(self, embed_dim, structural_feat_dim=3):
        super().__init__()
        self.sf_proj = nn.Linear(structural_feat_dim, embed_dim)
        cat_dim = embed_dim + embed_dim * 2 + embed_dim + embed_dim
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

    def forward(self, z, edge_index, structural_feats, cn_indices_list):
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

        combined = torch.cat([hadamard, pair_cat, cn_pool, sf_embed], dim=1)
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

    def decode(self, z, edge_index, structural_feats, cn_indices_list):
        return self.decoder(z, edge_index, structural_feats, cn_indices_list)

    def forward(self, data, pos_edge_index, neg_edge_index, pos_sf, neg_sf, pos_cn, neg_cn, return_attention=False):
        z, alphas = self.encode(data, return_attention=return_attention)
        pos_pred = self.decode(z, pos_edge_index, pos_sf, pos_cn)
        neg_pred = self.decode(z, neg_edge_index, neg_sf, neg_cn)
        return pos_pred, neg_pred, z, alphas

print(f"NCNEdgePredictor initialized")
model = NCNEdgePredictor(GNN_CONFIG).to(GNN_CONFIG["device"])
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")



class ReasoningTraceGenerator:
    def __init__(self, idx_to_class):
        self.idx_to_class = idx_to_class

    def generate(self, data, z, edge_index, edge_probs, alphas=None, structural_feats=None, cn_indices_list=None):
        traces = []
        pos_np = data.pos.cpu().numpy() if data.pos is not None else None
        x_np = data.x.cpu().numpy()

        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            prob = torch.sigmoid(edge_probs[i]).item()

            trace = {
                "edge_id": i,
                "source_node": src,
                "target_node": dst,
                "prediction_confidence": round(prob, 4),
                "predicted_link": prob > 0.5,
            }

            if pos_np is not None:
                dx = pos_np[dst, 0] - pos_np[src, 0]
                dy = pos_np[dst, 1] - pos_np[src, 1]
                spatial_dist = float(np.sqrt(dx ** 2 + dy ** 2))
                trace["spatial_distance"] = round(spatial_dist, 2)
                trace["spatial_angle_deg"] = round(float(np.degrees(np.arctan2(dy, dx))), 1)

            src_area = x_np[src, 2]
            dst_area = x_np[dst, 2]
            trace["area_ratio"] = round(min(src_area, dst_area) / max(src_area, dst_area + 1e-8), 4)
            trace["src_solidity"] = round(float(x_np[src, 7]), 4)
            trace["dst_solidity"] = round(float(x_np[dst, 7]), 4)

            src_embed = z[src].detach().cpu().numpy()
            dst_embed = z[dst].detach().cpu().numpy()
            cosine_sim = float(np.dot(src_embed, dst_embed) / (
                np.linalg.norm(src_embed) * np.linalg.norm(dst_embed) + 1e-8
            ))
            trace["embedding_cosine_similarity"] = round(cosine_sim, 4)

            if structural_feats is not None and i < structural_feats.size(0):
                trace["cn_count"] = round(float(structural_feats[i, 0]), 4)
                trace["jaccard"] = round(float(structural_feats[i, 1]), 4)
                trace["adamic_adar"] = round(float(structural_feats[i, 2]), 4)

            if cn_indices_list is not None and i < len(cn_indices_list):
                trace["common_neighbors"] = cn_indices_list[i]

            if alphas is not None and len(alphas) > 0:
                layer_means = [round(float(a.mean().item()), 4) for a in alphas]
                trace["attention_per_layer"] = layer_means

            reasons = []
            if trace.get("spatial_distance", 0) < 30:
                reasons.append("spatially proximate regions")
            else:
                reasons.append("spatially distant regions")

            if trace["area_ratio"] > 0.7:
                reasons.append("similar region sizes")
            else:
                reasons.append("dissimilar region sizes")

            if cosine_sim > 0.8:
                reasons.append("high embedding similarity suggests shared morphological features")
            elif cosine_sim > 0.5:
                reasons.append("moderate embedding similarity")
            else:
                reasons.append("low embedding similarity suggests distinct morphological profiles")

            cn_count = trace.get("cn_count", 0)
            if cn_count > 0.3:
                reasons.append(f"strong common neighbor overlap (CN={cn_count:.2f}, Jaccard={trace.get('jaccard', 0):.2f})")
            elif cn_count > 0:
                reasons.append(f"weak common neighbor signal (CN={cn_count:.2f})")
            else:
                reasons.append("no common neighbors")

            if prob > 0.8:
                verdict = "strong link"
            elif prob > 0.5:
                verdict = "moderate link"
            else:
                verdict = "weak/no link"

            trace["reasoning_text"] = (
                f"Edge ({src}->{dst}): {verdict} (conf={prob:.2f}). "
                f"Reasoning: {'; '.join(reasons)}."
            )

            traces.append(trace)

        return traces

reasoner = ReasoningTraceGenerator(idx_to_class)
print("ReasoningTraceGenerator initialized")

def prepare_edge_split(data, train_ratio=0.8):
    edge_index = data.edge_index

    undirected_edges = []
    seen = set()
    for i in range(edge_index.size(1)):
        s, t = edge_index[0, i].item(), edge_index[1, i].item()
        key = (min(s, t), max(s, t))
        if key not in seen:
            seen.add(key)
            undirected_edges.append(key)

    random.shuffle(undirected_edges)
    split = int(len(undirected_edges) * train_ratio)
    train_edges = undirected_edges[:split]
    test_edges = undirected_edges[split:]

    if len(test_edges) == 0 and len(train_edges) > 1:
        test_edges = [train_edges.pop()]

    def edges_to_index(edges):
        if not edges:
            return torch.zeros(2, 0, dtype=torch.long)
        src = [e[0] for e in edges] + [e[1] for e in edges]
        dst = [e[1] for e in edges] + [e[0] for e in edges]
        return torch.tensor([src, dst], dtype=torch.long)

    return edges_to_index(train_edges), edges_to_index(test_edges)


optimizer = optim.AdamW(
    model.parameters(),
    lr=GNN_CONFIG["gnn_lr"],
    weight_decay=GNN_CONFIG["gnn_weight_decay"],
)

train_graphs = graphs[:int(0.8 * len(graphs))]
test_graphs = graphs[int(0.8 * len(graphs)):]
if len(test_graphs) == 0:
    test_graphs = [graphs[-1]]
    train_graphs = graphs[:-1]

print(f"Train graphs: {len(train_graphs)} | Test graphs: {len(test_graphs)}")

steps_per_epoch = max(len(train_graphs), 1)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=GNN_CONFIG["gnn_lr"],
    steps_per_epoch=steps_per_epoch,
    epochs=GNN_CONFIG["gnn_epochs"],
)

def degree_biased_negative_sampling(edge_index, num_nodes, num_neg_samples):
    deg = torch.zeros(num_nodes, dtype=torch.float)
    for i in range(edge_index.size(1)):
        deg[edge_index[0, i].item()] += 1
        deg[edge_index[1, i].item()] += 1
    prob = (deg + 1.0)
    prob = prob / prob.sum()
    neg_src = torch.multinomial(prob, num_neg_samples, replacement=True)
    neg_dst = torch.multinomial(prob, num_neg_samples, replacement=True)
    mask = neg_src != neg_dst
    neg_src, neg_dst = neg_src[mask], neg_dst[mask]
    if neg_src.size(0) == 0:
        return negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=num_neg_samples)
    return torch.stack([neg_src, neg_dst], dim=0)

for epoch in range(GNN_CONFIG["gnn_epochs"]):
    model.train()
    epoch_loss = 0.0
    random.shuffle(train_graphs)

    for data in train_graphs:
        data = data.to(GNN_CONFIG["device"])
        if data.edge_index.size(1) < 2:
            continue

        train_ei, _ = prepare_edge_split(data, train_ratio=0.8)
        train_ei = train_ei.to(GNN_CONFIG["device"])

        num_nodes = data.x.size(0)
        neg_ei = degree_biased_negative_sampling(
            train_ei, num_nodes=num_nodes,
            num_neg_samples=train_ei.size(1),
        ).to(GNN_CONFIG["device"])

        pos_sf, pos_cn = sf_computer.compute(data.edge_index.cpu(), num_nodes, train_ei.cpu())
        neg_sf, neg_cn = sf_computer.compute(data.edge_index.cpu(), num_nodes, neg_ei.cpu())

        optimizer.zero_grad()
        pos_pred, neg_pred, z, _ = model(data, train_ei, neg_ei, pos_sf, neg_sf, pos_cn, neg_cn)

        pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        test_auc_list, test_ap_list = [], []
        with torch.no_grad():
            for data in test_graphs:
                data = data.to(GNN_CONFIG["device"])
                if data.edge_index.size(1) < 2:
                    continue
                _, test_ei = prepare_edge_split(data, train_ratio=0.8)
                if test_ei.size(1) == 0:
                    test_ei = data.edge_index
                test_ei = test_ei.to(GNN_CONFIG["device"])

                neg_ei = negative_sampling(
                    test_ei, num_nodes=data.x.size(0),
                    num_neg_samples=test_ei.size(1),
                ).to(GNN_CONFIG["device"])

                num_nodes = data.x.size(0)
                pos_sf, pos_cn = sf_computer.compute(data.edge_index.cpu(), num_nodes, test_ei.cpu())
                neg_sf, neg_cn = sf_computer.compute(data.edge_index.cpu(), num_nodes, neg_ei.cpu())

                pos_pred, neg_pred, _, _ = model(data, test_ei, neg_ei, pos_sf, neg_sf, pos_cn, neg_cn)
                preds = torch.cat([torch.sigmoid(pos_pred), torch.sigmoid(neg_pred)]).cpu().numpy()
                labels = np.concatenate([np.ones(pos_pred.size(0)), np.zeros(neg_pred.size(0))])

                if len(np.unique(labels)) > 1:
                    test_auc_list.append(roc_auc_score(labels, preds))
                    test_ap_list.append(average_precision_score(labels, preds))

        avg_auc = np.mean(test_auc_list) if test_auc_list else 0.0
        avg_ap = np.mean(test_ap_list) if test_ap_list else 0.0
        print(
            f"[GNN] Epoch {epoch+1}/{GNN_CONFIG['gnn_epochs']} | "
            f"Loss: {epoch_loss/max(len(train_graphs),1):.4f} | Test AUC: {avg_auc:.4f} | AP: {avg_ap:.4f}"
        )

print("GNN training complete")



model.eval()
all_preds_gnn = []
all_labels_gnn = []
sample_traces = []

with torch.no_grad():
    for data in test_graphs:
        data = data.to(GNN_CONFIG["device"])
        if data.edge_index.size(1) < 2:
            continue

        pos_ei = data.edge_index
        neg_ei = negative_sampling(
            pos_ei, num_nodes=data.x.size(0),
            num_neg_samples=pos_ei.size(1),
        ).to(GNN_CONFIG["device"])

        num_nodes = data.x.size(0)
        pos_sf, pos_cn = sf_computer.compute(data.edge_index.cpu(), num_nodes, pos_ei.cpu())
        neg_sf, neg_cn = sf_computer.compute(data.edge_index.cpu(), num_nodes, neg_ei.cpu())

        pos_pred, neg_pred, z, alphas = model(data, pos_ei, neg_ei, pos_sf, neg_sf, pos_cn, neg_cn, return_attention=True)

        all_preds_gnn.extend(torch.sigmoid(pos_pred).cpu().numpy().tolist())
        all_labels_gnn.extend([1] * pos_pred.size(0))
        all_preds_gnn.extend(torch.sigmoid(neg_pred).cpu().numpy().tolist())
        all_labels_gnn.extend([0] * neg_pred.size(0))

        if len(sample_traces) < 3:
            all_ei = torch.cat([pos_ei, neg_ei], dim=1)
            all_pred = torch.cat([pos_pred, neg_pred])
            all_sf = torch.cat([pos_sf, neg_sf], dim=0)
            all_cn = pos_cn + neg_cn
            traces = reasoner.generate(data, z, all_ei, all_pred, alphas=alphas, structural_feats=all_sf, cn_indices_list=all_cn)
            sample_traces.append(traces[:5])

all_preds_gnn = np.array(all_preds_gnn)
all_labels_gnn = np.array(all_labels_gnn)

if len(np.unique(all_labels_gnn)) > 1:
    final_auc = roc_auc_score(all_labels_gnn, all_preds_gnn)
    final_ap = average_precision_score(all_labels_gnn, all_preds_gnn)
    print(f"Final Test AUC-ROC: {final_auc:.4f}")
    print(f"Final Test AP:      {final_ap:.4f}")

pred_binary = (all_preds_gnn > 0.5).astype(int)
cm_gnn = confusion_matrix(all_labels_gnn, pred_binary)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_gnn, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["No Edge", "Edge"])
ax.set_yticklabels(["No Edge", "Edge"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm_gnn[i, j]), ha="center", va="center", fontsize=14)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Edge Prediction Confusion Matrix")
plt.colorbar(im)
plt.tight_layout()
plt.show()

print("\nSample Reasoning Traces:")
for graph_idx, traces in enumerate(sample_traces):
    print(f"\n--- Graph {graph_idx + 1} ---")
    for t in traces:
        print(f"  {t['reasoning_text']}")
        print(f"    cosine_sim={t['embedding_cosine_similarity']}, "
              f"area_ratio={t['area_ratio']}, "
              f"spatial_dist={t.get('spatial_distance', 'N/A')}")



print("\n=== End-to-End Demo ===\n")

demo_indices = random.sample(range(len(graphs)), min(4, len(graphs)))

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

model.eval()
with torch.no_grad():
    for col, idx in enumerate(demo_indices):
        data = graphs[idx].to(GNN_CONFIG["device"])
        mask_np = masks[idx].numpy()

        z, alphas = model.encode(data, return_attention=True)

        pos_ei = data.edge_index
        num_nodes = data.x.size(0)
        pos_sf, pos_cn = sf_computer.compute(data.edge_index.cpu(), num_nodes, pos_ei.cpu())
        edge_preds = model.decode(z, pos_ei, pos_sf, pos_cn)
        edge_probs = torch.sigmoid(edge_preds)

        traces = reasoner.generate(data, z, pos_ei, edge_preds, alphas=alphas, structural_feats=pos_sf, cn_indices_list=pos_cn)

        pos_np = data.pos.cpu().numpy()
        probs_np = edge_probs.cpu().numpy()

        axes[0, col].imshow(mask_np, cmap="gray", alpha=0.5)
        axes[0, col].scatter(pos_np[:, 0], pos_np[:, 1], c="red", s=60, zorder=5)
        for e in range(pos_ei.size(1)):
            s, t = pos_ei[0, e].item(), pos_ei[1, e].item()
            prob = probs_np[e]
            color = plt.cm.RdYlGn(prob)
            axes[0, col].plot(
                [pos_np[s, 0], pos_np[t, 0]],
                [pos_np[s, 1], pos_np[t, 1]],
                color=color, alpha=0.8, linewidth=2,
            )
        t_type = idx_to_class.get(tumor_types[idx], "unknown")
        axes[0, col].set_title(f"{t_type[:10]} | {data.x.size(0)} nodes", fontsize=9)
        axes[0, col].axis("off")

        G = nx.Graph()
        for n in range(data.x.size(0)):
            G.add_node(n)
        edge_colors = []
        for e in range(pos_ei.size(1)):
            s, t = pos_ei[0, e].item(), pos_ei[1, e].item()
            if s < t:
                G.add_edge(s, t)
                edge_colors.append(probs_np[e])

        nx_pos = {n: (pos_np[n, 0], -pos_np[n, 1]) for n in range(data.x.size(0))}
        nx.draw(
            G, nx_pos, ax=axes[1, col],
            node_size=120, node_color="red",
            edge_color=edge_colors, edge_cmap=plt.cm.RdYlGn,
            edge_vmin=0, edge_vmax=1,
            with_labels=True, font_size=7, width=2,
        )
        axes[1, col].set_title(f"Edge Confidence (NCN)", fontsize=9)

plt.suptitle("End-to-End: Mask -> Graph -> NCN Edge Prediction", fontsize=14)
plt.tight_layout()
plt.show()

print("\nSample Reasoning Traces from Demo:")
demo_data = graphs[demo_indices[0]].to(GNN_CONFIG["device"])
with torch.no_grad():
    z, alphas = model.encode(demo_data, return_attention=True)
    num_nodes = demo_data.x.size(0)
    demo_sf, demo_cn = sf_computer.compute(demo_data.edge_index.cpu(), num_nodes, demo_data.edge_index.cpu())
    ep = model.decode(z, demo_data.edge_index, demo_sf, demo_cn)
    demo_traces = reasoner.generate(demo_data, z, demo_data.edge_index, ep, alphas=alphas, structural_feats=demo_sf, cn_indices_list=demo_cn)

for t in demo_traces[:5]:
    print(f"\n  {t['reasoning_text']}")
    print(f"    confidence={t['prediction_confidence']}, "
          f"cosine_sim={t['embedding_cosine_similarity']}, "
          f"cn={t.get('cn_count', 'N/A')}, jaccard={t.get('jaccard', 'N/A')}")

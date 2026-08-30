import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import knn_graph, GATv2Conv, LayerNorm
from sklearn.metrics import roc_auc_score, average_precision_score

from config import SHARED, GNN, SUPERVOXEL
from supervoxel import process_case_supervoxels


class IntraNodeAggregator(nn.Module):
    def __init__(self, sv_feat_dim=None, embed_dim=None, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        sv_feat_dim = sv_feat_dim or SUPERVOXEL["sv_feat_dim"]
        embed_dim = embed_dim or GNN["embed_dim"]

        self.input_proj = nn.Linear(sv_feat_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 2, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, sv_features_list, device=None):
        device = device or SHARED["device"]
        n_nodes = len(sv_features_list)

        if n_nodes == 0:
            return torch.zeros(0, self.embed_dim, device=device), []

        # Handle nodes with no SVs (fallback to zero embedding)
        sv_attentions = []
        node_embeds = []

        for node_idx in range(n_nodes):
            sv_feats = sv_features_list[node_idx]  # (K_i, sv_feat_dim)

            if sv_feats is None or sv_feats.size(0) == 0:
                node_embeds.append(torch.zeros(self.embed_dim, device=device))
                sv_attentions.append(torch.zeros(0, device=device))
                continue

            sv_feats = sv_feats.to(device)
            projected = self.input_proj(sv_feats)  # (K_i, embed_dim)

            # Prepend [CLS] token
            cls = self.cls_token.expand(1, -1, -1).squeeze(0).to(device)  # (1, embed_dim)
            seq = torch.cat([cls, projected], dim=0).unsqueeze(0)  # (1, K_i+1, embed_dim)

            # Transformer
            out = self.transformer(seq)  # (1, K_i+1, embed_dim)
            cls_out = out[0, 0]  # (embed_dim,) — [CLS] output
            node_embeds.append(self.out_proj(cls_out))

            # Compute attention weights: dot product of [CLS] with each SV
            sv_out = out[0, 1:]  # (K_i, embed_dim)
            attn = torch.matmul(sv_out, cls_out)  # (K_i,)
            attn = F.softmax(attn, dim=0)
            sv_attentions.append(attn.detach())

        node_embeds = torch.stack(node_embeds, dim=0)  # (N_seg, embed_dim)
        return node_embeds, sv_attentions


def build_inter_edges(pos_3d, tissue_labels, strategy="compatibility_only", k=None):
    """Build inter-node edges between segmentation components.

    Args:
        pos_3d: (N, 3) tensor of node centroids.
        tissue_labels: list of int, tissue type per node (1=NCR, 2=ED, 3=ET).
        strategy: "compatibility_only" or "knn_filtered".
        k: number of neighbors for knn_filtered strategy.

    Returns:
        edge_index: (2, E) tensor of undirected edges.
    """
    n_nodes = pos_3d.size(0)
    all_src, all_dst = [], []

    if strategy == "knn_filtered" and k is not None and k >= 1 and n_nodes >= 2:
        actual_k = min(k, n_nodes - 1)
        if actual_k >= 1:
            ei = knn_graph(pos_3d, k=actual_k, loop=False)
            # Filter by tissue compatibility
            for idx in range(ei.size(1)):
                ni, nj = ei[0, idx].item(), ei[1, idx].item()
                ti, tj = tissue_labels[ni], tissue_labels[nj]
                compatible = (ti == tj) or (abs(ti - tj) <= 1) or ({ti, tj} == {1, 3})
                if compatible:
                    all_src.append(ni)
                    all_dst.append(nj)

    elif strategy == "compatibility_only":
        # Connect ALL tissue-compatible pairs (no distance filter)
        for ni in range(n_nodes):
            for nj in range(ni + 1, n_nodes):
                ti = tissue_labels[ni]
                tj = tissue_labels[nj]
                compatible = (ti == tj) or (abs(ti - tj) <= 1) or ({ti, tj} == {1, 3})
                if compatible:
                    all_src.extend([ni, nj])
                    all_dst.extend([nj, ni])

    if len(all_src) == 0:
        return torch.zeros(2, 0, dtype=torch.long)

    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    edge_index = to_undirected(edge_index)
    return edge_index


def build_hierarchical_graph(case_id, raw_4ch_3d, seg_mask_3d, config=None):
    """Build a hierarchical graph: seg nodes with SV internal structure.

    1. Run supervoxel pipeline (3D SLIC + pruning + assignment)
    2. For each seg component, collect contained SV features (with relative PE)
    3. Build inter-node edges via configurable strategy
    4. Return Data with SV features per node for IntraNodeAggregator

    Args:
        case_id: patient identifier string.
        raw_4ch_3d: (4, H, W, D) float32 numpy array.
        seg_mask_3d: (H, W, D) int numpy array (0=BG, 1=NCR, 2=ED, 3=ET).
        config: optional config override.

    Returns:
        Data object with fields:
          - x: placeholder (N_seg, sv_feat_dim) -- mean SV features per node
          - edge_index, edge_attr, pos, tissue_labels, slice_ids: as before
          - sv_features: list of (K_i, 25) tensors per node
          - sv_edge_indices: list of (2, E_i) tensors per node
          - n_svs_per_node: list of int
    """
    config = config or GNN
    sv_config = SUPERVOXEL

    sv_result = process_case_supervoxels(case_id, raw_4ch_3d, seg_mask_3d, sv_config)
    components = sv_result["components"]

    if len(components) < 2:
        empty_dim = sv_config["sv_feat_dim"]
        return Data(
            x=torch.zeros(max(len(components), 1), empty_dim),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            pos=torch.tensor([[0.5, 0.5, 0.5]]),
            edge_attr=torch.zeros(0, config["edge_attr_dim"]),
            sv_features=[],
            sv_edge_indices=[],
            n_svs_per_node=[],
        )

    n_nodes = len(components)
    H, W, D = seg_mask_3d.shape

    sv_features_list = []
    sv_edge_indices_list = []
    n_svs_list = []
    positions = []
    tissue_labels = []
    slice_ids_list = []

    for comp in components:
        comp_id = comp["id"]
        positions.append(comp["centroid"])
        tissue_labels.append(comp["tissue_label"])

        coords = np.argwhere(comp["mask"])
        mean_z = float(coords[:, 2].mean()) if len(coords) > 0 else 0.0
        slice_ids_list.append(int(mean_z))

        if comp_id in sv_result["sv_features_per_comp"]:
            sv_feats = sv_result["sv_features_per_comp"][comp_id]
            sv_features_list.append(sv_feats)
            n_svs_list.append(sv_feats.size(0))
        else:
            sv_features_list.append(torch.zeros(0, sv_config["sv_feat_dim"]))
            n_svs_list.append(0)

        if comp_id in sv_result["sv_edges_per_comp"]:
            sv_edge_indices_list.append(sv_result["sv_edges_per_comp"][comp_id])
        else:
            sv_edge_indices_list.append(torch.zeros(2, 0, dtype=torch.long))

    x_list = []
    for sv_f in sv_features_list:
        if sv_f.size(0) > 0:
            x_list.append(sv_f.mean(dim=0))
        else:
            x_list.append(torch.zeros(sv_config["sv_feat_dim"]))
    x = torch.stack(x_list, dim=0)

    pos_3d = torch.tensor(np.stack(positions), dtype=torch.float32)
    tissue_labels_t = torch.tensor(tissue_labels, dtype=torch.long)
    slice_ids_t = torch.tensor(slice_ids_list, dtype=torch.long)

    # Inter-node edges via configurable strategy
    edge_strategy = config.get("edge_strategy", "compatibility_only")
    k = config.get("k_neighbors", 3)
    edge_index = build_inter_edges(pos_3d, tissue_labels, strategy=edge_strategy, k=k)

    if edge_index.size(1) == 0:
        edge_attr = torch.zeros(0, config["edge_attr_dim"])
    else:
        edge_attr = []
        for e in range(edge_index.size(1)):
            si = edge_index[0, e].item()
            di = edge_index[1, e].item()
            diff = pos_3d[di] - pos_3d[si]
            dist = float(torch.norm(diff))
            angle = float(np.arctan2(diff[1].item(), diff[0].item()))
            z_gap = abs(slice_ids_list[si] - slice_ids_list[di]) / max(D, 1)
            same_tissue = 1.0 if tissue_labels[si] == tissue_labels[di] else 0.0
            edge_attr.append([dist, angle / np.pi, z_gap, same_tissue])
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        pos=pos_3d,
        edge_attr=edge_attr,
        tissue_labels=tissue_labels_t,
        slice_ids=slice_ids_t,
        sv_features=sv_features_list,
        sv_edge_indices=sv_edge_indices_list,
        n_svs_per_node=n_svs_list,
    )
    return data


def load_metadata(config=None):
    """Load brats_outputs metadata and organize by patient."""
    config = config or GNN
    metadata = torch.load(str(config["brats_output_dir"] / "metadata.pt"), weights_only=False)
    masks_dir = config["brats_output_dir"] / "masks"
    raw_dir = config["brats_output_dir"] / "raw_slices"

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

    return patient_slices, metadata


def build_all_graphs(config=None):
    """Build hierarchical graphs for all patients and split into train/val/test.

    Loads 3D volumes and builds hierarchical graphs with supervoxel
    internal structure for each patient.
    """
    config = config or GNN
    patient_slices, metadata = load_metadata(config)

    print(f"Loaded {len(metadata['case_ids'])} slices from {len(patient_slices)} patients")

    vol_dir = config["brats_output_dir"] / "volumes"

    print("Building hierarchical volumetric graphs...")
    graphs, splits = [], []

    for case_id, slice_list in patient_slices.items():
        raw_path = vol_dir / f"{case_id}_raw4ch.npy"
        seg_path = vol_dir / f"{case_id}_seg.npy"

        if raw_path.exists() and seg_path.exists():
            raw_4ch = np.load(str(raw_path))
            seg_mask = np.load(str(seg_path))
            g = build_hierarchical_graph(case_id, raw_4ch, seg_mask, config)
            del raw_4ch, seg_mask  # free memory immediately
        else:
            print(f"  WARNING: 3D volumes missing for {case_id}, skipping")
            continue

        g.case_id = case_id
        graphs.append(g)
        splits.append(slice_list[0]["split"])

    print(f"Built {len(graphs)} graphs")
    node_counts = [g.x.size(0) for g in graphs]
    edge_counts = [g.edge_index.size(1) for g in graphs]
    print(f"  Nodes: min={min(node_counts)}, max={max(node_counts)}, mean={np.mean(node_counts):.1f}")
    print(f"  Edges: min={min(edge_counts)}, max={max(edge_counts)}, mean={np.mean(edge_counts):.1f}")

    # Report SV stats if hierarchical
    sv_counts = [sum(g.n_svs_per_node) for g in graphs if hasattr(g, 'n_svs_per_node') and g.n_svs_per_node]
    if sv_counts:
        print(f"  SVs/graph: min={min(sv_counts)}, max={max(sv_counts)}, mean={np.mean(sv_counts):.1f}")

    train_graphs = [g for g, s in zip(graphs, splits) if s == "train"]
    val_graphs = [g for g, s in zip(graphs, splits) if s == "val"]
    test_graphs = [g for g, s in zip(graphs, splits) if s == "test"]
    print(f"  Split: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

    return train_graphs, val_graphs, test_graphs


# ══════════════════════════════════════════════════════════════════════
# Sub-phase 5B: Structural Features & OCN Architecture
# ══════════════════════════════════════════════════════════════════════

# ── Structural Features ──────────────────────────────────────────────

class StructuralFeatureComputer:
    """OCN-enhanced structural features: CN, Jaccard, AA + orthogonalized residual + path normalization.

    Inter-node (5-dim per edge):
      [cn_count, jaccard, adamic_adar, ocn_residual_norm, path_norm_cn]

    Intra-node topology (4-dim per node):
      [cn_density, connectivity_ratio, cn_variance, spectral_gap]
    """

    def compute(self, edge_index, num_nodes, candidate_edges,
                node_embeddings=None, tissue_labels=None, slice_ids=None):
        """Compute OCN-enhanced structural features for candidate edges.

        If node_embeddings is provided, computes orthogonalized residuals.
        Otherwise falls back to 5-dim with zeros for OCN features.
        """
        adj = defaultdict(set)
        for e in range(edge_index.size(1)):
            s, d = edge_index[0, e].item(), edge_index[1, e].item()
            adj[s].add(d)
            adj[d].add(s)

        max_deg = max((len(v) for v in adj.values()), default=1)

        # Precompute 2-hop reachability counts for path normalization
        two_hop = defaultdict(int)
        for node in range(num_nodes):
            for n1 in adj[node]:
                for n2 in adj[n1]:
                    if n2 != node:
                        two_hop[(node, n2)] = two_hop.get((node, n2), 0) + 1

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

            # OCN: orthogonalized residual
            ocn_residual = 0.0
            if node_embeddings is not None and i < node_embeddings.size(0) and j < node_embeddings.size(0):
                z_i = node_embeddings[i].detach().cpu().numpy()
                z_j = node_embeddings[j].detach().cpu().numpy()

                # Raw CN signal: sum of CN node embeddings
                if len(cn) > 0:
                    cn_list = list(cn)
                    cn_embeds = node_embeddings[cn_list].detach().cpu().numpy()
                    cn_signal = cn_embeds.mean(axis=0)

                    # Project out component explained by endpoint embeddings
                    pair_basis = np.stack([z_i, z_j])  # (2, D)
                    # Orthogonal projection: cn_signal - proj(cn_signal onto span(z_i, z_j))
                    try:
                        Q, _ = np.linalg.qr(pair_basis.T)  # (D, 2)
                        proj = Q @ (Q.T @ cn_signal)
                        residual = cn_signal - proj
                        ocn_residual = float(np.linalg.norm(residual))
                    except np.linalg.LinAlgError:
                        ocn_residual = float(np.linalg.norm(cn_signal))

            # Path normalization: CN count / 2-hop reachability
            path_reach = max(two_hop.get((i, j), 0), two_hop.get((j, i), 0), 1)
            path_norm_cn = len(cn) / path_reach

            feats.append([cn_count, jaccard, aa, ocn_residual, path_norm_cn])
            cn_indices_list.append(list(cn))

        return torch.tensor(feats, dtype=torch.float32), cn_indices_list

    def compute_intra_node_topology(self, sv_edge_indices_list, n_svs_per_node):
        """Compute topological fingerprint for each node's internal SV graph.

        For each seg node, characterize its internal connectivity:
          - cn_density: average CN count among internal SV pairs
          - connectivity_ratio: actual edges / max possible edges
          - cn_variance: variance of per-SV degree (homogeneous vs heterogeneous)
          - spectral_gap: algebraic connectivity (2nd smallest Laplacian eigenvalue)

        Args:
            sv_edge_indices_list: list of (2, E_i) tensors per node.
            n_svs_per_node: list of int, SV count per node.

        Returns:
            topo_feats: (N_seg, 4) tensor.
        """
        n_nodes = len(sv_edge_indices_list)
        topo = []

        for node_idx in range(n_nodes):
            ei = sv_edge_indices_list[node_idx]
            n_svs = n_svs_per_node[node_idx] if node_idx < len(n_svs_per_node) else 0

            if n_svs < 2 or ei.size(1) == 0:
                topo.append([0.0, 0.0, 0.0, 0.0])
                continue

            # Build adjacency
            adj = defaultdict(set)
            for e in range(ei.size(1)):
                s, d = ei[0, e].item(), ei[1, e].item()
                adj[s].add(d)
                adj[d].add(s)

            # CN density: avg CN count among all SV pairs
            cn_counts = []
            nodes = list(range(n_svs))
            for a in nodes:
                for b in nodes:
                    if a < b:
                        cn_counts.append(len(adj[a] & adj[b]))
            cn_density = float(np.mean(cn_counts)) if cn_counts else 0.0

            # Connectivity ratio
            max_edges = n_svs * (n_svs - 1) / 2
            actual_edges = ei.size(1) / 2  # undirected
            connectivity = actual_edges / max(max_edges, 1)

            # Degree variance
            degrees = [len(adj[n]) for n in nodes]
            cn_var = float(np.std(degrees)) if degrees else 0.0

            # Spectral gap (algebraic connectivity)
            spectral_gap = 0.0
            if n_svs >= 3:
                try:
                    from scipy.sparse import csr_matrix
                    from scipy.sparse.csgraph import laplacian as sp_laplacian
                    row = ei[0].numpy()
                    col = ei[1].numpy()
                    data = np.ones(len(row))
                    A = csr_matrix((data, (row, col)), shape=(n_svs, n_svs))
                    L = sp_laplacian(A, normed=False).toarray()
                    eigenvalues = np.sort(np.linalg.eigvalsh(L))
                    if len(eigenvalues) >= 2:
                        spectral_gap = float(eigenvalues[1])
                except Exception:
                    spectral_gap = 0.0

            topo.append([cn_density, connectivity, cn_var, spectral_gap])

        return torch.tensor(topo, dtype=torch.float32)


# ── GATv2 Encoder ────────────────────────────────────────────────────

class GATv2Encoder(nn.Module):
    """3-layer GATv2 with residual connections and edge-attr awareness."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, heads=4, edge_dim=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(
                hidden_dim, hidden_dim // heads, heads=heads,
                edge_dim=edge_dim, dropout=dropout, concat=True,
            ))
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


# ── Multi-Signal Edge Decoder ────────────────────────────────────────

class MultiSignalDecoder(nn.Module):
    """6-signal decoder: Hadamard + concat + CN pool + structural + tissue-pair + edge-type."""

    def __init__(self, embed_dim, structural_feat_dim=3, num_tissue_types=3):
        super().__init__()
        self.sf_proj = nn.Linear(structural_feat_dim, embed_dim)
        self.tissue_pair_embed = nn.Embedding(num_tissue_types * num_tissue_types, embed_dim)
        self.edge_type_embed = nn.Embedding(2, 32)
        self.num_tissue_types = num_tissue_types

        # hadamard(64) + concat(128) + cn_pool(64) + sf(64) + tissue(64) + edge_type(32) = 416
        cat_dim = embed_dim + embed_dim * 2 + embed_dim + embed_dim + embed_dim + 32

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


# ── Full Model ───────────────────────────────────────────────────────

class EdgePredictor(nn.Module):
    """Hierarchical OCN: IntraNodeAggregator + GATv2 encoder + multi-signal decoder."""

    def __init__(self, config=None):
        super().__init__()
        config = config or GNN
        self.aggregator = IntraNodeAggregator(
            sv_feat_dim=SUPERVOXEL["sv_feat_dim"],
            embed_dim=config["embed_dim"],
        )
        self.encoder = GATv2Encoder(
            in_dim=config["node_feat_dim"],  # 68 = 64 embed + 4 topo (added in Phase 3)
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


    def encode(self, data, return_attention=False):
        """Encode node features via SV aggregation + topology + GATv2."""
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.size(0) > 0 else None
        device = data.x.device

        # Aggregate SVs into node embeddings
        node_embeds, sv_attns = self.aggregator(data.sv_features, device=device)

        # Compute intra-node topology (4-dim per node)
        sf_computer = StructuralFeatureComputer()
        sv_ei_list = data.sv_edge_indices if hasattr(data, 'sv_edge_indices') else []
        n_svs = data.n_svs_per_node if hasattr(data, 'n_svs_per_node') else []
        if sv_ei_list and n_svs:
            topo_feats = sf_computer.compute_intra_node_topology(sv_ei_list, n_svs)
            topo_feats = topo_feats.to(device)
        else:
            topo_feats = torch.zeros(node_embeds.size(0), 4, device=device)

        x = torch.cat([node_embeds, topo_feats], dim=-1)  # (N, 68)

        z, alphas = self.encoder(x, data.edge_index, edge_attr=edge_attr, return_attention=return_attention)
        return z, alphas, sv_attns

    def decode(self, z, edge_index, structural_feats, cn_indices_list,
               src_tissue, dst_tissue, is_inter_slice):
        return self.decoder(z, edge_index, structural_feats, cn_indices_list,
                            src_tissue, dst_tissue, is_inter_slice)

    def forward(self, data, pos_ei, neg_ei, pos_sf, neg_sf, pos_cn, neg_cn,
                pos_src_t, pos_dst_t, neg_src_t, neg_dst_t,
                pos_inter, neg_inter, return_attention=False):
        z, alphas, sv_attns = self.encode(data, return_attention=return_attention)
        pos_pred = self.decode(z, pos_ei, pos_sf, pos_cn, pos_src_t, pos_dst_t, pos_inter)
        neg_pred = self.decode(z, neg_ei, neg_sf, neg_cn, neg_src_t, neg_dst_t, neg_inter)
        return pos_pred, neg_pred, z, alphas

    def gradient_saliency(self, data, edge_idx, structural_feats,
                          cn_indices, src_tissue, dst_tissue, is_inter):
        """Compute gradient-based saliency for a target edge prediction.

        Backpropagates through the prediction logit to produce importance
        scores at three hierarchy levels:
          1. Node-level: gradient magnitude w.r.t. node embeddings
          2. SV-level: gradient magnitude w.r.t. SV features per endpoint
          3. Edge-level: gradient magnitude w.r.t. edge attributes

        Args:
            data: PyG Data object (must have sv_features).
            edge_idx: int, index into data.edge_index.
            structural_feats: (E, sf_dim) structural features tensor.
            cn_indices: list of lists, common neighbor indices.
            src_tissue, dst_tissue, is_inter: per-edge metadata tensors.

        Returns:
            dict with:
              node_saliency: (N,) per-node importance
              sv_saliency: {node_id: (K,) per-SV importance} for src/dst
              edge_saliency: (edge_attr_dim,) importance per edge feature
        """
        self.eval()
        device = next(self.parameters()).device
        data = data.to(device)

        # Enable gradients on SV features
        sv_grads_available = []
        if hasattr(data, 'sv_features') and data.sv_features:
            for sv_f in data.sv_features:
                if sv_f.size(0) > 0:
                    sv_f.requires_grad_(True)
                    sv_grads_available.append(sv_f)

        # Enable gradients on edge_attr
        edge_attr_grad = None
        if (hasattr(data, 'edge_attr') and data.edge_attr is not None
                and data.edge_attr.size(0) > 0):
            data.edge_attr.requires_grad_(True)
            edge_attr_grad = data.edge_attr

        # Forward
        z, _, sv_attns = self.encode(data, return_attention=False)

        # Target edge prediction
        ei = data.edge_index[:, edge_idx:edge_idx+1]
        sf = structural_feats[edge_idx:edge_idx+1].to(device)
        cn = [cn_indices[edge_idx]]
        st = src_tissue[edge_idx:edge_idx+1].to(device)
        dt = dst_tissue[edge_idx:edge_idx+1].to(device)
        inter = is_inter[edge_idx:edge_idx+1].to(device)

        logit = self.decode(z, ei, sf, cn, st, dt, inter)
        logit.backward()

        # ── Node-level saliency ──
        # z doesn't have grad (it's computed), so use the input projection's grad
        # We compute per-node importance from the SV feature gradients
        n_nodes = data.x.size(0)
        node_saliency = torch.zeros(n_nodes, device=device)

        src_node = ei[0, 0].item()
        dst_node = ei[1, 0].item()

        # ── SV-level saliency ──
        sv_saliency = {}
        for node in [src_node, dst_node]:
            if (hasattr(data, 'sv_features') and node < len(data.sv_features)
                    and data.sv_features[node].grad is not None):
                grad_mag = data.sv_features[node].grad.abs().sum(dim=-1)
                sv_saliency[node] = grad_mag.detach().cpu()
                node_saliency[node] = grad_mag.sum().item()

        # ── Edge-level saliency ──
        edge_saliency = torch.zeros(data.edge_attr.size(1) if edge_attr_grad is not None else 0)
        if edge_attr_grad is not None and edge_attr_grad.grad is not None:
            edge_saliency = edge_attr_grad.grad[edge_idx].abs().detach().cpu()

        # Normalize node saliency
        node_saliency = node_saliency.detach().cpu()
        if node_saliency.max() > 0:
            node_saliency = node_saliency / node_saliency.max()

        # Normalize SV saliency per node
        for node in sv_saliency:
            if sv_saliency[node].max() > 0:
                sv_saliency[node] = sv_saliency[node] / sv_saliency[node].max()

        # Clean up
        self.zero_grad()
        for sv_f in sv_grads_available:
            if sv_f.grad is not None:
                sv_f.grad = None
            sv_f.requires_grad_(False)
        if edge_attr_grad is not None:
            if edge_attr_grad.grad is not None:
                edge_attr_grad.grad = None
            edge_attr_grad.requires_grad_(False)

        return {
            "node_saliency": node_saliency,
            "sv_saliency": sv_saliency,
            "edge_saliency": edge_saliency,
            "src_node": src_node,
            "dst_node": dst_node,
            "sv_attentions": sv_attns,
        }


# ══════════════════════════════════════════════════════════════════════
# Sub-phase 5C: Training & Evaluation
# ══════════════════════════════════════════════════════════════════════

# ── Training Utilities ────────────────────────────────────────────────

def get_edge_metadata(data, edge_index):
    """Extract tissue types and intra/inter-slice flag for each edge."""
    src_tissue = data.tissue_labels[edge_index[0]]
    dst_tissue = data.tissue_labels[edge_index[1]]
    is_inter = (data.slice_ids[edge_index[0]] != data.slice_ids[edge_index[1]]).long()
    return src_tissue, dst_tissue, is_inter


def degree_biased_negative_sampling(data, num_neg):
    """Sample negative edges biased toward high-degree nodes (harder negatives)."""
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


# ── Training Loop ────────────────────────────────────────────────────

def train_gnn(model, train_graphs, val_graphs, config=None):
    """Full GNN training loop with per-graph forward/backward."""
    config = config or GNN
    device = SHARED["device"]
    sf_computer = StructuralFeatureComputer()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["lr"],
        total_steps=config["epochs"] * len(train_graphs),
    )

    best_val_auc = 0.0
    checkpoint_path = config["checkpoint"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining GNN for {config['epochs']} epochs on {len(train_graphs)} graphs...")

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss, graphs_trained = 0.0, 0

        for g in train_graphs:
            if g.edge_index.size(1) < 2 or g.x.size(0) < 3:
                continue

            data = g.to(device)
            pos_ei = data.edge_index
            neg_ei = degree_biased_negative_sampling(g, pos_ei.size(1)).to(device)
            if neg_ei.size(1) == 0:
                continue

            pos_sf, pos_cn = sf_computer.compute(g.edge_index, g.x.size(0), pos_ei)
            neg_sf, neg_cn = sf_computer.compute(g.edge_index, g.x.size(0), neg_ei)
            pos_src_t, pos_dst_t, pos_inter = get_edge_metadata(g, pos_ei)
            neg_src_t, neg_dst_t, neg_inter = get_edge_metadata(g, neg_ei)

            pos_pred, neg_pred, z, _ = model(
                data, pos_ei, neg_ei,
                pos_sf, neg_sf, pos_cn, neg_cn,
                pos_src_t.to(device), pos_dst_t.to(device),
                neg_src_t.to(device), neg_dst_t.to(device),
                pos_inter.to(device), neg_inter.to(device),
            )

            loss = (
                F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred)) +
                F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            graphs_trained += 1

        avg_loss = epoch_loss / max(graphs_trained, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            metrics = evaluate_gnn(model, val_graphs, sf_computer)
            if metrics["auc"] is not None:
                print(
                    f"Epoch {epoch+1:3d}/{config['epochs']} | "
                    f"Loss: {avg_loss:.4f} | AUC: {metrics['auc']:.4f} | AP: {metrics['ap']:.4f} | "
                    f"Intra: {metrics['intra_auc']:.4f} | Inter: {metrics['inter_auc']:.4f}"
                )
                if metrics["auc"] > best_val_auc:
                    best_val_auc = metrics["auc"]
                    torch.save(model.state_dict(), str(checkpoint_path))
                    print(f"  -> Best model saved (AUC: {metrics['auc']:.4f})")

    print(f"Training complete. Best val AUC: {best_val_auc:.4f}")
    return best_val_auc


# ── Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_gnn(model, graphs, sf_computer=None):
    """Evaluate GNN on a set of graphs. Returns overall/intra/inter AUC + tissue-pair scores."""
    model.eval()
    sf_computer = sf_computer or StructuralFeatureComputer()
    device = SHARED["device"]

    all_labels, all_scores = [], []
    inter_labels, inter_scores = [], []
    intra_labels, intra_scores = [], []
    tissue_pair_scores = defaultdict(list)

    for g in graphs:
        if g.edge_index.size(1) < 2 or g.x.size(0) < 3:
            continue

        data = g.to(device)
        pos_ei = data.edge_index
        neg_ei = degree_biased_negative_sampling(g, pos_ei.size(1)).to(device)
        if neg_ei.size(1) == 0:
            continue

        pos_sf, pos_cn = sf_computer.compute(g.edge_index, g.x.size(0), pos_ei)
        neg_sf, neg_cn = sf_computer.compute(g.edge_index, g.x.size(0), neg_ei)
        pos_src_t, pos_dst_t, pos_inter = get_edge_metadata(g, pos_ei)
        neg_src_t, neg_dst_t, neg_inter = get_edge_metadata(g, neg_ei)

        pos_pred, neg_pred, _, _ = model(
            data, pos_ei, neg_ei,
            pos_sf, neg_sf, pos_cn, neg_cn,
            pos_src_t.to(device), pos_dst_t.to(device),
            neg_src_t.to(device), neg_dst_t.to(device),
            pos_inter.to(device), neg_inter.to(device),
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

        for i in range(len(pos_s)):
            src_t = GNN["tissue_labels"][pos_src_t[i].item()]
            dst_t = GNN["tissue_labels"][pos_dst_t[i].item()]
            tissue_pair_scores[f"{src_t}→{dst_t}"].append(float(pos_s[i]))

    if len(set(all_labels)) < 2:
        return {"auc": None, "ap": None, "intra_auc": None, "inter_auc": None,
                "tissue_pairs": {}, "all_labels": all_labels, "all_scores": all_scores}

    return {
        "auc": roc_auc_score(all_labels, all_scores),
        "ap": average_precision_score(all_labels, all_scores),
        "intra_auc": roc_auc_score(intra_labels, intra_scores) if len(set(intra_labels)) > 1 else 0.0,
        "inter_auc": roc_auc_score(inter_labels, inter_scores) if len(set(inter_labels)) > 1 else 0.0,
        "tissue_pairs": dict(tissue_pair_scores),
        "all_labels": all_labels,
        "all_scores": all_scores,
    }


# ══════════════════════════════════════════════════════════════════════
# Sub-phase 5D: Reasoning Traces & Visualization
# ══════════════════════════════════════════════════════════════════════

class ReasoningTraceGenerator:
    """Generate human-readable 3D reasoning traces for top-k edges."""

    def __init__(self, tissue_labels=None):
        self.tissue_labels = tissue_labels or GNN["tissue_labels"]

    def generate(self, data, z, pos_pred, sf, cn_list, src_tissue, dst_tissue, is_inter, top_k=10):
        scores = torch.sigmoid(pos_pred).detach().cpu().numpy()
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
            # Feature index depends on architecture: 35-dim old or 25-dim SV means
            feat_dim = src_feats.size(0)
            if feat_dim >= 16:
                src_t1c_mean = src_feats[15].item()  # T1c mean in old layout
                dst_t1c_mean = dst_feats[15].item()
            elif feat_dim >= 5:
                src_t1c_mean = src_feats[4].item()   # T1ce mean in SV layout (ch1 mean)
                dst_t1c_mean = dst_feats[4].item()
            else:
                src_t1c_mean = src_feats[0].item()
                dst_t1c_mean = dst_feats[0].item()

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


class HierarchicalExplainer:
    """3-level explanation for edge predictions in the hierarchical GNN.

    Level 1 (Structure): WHY are these two seg regions connected?
      - OCN structural features (CN, Jaccard, orthogonalized residual, path-norm)
      - Tissue-pair compatibility + spatial distance

    Level 2 (Supervoxel): WHICH parts of each region drove the connection?
      - IntraNodeAggregator attention weights → top-k supervoxels per endpoint

    Level 3 (Spatial): WHERE in the MRI is the evidence?
      - Map top-k supervoxels back to voxel coordinates → 3D heatmap
    """

    def __init__(self, tissue_labels=None):
        self.tissue_labels = tissue_labels or GNN["tissue_labels"]

    def explain_edge(self, data, edge_idx, z, sv_attentions, structural_feats,
                     cn_list, src_tissue, dst_tissue, is_inter, sv_labels_3d=None):
        """Generate a full 3-level explanation for one predicted edge.

        Args:
            data: PyG Data object.
            edge_idx: int, index into edge_index.
            z: (N, D) node embeddings.
            sv_attentions: list of (K_i,) attention weight tensors per node.
            structural_feats: (E, 5) OCN features for all edges.
            cn_list: list of CN index lists per edge.
            src_tissue, dst_tissue: (E,) tissue labels per edge.
            is_inter: (E,) inter-slice flags.
            sv_labels_3d: optional (H, W, D) SV label volume for heatmap.

        Returns:
            explanation: dict with keys 'text', 'level1', 'level2', 'level3', 'heatmap'.
        """
        i = data.edge_index[0, edge_idx].item()
        j = data.edge_index[1, edge_idx].item()

        src_t = self.tissue_labels.get(src_tissue[edge_idx].item(), "Unknown")
        dst_t = self.tissue_labels.get(dst_tissue[edge_idx].item(), "Unknown")
        src_slice = data.slice_ids[i].item()
        dst_slice = data.slice_ids[j].item()
        is_cross = is_inter[edge_idx].item() == 1

        # ── Level 1: Structural reasoning ──
        sf = structural_feats[edge_idx]
        cn_count = sf[0].item()
        jaccard = sf[1].item()
        aa = sf[2].item()
        ocn_residual = sf[3].item() if sf.size(0) > 3 else 0.0
        path_norm_cn = sf[4].item() if sf.size(0) > 4 else 0.0

        diff = data.pos[j] - data.pos[i]
        dist = float(torch.norm(diff))

        level1 = {
            "src_tissue": src_t,
            "dst_tissue": dst_t,
            "distance": dist,
            "cn_count": cn_count,
            "jaccard": jaccard,
            "adamic_adar": aa,
            "ocn_residual": ocn_residual,
            "path_norm_cn": path_norm_cn,
            "is_inter_slice": is_cross,
            "slice_gap": abs(src_slice - dst_slice),
        }

        # ── Level 2: Supervoxel attribution ──
        level2 = {"src_top_svs": [], "dst_top_svs": []}

        if sv_attentions and i < len(sv_attentions) and sv_attentions[i].numel() > 0:
            src_attn = sv_attentions[i]
            k = min(3, src_attn.size(0))
            top_vals, top_idx = src_attn.topk(k)
            level2["src_top_svs"] = [
                {"sv_local_idx": int(top_idx[t]), "attention": float(top_vals[t])}
                for t in range(k)
            ]
            level2["src_attn_entropy"] = float(-torch.sum(src_attn * torch.log(src_attn + 1e-8)))

        if sv_attentions and j < len(sv_attentions) and sv_attentions[j].numel() > 0:
            dst_attn = sv_attentions[j]
            k = min(3, dst_attn.size(0))
            top_vals, top_idx = dst_attn.topk(k)
            level2["dst_top_svs"] = [
                {"sv_local_idx": int(top_idx[t]), "attention": float(top_vals[t])}
                for t in range(k)
            ]
            level2["dst_attn_entropy"] = float(-torch.sum(dst_attn * torch.log(dst_attn + 1e-8)))

        # ── Level 3: Voxel-space heatmap ──
        heatmap = None
        if sv_labels_3d is not None and sv_attentions:
            heatmap = self._sv_attention_to_heatmap(
                i, j, sv_attentions, data, sv_labels_3d,
            )

        # ── Assemble text explanation ──
        text = self._format_explanation(level1, level2, i, j)

        return {
            "text": text,
            "level1": level1,
            "level2": level2,
            "level3": {"heatmap": heatmap},
            "src_node": i,
            "dst_node": j,
        }

    def _sv_attention_to_heatmap(self, src_node, dst_node, sv_attentions,
                                  data, sv_labels_3d):
        """Map SV attention weights back to voxel space.

        For each important SV, paint its voxels with the attention weight.
        Returns a heatmap of same shape as sv_labels_3d.
        """
        heatmap = np.zeros(sv_labels_3d.shape, dtype=np.float32)

        # We need the mapping from local SV index (within a seg node) to global SV ID
        # This mapping is stored in the supervoxel result but not in the Data object.
        # Fallback: use attention weights to paint all SVs in the node proportionally.
        for node_idx in [src_node, dst_node]:
            if node_idx >= len(sv_attentions) or sv_attentions[node_idx].numel() == 0:
                continue

            attn = sv_attentions[node_idx].cpu().numpy()

            # Get the SVs that belong to this seg node by checking which SVs
            # overlap with the node's tissue mask. We approximate by using
            # data.n_svs_per_node to know the count.
            if not hasattr(data, 'n_svs_per_node') or node_idx >= len(data.n_svs_per_node):
                continue

            n_svs = data.n_svs_per_node[node_idx]
            if n_svs == 0 or len(attn) != n_svs:
                continue

            # Find unique SV IDs in the tumor region that correspond to this node
            # This is an approximation — for exact mapping, the SV-to-comp assignment
            # would need to be stored in the Data object. For now, we use spatial proximity.
            node_pos = data.pos[node_idx].cpu().numpy()

            # Paint SVs by finding all unique SV labels and assigning attention
            # based on proximity to the node centroid
            unique_svs = np.unique(sv_labels_3d)
            unique_svs = unique_svs[unique_svs >= 0]  # exclude background (-1)

            H, W, D = sv_labels_3d.shape
            sv_centroids = []
            sv_ids = []
            for sv_id in unique_svs:
                mask = sv_labels_3d == sv_id
                if mask.sum() == 0:
                    continue
                coords = np.argwhere(mask)
                centroid = np.array([
                    coords[:, 0].mean() / H,
                    coords[:, 1].mean() / W,
                    coords[:, 2].mean() / D,
                ])
                sv_centroids.append(centroid)
                sv_ids.append(sv_id)

            if not sv_centroids:
                continue

            sv_centroids = np.stack(sv_centroids)
            dists = np.linalg.norm(sv_centroids - node_pos, axis=1)
            closest_indices = np.argsort(dists)[:n_svs]

            for local_idx, global_idx in enumerate(closest_indices):
                if local_idx < len(attn):
                    sv_id = sv_ids[global_idx]
                    sv_mask = sv_labels_3d == sv_id
                    heatmap[sv_mask] = max(heatmap[sv_mask].max(), attn[local_idx])

        return heatmap

    def _format_explanation(self, level1, level2, src_node, dst_node):
        """Format a human-readable text explanation."""
        L1 = level1
        strength = "strong" if L1["jaccard"] > 0.3 else "moderate" if L1["jaccard"] > 0.1 else "weak"

        lines = [
            f"Edge ({src_node}→{dst_node}): {L1['src_tissue']} → {L1['dst_tissue']}",
            f"",
            f"Level 1 — Structural Evidence ({strength}):",
            f"  Distance: {L1['distance']:.3f} | CN: {L1['cn_count']:.2f} | Jaccard: {L1['jaccard']:.3f}",
            f"  Adamic-Adar: {L1['adamic_adar']:.3f} | OCN residual: {L1['ocn_residual']:.3f}",
            f"  Path-norm CN: {L1['path_norm_cn']:.3f}",
        ]

        if L1["is_inter_slice"]:
            lines.append(f"  Cross-slice gap: {L1['slice_gap']} slices")

        if level2.get("src_top_svs"):
            lines.append(f"")
            lines.append(f"Level 2 — Supervoxel Attribution:")
            src_svs = level2["src_top_svs"]
            entropy = level2.get("src_attn_entropy", 0)
            focus = "focused" if entropy < 1.0 else "distributed"
            lines.append(f"  Source ({L1['src_tissue']}): {focus} attention (H={entropy:.2f})")
            for sv in src_svs:
                lines.append(f"    SV#{sv['sv_local_idx']}: weight={sv['attention']:.3f}")

        if level2.get("dst_top_svs"):
            dst_svs = level2["dst_top_svs"]
            entropy = level2.get("dst_attn_entropy", 0)
            focus = "focused" if entropy < 1.0 else "distributed"
            lines.append(f"  Target ({L1['dst_tissue']}): {focus} attention (H={entropy:.2f})")
            for sv in dst_svs:
                lines.append(f"    SV#{sv['sv_local_idx']}: weight={sv['attention']:.3f}")

        return "\n".join(lines)

    def explain_top_k(self, data, z, pos_pred, sv_attentions, structural_feats,
                      cn_list, src_tissue, dst_tissue, is_inter,
                      sv_labels_3d=None, top_k=5):
        """Generate explanations for the top-k highest-confidence edges."""
        scores = torch.sigmoid(pos_pred).detach().cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]

        explanations = []
        for idx in top_indices:
            conf = scores[idx]
            exp = self.explain_edge(
                data, idx, z, sv_attentions, structural_feats,
                cn_list, src_tissue, dst_tissue, is_inter, sv_labels_3d,
            )
            exp["confidence"] = float(conf)
            explanations.append(exp)

        return explanations


def plot_hierarchical_explanation(data, explanation, mri_slice_2d=None, seg_slice_2d=None,
                                  heatmap_slice_2d=None, slice_idx=None):
    """4-panel visualization of a hierarchical edge explanation.

    Panel 1: Graph structure with highlighted edge
    Panel 2: SV attention heatmap (if available)
    Panel 3: Structural feature breakdown
    Panel 4: Text explanation
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Panel 1: Graph with highlighted edge
    ax = axes[0]
    pos = data.pos.cpu().numpy()
    tl = data.tissue_labels.cpu().numpy()
    colors_map = {1: '#e74c3c', 2: '#2ecc71', 3: '#f39c12'}

    for tissue in [1, 2, 3]:
        mask = tl == tissue
        if mask.sum() > 0:
            ax.scatter(pos[mask, 0], pos[mask, 1], c=colors_map[tissue],
                       s=80, alpha=0.8, label=GNN["tissue_labels"].get(tissue, str(tissue)),
                       edgecolors='black', linewidths=0.5)

    # Draw all edges in gray
    ei = data.edge_index.cpu().numpy()
    for e in range(ei.shape[1]):
        si, di = ei[0, e], ei[1, e]
        ax.plot([pos[si, 0], pos[di, 0]], [pos[si, 1], pos[di, 1]],
                color='gray', alpha=0.15, linewidth=0.5)

    # Highlight explained edge
    src = explanation["src_node"]
    dst = explanation["dst_node"]
    ax.plot([pos[src, 0], pos[dst, 0]], [pos[src, 1], pos[dst, 1]],
            color='#3498db', linewidth=3, alpha=0.9, zorder=5)
    ax.scatter([pos[src, 0], pos[dst, 0]], [pos[src, 1], pos[dst, 1]],
               c='#3498db', s=150, zorder=6, edgecolors='white', linewidths=2)

    ax.set_title("Graph + Highlighted Edge", fontsize=10)
    ax.legend(fontsize=7)
    ax.set_aspect('equal')

    # Panel 2: SV attention heatmap
    ax2 = axes[1]
    if heatmap_slice_2d is not None:
        im = ax2.imshow(heatmap_slice_2d, cmap='hot', interpolation='nearest')
        if seg_slice_2d is not None:
            ax2.contour(seg_slice_2d, levels=[0.5, 1.5, 2.5], colors='cyan',
                        linewidths=0.5, alpha=0.7)
        plt.colorbar(im, ax=ax2, fraction=0.046)
        ax2.set_title(f"SV Attention Heatmap (z={slice_idx})", fontsize=10)
    else:
        # Show attention bar chart instead
        level2 = explanation["level2"]
        labels, weights = [], []
        for prefix, key in [("Src", "src_top_svs"), ("Dst", "dst_top_svs")]:
            for sv_info in level2.get(key, []):
                labels.append(f"{prefix} SV#{sv_info['sv_local_idx']}")
                weights.append(sv_info["attention"])
        if labels:
            bar_colors = ['#e74c3c' if l.startswith("Src") else '#2ecc71' for l in labels]
            ax2.barh(labels, weights, color=bar_colors, alpha=0.8)
            ax2.set_xlabel("Attention Weight")
            ax2.set_title("Top SV Attention Weights", fontsize=10)
            ax2.set_xlim(0, 1)
        else:
            ax2.text(0.5, 0.5, "No SV data", ha='center', va='center', fontsize=12)
            ax2.set_title("SV Attention", fontsize=10)

    # Panel 3: Structural features radar
    ax3 = axes[2]
    L1 = explanation["level1"]
    feat_names = ["CN", "Jaccard", "AA", "OCN\nResidual", "Path\nNorm CN"]
    feat_vals = [L1["cn_count"], L1["jaccard"], min(L1["adamic_adar"], 1.0),
                 min(L1["ocn_residual"], 1.0), min(L1["path_norm_cn"], 1.0)]

    x_pos = np.arange(len(feat_names))
    colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    ax3.bar(x_pos, feat_vals, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(feat_names, fontsize=8)
    ax3.set_ylim(0, 1.1)
    ax3.set_title("OCN Structural Features", fontsize=10)
    ax3.set_ylabel("Value")

    # Panel 4: Text explanation
    ax4 = axes[3]
    ax4.axis('off')
    ax4.text(0.05, 0.95, explanation["text"], transform=ax4.transAxes,
             fontsize=8, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    conf = explanation.get("confidence", 0)
    ax4.set_title(f"Explanation (conf={conf:.3f})", fontsize=10)

    plt.suptitle(
        f"Hierarchical Edge Explanation — {L1['src_tissue']} → {L1['dst_tissue']}",
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.show()


# ── Visualization ─────────────────────────────────────────────────────

def plot_3d_graphs(graphs, n=3):
    """3D scatter plot of tumor graphs with inter-slice edges highlighted."""
    fig = plt.figure(figsize=(18, 6))
    for plot_idx in range(min(n, len(graphs))):
        g = graphs[plot_idx]
        if g.edge_index.size(1) == 0:
            continue
        ax = fig.add_subplot(1, n, plot_idx + 1, projection='3d')
        pos = g.pos.numpy()
        tl = g.tissue_labels.numpy()
        colors = {1: 'red', 2: 'green', 3: 'gold'}
        for tissue in [1, 2, 3]:
            mask = tl == tissue
            if mask.sum() > 0:
                ax.scatter(pos[mask, 0], pos[mask, 1], pos[mask, 2],
                           c=colors[tissue], s=40, alpha=0.8,
                           label=GNN["tissue_labels"][tissue])
        ei = g.edge_index.numpy()
        for e in range(ei.shape[1]):
            si, di = ei[0, e], ei[1, e]
            is_inter = g.slice_ids[si].item() != g.slice_ids[di].item()
            color = 'blue' if is_inter else 'gray'
            alpha = 0.6 if is_inter else 0.15
            ax.plot([pos[si, 0], pos[di, 0]], [pos[si, 1], pos[di, 1]],
                    [pos[si, 2], pos[di, 2]], color=color, alpha=alpha, linewidth=0.5)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Slice")
        ax.set_title(f"{g.case_id[-7:]} | {g.x.size(0)} nodes", fontsize=9)
        ax.legend(fontsize=7)
    plt.suptitle("3D Tumor Graphs (blue=inter-slice, gray=intra-slice)", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_results(demo_graph, pos_pred, tissue_pair_scores, all_labels, all_scores):
    """3-panel visualization: slice overlay, tissue-pair bars, ROC curve."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: slice graph overlay
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
                       label=GNN["tissue_labels"][tissue], edgecolors='black', linewidths=0.5)
    pred_scores = torch.sigmoid(pos_pred).detach().cpu().numpy()
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

    # Panel 2: tissue-pair confidence
    ax2 = axes[1]
    pair_means = {p: np.mean(s) for p, s in tissue_pair_scores.items()}
    if pair_means:
        pairs = list(pair_means.keys())
        vals = [pair_means[p] for p in pairs]
        bar_colors = ['#e74c3c' if 'ET' in p else '#2ecc71' if 'ED' in p else '#3498db' for p in pairs]
        ax2.barh(pairs, vals, color=bar_colors, alpha=0.8)
        ax2.set_xlabel("Mean Confidence")
        ax2.set_title("Tissue-Pair Link Confidence", fontsize=10)
        ax2.set_xlim(0, 1)

    # Panel 3: ROC curve
    ax3 = axes[2]
    if len(all_labels) > 0 and len(set(all_labels)) > 1:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        overall_auc = roc_auc_score(all_labels, all_scores)
        ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f"AUC = {overall_auc:.3f}")
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax3.set_xlabel("FPR")
        ax3.set_ylabel("TPR")
        ax3.set_title("Test ROC Curve", fontsize=10)
        ax3.legend()

    plt.suptitle(f"3D GNN Analysis — {demo_graph.case_id}", fontsize=13)
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import ensure_dirs
    ensure_dirs()

    print(f"Device: {SHARED['device']}")

    # Build graphs
    train_graphs, val_graphs, test_graphs = build_all_graphs()

    # Visualize
    plot_3d_graphs(train_graphs)

    # Model
    model = EdgePredictor().to(SHARED["device"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"EdgePredictor | Parameters: {total_params:,}")

    # Train
    train_gnn(model, train_graphs, val_graphs)

    # Test
    model.load_state_dict(torch.load(str(GNN["checkpoint"]), weights_only=True))
    sf_computer = StructuralFeatureComputer()
    test_metrics = evaluate_gnn(model, test_graphs, sf_computer)

    print(f"\nTest Results:")
    print(f"  AUC-ROC: {test_metrics['auc']:.4f}")
    print(f"  AP:      {test_metrics['ap']:.4f}")
    print(f"  Intra:   {test_metrics['intra_auc']:.4f}")
    print(f"  Inter:   {test_metrics['inter_auc']:.4f}")
    print(f"\nTissue-pair confidence:")
    for pair, scores in sorted(test_metrics["tissue_pairs"].items(), key=lambda x: -np.mean(x[1])):
        print(f"  {pair}: {np.mean(scores):.3f} (n={len(scores)})")

    # Reasoning traces on a demo graph
    demo_graph = None
    for g in test_graphs:
        if g.edge_index.size(1) >= 10 and g.x.size(0) >= 5:
            demo_graph = g
            break
    if demo_graph is None and len(test_graphs) > 0:
        demo_graph = test_graphs[0]

    if demo_graph is not None and demo_graph.edge_index.size(1) >= 2:
        data = demo_graph.to(SHARED["device"])
        pos_ei = data.edge_index
        pos_sf, pos_cn = sf_computer.compute(demo_graph.edge_index, demo_graph.x.size(0), pos_ei)
        pos_src_t, pos_dst_t, pos_inter = get_edge_metadata(demo_graph, pos_ei)

        z, alphas, sv_attns = model.encode(data, return_attention=True)
        pos_pred = model.decode(
            z, pos_ei, pos_sf, pos_cn,
            pos_src_t.to(SHARED["device"]),
            pos_dst_t.to(SHARED["device"]),
            pos_inter.to(SHARED["device"]),
        )

        trace_gen = ReasoningTraceGenerator()
        traces = trace_gen.generate(
            demo_graph, z, pos_pred, pos_sf, pos_cn,
            pos_src_t, pos_dst_t, pos_inter, top_k=8,
        )

        print(f"\n--- Reasoning Traces for {demo_graph.case_id} ---")
        for t in traces:
            print(t)
            print()

        # Hierarchical 3-level explanations
        print(f"\n--- Hierarchical Explanations for {demo_graph.case_id} ---")
        explainer = HierarchicalExplainer()
        explanations = explainer.explain_top_k(
            demo_graph, z, pos_pred, sv_attns, pos_sf, pos_cn,
            pos_src_t, pos_dst_t, pos_inter, top_k=3,
        )

        for exp in explanations:
            print(exp["text"])
            print(f"  Confidence: {exp['confidence']:.3f}")
            print()
            plot_hierarchical_explanation(demo_graph, exp)

        plot_results(demo_graph, pos_pred, test_metrics["tissue_pairs"],
                     test_metrics["all_labels"], test_metrics["all_scores"])

    print("\n=== BraTS Hierarchical GNN Pipeline Complete ===")

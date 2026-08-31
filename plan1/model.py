import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import SHARED, GRAPH


# ── Patch-Level Transformer Embedder ─────────────────────────────────


class PatchEmbedder(nn.Module):
    def __init__(self, patch_dim=None, n_modalities=4, n_patch=None,
                 embed_dim=None, n_layers=None, n_heads=None,
                 seg_prior_dim=5, dropout=0.1):
        super().__init__()
        patch_dim = patch_dim or (GRAPH["patch_neighbors"] + 3)
        n_patch = n_patch or GRAPH["n_patch"]
        embed_dim = embed_dim or GRAPH["embed_dim"]
        n_layers = n_layers or GRAPH["transformer_layers"]
        n_heads = n_heads or GRAPH["transformer_heads"]

        self.embed_dim = embed_dim
        self.n_modalities = n_modalities
        self.n_patch = n_patch
        self.n_rows = n_patch * n_modalities  # 16 rows per SV

        # Linear projection: patch row → embed_dim
        self.input_proj = nn.Linear(patch_dim, embed_dim)

        # Learnable modality embeddings (shared across patches of same modality)
        self.modality_embed = nn.Embedding(n_modalities, embed_dim)

        # Learnable patch position embeddings
        self.patch_pos_embed = nn.Embedding(n_patch, embed_dim)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # Seg prior projection (optional, 0-dim if no seg priors)
        self.has_seg_prior = seg_prior_dim > 0
        if self.has_seg_prior:
            self.seg_proj = nn.Sequential(
                nn.Linear(seg_prior_dim, embed_dim),
                nn.GELU(),
            )
            # Final MLP: concat([CLS], seg_embed) → embed_dim
            self.out_proj = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.out_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )

    def forward(self, patch_tensors, seg_priors=None, return_attention=False):
        B = patch_tensors.size(0)
        device = patch_tensors.device

        # Project patch rows
        x = self.input_proj(patch_tensors)  # (B, n_rows, embed_dim)

        # Add modality embeddings: rows are ordered as
        # [patch0_mod0, patch0_mod1, ..., patch0_modM, patch1_mod0, ...]
        mod_ids = torch.arange(self.n_modalities, device=device)
        mod_ids = mod_ids.repeat(self.n_patch)  # (n_rows,)
        x = x + self.modality_embed(mod_ids).unsqueeze(0)  # broadcast over B

        # Add patch position embeddings
        patch_ids = torch.arange(self.n_patch, device=device)
        patch_ids = patch_ids.repeat_interleave(self.n_modalities)  # (n_rows,)
        x = x + self.patch_pos_embed(patch_ids).unsqueeze(0)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)  # (B, 1+n_rows, embed_dim)

        # Capture attention weights via manual layer-by-layer forward
        patch_attentions = None
        if return_attention:
            patch_attentions = []
            for layer in self.transformer.layers:
                # Manual forward through TransformerEncoderLayer
                # to capture self-attention weights
                x2, attn_w = layer.self_attn(
                    layer.norm1(x), layer.norm1(x), layer.norm1(x),
                    need_weights=True, average_attn_weights=False,
                )
                patch_attentions.append(attn_w.detach())
                # Complete the layer forward
                x = x + layer.dropout1(x2)
                x = x + layer._ff_block(layer.norm2(x))
        else:
            # Standard forward
            x = self.transformer(x)  # (B, 1+n_rows, embed_dim)

        cls_out = x[:, 0]  # (B, embed_dim)

        # Combine with seg prior if available
        if self.has_seg_prior and seg_priors is not None:
            seg_embed = self.seg_proj(seg_priors)  # (B, embed_dim)
            combined = torch.cat([cls_out, seg_embed], dim=-1)
            node_embeds = self.out_proj(combined)
        else:
            node_embeds = self.out_proj(cls_out)

        return node_embeds, patch_attentions


# ── Graph-Level GATv2 Encoder ────────────────────────────────────────


class GraphEncoder(nn.Module):
    def __init__(self, embed_dim=None, n_layers=None, n_heads=None,
                 edge_dim=4, pe_dim=None, dropout=0.1):
        super().__init__()
        from torch_geometric.nn import GATv2Conv, LayerNorm

        embed_dim = embed_dim or GRAPH["embed_dim"]
        n_layers = n_layers or GRAPH["gat_layers"]
        n_heads = n_heads or GRAPH["gat_heads"]
        pe_dim = pe_dim or GRAPH["laplacian_pe_dim"]

        self.embed_dim = embed_dim
        self.pe_dim = pe_dim

        # Laplacian PE projection
        self.pe_proj = nn.Linear(pe_dim, embed_dim)

        # GATv2 layers with residual connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GATv2Conv(
                embed_dim, embed_dim // n_heads, heads=n_heads,
                edge_dim=edge_dim, dropout=dropout, concat=True,
            ))
            self.norms.append(LayerNorm(embed_dim))

        # Multiscale fusion: concat all layer outputs → MLP → embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * n_layers, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        self.n_layers = n_layers
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None, lap_pe=None,
                return_attention=False):
        # Add Laplacian PE
        if lap_pe is not None:
            x = x + self.pe_proj(lap_pe)

        layer_outputs = []
        alphas = []

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            if return_attention:
                x, (_, alpha) = conv(
                    x, edge_index, edge_attr=edge_attr,
                    return_attention_weights=True,
                )
                alphas.append(alpha)
            else:
                x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = x + residual  # residual connection
            if i < self.n_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)

        # Multiscale fusion
        fused = torch.cat(layer_outputs, dim=-1)  # (N, embed_dim * n_layers)
        out = self.fusion(fused)  # (N, embed_dim)

        return out, alphas


# ── Task 1: Node Classification Head ────────────────────────────────


class NodeHead(nn.Module):
    def __init__(self, embed_dim=None, dropout=0.2):
        super().__init__()
        embed_dim = embed_dim or GRAPH["embed_dim"]
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


# ── Task 2: Edge Boundary Type Classification Head ──────────────────


class EdgeHead(nn.Module):
    def __init__(self, embed_dim=None, edge_dim=4, n_boundary_types=10,
                 dropout=0.2):
        super().__init__()
        embed_dim = embed_dim or GRAPH["embed_dim"]
        in_dim = embed_dim * 3 + edge_dim  # h_i + h_j + |h_i-h_j| + e_ij

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_boundary_types),
        )

    def forward(self, node_feats, edge_index, edge_attr):
        h_i = node_feats[edge_index[0]]       # (E, embed_dim)
        h_j = node_feats[edge_index[1]]       # (E, embed_dim)
        h_diff = torch.abs(h_i - h_j)         # (E, embed_dim)

        x = torch.cat([h_i, h_j, h_diff, edge_attr], dim=-1)  # (E, 3*D+4)
        return self.mlp(x)  # (E, 10)


# ── Laplacian Positional Encoding ────────────────────────────────────


def compute_laplacian_pe(edge_index, num_nodes, k=None):
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh

    k = k or GRAPH["laplacian_pe_dim"]

    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    row = edge_index[0]
    col = edge_index[1]
    data = np.ones(len(row), dtype=np.float64)

    # Build adjacency matrix
    A = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()

    # Degree matrix
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)

    # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
    from scipy.sparse import diags
    D_inv_sqrt = diags(deg_inv_sqrt)
    L = diags(np.ones(num_nodes)) - D_inv_sqrt @ A @ D_inv_sqrt

    # Compute k+1 smallest eigenvectors (skip trivial eigenvector 0)
    num_eig = min(k + 1, num_nodes - 1)
    if num_eig < 2:
        return torch.zeros(num_nodes, k, dtype=torch.float32)

    try:
        eigenvalues, eigenvectors = eigsh(L, k=num_eig, which="SM")
        # Skip first eigenvector (constant, eigenvalue ≈ 0)
        pe = eigenvectors[:, 1:k+1]

        # Pad if fewer eigenvectors than k
        if pe.shape[1] < k:
            pad = np.zeros((num_nodes, k - pe.shape[1]))
            pe = np.hstack([pe, pad])

        # Random sign flip for invariance
        signs = np.sign(pe[0])
        signs[signs == 0] = 1
        pe = pe * signs

    except Exception:
        pe = np.zeros((num_nodes, k))

    return torch.from_numpy(pe).float()


# ── Full Model ───────────────────────────────────────────────────────


class TumorRefiner(nn.Module):
    def __init__(self, config=None, use_seg_prior=True):
        super().__init__()
        cfg = config or GRAPH
        seg_dim = 5 if use_seg_prior else 0

        self.embedder = PatchEmbedder(
            patch_dim=cfg["patch_neighbors"] + 3,
            n_modalities=len(cfg["modalities"]),
            n_patch=cfg["n_patch"],
            embed_dim=cfg["embed_dim"],
            n_layers=cfg["transformer_layers"],
            n_heads=cfg["transformer_heads"],
            seg_prior_dim=seg_dim,
        )
        self.encoder = GraphEncoder(
            embed_dim=cfg["embed_dim"],
            n_layers=cfg["gat_layers"],
            n_heads=cfg["gat_heads"],
            edge_dim=4,  # [dx, dy, dz, dist]
            pe_dim=cfg["laplacian_pe_dim"],
        )
        self.node_head = NodeHead(embed_dim=cfg["embed_dim"])
        self.edge_head = EdgeHead(embed_dim=cfg["embed_dim"], edge_dim=4)
        self.use_seg_prior = use_seg_prior

    def forward(self, patch_tensors, seg_priors, edge_index, edge_attr,
                lap_pe=None, return_attention=False):
        # Stage 1: Patch-level embedding
        sp = seg_priors if self.use_seg_prior else None
        node_feats, patch_attns = self.embedder(
            patch_tensors, sp, return_attention=return_attention,
        )

        # Stage 2: Graph-level encoding
        graph_feats, graph_attns = self.encoder(
            node_feats, edge_index, edge_attr=edge_attr,
            lap_pe=lap_pe, return_attention=return_attention,
        )

        # Stage 3: Node classification (Task 1)
        node_logits = self.node_head(graph_feats)  # (N,)

        # Stage 4: Edge boundary classification (Task 2)
        edge_logits = self.edge_head(graph_feats, edge_index, edge_attr)

        attention_dict = {
            "graph": graph_attns,   # list of (E, n_heads) per GATv2 layer
            "patch": patch_attns,   # list of (B, n_heads, seq, seq) per TF layer
        }

        return node_logits, edge_logits, attention_dict

    def count_parameters(self):
        """Count total and per-component trainable parameters."""
        components = {
            "PatchEmbedder": self.embedder,
            "GraphEncoder": self.encoder,
            "NodeHead": self.node_head,
            "EdgeHead": self.edge_head,
        }
        total = 0
        breakdown = {}
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            breakdown[name] = n
            total += n
        breakdown["Total"] = total
        return breakdown


# ── Main (sanity check) ──────────────────────────────────────────────


if __name__ == "__main__":
    device = SHARED["device"]
    print(f"Device: {device}")
    print(f"Config: embed_dim={GRAPH['embed_dim']}, "
          f"transformer={GRAPH['transformer_layers']}L×{GRAPH['transformer_heads']}H, "
          f"gat={GRAPH['gat_layers']}L×{GRAPH['gat_heads']}H")
    print()

    # Build model
    model = TumorRefiner(use_seg_prior=True).to(device)
    params = model.count_parameters()
    print("Parameter count:")
    for name, n in params.items():
        print(f"  {name}: {n:,}")
    print()

    # Synthetic forward pass
    N = 100       # nodes (supervoxels)
    n_rows = GRAPH["n_patch"] * len(GRAPH["modalities"])  # 16
    patch_dim = GRAPH["patch_neighbors"] + 3               # 19
    E = 400       # edges

    patch_tensors = torch.randn(N, n_rows, patch_dim, device=device)
    seg_priors = torch.rand(N, 5, device=device)  # 4 probs + 1 entropy
    edge_index = torch.randint(0, N, (2, E), device=device)
    edge_attr = torch.randn(E, 4, device=device)
    lap_pe = torch.randn(N, GRAPH["laplacian_pe_dim"], device=device)

    print(f"Input shapes:")
    print(f"  patch_tensors: {patch_tensors.shape}")
    print(f"  seg_priors:    {seg_priors.shape}")
    print(f"  edge_index:    {edge_index.shape}")
    print(f"  edge_attr:     {edge_attr.shape}")
    print(f"  lap_pe:        {lap_pe.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        node_logits, edge_logits, attn_dict = model(
            patch_tensors, seg_priors, edge_index, edge_attr,
            lap_pe=lap_pe, return_attention=True,
        )

    node_probs = torch.sigmoid(node_logits)
    edge_preds = edge_logits.argmax(dim=-1)
    print(f"\nOutput:")
    print(f"  Node logits: {node_logits.shape}")
    print(f"  Node prob range: [{node_probs.min():.4f}, {node_probs.max():.4f}]")
    print(f"  Edge logits: {edge_logits.shape}")
    print(f"  Edge pred distribution: {torch.bincount(edge_preds, minlength=10).tolist()}")
    print(f"  GATv2 attention layers: {len(attn_dict['graph'])}")
    if attn_dict['patch']:
        print(f"  Patch attention layers: {len(attn_dict['patch'])}")
        for i, pa in enumerate(attn_dict['patch']):
            print(f"    Layer {i}: {pa.shape}")

    # Memory estimate
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\nModel memory: {param_bytes / 1e6:.1f} MB")

    # Test with real preprocessing output (if available)
    from .supervoxel import discover_cases, preprocess_case, BOUNDARY_TYPE_NAMES
    cases = discover_cases(GRAPH["data_root"])
    if cases:
        print(f"\n{'='*60}")
        print(f"Testing with real case: {cases[0]['case_id']}")
        print(f"{'='*60}")

        import time
        t0 = time.time()
        result = preprocess_case(cases[0])

        # Assemble tensors from preprocessing output
        retained = result["retained_labels"]
        N_real = len(retained)

        # Stack patch tensors
        patch_list = [torch.from_numpy(result["patches"][l]) for l in retained]
        patch_batch = torch.stack(patch_list, dim=0).to(device)

        # Seg priors (zeros if not available)
        if result["seg_features"] is not None:
            seg_list = []
            for l in retained:
                sf = result["seg_features"][l]
                feat = np.concatenate([sf["seg_feat"], [sf["seg_entropy"]]])
                seg_list.append(torch.from_numpy(feat))
            seg_batch = torch.stack(seg_list, dim=0).float().to(device)
        else:
            seg_batch = torch.zeros(N_real, 5, device=device)

        # Edge index & attr
        ei = torch.from_numpy(result["edge_index"]).to(device)
        ea = torch.from_numpy(result["edge_attr"]).to(device)

        # Laplacian PE
        lpe = compute_laplacian_pe(result["edge_index"], N_real).to(device)

        print(f"  Preprocessed in {time.time()-t0:.1f}s")
        print(f"  Nodes: {N_real}, Edges: {ei.shape[1]}")

        # Forward
        model.eval()
        with torch.no_grad():
            nl, el, _ = model(patch_batch, seg_batch, ei, ea, lap_pe=lpe)

        # Node results
        node_probs_real = torch.sigmoid(nl)
        gt_node = torch.tensor(
            [result["targets"][l]["y_cls"] for l in retained], device=device,
        )
        print(f"\n  Node: {nl.shape}, pred tumor={int((node_probs_real>0.5).sum())}, "
              f"GT tumor={int(gt_node.sum())}")

        # Edge results
        gt_edge_type = torch.from_numpy(
            result["edge_targets"]["y_edge_type"]
        ).to(device)
        edge_preds_real = el.argmax(dim=-1)
        correct = (edge_preds_real == gt_edge_type).float().mean()
        print(f"  Edge: {el.shape}, random-init accuracy={correct:.1%}")
        print(f"  GT boundary distribution:")
        gt_counts = torch.bincount(gt_edge_type, minlength=10)
        for i in range(10):
            if gt_counts[i] > 0:
                print(f"    {BOUNDARY_TYPE_NAMES[i]}: {gt_counts[i]} edges")

        print(f"\nEnd-to-end pipeline verified (Tasks 1+2). ✓")

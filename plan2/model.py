import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import SHARED, GRAPH

# Reuse backbone components from plan1
from plan1.model import (
    PatchEmbedder,
    GraphEncoder,
    EdgeHead,
    compute_laplacian_pe,
)


# ── Task 1: Tumor Proportion Regression Head ─────────────────────────


class RegressionHead(nn.Module):
    def __init__(self, embed_dim=None, K=None, dropout=0.2):
        super().__init__()
        embed_dim = embed_dim or GRAPH["embed_dim"]
        K = K or GRAPH["regression_ensemble_size"]

        self.K = K

        # K parallel MLP heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            for _ in range(K)
        ])

        # Attention gate: learns which ensemble members to trust
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, K),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
       # Each head produces a raw prediction
        preds = torch.stack([head(x).squeeze(-1) for head in self.heads],
                            dim=-1)  # (N, K)

        # Attention-weighted combination
        attn_weights = self.attention(x)  # (N, K)
        combined = (preds * attn_weights).sum(dim=-1)  # (N,)

        # Sigmoid to clamp to [0, 1]
        y_reg = torch.sigmoid(combined)

        # Also return individual sigmoid predictions for uncertainty
        ensemble_preds = torch.sigmoid(preds)

        return y_reg, ensemble_preds


# ── Task 3: Segmentation Uncertainty Prediction Head ─────────────────


class UncertaintyHead(nn.Module):
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


# ── Uncertainty-Weighted Multi-Task Loss ─────────────────────────────


class MultiTaskLoss(nn.Module):
    def __init__(self, init_log_vars=None):
        super().__init__()
        init_log_vars = init_log_vars or [
            GRAPH["init_log_var_reg"],
            GRAPH["init_log_var_edge"],
            GRAPH["init_log_var_unc"],
        ]
        # Learnable parameters: log(σ²) for each task
        self.log_vars = nn.ParameterList([
            nn.Parameter(torch.tensor(v, dtype=torch.float32))
            for v in init_log_vars
        ])

    def forward(self, losses):
        total = torch.tensor(0.0, device=losses[0].device)
        weights = []

        for i, (loss, log_var) in enumerate(zip(losses, self.log_vars)):
            precision = torch.exp(-log_var)  # 1/σ²
            total = total + 0.5 * precision * loss + 0.5 * log_var
            weights.append(float(precision.item()))

        return total, weights

    def get_sigmas(self):
        return [float(torch.exp(0.5 * lv).item()) for lv in self.log_vars]


# ── Full Model ───────────────────────────────────────────────────────


class MultiTaskRefiner(nn.Module):
    def __init__(self, config=None, use_seg_prior=True):
        super().__init__()
        cfg = config or GRAPH
        seg_dim = 5 if use_seg_prior else 0

        # Shared backbone (same architecture as Plan 1)
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
            edge_dim=4,
            pe_dim=cfg["laplacian_pe_dim"],
        )

        # Task heads
        self.regression_head = RegressionHead(
            embed_dim=cfg["embed_dim"],
            K=cfg["regression_ensemble_size"],
        )
        self.edge_head = EdgeHead(
            embed_dim=cfg["embed_dim"],
            edge_dim=4,
            n_boundary_types=cfg["n_boundary_types"],
        )
        self.uncertainty_head = UncertaintyHead(embed_dim=cfg["embed_dim"])

        # Multi-task loss
        self.loss_fn = MultiTaskLoss()

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

        # Task 1: Tumor proportion regression
        y_reg, ensemble_preds = self.regression_head(graph_feats)

        # Task 2: Edge boundary classification
        edge_logits = self.edge_head(graph_feats, edge_index, edge_attr)

        # Task 3: Segmentation uncertainty prediction
        unc_logits = self.uncertainty_head(graph_feats)

        outputs = {
            "y_reg": y_reg,
            "ensemble_preds": ensemble_preds,
            "edge_logits": edge_logits,
            "unc_logits": unc_logits,
        }

        attention_dict = {
            "graph": graph_attns,
            "patch": patch_attns,
        }

        return outputs, attention_dict

    def compute_loss(self, outputs, targets_dict):
        # Task 1: Smooth L1 (Huber) loss for regression
        L_reg = F.smooth_l1_loss(
            outputs["y_reg"], targets_dict["y_reg_target"],
        )

        # Task 2: Cross-entropy for edge classification
        edge_weight = targets_dict.get("edge_class_weights", None)
        L_edge = F.cross_entropy(
            outputs["edge_logits"], targets_dict["edge_type_target"],
            weight=edge_weight,
        )

        # Task 3: BCE for uncertainty prediction
        L_unc = F.binary_cross_entropy_with_logits(
            outputs["unc_logits"], targets_dict["unc_target"],
        )

        # Combine with learned uncertainty weights
        total, weights = self.loss_fn([L_reg, L_edge, L_unc])

        sigmas = self.loss_fn.get_sigmas()
        loss_dict = {
            "total": total.item(),
            "L_reg": L_reg.item(),
            "L_edge": L_edge.item(),
            "L_unc": L_unc.item(),
            "w_reg": weights[0],
            "w_edge": weights[1],
            "w_unc": weights[2],
            "σ_reg": sigmas[0],
            "σ_edge": sigmas[1],
            "σ_unc": sigmas[2],
        }

        return total, loss_dict

    def count_parameters(self):
        """Count total and per-component trainable parameters."""
        components = {
            "PatchEmbedder": self.embedder,
            "GraphEncoder": self.encoder,
            "RegressionHead": self.regression_head,
            "EdgeHead": self.edge_head,
            "UncertaintyHead": self.uncertainty_head,
            "MultiTaskLoss": self.loss_fn,
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
          f"ensemble_K={GRAPH['regression_ensemble_size']}, "
          f"transformer={GRAPH['transformer_layers']}L×{GRAPH['transformer_heads']}H, "
          f"gat={GRAPH['gat_layers']}L×{GRAPH['gat_heads']}H")
    print()

    # Build model
    model = MultiTaskRefiner(use_seg_prior=True).to(device)
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
    seg_priors = torch.rand(N, 5, device=device)
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
        outputs, attn_dict = model(
            patch_tensors, seg_priors, edge_index, edge_attr,
            lap_pe=lap_pe, return_attention=True,
        )

    print(f"\nTask 1 — Regression:")
    print(f"  y_reg: {outputs['y_reg'].shape}, "
          f"range=[{outputs['y_reg'].min():.4f}, {outputs['y_reg'].max():.4f}]")
    print(f"  ensemble_preds: {outputs['ensemble_preds'].shape}")
    ensemble_var = outputs['ensemble_preds'].var(dim=-1)
    print(f"  ensemble variance: mean={ensemble_var.mean():.6f}, "
          f"max={ensemble_var.max():.6f}")

    print(f"\nTask 2 — Edge Classification:")
    print(f"  edge_logits: {outputs['edge_logits'].shape}")
    edge_preds = outputs['edge_logits'].argmax(dim=-1)
    print(f"  pred distribution: {torch.bincount(edge_preds, minlength=10).tolist()}")

    print(f"\nTask 3 — Uncertainty:")
    unc_probs = torch.sigmoid(outputs['unc_logits'])
    print(f"  unc_logits: {outputs['unc_logits'].shape}, "
          f"prob range=[{unc_probs.min():.4f}, {unc_probs.max():.4f}]")

    print(f"\nAttention:")
    print(f"  GATv2 layers: {len(attn_dict['graph'])}")
    if attn_dict['patch']:
        print(f"  Patch layers: {len(attn_dict['patch'])}")
        for i, pa in enumerate(attn_dict['patch']):
            print(f"    Layer {i}: {pa.shape}")

    # Test loss computation
    print(f"\nLoss computation:")
    targets = {
        "y_reg_target": torch.rand(N, device=device),
        "edge_type_target": torch.randint(0, 10, (E,), device=device),
        "unc_target": torch.randint(0, 2, (N,), device=device).float(),
    }

    model.train()
    outputs_train, _ = model(
        patch_tensors, seg_priors, edge_index, edge_attr, lap_pe=lap_pe,
    )
    total_loss, loss_dict = model.compute_loss(outputs_train, targets)
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  L_reg={loss_dict['L_reg']:.4f} (σ={loss_dict['σ_reg']:.3f}, w={loss_dict['w_reg']:.3f})")
    print(f"  L_edge={loss_dict['L_edge']:.4f} (σ={loss_dict['σ_edge']:.3f}, w={loss_dict['w_edge']:.3f})")
    print(f"  L_unc={loss_dict['L_unc']:.4f} (σ={loss_dict['σ_unc']:.3f}, w={loss_dict['w_unc']:.3f})")

    # Backward pass check
    total_loss.backward()
    grad_norms = {name: p.grad.norm().item()
                  for name, p in model.named_parameters()
                  if p.grad is not None}
    print(f"  Gradient norms computed for {len(grad_norms)} parameters")
    print(f"  Log-var grads: ", end="")
    for lv in model.loss_fn.log_vars:
        print(f"  {lv.grad.item():.4f}", end="")
    print()

    # Memory estimate
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\nModel memory: {param_bytes / 1e6:.1f} MB")

    # ── Test with real BraTS case ──
    from .supervoxel import discover_cases, preprocess_case, BOUNDARY_TYPE_NAMES
    cases = discover_cases(GRAPH["data_root"])
    if cases:
        print(f"\n{'='*60}")
        print(f"Testing with real case: {cases[0]['case_id']}")
        print(f"{'='*60}")

        import time
        t0 = time.time()
        result = preprocess_case(cases[0])

        retained = result["retained_labels"]
        N_real = len(retained)

        patch_list = [torch.from_numpy(result["patches"][l]) for l in retained]
        patch_batch = torch.stack(patch_list, dim=0).to(device)

        if result["seg_features"] is not None:
            seg_list = []
            for l in retained:
                sf = result["seg_features"][l]
                feat = np.concatenate([sf["seg_feat"], [sf["seg_entropy"]]])
                seg_list.append(torch.from_numpy(feat))
            seg_batch = torch.stack(seg_list, dim=0).float().to(device)
        else:
            seg_batch = torch.zeros(N_real, 5, device=device)

        ei = torch.from_numpy(result["edge_index"]).to(device)
        ea = torch.from_numpy(result["edge_attr"]).to(device)
        lpe_real = compute_laplacian_pe(result["edge_index"], N_real).to(device)

        print(f"  Preprocessed in {time.time()-t0:.1f}s")
        print(f"  Nodes: {N_real}, Edges: {ei.shape[1]}")

        # Forward
        model.eval()
        with torch.no_grad():
            out, _ = model(patch_batch, seg_batch, ei, ea, lap_pe=lpe_real)

        # Task 1: Regression vs GT
        gt_reg = torch.tensor(
            [result["targets"][l]["y_reg"] for l in retained], device=device,
        )
        mae = (out["y_reg"] - gt_reg).abs().mean()
        print(f"\n  Task 1 (Regression):")
        print(f"    y_reg range: [{out['y_reg'].min():.4f}, {out['y_reg'].max():.4f}]")
        print(f"    GT range:    [{gt_reg.min():.4f}, {gt_reg.max():.4f}]")
        print(f"    MAE (random): {mae:.4f}")
        ens_var = out["ensemble_preds"].var(dim=-1)
        print(f"    Ensemble var: mean={ens_var.mean():.6f}")

        # Task 2: Edge
        gt_edge = torch.from_numpy(result["edge_targets"]["y_edge_type"]).to(device)
        edge_acc = (out["edge_logits"].argmax(-1) == gt_edge).float().mean()
        print(f"\n  Task 2 (Edge): accuracy={edge_acc:.1%}")

        # Task 3: Uncertainty — GT is "was seg model wrong?"
        unc_prob = torch.sigmoid(out["unc_logits"])
        print(f"\n  Task 3 (Uncertainty): prob range="
              f"[{unc_prob.min():.4f}, {unc_prob.max():.4f}]")

        print(f"\nEnd-to-end pipeline verified (Plan 2, Tasks 1+2+3). ✓")

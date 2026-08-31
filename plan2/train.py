"""
Plan 2 Training Script — Multi-Task GNN (Regression + Edge + Uncertainty)

Run from project root:
    python -m plan2.train
    python -m plan2.train --max-cases 20 --epochs 50
    python -m plan2.train --eval-only
"""

import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .config import SHARED, GRAPH
from .supervoxel import discover_cases, preprocess_case
from .model import MultiTaskRefiner, compute_laplacian_pe
from .explainability import explain_case


# ── Tensor Assembly ──────────────────────────────────────────────────


def assemble_tensors(result, device):
    retained = result["retained_labels"]
    N = len(retained)

    patch_batch = torch.stack(
        [torch.from_numpy(result["patches"][l]) for l in retained]
    ).to(device)

    if result["seg_features"] is not None:
        seg_list = []
        for l in retained:
            sf = result["seg_features"][l]
            feat = np.concatenate([sf["seg_feat"], [sf["seg_entropy"]]])
            seg_list.append(torch.from_numpy(feat))
        seg_batch = torch.stack(seg_list).float().to(device)
    else:
        seg_batch = torch.zeros(N, 5, device=device)

    ei = torch.from_numpy(result["edge_index"]).to(device)
    ea = torch.from_numpy(result["edge_attr"]).to(device)
    lpe = compute_laplacian_pe(result["edge_index"], N).to(device)

    # Task 1: Regression target
    gt_reg = torch.tensor(
        [result["targets"][l]["y_reg"] for l in retained],
        dtype=torch.float32, device=device,
    )

    # Task 2: Edge type target
    gt_edge = torch.from_numpy(result["edge_targets"]["y_edge_type"]).to(device)

    # Task 3: Uncertainty target (was seg model wrong?)
    if result["seg_features"] is not None:
        gt_unc = torch.tensor([
            1.0 if result["seg_features"][l]["seg_pred"]
            != result["targets"][l]["y_dominant"] else 0.0
            for l in retained
        ], dtype=torch.float32, device=device)
    else:
        gt_unc = torch.zeros(N, dtype=torch.float32, device=device)

    targets = {
        "y_reg_target": gt_reg,
        "edge_type_target": gt_edge,
        "unc_target": gt_unc,
    }
    return patch_batch, seg_batch, ei, ea, lpe, targets


# ── Training ─────────────────────────────────────────────────────────


def compute_edge_class_weights(graphs, device):
    counts = np.zeros(10, dtype=np.float64)
    for g in graphs:
        for t in g["edge_targets"]["y_edge_type"]:
            counts[int(t)] += 1
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts
    w = w / w.sum() * 10
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_one_epoch(model, graphs, optimizer, scheduler, edge_weights,
                    device, accum_steps=4):
    model.train()
    total_loss = 0.0
    metrics = {
        "L_reg": 0, "L_edge": 0, "L_unc": 0,
        "node_mae": 0, "n_node": 0,
        "edge_correct": 0, "n_edge": 0,
    }
    optimizer.zero_grad()
    processed = 0

    for i, result in enumerate(graphs):
        if result["edge_index"].shape[1] == 0:
            continue

        patches, seg, ei, ea, lpe, targets = assemble_tensors(result, device)
        targets["edge_class_weights"] = edge_weights

        outputs, _ = model(patches, seg, ei, ea, lap_pe=lpe)
        loss, ld = model.compute_loss(outputs, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        (loss / accum_steps).backward()
        processed += 1

        if (i + 1) % accum_steps == 0 or (i + 1) == len(graphs):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        metrics["L_reg"] += ld["L_reg"]
        metrics["L_edge"] += ld["L_edge"]
        metrics["L_unc"] += ld["L_unc"]
        metrics["node_mae"] += (
            outputs["y_reg"] - targets["y_reg_target"]
        ).abs().sum().item()
        metrics["n_node"] += len(targets["y_reg_target"])
        metrics["edge_correct"] += (
            outputs["edge_logits"].argmax(-1) == targets["edge_type_target"]
        ).sum().item()
        metrics["n_edge"] += len(targets["edge_type_target"])

    n = max(processed, 1)
    return {
        "loss": total_loss / n,
        "L_reg": metrics["L_reg"] / n,
        "L_edge": metrics["L_edge"] / n,
        "L_unc": metrics["L_unc"] / n,
        "mae": metrics["node_mae"] / max(metrics["n_node"], 1),
        "edge_acc": metrics["edge_correct"] / max(metrics["n_edge"], 1),
        "sigmas": model.loss_fn.get_sigmas(),
    }


@torch.no_grad()
def evaluate(model, graphs, edge_weights, device):
    model.eval()
    total_loss = 0.0
    metrics = {
        "node_mae": 0, "n_node": 0,
        "edge_correct": 0, "n_edge": 0,
    }
    processed = 0

    for result in graphs:
        if result["edge_index"].shape[1] == 0:
            continue

        patches, seg, ei, ea, lpe, targets = assemble_tensors(result, device)
        targets["edge_class_weights"] = edge_weights

        outputs, _ = model(patches, seg, ei, ea, lap_pe=lpe)
        loss, _ = model.compute_loss(outputs, targets)

        total_loss += loss.item()
        processed += 1
        metrics["node_mae"] += (
            outputs["y_reg"] - targets["y_reg_target"]
        ).abs().sum().item()
        metrics["n_node"] += len(targets["y_reg_target"])
        metrics["edge_correct"] += (
            outputs["edge_logits"].argmax(-1) == targets["edge_type_target"]
        ).sum().item()
        metrics["n_edge"] += len(targets["edge_type_target"])

    n = max(processed, 1)
    return {
        "loss": total_loss / n,
        "mae": metrics["node_mae"] / max(metrics["n_node"], 1),
        "edge_acc": metrics["edge_correct"] / max(metrics["n_edge"], 1),
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Plan 2: Multi-Task GNN")
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seg-prob-dir", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    device = SHARED["device"]
    epochs = args.epochs or GRAPH["epochs"]
    seg_prob_dir = args.seg_prob_dir or "brats_outputs/seg_probs"

    print(f"Device: {device}")
    print(f"Config: embed_dim={GRAPH['embed_dim']}, "
          f"ensemble_K={GRAPH['regression_ensemble_size']}, epochs={epochs}")
    print()

    # ── Preprocessing ──
    cases = discover_cases(GRAPH["data_root"])
    print(f"Discovered {len(cases)} BraTS cases")

    cases_to_use = cases[:args.max_cases]
    print(f"Preprocessing {len(cases_to_use)} cases...")

    all_graphs = []
    for i, case in enumerate(cases_to_use):
        t0 = time.time()
        result = preprocess_case(case, seg_prob_dir=seg_prob_dir)
        if result["edge_index"].shape[1] < 10:
            print(f"  [{i+1}/{len(cases_to_use)}] {case['case_id']}: "
                  f"SKIPPED ({len(result['retained_labels'])} nodes)")
            continue
        all_graphs.append(result)
        has_seg = "✓ seg" if result["seg_features"] is not None else "no seg"
        print(f"  [{i+1}/{len(cases_to_use)}] {case['case_id']}: "
              f"{len(result['retained_labels'])} nodes, "
              f"{result['edge_index'].shape[1]} edges, {has_seg} "
              f"({time.time()-t0:.1f}s)")

    print(f"\nUsable graphs: {len(all_graphs)}")

    split = int(0.8 * len(all_graphs))
    train_graphs = all_graphs[:split]
    val_graphs = all_graphs[split:]
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

    # ── Model ──
    model = MultiTaskRefiner(use_seg_prior=True).to(device)
    params = model.count_parameters()
    print(f"\nModel parameters:")
    for name, n in params.items():
        print(f"  {name}: {n:,}")

    if args.eval_only:
        if not GRAPH["checkpoint"].exists():
            print(f"\nCheckpoint not found: {GRAPH['checkpoint']}")
            return
        model.load_state_dict(torch.load(
            str(GRAPH["checkpoint"]), map_location=device, weights_only=True))
        print(f"\nLoaded checkpoint: {GRAPH['checkpoint']}")

        edge_weights = compute_edge_class_weights(all_graphs, device)
        vm = evaluate(model, val_graphs, edge_weights, device)
        print(f"\nVal: loss={vm['loss']:.4f} mae={vm['mae']:.4f} "
              f"edge={vm['edge_acc']:.1%}")

        print(f"\n{'='*60}")
        print("Running 5-level explainability...")
        report = explain_case(model, all_graphs[0], device=device)
        _print_report(report, all_graphs[0])
        return

    # ── Training ──
    edge_weights = compute_edge_class_weights(train_graphs, device)
    print(f"\nEdge class weights: {edge_weights.cpu().numpy().round(3)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=GRAPH["lr"], weight_decay=GRAPH["weight_decay"])
    total_steps = epochs * len(train_graphs) // GRAPH["accum_steps"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1))

    best_val_loss = float("inf")
    patience_counter = 0
    GRAPH["checkpoint"].parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for up to {epochs} epochs (patience={args.patience})...")

    for epoch in range(1, epochs + 1):
        random.shuffle(train_graphs)
        tm = train_one_epoch(
            model, train_graphs, optimizer, scheduler, edge_weights,
            device, GRAPH["accum_steps"],
        )

        if epoch % GRAPH["eval_every"] == 0 or epoch == 1:
            vm = evaluate(model, val_graphs, edge_weights, device)
            σ = tm["sigmas"]
            print(
                f"Ep {epoch:3d} | "
                f"Train loss={tm['loss']:.4f} mae={tm['mae']:.4f} "
                f"edge={tm['edge_acc']:.1%} | "
                f"Val loss={vm['loss']:.4f} mae={vm['mae']:.4f} "
                f"edge={vm['edge_acc']:.1%} | "
                f"σ=[{σ[0]:.3f},{σ[1]:.3f},{σ[2]:.3f}]"
            )

            if vm["loss"] < best_val_loss:
                best_val_loss = vm["loss"]
                patience_counter = 0
                torch.save(model.state_dict(), str(GRAPH["checkpoint"]))
            else:
                patience_counter += GRAPH["eval_every"]
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(best val loss: {best_val_loss:.4f})")
                    break

    model.load_state_dict(torch.load(
        str(GRAPH["checkpoint"]), map_location=device, weights_only=True))
    print(f"\nLoaded best model (val loss={best_val_loss:.4f})")

    # ── Explainability ──
    print(f"\n{'='*60}")
    print("Running 5-level explainability...")
    report = explain_case(model, all_graphs[0], device=device)
    _print_report(report, all_graphs[0])


def _print_report(report, result):
    """Print a 5-level explanation report."""

    # Level 1: Graph attention
    print(f"\n{'='*60}")
    print("Level 1: Graph Attention Traces")
    print("="*60)
    for idx, trace in list(report["graph_traces"].items())[:5]:
        label = trace["sv_label"]
        gt = result["targets"].get(label, {})
        reg = float(report["outputs"]["y_reg"][idx])
        print(f"  Node {idx} (SV {label}): reg={reg:.3f}, "
              f"GT={gt.get('y_reg', 0):.3f}, "
              f"entropy={trace['attention_entropy']:.3f}")
        for nb in trace["neighbors"][:3]:
            nb_gt = f"y_cls={nb.get('y_cls', '?')}" if 'y_cls' in nb else ""
            print(f"    → Neighbor {nb['node_idx']} (SV {nb['sv_label']}): "
                  f"attn={nb['mean_attention']:.4f}, {nb_gt}")

    # Level 2: Patch attention
    print(f"\n{'='*60}")
    print("Level 2: Patch Attention Traces")
    print("="*60)
    for idx, trace in list(report["patch_traces"].items())[:3]:
        print(f"  Node {idx}: ", end="")
        for m, name in enumerate(trace["modality_names"]):
            print(f"{name}={trace['per_modality_importance'][m]:.3f}", end="  ")
        print()

    # Level 3: Regression refinement
    reg = report["regression_refinement"]
    print(f"\n{'='*60}")
    print("Level 3: Regression Refinement")
    print("="*60)
    print(f"  MAE (GNN):  {reg['mae_gnn']:.4f}")
    print(f"  MAE (Seg):  {reg['mae_seg']:.4f}")
    print(f"  R²:         {reg['r2']:.4f}")
    print(f"  Zero-shot Dice: {reg['zero_shot_dice']:.4f}")
    print(f"  Corrections:    {reg['n_corrections']}")
    print(f"  Degradations:   {reg['n_degradations']}")
    print(f"  Ensemble var:   {reg['mean_ensemble_var']:.6f}")

    # Level 4: Task divergence
    div = report["task_divergence"]
    print(f"\n{'='*60}")
    print("Level 4: Task Divergence")
    print("="*60)
    for cat, count in sorted(div["category_counts"].items()):
        print(f"  {cat}: {count}")
    print(f"  Interesting nodes: {len(div['interesting_nodes'])}")
    for node in div["interesting_nodes"][:5]:
        print(f"    SV {node['sv_label']}: {node['category']} "
              f"(reg={node['y_reg']:.3f}, unc={node['unc_prob']:.3f}, "
              f"GT={node['gt_cls']})")

    # Level 5: Uncertainty
    unc = report["uncertainty_explanation"]
    print(f"\n{'='*60}")
    print("Level 5: Uncertainty-Driven Explanation")
    print("="*60)
    print(f"  High-unc SVs:      {unc['n_high_uncertainty']} "
          f"(threshold={unc['unc_threshold']})")
    print(f"  Mean unc prob:     {unc['mean_unc_prob']:.4f}")
    print(f"  Ensemble-unc corr: {unc['ensemble_unc_correlation']:.4f}")
    print(f"  AUROC:             {unc['auroc']:.4f}")
    print(f"  Actual seg error:  {unc['actual_seg_error_rate']:.1%}")
    for exp in unc["explanations"][:3]:
        print(f"\n  SV {exp['sv_label']}: unc={exp['unc_prob']:.3f}, "
              f"reg={exp['y_reg']:.3f}, var={exp['ensemble_var']:.6f}")
        if "seg_was_actually_wrong" in exp:
            print(f"    Seg wrong? {bool(exp['seg_was_actually_wrong'])}, "
                  f"seg_pred={exp['seg_pred']}, GT={exp['gt_dominant']}")
            if exp.get("regression_compensates"):
                print(f"    ✓ Regression compensates for seg error")

    print(f"\nDone ✓")


if __name__ == "__main__":
    main()

"""
Plan 1 Training Script — Binary Tumor Refinement + Edge Prediction

Run from project root:
    python -m plan1.train
    python -m plan1.train --max-cases 20 --epochs 50
    python -m plan1.train --eval-only
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
from .model import TumorRefiner, compute_laplacian_pe
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

    gt_node = torch.tensor(
        [result["targets"][l]["y_cls"] for l in retained],
        dtype=torch.float32, device=device,
    )
    gt_edge = torch.from_numpy(result["edge_targets"]["y_edge_type"]).to(device)

    return patch_batch, seg_batch, ei, ea, lpe, gt_node, gt_edge


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
                    device, lambda_node=1.0, lambda_edge=0.5, accum_steps=4):
    model.train()
    total_loss = 0.0
    n_node_correct, n_node_total = 0, 0
    n_edge_correct, n_edge_total = 0, 0
    optimizer.zero_grad()
    processed = 0

    for i, result in enumerate(graphs):
        if result["edge_index"].shape[1] == 0:
            continue

        patches, seg, ei, ea, lpe, gt_node, gt_edge = assemble_tensors(result, device)
        nl, el, _ = model(patches, seg, ei, ea, lap_pe=lpe)

        # Class-weighted BCE for imbalanced node labels
        pos_count = gt_node.sum()
        neg_count = len(gt_node) - pos_count
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)
        loss_node = F.binary_cross_entropy_with_logits(nl, gt_node, pos_weight=pos_weight)
        loss_edge = F.cross_entropy(el, gt_edge, weight=edge_weights)
        loss = lambda_node * loss_node + lambda_edge * loss_edge

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
        preds_node = (torch.sigmoid(nl) > 0.5).long()
        n_node_correct += (preds_node == gt_node.long()).sum().item()
        n_node_total += len(gt_node)
        n_edge_correct += (el.argmax(-1) == gt_edge).sum().item()
        n_edge_total += len(gt_edge)

    return {
        "loss": total_loss / max(processed, 1),
        "node_acc": n_node_correct / max(n_node_total, 1),
        "edge_acc": n_edge_correct / max(n_edge_total, 1),
    }


@torch.no_grad()
def evaluate(model, graphs, edge_weights, device):
    model.eval()
    total_loss = 0.0
    n_node_correct, n_node_total = 0, 0
    n_edge_correct, n_edge_total = 0, 0
    processed = 0

    for result in graphs:
        if result["edge_index"].shape[1] == 0:
            continue

        patches, seg, ei, ea, lpe, gt_node, gt_edge = assemble_tensors(result, device)
        nl, el, _ = model(patches, seg, ei, ea, lap_pe=lpe)

        loss_node = F.binary_cross_entropy_with_logits(nl, gt_node)
        loss_edge = F.cross_entropy(el, gt_edge, weight=edge_weights)
        loss = GRAPH["lambda_node"] * loss_node + GRAPH["lambda_edge"] * loss_edge

        total_loss += loss.item()
        processed += 1
        preds_node = (torch.sigmoid(nl) > 0.5).long()
        n_node_correct += (preds_node == gt_node.long()).sum().item()
        n_node_total += len(gt_node)
        n_edge_correct += (el.argmax(-1) == gt_edge).sum().item()
        n_edge_total += len(gt_edge)

    return {
        "loss": total_loss / max(processed, 1),
        "node_acc": n_node_correct / max(n_node_total, 1),
        "edge_acc": n_edge_correct / max(n_edge_total, 1),
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Plan 1: Train GNN")
    parser.add_argument("--max-cases", type=int, default=50,
                        help="Max BraTS cases to preprocess")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--seg-prob-dir", type=str, default=None,
                        help="Path to seg probability volumes (from segmentation.py)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Load checkpoint and run evaluation + explainability only")
    args = parser.parse_args()

    device = SHARED["device"]
    epochs = args.epochs or GRAPH["epochs"]
    seg_prob_dir = args.seg_prob_dir or "brats_outputs/seg_probs"

    print(f"Device: {device}")
    print(f"Config: embed_dim={GRAPH['embed_dim']}, epochs={epochs}")
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
        has_seg = "✓ seg priors" if result["seg_features"] is not None else "no seg priors"
        print(f"  [{i+1}/{len(cases_to_use)}] {case['case_id']}: "
              f"{len(result['retained_labels'])} nodes, "
              f"{result['edge_index'].shape[1]} edges, {has_seg} "
              f"({time.time()-t0:.1f}s)")

    print(f"\nUsable graphs: {len(all_graphs)}")

    # Split
    split = int(0.8 * len(all_graphs))
    train_graphs = all_graphs[:split]
    val_graphs = all_graphs[split:]
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

    # ── Model ──
    model = TumorRefiner(use_seg_prior=True).to(device)
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
        val_metrics = evaluate(model, val_graphs, edge_weights, device)
        print(f"\nVal: loss={val_metrics['loss']:.4f} "
              f"node={val_metrics['node_acc']:.1%} "
              f"edge={val_metrics['edge_acc']:.1%}")

        # Explainability
        print(f"\n{'='*60}")
        print("Running explainability on first case...")
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

    print(f"\nTraining for up to {epochs} epochs "
          f"(patience={args.patience})...")

    for epoch in range(1, epochs + 1):
        random.shuffle(train_graphs)
        tm = train_one_epoch(
            model, train_graphs, optimizer, scheduler, edge_weights,
            device, GRAPH["lambda_node"], GRAPH["lambda_edge"],
            GRAPH["accum_steps"],
        )

        if epoch % GRAPH["eval_every"] == 0 or epoch == 1:
            vm = evaluate(model, val_graphs, edge_weights, device)
            print(f"Epoch {epoch:3d} | "
                  f"Train loss={tm['loss']:.4f} node={tm['node_acc']:.1%} "
                  f"edge={tm['edge_acc']:.1%} | "
                  f"Val loss={vm['loss']:.4f} node={vm['node_acc']:.1%} "
                  f"edge={vm['edge_acc']:.1%}")

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

    # Reload best
    model.load_state_dict(torch.load(
        str(GRAPH["checkpoint"]), map_location=device, weights_only=True))
    print(f"\nLoaded best model (val loss={best_val_loss:.4f})")

    # ── Explainability ──
    print(f"\n{'='*60}")
    print("Running explainability on first case...")
    report = explain_case(model, all_graphs[0], device=device)
    _print_report(report, all_graphs[0])


def _print_report(report, result):
    """Print a 3-level explanation report."""

    # Level 1
    print(f"\n{'='*60}")
    print("Level 1: Graph Attention Traces")
    print("="*60)
    for idx, trace in list(report["graph_traces"].items())[:5]:
        label = trace["sv_label"]
        gt = result["targets"].get(label, {})
        prob = float(torch.sigmoid(report["node_logits"][idx]))
        print(f"  Node {idx} (SV {label}): p={prob:.3f}, "
              f"GT={gt.get('y_cls', '?')}, "
              f"entropy={trace['attention_entropy']:.3f}")
        for nb in trace["neighbors"][:3]:
            nb_gt = f"y_cls={nb.get('y_cls', '?')}" if 'y_cls' in nb else ""
            print(f"    → Neighbor {nb['node_idx']} (SV {nb['sv_label']}): "
                  f"attn={nb['mean_attention']:.4f}, {nb_gt}")

    # Level 2
    print(f"\n{'='*60}")
    print("Level 2: Patch Attention Traces")
    print("="*60)
    for idx, trace in list(report["patch_traces"].items())[:3]:
        print(f"  Node {idx}: ", end="")
        for m, name in enumerate(trace["modality_names"]):
            print(f"{name}={trace['per_modality_importance'][m]:.3f}", end="  ")
        print()

    # Level 3
    ref = report["refinement"]
    print(f"\n{'='*60}")
    print("Level 3: Refinement Trace")
    print("="*60)
    print(f"  GNN accuracy:  {ref['accuracy_gnn']:.1%}")
    print(f"  Seg accuracy:  {ref['accuracy_seg']:.1%}")
    print(f"  Corrections:   {ref['n_corrections']} "
          f"(rate: {ref['correction_rate']:.1%})")
    print(f"  Degradations:  {ref['n_degradations']} "
          f"(rate: {ref['degradation_rate']:.1%})")

    for c in ref["corrections"][:5]:
        print(f"    SV {c['sv_label']}: [{c['type']}] "
              f"GT={c['gt']}, GNN={c['gnn_pred']}(p={c['gnn_prob']:.3f}), "
              f"Seg={c['seg_pred']}")

    print(f"\nDone ✓")


if __name__ == "__main__":
    main()

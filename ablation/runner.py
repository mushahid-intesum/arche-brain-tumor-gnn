"""
Ablation Study Runner

Trains all ablation configurations across multiple seeds,
evaluates on test set, and saves results to JSON.

Usage:
    python -m ablation.runner                  # run all configs
    python -m ablation.runner --configs A_baseline D_full  # run specific configs
    python -m ablation.runner --seeds 42 123   # run with specific seeds
"""

import sys
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is on path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import SHARED, GNN
from gnn import (
    build_all_graphs,
    load_metadata,
    degree_biased_negative_sampling,
    get_edge_metadata,
    evaluate_gnn,
)
from ablation.flat_graph import build_3d_graph
from ablation.config import (
    CONFIGS, CONFIG_ORDER, SEEDS,
    EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP, OUTPUT_DIR,
)
from ablation.model import AblationModel, AblationStructuralComputer


# ── Utilities ─────────────────────────────────────────────────────────

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Training ──────────────────────────────────────────────────────────

def train_ablation(model, sf_computer, train_graphs, val_graphs,
                   epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                   config_name="", verbose=True):
    """Train one ablation model. Returns best val AUC and training time."""
    device = SHARED["device"]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    # Count usable graphs for scheduler
    usable = sum(1 for g in train_graphs
                 if g.edge_index.size(1) >= 2 and g.x.size(0) >= 3)
    total_steps = max(epochs * usable, 1)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
    )

    best_val_auc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            graphs_trained += 1

        avg_loss = epoch_loss / max(graphs_trained, 1)

        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            metrics = evaluate_gnn(model, val_graphs, sf_computer)
            if metrics["auc"] is not None and metrics["auc"] > best_val_auc:
                best_val_auc = metrics["auc"]

            if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                auc_str = f"{metrics['auc']:.4f}" if metrics["auc"] else "N/A"
                print(f"    Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {avg_loss:.4f} | Val AUC: {auc_str}")

    train_time = time.time() - start_time
    return best_val_auc, train_time


# ── Single Run ────────────────────────────────────────────────────────

def run_single(config_name, ablation_config, seed, hier_graphs, flat_graphs):
    """Run one config with one seed. Returns metrics dict."""
    set_seed(seed)
    device = SHARED["device"]

    # Select appropriate graph set
    if ablation_config.use_sv_aggregation:
        train_g, val_g, test_g = hier_graphs
    else:
        train_g, val_g, test_g = flat_graphs

    # Create model and structural computer
    model = AblationModel(ablation_config=ablation_config).to(device)
    sf_computer = AblationStructuralComputer(use_ocn=ablation_config.use_ocn_features)
    param_count = sum(p.numel() for p in model.parameters())

    print(f"  Seed {seed} | {model.describe()}")

    # Train
    best_val_auc, train_time = train_ablation(
        model, sf_computer, train_g, val_g,
        config_name=config_name,
    )

    # Evaluate on test set
    test_metrics = evaluate_gnn(model, test_g, sf_computer)

    result = {
        "config": config_name,
        "seed": seed,
        "param_count": param_count,
        "train_time_sec": round(train_time, 1),
        "best_val_auc": round(best_val_auc, 4) if best_val_auc else None,
        "test_auc": round(test_metrics["auc"], 4) if test_metrics["auc"] else None,
        "test_ap": round(test_metrics["ap"], 4) if test_metrics["ap"] else None,
        "test_intra_auc": round(test_metrics["intra_auc"], 4) if test_metrics["intra_auc"] else None,
        "test_inter_auc": round(test_metrics["inter_auc"], 4) if test_metrics["inter_auc"] else None,
    }

    if test_metrics["auc"]:
        print(f"    Test AUC: {test_metrics['auc']:.4f} | "
              f"AP: {test_metrics['ap']:.4f} | "
              f"Time: {train_time:.0f}s")

    return result


# ── Full Ablation Run ─────────────────────────────────────────────────

def run_ablation(config_names=None, seeds=None):
    """Run the full ablation study.

    Args:
        config_names: list of config keys to run. None = all.
        seeds: list of seeds. None = all from config.
    """
    config_names = config_names or CONFIG_ORDER
    seeds = seeds or SEEDS

    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)
    print(f"Configs: {config_names}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {EPOCHS} | LR: {LR} | WD: {WEIGHT_DECAY}")
    print(f"Device: {SHARED['device']}")

    # ── Build graphs once ──
    needs_hier = any(CONFIGS[c].use_sv_aggregation for c in config_names)
    needs_flat = any(not CONFIGS[c].use_sv_aggregation for c in config_names)

    hier_graphs = None
    flat_graphs = None

    if needs_hier:
        print("\nBuilding hierarchical graphs...")
        hier_graphs = build_all_graphs()

    if needs_flat:
        print("\nBuilding flat graphs...")
        # Build flat graphs using the ablation flat_graph module
        patient_slices, metadata = load_metadata()
        from collections import defaultdict
        splits_map = defaultdict(str)
        for case_id, slices in patient_slices.items():
            splits_map[case_id] = slices[0]["split"]

        flat_all = []
        flat_splits = []
        for case_id, slice_list in patient_slices.items():
            g = build_3d_graph(case_id, slice_list)
            g.case_id = case_id
            flat_all.append(g)
            flat_splits.append(splits_map[case_id])

        flat_train = [g for g, s in zip(flat_all, flat_splits) if s == "train"]
        flat_val = [g for g, s in zip(flat_all, flat_splits) if s == "val"]
        flat_test = [g for g, s in zip(flat_all, flat_splits) if s == "test"]
        flat_graphs = (flat_train, flat_val, flat_test)
        print(f"  Flat graphs: train={len(flat_train)}, val={len(flat_val)}, test={len(flat_test)}")

    # ── Run each config x seed ──
    all_results = []
    total_start = time.time()

    for config_name in config_names:
        ablation_config = CONFIGS[config_name]
        print(f"\n{'─' * 60}")
        print(f"Config: {config_name}")
        print(f"  SV={ablation_config.use_sv_aggregation} | "
              f"Topo={ablation_config.use_intra_topology} | "
              f"OCN={ablation_config.use_ocn_features}")
        print(f"{'─' * 60}")

        for seed in seeds:
            result = run_single(
                config_name, ablation_config, seed,
                hier_graphs, flat_graphs,
            )
            all_results.append(result)

    # ── Save results ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.json"
    with open(str(results_path), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Print summary table ──
    print_summary(all_results, config_names)

    total_time = time.time() - total_start
    print(f"\nTotal ablation time: {total_time/60:.1f} minutes")

    return all_results


def print_summary(all_results, config_names):
    """Print a summary table of mean and std across seeds."""
    print(f"\n{'=' * 70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Config':<16s} {'AUC':>12s} {'AP':>12s} "
          f"{'Intra':>10s} {'Inter':>10s} {'Params':>10s} {'Time(s)':>8s}")
    print("-" * 80)

    for config_name in config_names:
        runs = [r for r in all_results if r["config"] == config_name]
        if not runs:
            continue

        aucs = [r["test_auc"] for r in runs if r["test_auc"] is not None]
        aps = [r["test_ap"] for r in runs if r["test_ap"] is not None]
        intras = [r["test_intra_auc"] for r in runs if r["test_intra_auc"] is not None]
        inters = [r["test_inter_auc"] for r in runs if r["test_inter_auc"] is not None]
        times = [r["train_time_sec"] for r in runs]
        params = runs[0]["param_count"]

        def fmt(vals):
            if not vals:
                return "N/A"
            return f"{np.mean(vals):.4f}+/-{np.std(vals):.4f}"

        print(f"{config_name:<16s} {fmt(aucs):>12s} {fmt(aps):>12s} "
              f"{fmt(intras):>10s} {fmt(inters):>10s} "
              f"{params:>10,d} {np.mean(times):>7.0f}s")


# ── CLI Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--configs", nargs="+", default=None,
        choices=list(CONFIGS.keys()),
        help="Which configs to run (default: all)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help=f"Which seeds to use (default: {SEEDS})",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    args = parser.parse_args()

    if args.epochs != EPOCHS:
        # Override module-level constant for this run
        import ablation.config as ac
        ac.EPOCHS = args.epochs
        globals()["EPOCHS"] = args.epochs

    run_ablation(config_names=args.configs, seeds=args.seeds)

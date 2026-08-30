"""
Explainability Study Runner

Trains all experiment configurations across multiple seeds, evaluates
link prediction performance AND explainability metrics, and runs
post-hoc baseline comparisons.

All parameters are controlled via constants in explainability/config.py.
Usage: python -m explainability.runner
"""

import sys
import json
import time
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is on path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import SHARED, GNN, SUPERVOXEL
from gnn import (
    build_hierarchical_graph,
    build_all_graphs,
    load_metadata,
    degree_biased_negative_sampling,
    get_edge_metadata,
    evaluate_gnn,
    StructuralFeatureComputer,
    EdgePredictor,
)
from ablation.flat_graph import build_3d_graph
from ablation.model import AblationModel, AblationStructuralComputer
from ablation.config import AblationConfig

from explainability.config import (
    CONFIGS, CONFIG_ORDER, ExplainabilityConfig,
    KNN_CONFIGS, ABLATION_CONFIGS, POSTHOC_METHODS,
    SEEDS, EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP, OUTPUT_DIR,
)
from explainability.metrics import ExplainabilityMetrics
from explainability.posthoc import (
    GNNExplainerWrapper,
    GradCAMExplainer,
    AttentionOnlyExplainer,
)


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


def config_to_gnn_override(exp_config: ExplainabilityConfig) -> dict:
    """Create a GNN config dict override from an ExplainabilityConfig."""
    override = dict(GNN)
    override["edge_strategy"] = exp_config.edge_strategy
    override["k_neighbors"] = exp_config.k_neighbors
    override["num_layers"] = exp_config.num_gnn_layers
    return override


def config_to_ablation_config(exp_config: ExplainabilityConfig) -> AblationConfig:
    """Convert an ExplainabilityConfig to an AblationConfig for model creation."""
    return AblationConfig(
        use_sv_aggregation=exp_config.use_sv_aggregation,
        use_ocn_features=exp_config.use_ocn_features,
        use_intra_topology=exp_config.use_intra_topology,
        name=exp_config.name,
    )


# ── Graph Building ────────────────────────────────────────────────────

def build_graphs_for_config(exp_config: ExplainabilityConfig):
    """Build train/val/test graphs with the specified edge strategy.

    Uses hierarchical graphs for configs with SV aggregation,
    flat graphs for baseline configs.
    """
    gnn_override = config_to_gnn_override(exp_config)

    if exp_config.use_sv_aggregation:
        # Hierarchical graphs with configurable edge strategy
        return build_all_graphs(config=gnn_override)
    else:
        # Flat graphs for baseline (A_baseline)
        patient_slices, metadata = load_metadata()
        splits_map = {}
        for case_id, slices in patient_slices.items():
            splits_map[case_id] = slices[0]["split"]

        flat_all, flat_splits = [], []
        for case_id, slice_list in patient_slices.items():
            g = build_3d_graph(case_id, slice_list)
            g.case_id = case_id
            flat_all.append(g)
            flat_splits.append(splits_map[case_id])

        train = [g for g, s in zip(flat_all, flat_splits) if s == "train"]
        val = [g for g, s in zip(flat_all, flat_splits) if s == "val"]
        test = [g for g, s in zip(flat_all, flat_splits) if s == "test"]
        return train, val, test


# ── Training ──────────────────────────────────────────────────────────

def train_model(model, sf_computer, train_graphs, val_graphs,
                epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                config_name="", verbose=True):
    """Train one model configuration. Returns best val AUC and training time."""
    device = SHARED["device"]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    usable = sum(1 for g in train_graphs
                 if g.edge_index.size(1) >= 2 and g.x.size(0) >= 3)
    total_steps = max(epochs * usable, 1)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
    )

    best_val_auc = 0.0
    best_state = None
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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            metrics = evaluate_gnn(model, val_graphs, sf_computer)
            if metrics["auc"] is not None and metrics["auc"] > best_val_auc:
                best_val_auc = metrics["auc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                auc_str = f"{metrics['auc']:.4f}" if metrics["auc"] else "N/A"
                print(f"    Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {avg_loss:.4f} | Val AUC: {auc_str}")

    train_time = time.time() - start_time

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_auc, train_time


# ── Explainability Evaluation ─────────────────────────────────────────

def evaluate_explainability(model, test_graphs, sf_computer, device, top_k_svs=3):
    """Run all explainability metrics on the test set.

    Returns:
        all_edge_metrics: list of per-edge metric dicts
        summary: aggregated statistics
    """
    xai = ExplainabilityMetrics(model, device=device)
    all_edge_metrics = []

    for g in test_graphs:
        if g.edge_index.size(1) < 2 or g.x.size(0) < 3:
            continue
        edge_metrics = xai.evaluate_graph(g, top_k_svs=top_k_svs)
        for m in edge_metrics:
            m["case_id"] = getattr(g, "case_id", "unknown")
        all_edge_metrics.extend(edge_metrics)

    summary = ExplainabilityMetrics.aggregate_metrics(all_edge_metrics)
    return all_edge_metrics, summary


def evaluate_posthoc(model, test_graphs, sf_computer, device, top_k_svs=3):
    """Run post-hoc baselines on the test set and compare.

    Returns dict mapping method_name -> {edge_metrics, summary}.
    """
    results = {}

    # ── Intrinsic (our 3-level system) ──
    all_intrinsic, intrinsic_summary = evaluate_explainability(
        model, test_graphs, sf_computer, device, top_k_svs=top_k_svs,
    )
    results["intrinsic"] = {
        "summary": intrinsic_summary,
        "explanations": [
            {"top_sv_indices": m["top_sv_indices"], "sf_ranking": m["sf_ranking"]}
            for m in all_intrinsic
        ],
    }

    # ── GNNExplainer ──
    print("    Running GNNExplainer...")
    gnn_exp = GNNExplainerWrapper(model, device=device, epochs=50)
    gnn_exp_explanations = []

    for g in test_graphs:
        if g.edge_index.size(1) < 2 or g.x.size(0) < 3:
            continue
        data = g.to(device)

        with torch.no_grad():
            z, _, sv_attns = model.encode(data, return_attention=True)

        sf, cn = sf_computer.compute(g.edge_index, g.x.size(0), g.edge_index,
                                      node_embeddings=z, tissue_labels=g.tissue_labels,
                                      slice_ids=g.slice_ids)
        sf = sf.to(device)
        src_t, dst_t, is_inter = get_edge_metadata(g, g.edge_index)

        # Explain a sample of edges (max 5 per graph for efficiency)
        n_explain = min(5, g.edge_index.size(1))
        for idx in range(n_explain):
            explanation = gnn_exp.explain_edge(
                data, idx, z=z, structural_feats=sf, cn_indices=cn,
                src_tissue=src_t, dst_tissue=dst_t, is_inter=is_inter,
            )
            std_expl = gnn_exp.to_standard_explanation(
                explanation, data, sv_attns, sf, idx, top_k_svs=top_k_svs,
            )
            gnn_exp_explanations.append(std_expl)

    results["gnn_explainer"] = {"explanations": gnn_exp_explanations}

    # ── Grad-CAM ──
    print("    Running Grad-CAM...")
    grad_cam = GradCAMExplainer(model, device=device)
    grad_cam_explanations = []

    for g in test_graphs:
        if g.edge_index.size(1) < 2 or g.x.size(0) < 3:
            continue
        data = g.to(device)

        sf, cn = sf_computer.compute(g.edge_index, g.x.size(0), g.edge_index)
        sf = sf.to(device)
        src_t, dst_t, is_inter = get_edge_metadata(g, g.edge_index)

        n_explain = min(5, g.edge_index.size(1))
        for idx in range(n_explain):
            explanation = grad_cam.explain_edge(
                data, idx, structural_feats=sf, cn_indices=cn,
                src_tissue=src_t, dst_tissue=dst_t, is_inter=is_inter,
            )
            std_expl = grad_cam.to_standard_explanation(
                explanation, data, sf, idx, top_k_svs=top_k_svs,
            )
            grad_cam_explanations.append(std_expl)

    results["grad_cam"] = {"explanations": grad_cam_explanations}

    # ── Attention-only ──
    print("    Running attention-only baseline...")
    attn_only = AttentionOnlyExplainer(model, device=device)
    attn_only_explanations = []

    for g in test_graphs:
        if g.edge_index.size(1) < 2 or g.x.size(0) < 3:
            continue
        data = g.to(device)

        with torch.no_grad():
            _, _, sv_attns = model.encode(data, return_attention=True)

        sf, cn = sf_computer.compute(g.edge_index, g.x.size(0), g.edge_index)
        sf = sf.to(device)

        n_explain = min(5, g.edge_index.size(1))
        for idx in range(n_explain):
            explanation = attn_only.explain_edge(data, idx)
            std_expl = attn_only.to_standard_explanation(
                explanation, data, sv_attns, sf, idx, top_k_svs=top_k_svs,
            )
            attn_only_explanations.append(std_expl)

    results["attention_only"] = {"explanations": attn_only_explanations}

    # ── Stability: compare intrinsic vs each post-hoc ──
    for method in ["gnn_explainer", "grad_cam", "attention_only"]:
        n_compare = min(len(results["intrinsic"]["explanations"]),
                        len(results[method]["explanations"]))
        if n_compare > 0:
            stability = ExplainabilityMetrics.stability(
                results["intrinsic"]["explanations"][:n_compare],
                results[method]["explanations"][:n_compare],
            )
            results[method]["vs_intrinsic_stability"] = stability

    return results


# ── Single Run ────────────────────────────────────────────────────────

def run_single(config_name, exp_config, seed, graphs_cache):
    """Run one config with one seed. Returns metrics dict."""
    set_seed(seed)
    device = SHARED["device"]

    # Get or build graphs
    cache_key = (exp_config.edge_strategy, exp_config.k_neighbors, exp_config.use_sv_aggregation)
    if cache_key not in graphs_cache:
        graphs_cache[cache_key] = build_graphs_for_config(exp_config)
    train_g, val_g, test_g = graphs_cache[cache_key]

    # Create model
    gnn_override = config_to_gnn_override(exp_config)
    abl_config = config_to_ablation_config(exp_config)
    model = AblationModel(ablation_config=abl_config, gnn_config=gnn_override).to(device)
    sf_computer = AblationStructuralComputer(use_ocn=exp_config.use_ocn_features)
    param_count = sum(p.numel() for p in model.parameters())

    print(f"  Seed {seed} | {model.describe()}")

    # Train
    best_val_auc, train_time = train_model(
        model, sf_computer, train_g, val_g,
        config_name=config_name,
    )

    # Link prediction performance
    test_metrics = evaluate_gnn(model, test_g, sf_computer)

    # Explainability metrics
    _, xai_summary = evaluate_explainability(model, test_g, sf_computer, device)

    result = {
        "config": config_name,
        "seed": seed,
        "edge_strategy": exp_config.edge_strategy,
        "k_neighbors": exp_config.k_neighbors,
        "num_gnn_layers": exp_config.num_gnn_layers,
        "use_sv": exp_config.use_sv_aggregation,
        "use_topo": exp_config.use_intra_topology,
        "use_ocn": exp_config.use_ocn_features,
        "param_count": param_count,
        "train_time_sec": round(train_time, 1),
        # Link prediction
        "best_val_auc": round(best_val_auc, 4) if best_val_auc else None,
        "test_auc": round(test_metrics["auc"], 4) if test_metrics["auc"] else None,
        "test_ap": round(test_metrics["ap"], 4) if test_metrics["ap"] else None,
        "test_intra_auc": round(test_metrics["intra_auc"], 4) if test_metrics.get("intra_auc") else None,
        "test_inter_auc": round(test_metrics["inter_auc"], 4) if test_metrics.get("inter_auc") else None,
        # Explainability
        **{f"xai_{k}": round(v, 4) for k, v in xai_summary.items()},
    }

    if test_metrics["auc"]:
        print(f"    Test AUC: {test_metrics['auc']:.4f} | "
              f"AP: {test_metrics['ap']:.4f} | "
              f"Fid+: {xai_summary.get('combined_drop_mean', 0):.4f} | "
              f"Time: {train_time:.0f}s")

    return result, model, (train_g, val_g, test_g), sf_computer


# ── Full Study ────────────────────────────────────────────────────────

def run_explainability_study(config_names=None, seeds=None, run_posthoc=True):
    """Run the full explainability study.

    Args:
        config_names: list of config keys to run. None = all.
        seeds: list of seeds. None = all from config.
        run_posthoc: whether to run post-hoc baselines on D_full.
    """
    config_names = config_names or CONFIG_ORDER
    seeds = seeds or SEEDS

    # Deduplicate: D_full == K0_compat_L2
    if "D_full" in config_names and "K0_compat_L2" in config_names:
        config_names = [c for c in config_names if c != "D_full"]
        print("Note: D_full == K0_compat_L2, deduplicating (training only once).")

    print("=" * 70)
    print("EXPLAINABILITY STUDY")
    print("=" * 70)
    print(f"Configs: {config_names}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {EPOCHS} | LR: {LR} | WD: {WEIGHT_DECAY}")
    print(f"Post-hoc baselines: {run_posthoc}")
    print(f"Device: {SHARED['device']}")

    # Cache built graphs (keyed by edge strategy + k + sv flag)
    graphs_cache = {}
    all_results = []
    total_start = time.time()

    # ── Train and evaluate each config x seed ──
    last_full_model = None
    last_full_graphs = None
    last_full_sf_computer = None

    for config_name in config_names:
        exp_config = CONFIGS[config_name]
        print(f"\n{'─' * 60}")
        print(f"Config: {config_name}")
        print(f"  Strategy={exp_config.edge_strategy} | k={exp_config.k_neighbors} | "
              f"Layers={exp_config.num_gnn_layers}")
        print(f"  SV={exp_config.use_sv_aggregation} | "
              f"Topo={exp_config.use_intra_topology} | "
              f"OCN={exp_config.use_ocn_features}")
        print(f"{'─' * 60}")

        for seed in seeds:
            result, model, graphs, sf_comp = run_single(
                config_name, exp_config, seed, graphs_cache,
            )
            all_results.append(result)

            # Keep last D_full/K0_compat_L2 model for post-hoc
            is_full = (exp_config.use_sv_aggregation and exp_config.use_ocn_features
                       and exp_config.use_intra_topology
                       and exp_config.edge_strategy == "compatibility_only"
                       and exp_config.num_gnn_layers == 2)
            if is_full:
                last_full_model = model
                last_full_graphs = graphs
                last_full_sf_computer = sf_comp

    # ── Post-hoc comparison on D_full ──
    posthoc_results = None
    if run_posthoc and last_full_model is not None:
        print(f"\n{'=' * 60}")
        print("POST-HOC BASELINES (on D_full / K0_compat_L2)")
        print(f"{'=' * 60}")

        _, _, test_g = last_full_graphs
        posthoc_results = evaluate_posthoc(
            last_full_model, test_g, last_full_sf_computer, SHARED["device"],
        )

        # Print comparison summary
        print("\n  Post-hoc vs Intrinsic stability:")
        for method in ["gnn_explainer", "grad_cam", "attention_only"]:
            if "vs_intrinsic_stability" in posthoc_results.get(method, {}):
                stab = posthoc_results[method]["vs_intrinsic_stability"]
                print(f"    {method:20s} | SV Jaccard: {stab['sv_jaccard']:.4f} | "
                      f"SF Corr: {stab['sf_correlation']:.4f}")

    # ── Save results ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / "results.json"
    with open(str(results_path), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    if posthoc_results:
        posthoc_path = OUTPUT_DIR / "posthoc_results.json"
        # Serialize: strip non-serializable tensors
        serializable = {}
        for method, data in posthoc_results.items():
            serializable[method] = {
                "summary": data.get("summary", {}),
                "vs_intrinsic_stability": data.get("vs_intrinsic_stability", {}),
                "n_explanations": len(data.get("explanations", [])),
            }
        with open(str(posthoc_path), "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"Post-hoc results saved to {posthoc_path}")

    # ── Print summary table ──
    print_summary(all_results, config_names)

    total_time = time.time() - total_start
    print(f"\nTotal study time: {total_time/60:.1f} minutes")

    return all_results, posthoc_results


def print_summary(all_results, config_names):
    """Print a summary table of mean and std across seeds."""
    print(f"\n{'=' * 100}")
    print("EXPLAINABILITY STUDY RESULTS")
    print(f"{'=' * 100}")
    print(f"{'Config':<16s} {'AUC':>10s} {'AP':>10s} "
          f"{'Fid+(comb)':>12s} {'Fid-(L2)':>12s} {'Sparsity':>10s} "
          f"{'Cmplx':>8s} {'Time(s)':>8s}")
    print("-" * 100)

    for config_name in config_names:
        runs = [r for r in all_results if r["config"] == config_name]
        if not runs:
            continue

        def fmt(key):
            vals = [r[key] for r in runs if r.get(key) is not None]
            if not vals:
                return "N/A"
            return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"

        aucs = fmt("test_auc")
        aps = fmt("test_ap")
        fid_plus = fmt("xai_combined_drop_mean")
        fid_minus = fmt("xai_level2_retention_mean")
        sparsity = fmt("xai_sv_sparsity_mean")
        complexity = fmt("xai_complexity_mean")
        times = [r["train_time_sec"] for r in runs]

        print(f"{config_name:<16s} {aucs:>10s} {aps:>10s} "
              f"{fid_plus:>12s} {fid_minus:>12s} {sparsity:>10s} "
              f"{complexity:>8s} {np.mean(times):>7.0f}s")


# ── Entry Point ───────────────────────────────────────────────────────
#
# All parameters controlled via constants in explainability/config.py:
#   CONFIG_ORDER, SEEDS, EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP
#
# To run a subset, edit those constants before running.

if __name__ == "__main__":
    run_explainability_study(
        config_names=CONFIG_ORDER,
        seeds=SEEDS,
        run_posthoc=True,
    )

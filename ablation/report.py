"""
Ablation Study Report Generator

Loads results from ablation/runner.py and generates:
  1. Summary table (mean +/- std across seeds)
  2. Bar charts with error bars (AUC, AP, Intra/Inter)
  3. Tissue-pair heatmap per config
  4. Paired t-tests for statistical significance

Usage:
    python -m ablation.report                          # default path
    python -m ablation.report --results path/to/results.json
    python -m ablation.report --save                   # save plots to disk
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ablation.config import CONFIG_ORDER, OUTPUT_DIR


# ── Data Loading ──────────────────────────────────────────────────────

def load_results(path=None):
    """Load ablation results JSON."""
    path = path or (OUTPUT_DIR / "results.json")
    with open(str(path)) as f:
        return json.load(f)


def group_by_config(results):
    """Group results by config name. Returns dict of config -> list of runs."""
    grouped = {}
    for r in results:
        cfg = r["config"]
        if cfg not in grouped:
            grouped[cfg] = []
        grouped[cfg].append(r)
    return grouped


# ── Summary Table ─────────────────────────────────────────────────────

def print_summary_table(results):
    """Print formatted summary table."""
    grouped = group_by_config(results)

    print()
    print("=" * 90)
    print("ABLATION STUDY RESULTS")
    print("=" * 90)
    print()

    header = (f"{'Config':<16s} | {'AUC-ROC':>16s} | {'AP':>16s} | "
              f"{'Intra-AUC':>16s} | {'Inter-AUC':>16s}")
    print(header)
    print("-" * 90)

    for cfg_name in CONFIG_ORDER:
        if cfg_name not in grouped:
            continue
        runs = grouped[cfg_name]

        aucs = [r["test_auc"] for r in runs if r["test_auc"] is not None]
        aps = [r["test_ap"] for r in runs if r["test_ap"] is not None]
        intras = [r["test_intra_auc"] for r in runs if r["test_intra_auc"] is not None]
        inters = [r["test_inter_auc"] for r in runs if r["test_inter_auc"] is not None]

        def fmt(vals):
            if not vals:
                return "N/A"
            return f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"

        print(f"{cfg_name:<16s} | {fmt(aucs):>16s} | {fmt(aps):>16s} | "
              f"{fmt(intras):>16s} | {fmt(inters):>16s}")

    print("-" * 90)

    # Additional info
    print()
    print(f"{'Config':<16s} | {'Params':>10s} | {'Train Time':>12s} | {'Best Val AUC':>14s}")
    print("-" * 60)
    for cfg_name in CONFIG_ORDER:
        if cfg_name not in grouped:
            continue
        runs = grouped[cfg_name]
        params = runs[0]["param_count"]
        times = [r["train_time_sec"] for r in runs]
        val_aucs = [r["best_val_auc"] for r in runs if r["best_val_auc"] is not None]

        time_str = f"{np.mean(times):.0f} +/- {np.std(times):.0f}s"
        val_str = f"{np.mean(val_aucs):.4f}" if val_aucs else "N/A"
        print(f"{cfg_name:<16s} | {params:>10,d} | {time_str:>12s} | {val_str:>14s}")

    print()


# ── Statistical Significance ─────────────────────────────────────────

def run_significance_tests(results):
    """Run paired t-tests between key config pairs.

    Tests the following hypotheses:
      - D_full > A_baseline (full model beats baseline)
      - B_sv_topo > A_baseline (SV+topo beats baseline)
      - B_sv_topo > B_prime_sv (topology adds value)
      - C_ocn_only > A_baseline (OCN beats baseline)
      - D_full > B_sv_topo (adding OCN helps)
      - D_full > C_ocn_only (adding SV helps)
    """
    grouped = group_by_config(results)

    test_pairs = [
        ("D_full", "A_baseline", "Full model vs Baseline"),
        ("B_sv_topo", "A_baseline", "SV+Topo vs Baseline"),
        ("B_prime_sv", "A_baseline", "SV-only vs Baseline"),
        ("B_sv_topo", "B_prime_sv", "Topology contribution"),
        ("C_ocn_only", "A_baseline", "OCN vs Baseline"),
        ("D_full", "B_sv_topo", "Adding OCN to SV+Topo"),
        ("D_full", "C_ocn_only", "Adding SV to OCN"),
    ]

    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE (paired t-test, one-sided)")
    print("=" * 70)
    print()
    print(f"{'Comparison':<30s} | {'Mean Diff':>10s} | {'t-stat':>8s} | {'p-value':>8s} | {'Sig?':>5s}")
    print("-" * 70)

    for better, worse, description in test_pairs:
        if better not in grouped or worse not in grouped:
            print(f"{description:<30s} | {'SKIP':>10s} |")
            continue

        better_aucs = sorted([r["test_auc"] for r in grouped[better]
                              if r["test_auc"] is not None])
        worse_aucs = sorted([r["test_auc"] for r in grouped[worse]
                             if r["test_auc"] is not None])

        # Match by seed order (both sorted by seed in runner)
        n = min(len(better_aucs), len(worse_aucs))
        if n < 2:
            print(f"{description:<30s} | {'N<2':>10s} |")
            continue

        b = np.array(better_aucs[:n])
        w = np.array(worse_aucs[:n])
        diff = b - w
        mean_diff = np.mean(diff)

        # One-sided paired t-test: H1: better > worse
        t_stat, p_two = stats.ttest_rel(b, w)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

        sig = "***" if p_one < 0.001 else "**" if p_one < 0.01 else "*" if p_one < 0.05 else ""

        print(f"{description:<30s} | {mean_diff:>+10.4f} | {t_stat:>8.3f} | {p_one:>8.4f} | {sig:>5s}")

    print()
    print("Significance levels: * p<0.05, ** p<0.01, *** p<0.001")
    print()


# ── Plots ─────────────────────────────────────────────────────────────

def plot_auc_comparison(results, save_path=None):
    """Bar chart of AUC-ROC and AP with error bars."""
    grouped = group_by_config(results)

    configs = [c for c in CONFIG_ORDER if c in grouped]
    auc_means, auc_stds = [], []
    ap_means, ap_stds = [], []

    for cfg in configs:
        runs = grouped[cfg]
        aucs = [r["test_auc"] for r in runs if r["test_auc"] is not None]
        aps = [r["test_ap"] for r in runs if r["test_ap"] is not None]
        auc_means.append(np.mean(aucs) if aucs else 0)
        auc_stds.append(np.std(aucs) if aucs else 0)
        ap_means.append(np.mean(aps) if aps else 0)
        ap_stds.append(np.std(aps) if aps else 0)

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, auc_means, width, yerr=auc_stds,
                   label="AUC-ROC", color="#3498db", alpha=0.85,
                   edgecolor="black", linewidth=0.5, capsize=4)
    bars2 = ax.bar(x + width/2, ap_means, width, yerr=ap_stds,
                   label="Average Precision", color="#2ecc71", alpha=0.85,
                   edgecolor="black", linewidth=0.5, capsize=4)

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Ablation Study: AUC-ROC and AP by Configuration", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.3f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_intra_inter_comparison(results, save_path=None):
    """Grouped bar chart of intra-slice vs inter-slice AUC."""
    grouped = group_by_config(results)
    configs = [c for c in CONFIG_ORDER if c in grouped]

    intra_means, intra_stds = [], []
    inter_means, inter_stds = [], []

    for cfg in configs:
        runs = grouped[cfg]
        intras = [r["test_intra_auc"] for r in runs if r["test_intra_auc"] is not None]
        inters = [r["test_inter_auc"] for r in runs if r["test_inter_auc"] is not None]
        intra_means.append(np.mean(intras) if intras else 0)
        intra_stds.append(np.std(intras) if intras else 0)
        inter_means.append(np.mean(inters) if inters else 0)
        inter_stds.append(np.std(inters) if inters else 0)

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width/2, intra_means, width, yerr=intra_stds,
           label="Intra-slice AUC", color="#e74c3c", alpha=0.85,
           edgecolor="black", linewidth=0.5, capsize=4)
    ax.bar(x + width/2, inter_means, width, yerr=inter_stds,
           label="Inter-slice AUC", color="#9b59b6", alpha=0.85,
           edgecolor="black", linewidth=0.5, capsize=4)

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("Ablation Study: Intra-slice vs Inter-slice Performance", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_component_contribution(results, save_path=None):
    """Stacked delta chart showing how each component improves over baseline."""
    grouped = group_by_config(results)

    if "A_baseline" not in grouped:
        print("Cannot plot component contribution without A_baseline results.")
        return

    baseline_auc = np.mean([r["test_auc"] for r in grouped["A_baseline"]
                            if r["test_auc"] is not None])

    components = []
    deltas = []
    colors = []

    comparisons = [
        ("B_prime_sv", "SV Aggregation", "#3498db"),
        ("B_sv_topo", "SV + Topology", "#2ecc71"),
        ("C_ocn_only", "OCN Features", "#e74c3c"),
        ("D_full", "Full Model", "#f39c12"),
    ]

    for cfg_name, label, color in comparisons:
        if cfg_name not in grouped:
            continue
        cfg_auc = np.mean([r["test_auc"] for r in grouped[cfg_name]
                           if r["test_auc"] is not None])
        delta = cfg_auc - baseline_auc
        components.append(label)
        deltas.append(delta)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(components, deltas, color=colors, alpha=0.85,
                   edgecolor="black", linewidth=0.5)

    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("AUC Improvement over Baseline", fontsize=12)
    ax.set_title(f"Component Contributions (Baseline AUC = {baseline_auc:.4f})", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    for bar, delta in zip(bars, deltas):
        x_pos = bar.get_width()
        sign = "+" if delta >= 0 else ""
        ax.annotate(f"{sign}{delta:.4f}",
                    xy=(x_pos, bar.get_y() + bar.get_height() / 2),
                    xytext=(5 if delta >= 0 else -5, 0),
                    textcoords="offset points",
                    ha="left" if delta >= 0 else "right",
                    va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_seed_variance(results, save_path=None):
    """Box plot showing variance across seeds for each config."""
    grouped = group_by_config(results)
    configs = [c for c in CONFIG_ORDER if c in grouped]

    data_to_plot = []
    labels = []
    for cfg in configs:
        aucs = [r["test_auc"] for r in grouped[cfg] if r["test_auc"] is not None]
        if aucs:
            data_to_plot.append(aucs)
            labels.append(cfg.replace("_", "\n"))

    if not data_to_plot:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.5)

    box_colors = ["#3498db", "#2ecc71", "#1abc9c", "#e74c3c", "#f39c12"]
    for patch, color in zip(bp["boxes"], box_colors[:len(data_to_plot)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Test AUC-ROC", fontsize=12)
    ax.set_title("Ablation Study: AUC Distribution Across Seeds", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ── Full Report ───────────────────────────────────────────────────────

def generate_report(results_path=None, save_plots=False):
    """Generate the full ablation report."""
    results = load_results(results_path)
    print(f"Loaded {len(results)} run results.")

    # Summary table
    print_summary_table(results)

    # Statistical significance
    run_significance_tests(results)

    # Plots
    plot_dir = OUTPUT_DIR / "plots" if save_plots else None
    if plot_dir:
        plot_dir.mkdir(parents=True, exist_ok=True)

    plot_auc_comparison(
        results,
        save_path=plot_dir / "auc_comparison.png" if plot_dir else None,
    )
    plot_intra_inter_comparison(
        results,
        save_path=plot_dir / "intra_inter_comparison.png" if plot_dir else None,
    )
    plot_component_contribution(
        results,
        save_path=plot_dir / "component_contribution.png" if plot_dir else None,
    )
    plot_seed_variance(
        results,
        save_path=plot_dir / "seed_variance.png" if plot_dir else None,
    )


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ablation study report")
    parser.add_argument(
        "--results", type=str, default=None,
        help=f"Path to results.json (default: {OUTPUT_DIR / 'results.json'})",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save plots to disk instead of just showing them",
    )
    args = parser.parse_args()

    generate_report(
        results_path=args.results,
        save_plots=args.save,
    )

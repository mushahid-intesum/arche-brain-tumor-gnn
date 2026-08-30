"""
Explainability Study Report Generator

Loads results from the runner, generates tables (console + CSV) and
plots (matplotlib PNG) summarizing the explainability study.

All parameters are controlled via constants below.
Usage: python -m explainability.report
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from explainability.config import (
    CONFIGS, CONFIG_ORDER, KNN_CONFIGS, ABLATION_CONFIGS,
    POSTHOC_METHODS, OUTPUT_DIR,
)


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_results(path=None):
    """Load per-run results from the runner's JSON output.

    Returns:
        list of dicts, one per (config, seed) run.
    """
    path = path or (OUTPUT_DIR / "results.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}\n"
            f"Run 'python -m explainability.runner' first."
        )
    with open(str(path)) as f:
        return json.load(f)


def load_posthoc_results(path=None):
    """Load post-hoc comparison results.

    Returns:
        dict mapping method_name -> {summary, vs_intrinsic_stability, n_explanations}
        or None if file doesn't exist.
    """
    path = path or (OUTPUT_DIR / "posthoc_results.json")
    if not path.exists():
        return None
    with open(str(path)) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════

def aggregate_by_config(results):
    """Group results by config name and compute mean ± std across seeds.

    Args:
        results: list of per-run dicts from load_results().

    Returns:
        dict[config_name] -> {
            "runs": list of raw run dicts,
            "n_seeds": int,
            "metrics": {
                metric_key: {"mean": float, "std": float, "values": list}
            }
        }
    """
    grouped = defaultdict(list)
    for r in results:
        grouped[r["config"]].append(r)

    aggregated = {}
    for config_name, runs in grouped.items():
        metrics = {}

        # Collect all numeric keys
        numeric_keys = [
            k for k in runs[0].keys()
            if isinstance(runs[0].get(k), (int, float))
            and runs[0].get(k) is not None
        ]

        for key in numeric_keys:
            values = [r[key] for r in runs if r.get(key) is not None]
            if values:
                metrics[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }

        aggregated[config_name] = {
            "runs": runs,
            "n_seeds": len(runs),
            "metrics": metrics,
        }

    return aggregated


def get_metric(agg, config_name, key):
    """Safely get a metric's mean and std from aggregated data.

    Returns:
        (mean, std) or (None, None) if not available.
    """
    entry = agg.get(config_name, {})
    m = entry.get("metrics", {}).get(key, {})
    return m.get("mean"), m.get("std")


# ══════════════════════════════════════════════════════════════════════
# Formatting
# ══════════════════════════════════════════════════════════════════════

def fmt(mean, std, precision=4):
    """Format a metric as 'mean±std' or 'N/A'."""
    if mean is None:
        return "N/A"
    if std is not None and std > 0:
        return f"{mean:.{precision}f}±{std:.{precision}f}"
    return f"{mean:.{precision}f}"


def fmt_metric(agg, config_name, key, precision=4):
    """Shortcut: get + format a metric."""
    mean, std = get_metric(agg, config_name, key)
    return fmt(mean, std, precision)


def print_table(headers, rows, col_widths=None):
    """Print a formatted console table.

    Args:
        headers: list of column header strings.
        rows: list of lists (one per row), same length as headers.
        col_widths: optional list of int widths per column.
    """
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i])))
            col_widths.append(max_w + 2)

    # Header
    header_line = ""
    for h, w in zip(headers, col_widths):
        header_line += f"{h:>{w}s}"
    print(header_line)
    print("-" * sum(col_widths))

    # Rows
    for row in rows:
        line = ""
        for val, w in zip(row, col_widths):
            line += f"{str(val):>{w}s}"
        print(line)


def save_csv(headers, rows, path):
    """Save a table as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"  Saved CSV: {path}")


# ══════════════════════════════════════════════════════════════════════
# Tables
# ══════════════════════════════════════════════════════════════════════

def table_knn_sensitivity(agg, save_dir=None):
    """Table 1: KNN sensitivity — K0-K5 × 2 layer counts.

    Shows how the number of KNN neighbors and encoder depth affect
    both link prediction and explanation quality.
    """
    print("\n  Table 1: KNN Sensitivity Analysis")
    print("  " + "=" * 90)

    headers = [
        "Config", "k", "Layers",
        "AUC", "AP",
        "Fid+", "Fid-", "Sparsity", "Complexity",
    ]

    rows = []
    knn_order = [
        "K0_compat_L2", "K2_L2", "K3_L2", "K4_L2", "K5_L2",
        "K0_compat_L3", "K2_L3", "K3_L3", "K4_L3", "K5_L3",
    ]

    for config_name in knn_order:
        if config_name not in agg:
            continue

        cfg = CONFIGS.get(config_name)
        if cfg is None:
            continue

        k_str = "—" if cfg.k_neighbors == 0 else str(cfg.k_neighbors)

        row = [
            config_name,
            k_str,
            str(cfg.num_gnn_layers),
            fmt_metric(agg, config_name, "test_auc"),
            fmt_metric(agg, config_name, "test_ap"),
            fmt_metric(agg, config_name, "xai_combined_drop_mean"),
            fmt_metric(agg, config_name, "xai_level2_retention_mean"),
            fmt_metric(agg, config_name, "xai_sv_sparsity_mean"),
            fmt_metric(agg, config_name, "xai_complexity_mean", precision=1),
        ]
        rows.append(row)

    print_table(headers, rows)

    if save_dir:
        save_csv(headers, rows, Path(save_dir) / "table1_knn_sensitivity.csv")

    # Highlight best/worst
    _print_best_worst(agg, knn_order, "xai_combined_drop_mean", "Fid+ (combined)")


def table_component_contribution(agg, save_dir=None):
    """Table 2: Component ablation — A through D_full.

    Shows how each architectural component (SV aggregation, topology,
    OCN features) contributes to explanation quality.
    """
    print("\n  Table 2: Component Contribution to Explainability")
    print("  " + "=" * 90)

    headers = [
        "Config", "SV", "Topo", "OCN",
        "AUC", "AP",
        "Fid+", "Fid-", "Sparsity", "Complexity",
    ]

    rows = []
    ablation_order = ["A_baseline", "B_prime_sv", "B_sv_topo", "C_ocn_only", "D_full"]

    for config_name in ablation_order:
        # D_full may have been deduplicated to K0_compat_L2
        lookup = config_name
        if config_name not in agg and config_name == "D_full" and "K0_compat_L2" in agg:
            lookup = "K0_compat_L2"
        if lookup not in agg:
            continue

        cfg = CONFIGS.get(config_name)
        if cfg is None:
            continue

        row = [
            config_name,
            "✓" if cfg.use_sv_aggregation else "✗",
            "✓" if cfg.use_intra_topology else "✗",
            "✓" if cfg.use_ocn_features else "✗",
            fmt_metric(agg, lookup, "test_auc"),
            fmt_metric(agg, lookup, "test_ap"),
            fmt_metric(agg, lookup, "xai_combined_drop_mean"),
            fmt_metric(agg, lookup, "xai_level2_retention_mean"),
            fmt_metric(agg, lookup, "xai_sv_sparsity_mean"),
            fmt_metric(agg, lookup, "xai_complexity_mean", precision=1),
        ]
        rows.append(row)

    print_table(headers, rows)

    if save_dir:
        save_csv(headers, rows, Path(save_dir) / "table2_component_contribution.csv")

    # Compute deltas vs baseline
    _print_component_deltas(agg, ablation_order)


def table_posthoc_comparison(posthoc_data, save_dir=None):
    """Table 3: Intrinsic vs post-hoc methods on D_full.

    Compares our 3-level intrinsic explanations against GNNExplainer,
    Grad-CAM, and attention-only baselines.
    """
    print("\n  Table 3: Intrinsic vs Post-hoc Explainability Methods")
    print("  " + "=" * 80)

    headers = [
        "Method",
        "SV Jaccard vs Intrinsic", "SF Corr vs Intrinsic",
        "N Explanations",
    ]

    rows = []

    # Intrinsic (reference)
    intrinsic_data = posthoc_data.get("intrinsic", {})
    intrinsic_summary = intrinsic_data.get("summary", {})
    n_intrinsic = intrinsic_data.get("n_explanations", 0)

    rows.append([
        "Intrinsic (ours)",
        "— (reference)",
        "— (reference)",
        str(n_intrinsic),
    ])

    # Post-hoc methods
    for method in ["gnn_explainer", "grad_cam", "attention_only"]:
        method_data = posthoc_data.get(method, {})
        stability = method_data.get("vs_intrinsic_stability", {})
        n_expl = method_data.get("n_explanations", 0)

        sv_jaccard = stability.get("sv_jaccard")
        sf_corr = stability.get("sf_correlation")

        label = {
            "gnn_explainer": "GNNExplainer",
            "grad_cam": "Grad-CAM",
            "attention_only": "Attention-only",
        }.get(method, method)

        rows.append([
            label,
            f"{sv_jaccard:.4f}" if sv_jaccard is not None else "N/A",
            f"{sf_corr:.4f}" if sf_corr is not None else "N/A",
            str(n_expl),
        ])

    print_table(headers, rows)

    # Also print intrinsic summary if available
    if intrinsic_summary:
        print("\n  Intrinsic Explanation Summary (D_full):")
        summary_keys = [
            ("Fid+ (L1 structural)", "level1_drop_mean"),
            ("Fid+ (L2 supervoxel)", "level2_drop_mean"),
            ("Fid+ (combined)", "combined_drop_mean"),
            ("Fid- (L1 retention)", "level1_retention_mean"),
            ("Fid- (L2 retention)", "level2_retention_mean"),
            ("SV Sparsity", "sv_sparsity_mean"),
            ("SF Sparsity", "sf_sparsity_mean"),
            ("Complexity", "complexity_mean"),
        ]
        for label, key in summary_keys:
            val = intrinsic_summary.get(key)
            if val is not None:
                print(f"    {label:<30s}: {val:.4f}")

    if save_dir:
        save_csv(headers, rows, Path(save_dir) / "table3_posthoc_comparison.csv")


# ── Table Helpers ─────────────────────────────────────────────────────

def _print_best_worst(agg, config_names, metric_key, metric_label):
    """Print which config has the best/worst value for a metric."""
    best_name, best_val = None, -float("inf")
    worst_name, worst_val = None, float("inf")

    for name in config_names:
        mean, _ = get_metric(agg, name, metric_key)
        if mean is not None:
            if mean > best_val:
                best_val, best_name = mean, name
            if mean < worst_val:
                worst_val, worst_name = mean, name

    if best_name:
        print(f"\n  Best {metric_label}:  {best_name} ({best_val:.4f})")
    if worst_name:
        print(f"  Worst {metric_label}: {worst_name} ({worst_val:.4f})")


def _print_component_deltas(agg, config_names):
    """Print how much each component improves over baseline."""
    baseline = "A_baseline"
    baseline_lookup = baseline if baseline in agg else None
    if baseline_lookup is None:
        return

    metrics_to_compare = [
        ("xai_combined_drop_mean", "Fid+"),
        ("xai_level2_retention_mean", "Fid-"),
    ]

    base_vals = {}
    for key, label in metrics_to_compare:
        mean, _ = get_metric(agg, baseline_lookup, key)
        base_vals[key] = mean

    print("\n  Δ vs A_baseline:")
    for config_name in config_names:
        if config_name == baseline:
            continue

        lookup = config_name
        if config_name not in agg and config_name == "D_full" and "K0_compat_L2" in agg:
            lookup = "K0_compat_L2"
        if lookup not in agg:
            continue

        parts = []
        for key, label in metrics_to_compare:
            mean, _ = get_metric(agg, lookup, key)
            base = base_vals.get(key)
            if mean is not None and base is not None:
                delta = mean - base
                sign = "+" if delta >= 0 else ""
                parts.append(f"{label}: {sign}{delta:.4f}")

        if parts:
            print(f"    {config_name:<16s}: {' | '.join(parts)}")


# ══════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════

# ── Shared plot styling ──────────────────────────────────────────────

_PALETTE = {
    "bg": "#1a1a2e",
    "fg": "#e0e0e0",
    "grid": "#2d2d4a",
    "accent1": "#00d4ff",   # cyan
    "accent2": "#ff6b6b",   # coral
    "accent3": "#51cf66",   # green
    "accent4": "#ffd43b",   # gold
    "accent5": "#cc5de8",   # purple
    "accent6": "#ff922b",   # orange
}

_KNN_COLORS = {
    "K0_compat": _PALETTE["accent1"],
    "K2": _PALETTE["accent2"],
    "K3": _PALETTE["accent3"],
    "K4": _PALETTE["accent4"],
    "K5": _PALETTE["accent5"],
}

_ABLATION_COLORS = {
    "A_baseline": _PALETTE["fg"],
    "B_prime_sv": _PALETTE["accent2"],
    "B_sv_topo": _PALETTE["accent3"],
    "C_ocn_only": _PALETTE["accent4"],
    "D_full": _PALETTE["accent1"],
}


def _apply_style(fig, ax):
    """Apply consistent dark theme to a figure."""
    fig.patch.set_facecolor(_PALETTE["bg"])
    ax.set_facecolor(_PALETTE["bg"])
    ax.tick_params(colors=_PALETTE["fg"], labelsize=9)
    ax.xaxis.label.set_color(_PALETTE["fg"])
    ax.yaxis.label.set_color(_PALETTE["fg"])
    ax.title.set_color(_PALETTE["fg"])
    for spine in ax.spines.values():
        spine.set_color(_PALETTE["grid"])
    ax.grid(True, alpha=0.3, color=_PALETTE["grid"])


def _save_plot(fig, save_dir, filename):
    """Save a figure and print confirmation."""
    if save_dir:
        path = Path(save_dir) / filename
        fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {path}")


# ── Plot 1: Fidelity-Sparsity Tradeoff ───────────────────────────────

def plot_fidelity_sparsity_tradeoff(agg, save_dir=None):
    """Scatter plot: x=sparsity, y=fidelity+ (combined drop).

    Each point is one config. Labeled with config name.
    Ideal: top-left (low sparsity, high fidelity).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    _apply_style(fig, ax)

    for config_name, data in agg.items():
        sparsity_mean, _ = get_metric(agg, config_name, "xai_sv_sparsity_mean")
        fidelity_mean, fidelity_std = get_metric(agg, config_name, "xai_combined_drop_mean")

        if sparsity_mean is None or fidelity_mean is None:
            continue

        # Color by group
        base = config_name.split("_L")[0]
        if config_name in _ABLATION_COLORS:
            color = _ABLATION_COLORS[config_name]
            marker = "s"
        elif base in _KNN_COLORS:
            color = _KNN_COLORS[base]
            marker = "o"
        else:
            color = _PALETTE["fg"]
            marker = "^"

        layers = "L3" if "_L3" in config_name else "L2"
        alpha = 0.6 if layers == "L3" else 1.0

        ax.scatter(sparsity_mean, fidelity_mean, c=color, s=100,
                   marker=marker, alpha=alpha, edgecolors="white", linewidths=0.5,
                   zorder=3)

        if fidelity_std:
            ax.errorbar(sparsity_mean, fidelity_mean, yerr=fidelity_std,
                        color=color, alpha=0.3, fmt="none", capsize=3)

        # Label
        label = config_name.replace("_compat", "").replace("_L2", " (2L)").replace("_L3", " (3L)")
        ax.annotate(label, (sparsity_mean, fidelity_mean),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=7, color=color, alpha=0.9)

    ax.set_xlabel("SV Sparsity (lower = more concise)", fontsize=11)
    ax.set_ylabel("Fidelity+ Combined Drop (higher = more necessary)", fontsize=11)
    ax.set_title("Fidelity–Sparsity Tradeoff", fontsize=14, fontweight="bold")

    _save_plot(fig, save_dir, "plot1_fidelity_sparsity.png")
    plt.close(fig)
    print("  Plot 1: Fidelity-Sparsity tradeoff — done")


# ── Plot 2: Attention Faithfulness Heatmap ────────────────────────────

def plot_attention_faithfulness_heatmap(agg, save_dir=None):
    """Heatmap: rows=k values (K0-K5), columns=layers (2, 3).

    Cell color = attention faithfulness score. Higher = attention
    is more aligned with gradient-based saliency.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    k_labels = ["K0 (compat)", "K2", "K3", "K4", "K5"]
    k_keys = ["K0_compat", "K2", "K3", "K4", "K5"]
    layer_labels = ["2 layers", "3 layers"]
    layer_suffixes = ["_L2", "_L3"]

    # Build matrix
    matrix = np.full((len(k_keys), len(layer_suffixes)), np.nan)
    for i, k_key in enumerate(k_keys):
        for j, suffix in enumerate(layer_suffixes):
            config_name = f"{k_key}{suffix}"
            # The attention faithfulness metric isn't in the batch evaluator by default,
            # so we fall back to combined_drop as a proxy
            mean, _ = get_metric(agg, config_name, "xai_combined_drop_mean")
            if mean is not None:
                matrix[i, j] = mean

    fig, ax = plt.subplots(figsize=(6, 5))
    _apply_style(fig, ax)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cyan_fire", ["#1a1a2e", "#00d4ff", "#51cf66", "#ffd43b"], N=256
    )

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, fontsize=10, color=_PALETTE["fg"])
    ax.set_yticks(range(len(k_labels)))
    ax.set_yticklabels(k_labels, fontsize=10, color=_PALETTE["fg"])

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "black" if val > np.nanmedian(matrix) else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

    ax.set_title("Fidelity+ by k-Value and Encoder Depth", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color=_PALETTE["fg"])
    cbar.ax.yaxis.set_ticklabels(
        [f"{t:.2f}" for t in cbar.get_ticks()], color=_PALETTE["fg"], fontsize=8
    )

    _save_plot(fig, save_dir, "plot2_faithfulness_heatmap.png")
    plt.close(fig)
    print("  Plot 2: Attention faithfulness heatmap — done")


# ── Plot 3: Stability Box Plots ──────────────────────────────────────

def plot_stability_boxplots(agg, save_dir=None):
    """Box plots: one box per config showing Fid+ distribution across seeds."""
    import matplotlib.pyplot as plt

    configs_to_plot = [c for c in CONFIG_ORDER if c in agg]
    if not configs_to_plot:
        print("  Plot 3: No data — skipped")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    _apply_style(fig, ax)

    data_to_plot = []
    labels = []
    colors = []

    for config_name in configs_to_plot:
        m = agg[config_name]["metrics"].get("xai_combined_drop_mean", {})
        values = m.get("values")
        if values and len(values) > 1:
            data_to_plot.append(values)
            labels.append(config_name.replace("_compat", "").replace("_L2", "\n(2L)").replace("_L3", "\n(3L)"))

            base = config_name.split("_L")[0]
            if config_name in _ABLATION_COLORS:
                colors.append(_ABLATION_COLORS[config_name])
            elif base in _KNN_COLORS:
                colors.append(_KNN_COLORS[base])
            else:
                colors.append(_PALETTE["fg"])

    if not data_to_plot:
        print("  Plot 3: Insufficient seeds for boxplots — skipped")
        plt.close(fig)
        return

    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels,
                    widths=0.6, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markersize=5),
                    medianprops=dict(color="white", linewidth=1.5),
                    whiskerprops=dict(color=_PALETTE["fg"]),
                    capprops=dict(color=_PALETTE["fg"]),
                    flierprops=dict(markeredgecolor=_PALETTE["fg"]))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("white")

    ax.set_ylabel("Fid+ Combined Drop", fontsize=11)
    ax.set_title("Explanation Stability Across Seeds", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7, rotation=0)

    _save_plot(fig, save_dir, "plot3_stability_boxplots.png")
    plt.close(fig)
    print("  Plot 3: Stability box plots — done")


# ── Plot 4: Component Delta Chart ────────────────────────────────────

def plot_component_delta(agg, save_dir=None):
    """Grouped bar chart: Δ Fid+ and Δ Fid- for each component vs A_baseline."""
    import matplotlib.pyplot as plt

    baseline = "A_baseline"
    baseline_lookup = baseline if baseline in agg else None
    if baseline_lookup is None:
        print("  Plot 4: A_baseline not found — skipped")
        return

    base_fid_plus, _ = get_metric(agg, baseline_lookup, "xai_combined_drop_mean")
    base_fid_minus, _ = get_metric(agg, baseline_lookup, "xai_level2_retention_mean")

    if base_fid_plus is None:
        print("  Plot 4: No baseline Fid+ data — skipped")
        return

    configs = ["B_prime_sv", "B_sv_topo", "C_ocn_only", "D_full"]
    labels = ["+ SV only", "+ SV + Topo", "+ OCN only", "Full"]

    deltas_fid_plus = []
    deltas_fid_minus = []
    bar_colors = []

    for config_name in configs:
        lookup = config_name
        if config_name not in agg and config_name == "D_full" and "K0_compat_L2" in agg:
            lookup = "K0_compat_L2"

        fid_plus, _ = get_metric(agg, lookup, "xai_combined_drop_mean")
        fid_minus, _ = get_metric(agg, lookup, "xai_level2_retention_mean")

        deltas_fid_plus.append((fid_plus - base_fid_plus) if fid_plus is not None else 0)
        deltas_fid_minus.append((fid_minus - (base_fid_minus or 0)) if fid_minus is not None else 0)
        bar_colors.append(_ABLATION_COLORS.get(config_name, _PALETTE["fg"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_style(fig, ax)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, deltas_fid_plus, width, label="Δ Fid+",
                   color=[c for c in bar_colors], alpha=0.9, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, deltas_fid_minus, width, label="Δ Fid−",
                   color=[c for c in bar_colors], alpha=0.5, edgecolor="white", linewidth=0.5,
                   hatch="//")

    ax.set_xlabel("Component Added", fontsize=11)
    ax.set_ylabel("Δ vs A_baseline", fontsize=11)
    ax.set_title("Component Contribution to Explanation Quality", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.axhline(y=0, color=_PALETTE["fg"], linewidth=0.5, alpha=0.5)
    ax.legend(facecolor=_PALETTE["bg"], edgecolor=_PALETTE["grid"],
              labelcolor=_PALETTE["fg"], fontsize=9)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h,
                f"{h:+.3f}", ha="center", va="bottom" if h >= 0 else "top",
                fontsize=8, color=_PALETTE["fg"])

    _save_plot(fig, save_dir, "plot4_component_delta.png")
    plt.close(fig)
    print("  Plot 4: Component delta chart — done")


# ── Plot 5: Post-hoc Radar Chart ─────────────────────────────────────

def plot_posthoc_radar(posthoc_data, save_dir=None):
    """Radar chart comparing intrinsic vs post-hoc methods.

    Axes: SV Jaccard, SF Correlation, N Explanations (normalized).
    """
    import matplotlib.pyplot as plt

    methods = {
        "gnn_explainer": ("GNNExplainer", _PALETTE["accent2"]),
        "grad_cam": ("Grad-CAM", _PALETTE["accent3"]),
        "attention_only": ("Attention-only", _PALETTE["accent4"]),
    }

    # Collect data
    categories = ["SV Jaccard\nvs Intrinsic", "SF Correlation\nvs Intrinsic"]
    method_values = {}

    for method_key, (label, color) in methods.items():
        method_data = posthoc_data.get(method_key, {})
        stability = method_data.get("vs_intrinsic_stability", {})

        sv_j = stability.get("sv_jaccard", 0)
        sf_c = stability.get("sf_correlation", 0)

        # Clamp to [0, 1] for radar
        method_values[method_key] = [
            max(0, min(1, sv_j)),
            max(0, min(1, (sf_c + 1) / 2)),  # map [-1,1] -> [0,1]
        ]

    if not method_values:
        print("  Plot 5: No post-hoc data — skipped")
        return

    # Radar plot
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(_PALETTE["bg"])
    ax.set_facecolor(_PALETTE["bg"])

    # Grid styling
    ax.set_thetagrids(np.degrees(angles[:-1]), categories,
                      color=_PALETTE["fg"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       color=_PALETTE["fg"], fontsize=7)
    ax.spines["polar"].set_color(_PALETTE["grid"])
    ax.grid(color=_PALETTE["grid"], alpha=0.3)

    # Intrinsic reference (perfect = 1.0 on all axes)
    ref_values = [1.0] * n_cats + [1.0]
    ax.plot(angles, ref_values, color=_PALETTE["accent1"], linewidth=2,
            linestyle="--", alpha=0.5, label="Intrinsic (reference)")
    ax.fill(angles, ref_values, color=_PALETTE["accent1"], alpha=0.05)

    # Post-hoc methods
    for method_key, (label, color) in methods.items():
        values = method_values.get(method_key, [0] * n_cats)
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.15)
        ax.scatter(angles[:-1], values[:-1], color=color, s=50, zorder=5,
                   edgecolors="white", linewidths=0.5)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              facecolor=_PALETTE["bg"], edgecolor=_PALETTE["grid"],
              labelcolor=_PALETTE["fg"], fontsize=9)
    ax.set_title("Post-hoc vs Intrinsic Explanations",
                 fontsize=14, fontweight="bold", color=_PALETTE["fg"], pad=20)

    _save_plot(fig, save_dir, "plot5_posthoc_radar.png")
    plt.close(fig)
    print("  Plot 5: Post-hoc radar chart — done")


# ── Plot 6: Example Explanation Visualization ─────────────────────────

def plot_example_explanations(model=None, test_graphs=None, save_dir=None):
    """Placeholder for 4-panel explanation visualizations.

    This requires a trained model and test data at generation time,
    which is handled separately by the runner. This function generates
    a summary panel from saved explanation data if available.
    """
    import matplotlib.pyplot as plt

    if save_dir is None:
        print("  Plot 6: No save dir — skipped")
        return

    # Check for pre-saved explanation data
    expl_path = OUTPUT_DIR / "example_explanations.json"
    if not expl_path.exists():
        print("  Plot 6: No example_explanations.json found — skipped")
        print("  (Run the runner with --save-examples to generate)")
        return

    with open(str(expl_path)) as f:
        examples = json.load(f)

    n_examples = min(4, len(examples))
    if n_examples == 0:
        print("  Plot 6: No examples — skipped")
        return

    fig, axes = plt.subplots(1, n_examples, figsize=(5 * n_examples, 5))
    fig.patch.set_facecolor(_PALETTE["bg"])

    if n_examples == 1:
        axes = [axes]

    for idx, (ax, example) in enumerate(zip(axes, examples[:n_examples])):
        ax.set_facecolor(_PALETTE["bg"])
        ax.set_title(f"Edge {idx + 1}: {example.get('src_tissue', '?')} → {example.get('dst_tissue', '?')}",
                     color=_PALETTE["fg"], fontsize=10, fontweight="bold")

        # Visualize SV attention as horizontal bars
        sv_attns = example.get("sv_attentions", [])
        if sv_attns:
            y_pos = range(len(sv_attns))
            bars = ax.barh(y_pos, sv_attns, color=_PALETTE["accent1"], alpha=0.8,
                          edgecolor="white", linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"SV {i}" for i in range(len(sv_attns))],
                              fontsize=7, color=_PALETTE["fg"])
            ax.set_xlabel("Attention Weight", fontsize=9, color=_PALETTE["fg"])
            ax.tick_params(colors=_PALETTE["fg"], labelsize=7)
        else:
            ax.text(0.5, 0.5, "No SV data", ha="center", va="center",
                    color=_PALETTE["fg"], fontsize=11, transform=ax.transAxes)

        for spine in ax.spines.values():
            spine.set_color(_PALETTE["grid"])

    fig.suptitle("Example Edge Explanations", fontsize=14, fontweight="bold",
                 color=_PALETTE["fg"])
    fig.tight_layout()

    _save_plot(fig, save_dir, "plot6_example_explanations.png")
    plt.close(fig)
    print("  Plot 6: Example explanations — done")


# ══════════════════════════════════════════════════════════════════════
# Master Report Generator (wired in Phase 4D)
# ══════════════════════════════════════════════════════════════════════

def generate_full_report(results_path=None, posthoc_path=None,
                         tables=True, plots=True, save_format="console"):
    """Generate the complete explainability report.

    Args:
        results_path: path to results.json (default: OUTPUT_DIR/results.json)
        posthoc_path: path to posthoc_results.json
        tables: whether to generate tables
        plots: whether to generate plots
        save_format: "console", "csv", or "latex"
    """
    import time

    results = load_results(results_path)
    agg = aggregate_by_config(results)
    posthoc = load_posthoc_results(posthoc_path)

    save_dir = OUTPUT_DIR / "report"
    save_dir.mkdir(parents=True, exist_ok=True)

    do_save = save_format in ("csv", "latex")

    print("=" * 70)
    print("EXPLAINABILITY STUDY REPORT")
    print("=" * 70)
    print(f"Loaded {len(results)} runs across {len(agg)} configs")
    print(f"Save format: {save_format}")
    print(f"Output dir: {save_dir}")
    print(f"Posthoc data: {'loaded' if posthoc else 'not found'}")

    if tables:
        t0 = time.time()
        print("\n── Tables ─────────────────────────────────────────────")

        table_knn_sensitivity(agg, save_dir if do_save else None)
        table_component_contribution(agg, save_dir if do_save else None)

        if posthoc:
            table_posthoc_comparison(posthoc, save_dir if do_save else None)
        else:
            print("\n  Table 3: Skipped (no post-hoc data)")

        print(f"\n  Tables generated in {time.time() - t0:.1f}s")

    if plots:
        t0 = time.time()
        print("\n── Plots ──────────────────────────────────────────────")

        plot_fidelity_sparsity_tradeoff(agg, save_dir=save_dir)
        plot_attention_faithfulness_heatmap(agg, save_dir=save_dir)
        plot_stability_boxplots(agg, save_dir=save_dir)
        plot_component_delta(agg, save_dir=save_dir)

        if posthoc:
            plot_posthoc_radar(posthoc, save_dir=save_dir)
        else:
            print("  Plot 5: Skipped (no post-hoc data)")

        plot_example_explanations(save_dir=save_dir)

        print(f"\n  Plots generated in {time.time() - t0:.1f}s")

    # ── Summary statistics ──
    print("\n── Summary ────────────────────────────────────────────")
    _print_overall_summary(agg)

    print(f"\nReport complete. Output: {save_dir}")
    return agg, posthoc


def _print_overall_summary(agg):
    """Print high-level summary of the best configurations."""
    if not agg:
        print("  No data to summarize.")
        return

    # Find best config for each key metric
    best_configs = {}
    metric_keys = [
        ("test_auc", "Best AUC"),
        ("xai_combined_drop_mean", "Best Fid+ (necessity)"),
        ("xai_level2_retention_mean", "Best Fid- (sufficiency)"),
        ("xai_sv_sparsity_mean", "Most Sparse"),
        ("xai_complexity_mean", "Least Complex"),
    ]

    for key, label in metric_keys:
        best_name, best_val = None, None
        # For sparsity and complexity, lower is better
        minimize = key in ("xai_sv_sparsity_mean", "xai_complexity_mean")

        for config_name in agg:
            mean, _ = get_metric(agg, config_name, key)
            if mean is None:
                continue
            if best_val is None:
                best_val, best_name = mean, config_name
            elif minimize and mean < best_val:
                best_val, best_name = mean, config_name
            elif not minimize and mean > best_val:
                best_val, best_name = mean, config_name

        if best_name:
            best_configs[label] = (best_name, best_val)
            print(f"  {label:<28s}: {best_name} ({best_val:.4f})")

    # Config count summary
    print(f"\n  Total configs evaluated: {len(agg)}")
    total_runs = sum(d['n_seeds'] for d in agg.values())
    print(f"  Total runs: {total_runs}")


# ── Smoke Test ────────────────────────────────────────────────────────

def _run_smoke_test():
    """Generate dummy data and validate the full report pipeline."""
    import tempfile

    print("=" * 70)
    print("SMOKE TEST: Report Pipeline")
    print("=" * 70)

    # Create dummy results
    dummy_results = []
    configs_to_test = ["K0_compat_L2", "K3_L2", "K5_L2", "A_baseline", "D_full"]

    for config_name in configs_to_test:
        for seed in [42, 123]:
            cfg = CONFIGS.get(config_name)
            if cfg is None:
                continue
            dummy_results.append({
                "config": config_name,
                "seed": seed,
                "edge_strategy": cfg.edge_strategy,
                "k_neighbors": cfg.k_neighbors,
                "num_gnn_layers": cfg.num_gnn_layers,
                "use_sv": cfg.use_sv_aggregation,
                "use_topo": cfg.use_intra_topology,
                "use_ocn": cfg.use_ocn_features,
                "param_count": 150000 + seed,
                "train_time_sec": 120.0 + seed * 0.1,
                "best_val_auc": 0.75 + np.random.uniform(-0.05, 0.05),
                "test_auc": 0.72 + np.random.uniform(-0.05, 0.05),
                "test_ap": 0.68 + np.random.uniform(-0.05, 0.05),
                "test_intra_auc": 0.70 + np.random.uniform(-0.05, 0.05),
                "test_inter_auc": 0.65 + np.random.uniform(-0.05, 0.05),
                "xai_combined_drop_mean": 0.15 + np.random.uniform(-0.05, 0.05),
                "xai_level1_drop_mean": 0.08 + np.random.uniform(-0.02, 0.02),
                "xai_level2_drop_mean": 0.10 + np.random.uniform(-0.03, 0.03),
                "xai_level1_retention_mean": 0.30 + np.random.uniform(-0.05, 0.05),
                "xai_level2_retention_mean": 0.55 + np.random.uniform(-0.1, 0.1),
                "xai_sv_sparsity_mean": 0.40 + np.random.uniform(-0.1, 0.1),
                "xai_sf_sparsity_mean": 0.60 + np.random.uniform(-0.1, 0.1),
                "xai_complexity_mean": 8.0 + np.random.uniform(-2, 2),
            })

    dummy_posthoc = {
        "intrinsic": {
            "summary": {
                "combined_drop_mean": 0.18,
                "level1_drop_mean": 0.09,
                "level2_drop_mean": 0.12,
                "level1_retention_mean": 0.32,
                "level2_retention_mean": 0.58,
                "sv_sparsity_mean": 0.38,
                "sf_sparsity_mean": 0.55,
                "complexity_mean": 7.5,
            },
            "n_explanations": 50,
        },
        "gnn_explainer": {
            "vs_intrinsic_stability": {"sv_jaccard": 0.42, "sf_correlation": 0.35},
            "n_explanations": 25,
        },
        "grad_cam": {
            "vs_intrinsic_stability": {"sv_jaccard": 0.55, "sf_correlation": 0.48},
            "n_explanations": 25,
        },
        "attention_only": {
            "vs_intrinsic_stability": {"sv_jaccard": 0.28, "sf_correlation": 0.15},
            "n_explanations": 25,
        },
    }

    # Save dummy data to temp location
    test_dir = OUTPUT_DIR / "_smoke_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    results_path = test_dir / "results.json"
    with open(str(results_path), "w") as f:
        json.dump(dummy_results, f, indent=2)

    posthoc_path = test_dir / "posthoc_results.json"
    with open(str(posthoc_path), "w") as f:
        json.dump(dummy_posthoc, f, indent=2)

    print(f"Wrote dummy data to {test_dir}")
    print(f"  {len(dummy_results)} runs, {len(configs_to_test)} configs")
    print()

    # Run the report
    try:
        generate_full_report(
            results_path=results_path,
            posthoc_path=posthoc_path,
            tables=True,
            plots=True,
            save_format="csv",
        )
        print("\n" + "=" * 70)
        print("SMOKE TEST PASSED")
        print("=" * 70)
    except Exception as e:
        print(f"\nSMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

# ── Entry Point ───────────────────────────────────────────────────────
#
# Edit these constants to control what the report generates.
# Results/posthoc paths default to OUTPUT_DIR from explainability/config.py.

# Report configuration constants
RUN_SMOKE_TEST = False          # Set True to validate pipeline with dummy data
GENERATE_TABLES = True          # Set False to skip tables
GENERATE_PLOTS = True           # Set False to skip plots
SAVE_FORMAT = "csv"             # "console", "csv", or "latex"

if __name__ == "__main__":
    if RUN_SMOKE_TEST:
        _run_smoke_test()
    else:
        generate_full_report(
            results_path=None,       # uses OUTPUT_DIR / "results.json"
            posthoc_path=None,       # uses OUTPUT_DIR / "posthoc_results.json"
            tables=GENERATE_TABLES,
            plots=GENERATE_PLOTS,
            save_format=SAVE_FORMAT,
        )

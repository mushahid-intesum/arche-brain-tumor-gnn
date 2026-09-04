"""
Report Generator for Plan 3a.

Generates a comprehensive markdown report from ablation results:
  - Per-experiment C-Index comparison table
  - Faithfulness audit summary
  - Concept correlation heatmap (text-based)
  - Training curves comparison
  - Architecture diagram
  - Key findings & ablation insights
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def generate_report(
    results_path: str = None,
    output_path: str = None,
) -> str:
    """
    Generate markdown report from ablation results JSON.

    Args:
        results_path: path to ablation_results.json
        output_path: where to write the markdown report

    Returns:
        report_text: the full markdown content
    """
    if results_path is None:
        results_path = str(
            Path(__file__).resolve().parent.parent / "checkpoints" / "ablation_results.json"
        )
    if output_path is None:
        output_path = str(
            Path(__file__).resolve().parent.parent / "RESULTS.md"
        )

    with open(results_path) as f:
        results = json.load(f)

    report = []
    report.append("# Plan 3a: Hypergraph Concept Bottleneck GNN — Results Report")
    report.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    # ── Architecture Overview ────────────────────────────────────────
    report.append("## 1. Architecture Overview\n")
    report.append("```")
    report.append("MRI Patches (N, 6, 16×16)           Clinical Features (18-dim)")
    report.append("    │                                        │")
    report.append("    ▼                                        │")
    report.append("PatchEncoder (MLP)                           │")
    report.append("    │                                        │")
    report.append("    ▼                                        │")
    report.append("SheafHGNN (3 layers)                         │")
    report.append("    │ ←── Topological + Feature              │")
    report.append("    │     Hyperedges                         │")
    report.append("    ▼                                        │")
    report.append("ConceptBottleneck (8 concepts)               │")
    report.append("    │ ←── HECRL inter-concept attention      │")
    report.append("    │                                        │")
    report.append("    ▼ (optional)                             │")
    report.append("MultiGranularTree                            │")
    report.append("    │ L0→L1→L2→L3 + AdaptiveRouter          │")
    report.append("    │                                        │")
    report.append("    ▼                                        ▼")
    report.append("MultiModalFusion ◄────────── ClinicalEncoder")
    report.append("    │ ←── DynamicWeighting                   ")
    report.append("    │     (mono+holo confidence)             ")
    report.append("    ▼                                        ")
    report.append("SurvivalHead → Hazard Logits (4 bins)       ")
    report.append("```\n")

    # ── Experiment Results ───────────────────────────────────────────
    report.append("## 2. Ablation Results\n")
    report.append("### 2.1 C-Index Comparison\n")
    report.append("| Exp | Configuration | C-Index | Params |")
    report.append("|-----|--------------|---------|--------|")
    for r in results:
        ci = f"{r['mean_c_index']:.4f} ± {r['std_c_index']:.4f}"
        report.append(f"| **{r['experiment']}** | {r['name'][4:]} | {ci} | {r['n_params']:,} |")
    report.append("")

    # Best experiment
    if results:
        best = max(results, key=lambda x: x["mean_c_index"])
        report.append(f"> **Best**: {best['experiment']} ({best['name']}) "
                      f"with C-Index = {best['mean_c_index']:.4f}\n")

    # ── Per-fold breakdown ───────────────────────────────────────────
    report.append("### 2.2 Per-Fold C-Index\n")
    # Header
    header = "| Exp |"
    divider = "|-----|"
    for fold_idx in range(len(results[0]["fold_results"]) if results else 0):
        header += f" Fold {fold_idx+1} |"
        divider += "--------|"
    header += " Mean |"
    divider += "------|"
    report.append(header)
    report.append(divider)

    for r in results:
        row = f"| {r['experiment']} |"
        for fr in r["fold_results"]:
            row += f" {fr['best_c_index']:.4f} |"
        row += f" {r['mean_c_index']:.4f} |"
        report.append(row)
    report.append("")

    # ── Faithfulness ─────────────────────────────────────────────────
    has_faith = any("faithfulness" in r for r in results)
    if has_faith:
        report.append("### 2.3 Faithfulness Audit\n")
        report.append("| Exp | EST Rejection | Fid⁻ Rejection | Suf Rejection | Overall |")
        report.append("|-----|:------------:|:---------------:|:-------------:|:-------:|")
        for r in results:
            if "faithfulness" in r:
                f = r["faithfulness"]
                report.append(
                    f"| {r['experiment']} | "
                    f"{f['est_rejection']:.0%} | "
                    f"{f['fid_minus_rejection']:.0%} | "
                    f"{f['sufficiency_rejection']:.0%} | "
                    f"{f['overall_rejection']:.0%} |"
                )
        report.append("")
        report.append("> Lower rejection = more faithful explanations. "
                      "0% = all explanations pass the EST audit.\n")

    # ── Training Dynamics ────────────────────────────────────────────
    report.append("### 2.4 Training Dynamics\n")
    for r in results:
        if r["fold_results"] and r["fold_results"][0].get("history"):
            hist = r["fold_results"][0]["history"]
            report.append(f"**{r['experiment']}** (Fold 1):\n")
            report.append("| Epoch | Train Loss | Val C-Index | Concept r |")
            report.append("|-------|-----------|-------------|-----------|")
            for h in hist[:5]:  # first 5 epochs
                report.append(
                    f"| {h['epoch']} | {h['train_loss']:.4f} | "
                    f"{h['val_c_index']:.4f} | {h.get('concept_corr', 0):.3f} |"
                )
            if len(hist) > 5:
                last = hist[-1]
                report.append(
                    f"| {last['epoch']} | {last['train_loss']:.4f} | "
                    f"{last['val_c_index']:.4f} | {last.get('concept_corr', 0):.3f} |"
                )
            report.append("")

    # ── Ablation Insights ────────────────────────────────────────────
    report.append("## 3. Ablation Insights\n")

    if len(results) >= 2:
        # Compare experiments pairwise
        sorted_r = sorted(results, key=lambda x: x["mean_c_index"], reverse=True)

        report.append("### Key Findings\n")

        # Hypergraph vs kNN
        e1 = next((r for r in results if r["experiment"] == "E1"), None)
        e2 = next((r for r in results if r["experiment"] == "E2"), None)
        if e1 and e2:
            delta = e2["mean_c_index"] - e1["mean_c_index"]
            direction = "improves" if delta > 0 else "decreases"
            report.append(f"1. **Hypergraph effect**: Switching from kNN (E1) to sheaf "
                          f"hypergraph (E2) {direction} C-Index by "
                          f"{abs(delta):.4f}")

        # Concept bottleneck
        e2 = next((r for r in results if r["experiment"] == "E2"), None)
        e3 = next((r for r in results if r["experiment"] == "E3"), None)
        if e2 and e3:
            delta = e3["mean_c_index"] - e2["mean_c_index"]
            direction = "improves" if delta > 0 else "decreases"
            report.append(f"2. **Concept bottleneck effect**: Adding concept supervision (E3) "
                          f"{direction} C-Index by {abs(delta):.4f}")

        # Fusion
        e3 = next((r for r in results if r["experiment"] == "E3"), None)
        e4 = next((r for r in results if r["experiment"] == "E4"), None)
        if e3 and e4:
            delta = e4["mean_c_index"] - e3["mean_c_index"]
            direction = "improves" if delta > 0 else "decreases"
            report.append(f"3. **Multimodal fusion effect**: Adding clinical data (E4) "
                          f"{direction} C-Index by {abs(delta):.4f}")

        # Tree
        e4 = next((r for r in results if r["experiment"] == "E4"), None)
        e5 = next((r for r in results if r["experiment"] == "E5"), None)
        if e4 and e5:
            delta = e5["mean_c_index"] - e4["mean_c_index"]
            direction = "improves" if delta > 0 else "decreases"
            report.append(f"4. **Multi-granular tree effect**: Adding TIF hierarchy (E5) "
                          f"{direction} C-Index by {abs(delta):.4f}")
        report.append("")

    # ── Methodology ──────────────────────────────────────────────────
    report.append("## 4. Methodology\n")
    report.append("### Dataset")
    if results:
        report.append(f"- **Patients**: {results[0]['n_patients']}")
        report.append(f"- **Folds**: {results[0]['n_folds']}-fold CV")
        report.append(f"- **Epochs**: {results[0]['epochs']}")
    report.append("- **Modalities**: T1-pre, T1-post, T2, FLAIR, DTI, Perfusion")
    report.append("- **Task**: Survival prediction (discrete-time NLL loss)")
    report.append("- **Primary metric**: Harrell's C-Index\n")

    report.append("### Concepts (self-supervised, no segmentation GT)")
    report.append("| # | Concept | Source |")
    report.append("|---|---------|--------|")
    report.append("| c1 | Enhancement ratio | log(1 + T1-post/T1-pre) |")
    report.append("| c2 | FLAIR z-score | FLAIR_mean / FLAIR_std |")
    report.append("| c3 | T2 abnormality | T2 × FLAIR interaction |")
    report.append("| c4 | DTI mean diffusivity | Mean DTI signal |")
    report.append("| c5 | DTI FA proxy | DTI coefficient of variation |")
    report.append("| c6 | Intensity heterogeneity | Cross-modality std |")
    report.append("| c7 | Boundary complexity | Graph-learned (SHGNN) |")
    report.append("| c8 | Spatial location | Normalized z-coordinate |")
    report.append("")

    report.append("### References")
    report.append("- **HyperCBM** (NeurIPS 2026): Concept bottleneck + HECRL")
    report.append("- **MRePath** (IJCAI 2025): Sheaf hypergraph + dynamic modality rebalancing")
    report.append("- **SE-GNN Audit** (ICLR 2026): EST faithfulness metric")
    report.append("- **TIF** (arXiv 2505.00364): Multi-granular tree interpretability")
    report.append("")

    # Write
    report_text = "\n".join(report)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)
    print(f"Report written to {output_path}")

    return report_text


if __name__ == "__main__":
    from plan3a.config import RESULTS_JSON, REPORT_OUTPUT
    generate_report(RESULTS_JSON, REPORT_OUTPUT)

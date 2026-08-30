"""
Common Neighbor Statistics Analysis

Computes CN statistics for:
  1. Inter-node graphs (edges between segmentation components)
  2. Intra-node graphs (edges between supervoxels within each component)

Run: python3 cn_analysis.py
Requires: built graphs from gnn.build_all_graphs()
"""

import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

from config import SHARED, GNN, SUPERVOXEL


def compute_cn_stats_for_graph(edge_index, num_nodes, label="graph"):
    """Compute CN statistics for a single graph.

    Returns a dict with:
      - per-edge CN counts
      - per-edge Jaccard coefficients
      - per-edge Adamic-Adar indices
      - degree distribution
      - summary statistics
    """
    adj = defaultdict(set)
    for e in range(edge_index.size(1)):
        s, d = edge_index[0, e].item(), edge_index[1, e].item()
        adj[s].add(d)
        adj[d].add(s)

    degrees = [len(adj[n]) for n in range(num_nodes)]
    max_deg = max(degrees) if degrees else 1

    cn_counts = []
    jaccards = []
    adamic_adars = []

    # Collect unique undirected edges
    seen = set()
    for e in range(edge_index.size(1)):
        i, j = edge_index[0, e].item(), edge_index[1, e].item()
        pair = (min(i, j), max(i, j))
        if pair in seen:
            continue
        seen.add(pair)

        cn = adj[i] & adj[j]
        union = adj[i] | adj[j]

        cn_counts.append(len(cn))
        jaccards.append(len(cn) / max(len(union), 1))

        aa = 0.0
        for w in cn:
            deg_w = len(adj[w])
            if deg_w > 1:
                aa += 1.0 / np.log(deg_w)
        adamic_adars.append(aa)

    cn_counts = np.array(cn_counts, dtype=float)
    jaccards = np.array(jaccards)
    adamic_adars = np.array(adamic_adars)
    degrees = np.array(degrees, dtype=float)

    def safe_stats(arr, name):
        if len(arr) == 0:
            return {f"{name}_mean": 0, f"{name}_std": 0, f"{name}_min": 0,
                    f"{name}_max": 0, f"{name}_median": 0, f"{name}_nonzero_frac": 0}
        return {
            f"{name}_mean": float(np.mean(arr)),
            f"{name}_std": float(np.std(arr)),
            f"{name}_min": float(np.min(arr)),
            f"{name}_max": float(np.max(arr)),
            f"{name}_median": float(np.median(arr)),
            f"{name}_nonzero_frac": float(np.count_nonzero(arr) / max(len(arr), 1)),
        }

    stats = {
        "num_nodes": num_nodes,
        "num_edges_undirected": len(seen),
        **safe_stats(cn_counts, "cn"),
        **safe_stats(jaccards, "jaccard"),
        **safe_stats(adamic_adars, "aa"),
        **safe_stats(degrees, "degree"),
    }
    return stats


def analyze_inter_node(graphs):
    """Analyze CN statistics across all inter-node (seg component) graphs."""
    all_stats = []
    for g in graphs:
        if g.edge_index.size(1) < 2:
            continue
        stats = compute_cn_stats_for_graph(g.edge_index, g.x.size(0))
        stats["case_id"] = g.case_id if hasattr(g, 'case_id') else "unknown"
        all_stats.append(stats)
    return all_stats


def analyze_intra_node(graphs):
    """Analyze CN statistics across all intra-node (supervoxel) graphs."""
    all_stats = []
    for g in graphs:
        if not hasattr(g, 'sv_edge_indices') or not g.sv_edge_indices:
            continue

        case_id = g.case_id if hasattr(g, 'case_id') else "unknown"
        tissue_labels = g.tissue_labels.numpy() if hasattr(g, 'tissue_labels') else []
        n_svs = g.n_svs_per_node if hasattr(g, 'n_svs_per_node') else []

        for node_idx, ei in enumerate(g.sv_edge_indices):
            if ei.size(1) < 2:
                continue
            n = n_svs[node_idx] if node_idx < len(n_svs) else 0
            if n < 2:
                continue

            stats = compute_cn_stats_for_graph(ei, n)
            stats["case_id"] = case_id
            stats["node_idx"] = node_idx
            stats["tissue_label"] = int(tissue_labels[node_idx]) if node_idx < len(tissue_labels) else -1
            all_stats.append(stats)
    return all_stats


def format_report(inter_stats, intra_stats):
    """Format a text report of CN statistics."""
    lines = []
    lines.append("=" * 70)
    lines.append("COMMON NEIGHBOR STATISTICS REPORT")
    lines.append("=" * 70)

    # Inter-node summary
    lines.append("\n## INTER-NODE GRAPHS (Segmentation Components)")
    lines.append(f"Number of graphs analyzed: {len(inter_stats)}")

    if inter_stats:
        keys = ["cn_mean", "cn_max", "cn_nonzero_frac", "jaccard_mean",
                "aa_mean", "degree_mean", "num_nodes", "num_edges_undirected"]
        for key in keys:
            vals = [s[key] for s in inter_stats if key in s]
            if vals:
                lines.append(f"  {key:30s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                             f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    # Intra-node summary
    lines.append("\n## INTRA-NODE GRAPHS (Supervoxels within Components)")
    lines.append(f"Number of intra-node graphs analyzed: {len(intra_stats)}")

    if intra_stats:
        tissue_names = {1: "NCR", 2: "ED", 3: "ET"}

        keys = ["cn_mean", "cn_max", "cn_nonzero_frac", "jaccard_mean",
                "aa_mean", "degree_mean", "num_nodes", "num_edges_undirected"]
        lines.append("\n  Overall:")
        for key in keys:
            vals = [s[key] for s in intra_stats if key in s]
            if vals:
                lines.append(f"    {key:30s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                             f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

        # Per-tissue breakdown
        for tissue_id, tissue_name in tissue_names.items():
            tissue_stats = [s for s in intra_stats if s.get("tissue_label") == tissue_id]
            if not tissue_stats:
                continue
            lines.append(f"\n  {tissue_name} (n={len(tissue_stats)}):")
            for key in keys:
                vals = [s[key] for s in tissue_stats if key in s]
                if vals:
                    lines.append(f"    {key:30s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    # Comparison
    lines.append("\n## INTER vs INTRA COMPARISON")
    if inter_stats and intra_stats:
        inter_cn = np.mean([s["cn_mean"] for s in inter_stats])
        intra_cn = np.mean([s["cn_mean"] for s in intra_stats])
        inter_jac = np.mean([s["jaccard_mean"] for s in inter_stats])
        intra_jac = np.mean([s["jaccard_mean"] for s in intra_stats])
        inter_deg = np.mean([s["degree_mean"] for s in inter_stats])
        intra_deg = np.mean([s["degree_mean"] for s in intra_stats])

        lines.append(f"  {'Metric':30s}  {'Inter-Node':>12s}  {'Intra-Node':>12s}")
        lines.append(f"  {'-'*30}  {'-'*12}  {'-'*12}")
        lines.append(f"  {'Mean CN count':30s}  {inter_cn:12.4f}  {intra_cn:12.4f}")
        lines.append(f"  {'Mean Jaccard':30s}  {inter_jac:12.4f}  {intra_jac:12.4f}")
        lines.append(f"  {'Mean Degree':30s}  {inter_deg:12.4f}  {intra_deg:12.4f}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    import gnn

    print(f"Device: {SHARED['device']}")

    # Build graphs (hierarchical if volumes exist, else flat)
    train_graphs, val_graphs, test_graphs = gnn.build_all_graphs()
    all_graphs = train_graphs + val_graphs + test_graphs

    print(f"\nAnalyzing {len(all_graphs)} graphs...")

    inter_stats = analyze_inter_node(all_graphs)
    intra_stats = analyze_intra_node(all_graphs)

    report = format_report(inter_stats, intra_stats)
    print(report)

    # Save report
    output_path = Path("brats_outputs/cn_statistics_report.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")

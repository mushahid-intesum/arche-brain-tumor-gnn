import torch
import numpy as np

from .config import GRAPH

# Reuse Levels 1 and 2 from plan1
from plan1.explainability import (
    graph_attention_trace,
    patch_attention_trace,
)


# ── Level 3: Regression Refinement Trace (adapted from Plan 1) ───────


def regression_refinement_trace(outputs, targets, retained_labels,
                                seg_features=None, idx_to_label=None,
                                edge_index=None, graph_attentions=None):
    N = len(retained_labels)
    y_reg = outputs["y_reg"].cpu().numpy()
    ensemble_var = outputs["ensemble_preds"].var(dim=-1).cpu().numpy()

    gt_reg = np.array([targets[retained_labels[i]]["y_reg"] for i in range(N)])
    gt_cls = np.array([targets[retained_labels[i]]["y_cls"] for i in range(N)])

    # Seg model prediction: derive tumor proportion proxy from seg_feat
    if seg_features is not None:
        seg_tumor_prob = np.array([
            1.0 - seg_features[retained_labels[i]]["seg_feat"][0]  # 1 - P(BG)
            for i in range(N)
        ])
    else:
        seg_tumor_prob = np.zeros(N)

    # Regression error
    mae = float(np.abs(y_reg - gt_reg).mean())
    r2_denom = np.var(gt_reg) * N
    r2 = float(1.0 - np.sum((y_reg - gt_reg)**2) / max(r2_denom, 1e-8))

    # Seg error
    seg_mae = float(np.abs(seg_tumor_prob - gt_reg).mean())

    # Find nodes where GNN substantially improves over seg model
    gnn_error = np.abs(y_reg - gt_reg)
    seg_error = np.abs(seg_tumor_prob - gt_reg)
    improvement = seg_error - gnn_error  # positive = GNN better

    # Corrections: GNN much better than seg
    correction_mask = improvement > 0.1
    # Degradations: GNN much worse than seg
    degradation_mask = improvement < -0.1

    corrections = []
    for i in range(N):
        if not correction_mask[i] and not degradation_mask[i]:
            continue

        label = retained_labels[i]
        entry = {
            "node_idx": i,
            "sv_label": label,
            "gt_reg": float(gt_reg[i]),
            "gt_cls": int(gt_cls[i]),
            "gnn_reg": float(y_reg[i]),
            "seg_tumor_prob": float(seg_tumor_prob[i]),
            "gnn_error": float(gnn_error[i]),
            "seg_error": float(seg_error[i]),
            "improvement": float(improvement[i]),
            "ensemble_var": float(ensemble_var[i]),
            "type": "correction" if correction_mask[i] else "degradation",
            "gt_dominant": targets[label]["y_dominant"],
            "centroid": targets[label]["centroid"].tolist(),
        }

        # Attention trace for corrections
        if (correction_mask[i] and edge_index is not None
                and graph_attentions is not None):
            trace = graph_attention_trace(
                i, edge_index, graph_attentions,
                targets=targets, idx_to_label=idx_to_label, top_k=3,
            )
            entry["attention_trace"] = trace["neighbors"]
            entry["attention_entropy"] = trace["attention_entropy"]

        corrections.append(entry)

    # Zero-shot Dice: threshold regression at τ to recover binary
    tau = GRAPH.get("tau", 0.15)
    gnn_binary = (y_reg > tau).astype(int)
    tp = ((gnn_binary == 1) & (gt_cls == 1)).sum()
    fp = ((gnn_binary == 1) & (gt_cls == 0)).sum()
    fn = ((gnn_binary == 0) & (gt_cls == 1)).sum()
    dice = float(2 * tp / max(2 * tp + fp + fn, 1))

    return {
        "total_nodes": N,
        "mae_gnn": mae,
        "mae_seg": seg_mae,
        "r2": r2,
        "zero_shot_dice": dice,
        "n_corrections": int(correction_mask.sum()),
        "n_degradations": int(degradation_mask.sum()),
        "mean_ensemble_var": float(ensemble_var.mean()),
        "corrections": corrections,
    }


# ── Level 4: Task-Specific Divergence ────────────────────────────────


def task_divergence_trace(outputs, targets, retained_labels,
                          seg_features=None, idx_to_label=None):
    N = len(retained_labels)
    y_reg = outputs["y_reg"].cpu().numpy()
    unc_prob = torch.sigmoid(outputs["unc_logits"]).cpu().numpy()

    # For edges, compute per-node dominant edge type
    edge_logits = outputs["edge_logits"]
    edge_index = outputs.get("_edge_index")  # injected by explain_case

    # Per-node incident edge analysis
    edge_preds = edge_logits.argmax(dim=-1).cpu().numpy() if edge_logits is not None else None

    node_edge_profile = {}
    if edge_preds is not None and edge_index is not None:
        ei = edge_index.cpu().numpy() if torch.is_tensor(edge_index) else edge_index
        for i in range(N):
            mask = (ei[0] == i) | (ei[1] == i)
            incident = edge_preds[mask]
            if len(incident) > 0:
                type_counts = np.bincount(incident, minlength=10)
                dominant_type = int(type_counts.argmax())
                has_nontrivial = int((incident > 0).sum())  # non-BG↔BG
            else:
                type_counts = np.zeros(10, dtype=int)
                dominant_type = 0
                has_nontrivial = 0
            node_edge_profile[i] = {
                "dominant_type": dominant_type,
                "has_nontrivial_edges": has_nontrivial,
                "type_counts": type_counts,
            }

    # Classify each node's divergence pattern
    tau = GRAPH.get("tau", 0.15)
    unc_threshold = 0.5
    node_divergences = []
    category_counts = {}

    for i in range(N):
        label = retained_labels[i]
        reg_high = y_reg[i] > tau
        unc_high = unc_prob[i] > unc_threshold
        ep = node_edge_profile.get(i, {})
        dom_type = ep.get("dominant_type", 0)
        has_nontrivial = ep.get("has_nontrivial_edges", 0) > 0

        # Classify divergence
        if reg_high and not unc_high and dom_type > 0:
            category = "AGREEMENT_CONFIDENT"
        elif not reg_high and not unc_high and dom_type == 0:
            category = "AGREEMENT_NEGATIVE"
        elif reg_high and unc_high:
            category = "DISAGREEMENT_REG_UNC"
        elif reg_high and dom_type == 0:
            category = "DISAGREEMENT_EDGE"
        elif has_nontrivial and unc_high:
            category = "UNCERTAIN_BOUNDARY"
        else:
            category = "MIXED"

        category_counts[category] = category_counts.get(category, 0) + 1

        gt = targets.get(label, {})
        entry = {
            "node_idx": i,
            "sv_label": label,
            "category": category,
            "y_reg": float(y_reg[i]),
            "unc_prob": float(unc_prob[i]),
            "dominant_edge_type": dom_type,
            "has_nontrivial_edges": has_nontrivial,
            "gt_cls": gt.get("y_cls", -1),
            "gt_reg": gt.get("y_reg", -1),
            "gt_dominant": gt.get("y_dominant", -1),
        }

        # Add seg context
        if seg_features and label in seg_features:
            sf = seg_features[label]
            entry["seg_pred"] = sf["seg_pred"]
            entry["seg_entropy"] = float(sf["seg_entropy"])

        node_divergences.append(entry)

    # Find the most clinically interesting nodes: DISAGREEMENT categories
    interesting = [n for n in node_divergences
                   if n["category"].startswith("DISAGREEMENT")
                   or n["category"] == "UNCERTAIN_BOUNDARY"]

    return {
        "total_nodes": N,
        "category_counts": category_counts,
        "interesting_nodes": interesting[:20],  # cap for readability
        "all_divergences": node_divergences,
    }


# ── Level 5: Uncertainty-Driven Explanation ──────────────────────────


def uncertainty_explanation_trace(outputs, targets, retained_labels,
                                  seg_features=None, idx_to_label=None,
                                  edge_index=None, graph_attentions=None,
                                  unc_threshold=0.6, top_k=10):
    N = len(retained_labels)
    unc_prob = torch.sigmoid(outputs["unc_logits"]).cpu().numpy()
    y_reg = outputs["y_reg"].cpu().numpy()
    ens_var = outputs["ensemble_preds"].var(dim=-1).cpu().numpy()

    # Find high-uncertainty nodes
    high_unc_indices = np.where(unc_prob > unc_threshold)[0]
    # Sort by uncertainty descending
    high_unc_indices = high_unc_indices[np.argsort(unc_prob[high_unc_indices])[::-1]]
    high_unc_indices = high_unc_indices[:top_k]

    explanations = []
    for i in high_unc_indices:
        label = retained_labels[i]
        gt = targets.get(label, {})

        entry = {
            "node_idx": int(i),
            "sv_label": label,
            "unc_prob": float(unc_prob[i]),
            "y_reg": float(y_reg[i]),
            "ensemble_var": float(ens_var[i]),
            "gt_reg": float(gt.get("y_reg", -1)),
            "gt_cls": int(gt.get("y_cls", -1)),
            "gt_dominant": int(gt.get("y_dominant", -1)),
            "centroid": gt.get("centroid", np.zeros(3)).tolist(),
        }

        # Was the seg model actually wrong?
        if seg_features and label in seg_features:
            sf = seg_features[label]
            seg_pred = sf["seg_pred"]
            gt_dom = gt.get("y_dominant", 0)
            seg_was_wrong = int(seg_pred != gt_dom)
            entry["seg_pred"] = seg_pred
            entry["seg_entropy"] = float(sf["seg_entropy"])
            entry["seg_was_actually_wrong"] = seg_was_wrong
            entry["seg_feat"] = sf["seg_feat"].tolist()

            # Does regression compensate?
            # If seg says BG but regression says tumor (y_reg > τ)
            tau = GRAPH.get("tau", 0.15)
            seg_says_bg = (seg_pred == 0)
            reg_says_tumor = (y_reg[i] > tau)
            entry["regression_compensates"] = bool(
                seg_was_wrong and seg_says_bg and reg_says_tumor
            )

        # Graph attention trace: what neighbors influence this uncertain node?
        if edge_index is not None and graph_attentions is not None:
            attn_trace = graph_attention_trace(
                int(i), edge_index, graph_attentions,
                targets=targets, idx_to_label=idx_to_label, top_k=5,
            )
            entry["attention_trace"] = attn_trace["neighbors"]
            entry["attention_entropy"] = attn_trace["attention_entropy"]

            # Neighbor consistency: are neighbors of uncertain node also uncertain?
            neighbor_unc = []
            for nb in attn_trace["neighbors"]:
                nb_idx = nb["node_idx"]
                if nb_idx < N:
                    neighbor_unc.append(float(unc_prob[nb_idx]))
            entry["neighbor_mean_unc"] = float(np.mean(neighbor_unc)) if neighbor_unc else 0.0

        explanations.append(entry)

    # Global uncertainty statistics
    # Correlation between ensemble variance and uncertainty probability
    if N > 1:
        corr = float(np.corrcoef(ens_var, unc_prob)[0, 1])
    else:
        corr = 0.0

    # How well does the uncertainty head detect actual seg errors?
    if seg_features is not None:
        gt_unc = np.array([
            1 if seg_features[retained_labels[i]]["seg_pred"]
            != targets[retained_labels[i]]["y_dominant"]
            else 0
            for i in range(N)
        ])
        # AUROC approximation: how well does unc_prob rank actual errors?
        from sklearn.metrics import roc_auc_score
        try:
            auroc = float(roc_auc_score(gt_unc, unc_prob))
        except (ValueError, ImportError):
            auroc = -1.0
        actual_error_rate = float(gt_unc.mean())
    else:
        auroc = -1.0
        actual_error_rate = -1.0

    return {
        "total_nodes": N,
        "n_high_uncertainty": len(high_unc_indices),
        "unc_threshold": unc_threshold,
        "mean_unc_prob": float(unc_prob.mean()),
        "ensemble_unc_correlation": corr,
        "auroc": auroc,
        "actual_seg_error_rate": actual_error_rate,
        "explanations": explanations,
    }


# ── Full Explanation Report (5 levels) ───────────────────────────────


def explain_case(model, result, device=None, top_k_nodes=10):
    from plan1.model import compute_laplacian_pe

    device = device or GRAPH.get("device", "cpu")
    retained = result["retained_labels"]
    N = len(retained)

    # Assemble tensors
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
        seg_batch = torch.zeros(N, 5, device=device)

    ei = torch.from_numpy(result["edge_index"]).to(device)
    ea = torch.from_numpy(result["edge_attr"]).to(device)
    lpe = compute_laplacian_pe(result["edge_index"], N).to(device)

    # Forward with attention
    model.eval()
    with torch.no_grad():
        outputs, attn_dict = model(
            patch_batch, seg_batch, ei, ea,
            lap_pe=lpe, return_attention=True,
        )

    # Inject edge_index for task_divergence_trace
    outputs["_edge_index"] = ei

    # ── Level 1: Graph attention traces ──
    # Focus on tumor SVs and high-regression predictions
    tumor_indices = [i for i in range(N)
                     if result["targets"][retained[i]]["y_cls"] == 1]
    high_reg = outputs["y_reg"].cpu().argsort(descending=True)[:top_k_nodes].tolist()
    explain_indices = list(set(tumor_indices + high_reg))[:top_k_nodes]

    graph_traces = {}
    for idx in explain_indices:
        graph_traces[idx] = graph_attention_trace(
            idx, ei, attn_dict["graph"],
            targets=result["targets"],
            idx_to_label=result["idx_to_label"],
        )

    # ── Level 2: Patch attention traces ──
    patch_traces = {}
    if attn_dict["patch"]:
        for idx in explain_indices:
            patch_traces[idx] = patch_attention_trace(
                idx, attn_dict["patch"],
            )

    # ── Level 3: Regression refinement trace ──
    reg_trace = regression_refinement_trace(
        outputs, result["targets"], retained,
        seg_features=result["seg_features"],
        idx_to_label=result["idx_to_label"],
        edge_index=ei, graph_attentions=attn_dict["graph"],
    )

    # ── Level 4: Task divergence trace ──
    div_trace = task_divergence_trace(
        outputs, result["targets"], retained,
        seg_features=result["seg_features"],
        idx_to_label=result["idx_to_label"],
    )

    # ── Level 5: Uncertainty-driven explanation ──
    unc_trace = uncertainty_explanation_trace(
        outputs, result["targets"], retained,
        seg_features=result["seg_features"],
        idx_to_label=result["idx_to_label"],
        edge_index=ei, graph_attentions=attn_dict["graph"],
    )

    return {
        "case_id": result["case_id"],
        "graph_traces": graph_traces,
        "patch_traces": patch_traces,
        "regression_refinement": reg_trace,
        "task_divergence": div_trace,
        "uncertainty_explanation": unc_trace,
        "outputs": {k: v.cpu() if torch.is_tensor(v) else v
                    for k, v in outputs.items() if k != "_edge_index"},
    }


# ── Main (test with real data) ───────────────────────────────────────


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from .model import MultiTaskRefiner
    from .supervoxel import discover_cases, preprocess_case
    from .config import SHARED

    device = SHARED["device"]
    print(f"Device: {device}")

    model = MultiTaskRefiner(use_seg_prior=True).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params\n")

    cases = discover_cases(GRAPH["data_root"])
    if not cases:
        print("No BraTS cases found.")
        exit(1)

    print(f"Processing: {cases[0]['case_id']}")
    import time
    t0 = time.time()
    result = preprocess_case(cases[0])
    print(f"Preprocessed in {time.time()-t0:.1f}s\n")

    t1 = time.time()
    report = explain_case(model, result, device=device)
    print(f"Explanation generated in {time.time()-t1:.1f}s\n")

    # ── Level 1: Graph traces ──
    print("=" * 60)
    print("Level 1: Graph Attention Traces")
    print("=" * 60)
    for idx, trace in list(report["graph_traces"].items())[:3]:
        label = trace["sv_label"]
        gt = result["targets"].get(label, {})
        reg = float(report["outputs"]["y_reg"][idx])
        print(f"\n  Node {idx} (SV {label}): reg={reg:.3f}, GT={gt.get('y_reg', '?'):.3f}, "
              f"entropy={trace['attention_entropy']:.3f}")
        for nb in trace["neighbors"][:3]:
            nb_gt = f"y_cls={nb.get('y_cls', '?')}" if 'y_cls' in nb else ""
            print(f"    → Neighbor {nb['node_idx']} (SV {nb['sv_label']}): "
                  f"attn={nb['mean_attention']:.4f}, {nb_gt}")

    # ── Level 2: Patch traces ──
    print(f"\n{'='*60}")
    print("Level 2: Patch Attention Traces")
    print("=" * 60)
    for idx, trace in list(report["patch_traces"].items())[:2]:
        print(f"\n  Node {idx}: {trace['attention_layers']} layers")
        print(f"    Modality importance: ", end="")
        for m, name in enumerate(trace["modality_names"]):
            print(f"{name}={trace['per_modality_importance'][m]:.3f}", end="  ")
        print()

    # ── Level 3: Regression refinement ──
    reg = report["regression_refinement"]
    print(f"\n{'='*60}")
    print("Level 3: Regression Refinement")
    print("=" * 60)
    print(f"  MAE (GNN):  {reg['mae_gnn']:.4f}")
    print(f"  MAE (Seg):  {reg['mae_seg']:.4f}")
    print(f"  R²:         {reg['r2']:.4f}")
    print(f"  Zero-shot Dice: {reg['zero_shot_dice']:.4f}")
    print(f"  Corrections:    {reg['n_corrections']}")
    print(f"  Degradations:   {reg['n_degradations']}")
    print(f"  Ensemble var:   {reg['mean_ensemble_var']:.6f}")

    # ── Level 4: Task divergence ──
    div = report["task_divergence"]
    print(f"\n{'='*60}")
    print("Level 4: Task Divergence")
    print("=" * 60)
    print(f"  Category distribution:")
    for cat, count in sorted(div["category_counts"].items()):
        print(f"    {cat}: {count}")
    print(f"  Interesting nodes (disagreements): {len(div['interesting_nodes'])}")
    for node in div["interesting_nodes"][:5]:
        print(f"    SV {node['sv_label']}: {node['category']} "
              f"(reg={node['y_reg']:.3f}, unc={node['unc_prob']:.3f}, "
              f"GT={node['gt_cls']})")

    # ── Level 5: Uncertainty explanation ──
    unc = report["uncertainty_explanation"]
    print(f"\n{'='*60}")
    print("Level 5: Uncertainty-Driven Explanation")
    print("=" * 60)
    print(f"  High-uncertainty SVs: {unc['n_high_uncertainty']} "
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

    print(f"\nPlan 2 Explainability pipeline verified (5 levels). ✓")

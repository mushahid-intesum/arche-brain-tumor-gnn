import torch
import numpy as np

from .config import GRAPH


# ── Graph-Level Attention Traces ─────────────────────────────────────


def graph_attention_trace(node_idx, edge_index, graph_attentions,
                          targets=None, idx_to_label=None, top_k=5):
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index)

    # Find all edges where node_idx is the destination
    dst_mask = (edge_index[1] == node_idx)
    incoming_indices = torch.where(dst_mask)[0]

    if len(incoming_indices) == 0:
        return {
            "node_idx": node_idx,
            "sv_label": idx_to_label.get(node_idx) if idx_to_label else None,
            "neighbors": [],
            "attention_entropy": 0.0,
            "total_incoming_edges": 0,
        }

    # Source nodes for incoming edges
    src_nodes = edge_index[0, incoming_indices].cpu().numpy()
    unique_src = np.unique(src_nodes)

    # Aggregate attention per source node across layers and heads
    neighbor_attn = {}
    for src in unique_src:
        layer_attns = []
        for layer_idx, attn in enumerate(graph_attentions):
            # attn shape: (E, n_heads) — attention for each edge per head
            # Find edges from src → node_idx
            src_mask = (edge_index[0] == src) & (edge_index[1] == node_idx)
            edge_positions = torch.where(src_mask)[0]
            if len(edge_positions) > 0:
                # Average attention across heads for this edge
                edge_attn = attn[edge_positions].mean().item()
                layer_attns.append(edge_attn)
            else:
                layer_attns.append(0.0)

        neighbor_attn[int(src)] = {
            "per_layer": layer_attns,
            "mean": float(np.mean(layer_attns)),
        }

    # Sort by mean attention (descending)
    ranked = sorted(neighbor_attn.items(), key=lambda x: x[1]["mean"],
                    reverse=True)

    # Compute attention entropy (how focused vs. diffuse)
    attn_values = np.array([v["mean"] for _, v in ranked])
    attn_sum = attn_values.sum()
    if attn_sum > 0:
        attn_probs = attn_values / attn_sum
        attn_probs = attn_probs[attn_probs > 0]
        entropy = float(-np.sum(attn_probs * np.log(attn_probs)))
    else:
        entropy = 0.0

    # Build neighbor list with optional GT context
    neighbors = []
    for src, attn_info in ranked[:top_k]:
        entry = {
            "node_idx": src,
            "sv_label": idx_to_label.get(src) if idx_to_label else None,
            "mean_attention": attn_info["mean"],
            "per_layer_attention": attn_info["per_layer"],
        }
        # Add GT context if available
        if targets and idx_to_label:
            label = idx_to_label.get(src)
            if label and label in targets:
                t = targets[label]
                entry["y_cls"] = t["y_cls"]
                entry["y_dominant"] = t["y_dominant"]
                entry["centroid"] = t["centroid"].tolist()
        neighbors.append(entry)

    return {
        "node_idx": node_idx,
        "sv_label": idx_to_label.get(node_idx) if idx_to_label else None,
        "neighbors": neighbors,
        "attention_entropy": entropy,
        "total_incoming_edges": len(incoming_indices),
    }


# ── Patch-Level Attention Traces ─────────────────────────────────────


def patch_attention_trace(node_idx, patch_attentions, n_modalities=4,
                          n_patch=None, modality_names=None):
    n_patch = n_patch or GRAPH["n_patch"]
    modality_names = modality_names or GRAPH["modalities"]
    n_rows = n_patch * n_modalities

    if not patch_attentions:
        return {
            "node_idx": node_idx,
            "cls_to_patch_attention": np.zeros(n_rows),
            "per_patch_importance": np.zeros(n_patch),
            "per_modality_importance": np.zeros(n_modalities),
            "top_patches": [],
            "attention_layers": 0,
        }

    # Average [CLS]→patch attention across layers and heads
    # Attention shape: (B, n_heads, seq_len, seq_len)
    # seq_len = 1 (CLS) + n_rows (patches)
    # CLS is at position 0
    cls_attn_layers = []
    for attn in patch_attentions:
        # attn[node_idx]: (n_heads, seq_len, seq_len)
        # CLS → patches: row 0, columns 1:n_rows+1
        node_attn = attn[node_idx]  # (n_heads, seq_len, seq_len)
        cls_to_all = node_attn[:, 0, 1:]  # (n_heads, n_rows) — CLS to patches
        cls_attn_layers.append(cls_to_all.cpu().numpy())

    # Average across layers and heads
    avg_attn = np.mean(cls_attn_layers, axis=(0, 1))  # (n_rows,)

    # Aggregate per patch centroid (sum over modalities within each patch)
    per_patch = np.zeros(n_patch)
    for p in range(n_patch):
        start = p * n_modalities
        end = start + n_modalities
        per_patch[p] = avg_attn[start:end].sum()

    # Aggregate per modality (sum over patches within each modality)
    per_mod = np.zeros(n_modalities)
    for m in range(n_modalities):
        indices = [p * n_modalities + m for p in range(n_patch)]
        per_mod[m] = avg_attn[indices].sum()

    # Rank individual patch rows
    ranked_indices = np.argsort(avg_attn)[::-1]
    top_patches = []
    for idx in ranked_indices[:8]:  # top 8 rows
        patch_id = idx // n_modalities
        mod_id = idx % n_modalities
        top_patches.append({
            "row_idx": int(idx),
            "patch_id": int(patch_id),
            "modality_id": int(mod_id),
            "modality_name": modality_names[mod_id] if mod_id < len(modality_names) else f"mod_{mod_id}",
            "attention": float(avg_attn[idx]),
        })

    return {
        "node_idx": node_idx,
        "cls_to_patch_attention": avg_attn,
        "per_patch_importance": per_patch,
        "per_modality_importance": per_mod,
        "modality_names": modality_names,
        "top_patches": top_patches,
        "attention_layers": len(patch_attentions),
    }


# ── Refinement-Level Traces ──────────────────────────────────────────


def refinement_trace(node_logits, targets, retained_labels,
                     seg_features=None, idx_to_label=None,
                     edge_index=None, graph_attentions=None,
                     threshold=0.5):

    N = len(retained_labels)
    gnn_preds = (torch.sigmoid(node_logits) > threshold).cpu().numpy()

    gt_labels = np.array([targets[retained_labels[i]]["y_cls"]
                          for i in range(N)])

    # Seg model predictions (binary: is dominant class != BG?)
    if seg_features is not None:
        seg_preds = np.array([
            1 if seg_features[retained_labels[i]]["seg_pred"] != 0 else 0
            for i in range(N)
        ])
    else:
        seg_preds = np.zeros(N, dtype=int)

    # Accuracy
    gnn_correct = (gnn_preds == gt_labels).sum()
    seg_correct = (seg_preds == gt_labels).sum()
    accuracy_gnn = float(gnn_correct / max(N, 1))
    accuracy_seg = float(seg_correct / max(N, 1))

    # Find corrections and degradations
    seg_wrong = (seg_preds != gt_labels)
    seg_right = (seg_preds == gt_labels)
    gnn_right = (gnn_preds == gt_labels)
    gnn_wrong = (gnn_preds != gt_labels)

    corrections_mask = seg_wrong & gnn_right  # seg wrong, GNN fixed it
    degradations_mask = seg_right & gnn_wrong  # seg right, GNN broke it

    n_seg_errors = seg_wrong.sum()
    correction_rate = float(corrections_mask.sum() / max(n_seg_errors, 1))
    n_seg_correct = seg_right.sum()
    degradation_rate = float(degradations_mask.sum() / max(n_seg_correct, 1))

    # Build correction details
    corrections = []
    for i in range(N):
        if not corrections_mask[i] and not degradations_mask[i]:
            continue

        label = retained_labels[i]
        entry = {
            "node_idx": i,
            "sv_label": label,
            "gt": int(gt_labels[i]),
            "gnn_pred": int(gnn_preds[i]),
            "gnn_prob": float(torch.sigmoid(node_logits[i]).item()),
            "seg_pred": int(seg_preds[i]),
            "type": "correction" if corrections_mask[i] else "degradation",
            "gt_dominant": targets[label]["y_dominant"],
            "centroid": targets[label]["centroid"].tolist(),
        }

        # Add seg prior context
        if seg_features and label in seg_features:
            sf = seg_features[label]
            entry["seg_entropy"] = float(sf["seg_entropy"])
            entry["seg_feat"] = sf["seg_feat"].tolist()

        # Trace back through graph attention for corrections
        if (corrections_mask[i] and edge_index is not None
                and graph_attentions is not None):
            trace = graph_attention_trace(
                i, edge_index, graph_attentions,
                targets=targets, idx_to_label=idx_to_label, top_k=3,
            )
            entry["attention_trace"] = trace["neighbors"]
            entry["attention_entropy"] = trace["attention_entropy"]

        corrections.append(entry)

    return {
        "total_nodes": N,
        "accuracy_gnn": accuracy_gnn,
        "accuracy_seg": accuracy_seg,
        "correction_rate": correction_rate,
        "degradation_rate": degradation_rate,
        "n_corrections": int(corrections_mask.sum()),
        "n_degradations": int(degradations_mask.sum()),
        "corrections": corrections,
    }


# ── Full Explanation Report ──────────────────────────────────────────


def explain_case(model, result, device=None, top_k_nodes=10):
    from .model import compute_laplacian_pe

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
        node_logits, edge_logits, attn_dict = model(
            patch_batch, seg_batch, ei, ea,
            lap_pe=lpe, return_attention=True,
        )

    # ── Level 1: Graph attention traces ──
    # Focus on tumor SVs and high-probability predictions
    probs = torch.sigmoid(node_logits).cpu()
    tumor_indices = [i for i in range(N)
                     if result["targets"][retained[i]]["y_cls"] == 1]
    high_prob_indices = torch.argsort(probs, descending=True)[:top_k_nodes].tolist()
    explain_indices = list(set(tumor_indices + high_prob_indices))[:top_k_nodes]

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

    # ── Level 3: Refinement trace ──
    ref_trace = refinement_trace(
        node_logits, result["targets"], retained,
        seg_features=result["seg_features"],
        idx_to_label=result["idx_to_label"],
        edge_index=ei, graph_attentions=attn_dict["graph"],
    )

    return {
        "case_id": result["case_id"],
        "graph_traces": graph_traces,
        "patch_traces": patch_traces,
        "refinement": ref_trace,
        "node_logits": node_logits.cpu(),
        "edge_logits": edge_logits.cpu(),
    }


# ── Main (test with synthetic/real data) ─────────────────────────────


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from .model import TumorRefiner, compute_laplacian_pe
    from .supervoxel import discover_cases, preprocess_case
    from .config import SHARED

    device = SHARED["device"]
    print(f"Device: {device}")

    # Build model (untrained)
    model = TumorRefiner(use_seg_prior=True).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params\n")

    # Load real case
    cases = discover_cases(GRAPH["data_root"])
    if not cases:
        print("No BraTS cases found.")
        exit(1)

    print(f"Processing: {cases[0]['case_id']}")
    import time
    t0 = time.time()
    result = preprocess_case(cases[0])
    print(f"Preprocessed in {time.time()-t0:.1f}s\n")

    # Generate explanation report
    t1 = time.time()
    report = explain_case(model, result, device=device)
    print(f"Explanation generated in {time.time()-t1:.1f}s\n")

    # ── Print Level 1: Graph traces ──
    print("=" * 60)
    print("Level 1: Graph Attention Traces")
    print("=" * 60)
    for idx, trace in report["graph_traces"].items():
        label = trace["sv_label"]
        gt = result["targets"].get(label, {})
        gt_str = f"GT={gt.get('y_cls', '?')}" if gt else ""
        prob = float(torch.sigmoid(report["node_logits"][idx]))
        print(f"\n  Node {idx} (SV {label}): p={prob:.3f}, {gt_str}, "
              f"entropy={trace['attention_entropy']:.3f}")
        for nb in trace["neighbors"][:3]:
            nb_gt = f"y_cls={nb.get('y_cls', '?')}" if 'y_cls' in nb else ""
            print(f"    → Neighbor {nb['node_idx']} (SV {nb['sv_label']}): "
                  f"attn={nb['mean_attention']:.4f}, {nb_gt}")

    # ── Print Level 2: Patch traces ──
    print(f"\n{'='*60}")
    print("Level 2: Patch Attention Traces")
    print("=" * 60)
    for idx, trace in list(report["patch_traces"].items())[:3]:
        print(f"\n  Node {idx}: {trace['attention_layers']} attention layers")
        print(f"    Per-modality importance: ", end="")
        for m, name in enumerate(trace["modality_names"]):
            print(f"{name}={trace['per_modality_importance'][m]:.3f}", end="  ")
        print()
        print(f"    Per-patch importance: {trace['per_patch_importance']}")
        print(f"    Top patches:")
        for tp in trace["top_patches"][:4]:
            print(f"      patch={tp['patch_id']}, mod={tp['modality_name']}, "
                  f"attn={tp['attention']:.4f}")

    # ── Print Level 3: Refinement ──
    ref = report["refinement"]
    print(f"\n{'='*60}")
    print("Level 3: Refinement Trace")
    print("=" * 60)
    print(f"  GNN accuracy:  {ref['accuracy_gnn']:.1%}")
    print(f"  Seg accuracy:  {ref['accuracy_seg']:.1%}")
    print(f"  Corrections:   {ref['n_corrections']} "
          f"(rate: {ref['correction_rate']:.1%} of seg errors)")
    print(f"  Degradations:  {ref['n_degradations']} "
          f"(rate: {ref['degradation_rate']:.1%} of seg correct)")
    if ref["corrections"]:
        print(f"\n  Detailed corrections/degradations:")
        for c in ref["corrections"][:5]:
            print(f"    SV {c['sv_label']}: [{c['type']}] "
                  f"GT={c['gt']}, GNN={c['gnn_pred']}(p={c['gnn_prob']:.3f}), "
                  f"Seg={c['seg_pred']}")

    print(f"\nExplainability pipeline verified. ✓")

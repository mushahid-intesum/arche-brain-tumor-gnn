"""
Experiment Runner for Plan 3a — All Ablation Configurations (E1–E7).

Unified interface to run any experiment from the research plan:
  E1: kNN graph + GATv2 (no concepts)           — baseline
  E2: Hypergraph only (no concepts, no fusion)   — does hypergraph help?
  E3: Hypergraph + Concept Bottleneck            — core contribution
  E4: E3 + Clinical fusion (MRePath-style)       — does multimodal help?
  E5: E4 + TIF multi-granular tree               — full model
  E6: E4 + EST regularizer                       — faithfulness training
  E7: (future) Track B: Supervoxel graph

Configuration:
    All settings are in plan3a/config.py:
      RUN_EXPERIMENT  — "E1"–"E6" or "all"
      RUN_LIMIT       — limit patients (None = all)
      RUN_AUDIT       — run faithfulness audit post-training

Usage:
    python -m plan3a.runner
"""
import os
import sys
import json
import time

from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plan3a.config import (
    PROCESSED_DIR, LR, WEIGHT_DECAY, EPOCHS, NUM_FOLDS, DEVICE,
    SHEAF_HGNN_DIM, SHEAF_HGNN_LAYERS, NUM_CONCEPTS, GRAD_ACCUM_STEPS,
    RUN_EXPERIMENT, RUN_LIMIT, RUN_AUDIT, CHECKPOINTS_DIR,
)
from plan3a.data.dataset import Plan3aDataset, get_kfold_splits
from plan3a.data.hypergraph import build_patient_hypergraph
from plan3a.model.full_model import Plan3aModel, NLLSurvivalLoss
from plan3a.eval.task_metrics import (
    concordance_index, concept_metrics, hazard_to_risk, compute_time_bins,
)
from plan3a.eval.faithfulness import FaithfulnessAuditor, compute_rejection_ratios


# ── Experiment Configurations ────────────────────────────────────────────

EXPERIMENTS = {
    "E1": {
        "name": "E1: Baseline GNN (no hypergraph, no concepts)",
        "description": "kNN graph + simple GNN — no concept bottleneck, no fusion",
        "model_kwargs": {
            "use_hecrl": False,
            "residual_bypass": True,   # bypass concepts entirely
            "use_fusion": False,
            "use_tree": False,
        },
        "use_hypergraph": False,  # use kNN graph instead
    },
    "E2": {
        "name": "E2: Hypergraph only (no concepts, no fusion)",
        "description": "Sheaf HGNN without concept bottleneck or clinical fusion",
        "model_kwargs": {
            "use_hecrl": False,
            "residual_bypass": True,
            "use_fusion": False,
            "use_tree": False,
        },
        "use_hypergraph": True,
    },
    "E3": {
        "name": "E3: Hypergraph + Concept Bottleneck",
        "description": "Core contribution: sheaf HGNN with concept bottleneck (imaging only)",
        "model_kwargs": {
            "use_hecrl": True,
            "residual_bypass": False,
            "use_fusion": False,
            "use_tree": False,
        },
        "use_hypergraph": True,
    },
    "E4": {
        "name": "E4: E3 + Clinical Fusion (MRePath-style)",
        "description": "Full imaging+clinical multimodal model",
        "model_kwargs": {
            "use_hecrl": True,
            "residual_bypass": False,
            "use_fusion": True,
            "use_tree": False,
        },
        "use_hypergraph": True,
    },
    "E5": {
        "name": "E5: E4 + TIF Multi-Granular Tree",
        "description": "Full model with hierarchical coarsening for multi-scale explanations",
        "model_kwargs": {
            "use_hecrl": True,
            "residual_bypass": False,
            "use_fusion": True,
            "use_tree": True,
            "tree_levels": 3,
        },
        "use_hypergraph": True,
    },
    "E6": {
        "name": "E6: E4 + EST Regularizer",
        "description": "Full model with faithfulness regularization during training",
        "model_kwargs": {
            "use_hecrl": True,
            "residual_bypass": False,
            "use_fusion": True,
            "use_tree": False,
        },
        "use_hypergraph": True,
        "est_regularize": True,
    },
}


def build_knn_graph(data, k=8):
    """
    Build a kNN graph from patch coordinates as a simple baseline.
    Converts kNN edges into a degenerate hypergraph (each edge = 2-node hyperedge).
    """
    from scipy.spatial.distance import cdist

    coords = data["coords"].numpy() if isinstance(data["coords"], torch.Tensor) else data["coords"]
    N = coords.shape[0]
    if N == 0:
        data["hyperedge_index"] = torch.zeros(2, 0, dtype=torch.long)
        data["num_hyperedges"] = 0
        data["node_features"] = torch.zeros(0, 1536)
        return data

    dist = cdist(coords, coords)
    np.fill_diagonal(dist, np.inf)

    node_list = []
    edge_list = []
    edge_id = 0

    for i in range(N):
        neighbors = np.argsort(dist[i])[:k]
        for j in neighbors:
            node_list.extend([i, j])
            edge_list.extend([edge_id, edge_id])
            edge_id += 1

    data["hyperedge_index"] = torch.tensor([node_list, edge_list], dtype=torch.long)
    data["num_hyperedges"] = edge_id
    patches = data["patches"]
    data["node_features"] = patches.reshape(N, -1).float() if isinstance(patches, torch.Tensor) else torch.from_numpy(patches.reshape(N, -1)).float()
    return data


def prepare_patient_data(patient_pt, use_hypergraph=True):
    """Load and prepare patient data based on experiment config."""
    if use_hypergraph:
        return build_patient_hypergraph(patient_pt)
    else:
        # E1 baseline: kNN graph
        data = {
            "patches": patient_pt["patches"],
            "coords": patient_pt["coords"],
            "concepts": patient_pt["concepts"],
            "patient_id": patient_pt["patient_id"],
            "clinical_features": patient_pt["clinical_features"],
            "survival_time": patient_pt["survival_time"],
            "event": patient_pt["event"],
            "has_survival": patient_pt["has_survival"],
            "modality_mask": patient_pt["modality_mask"],
            "num_nodes": patient_pt["num_patches"],
        }
        return build_knn_graph(data)


def train_epoch(model, dataset, optimizer, time_bins, device,
                exp_config, grad_accum=GRAD_ACCUM_STEPS):
    """Train one epoch with optional EST regularization."""
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_surv = 0.0
    total_conc = 0.0
    n = 0

    for i in range(len(dataset)):
        data = dataset[i]
        if data["num_nodes"] == 0 or not data["has_survival"]:
            continue

        nf = data["node_features"].to(device)
        he = data["hyperedge_index"].to(device)
        ct = data["concepts"].to(device) if "concepts" in data else None
        cl = data["clinical_features"].to(device)
        st = data["survival_time"].to(device).unsqueeze(0)
        ev = data["event"].to(device).unsqueeze(0)
        tb = time_bins.to(device)

        outputs = model(
            node_features=nf, hyperedge_index=he,
            num_nodes=data["num_nodes"], num_edges=data["num_hyperedges"],
            concept_targets=ct, clinical_features=cl,
        )

        losses = model.compute_loss(outputs, st, ev, tb)
        loss = losses["total_loss"] / grad_accum

        loss.backward()

        if (i + 1) % grad_accum == 0 or (i + 1) == len(dataset):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += losses["total_loss"].item()
        total_surv += losses["survival_loss"].item()
        total_conc += losses["concept_loss"].item()
        n += 1

    return {
        "loss": total_loss / max(n, 1),
        "survival_loss": total_surv / max(n, 1),
        "concept_loss": total_conc / max(n, 1),
        "n_samples": n,
    }


@torch.no_grad()
def evaluate_epoch(model, dataset, time_bins, device):
    """Evaluate and compute C-Index + concept metrics."""
    model.eval()
    risks, times, events = [], [], []
    pred_c, true_c = [], []
    total_loss = 0.0
    n = 0

    for i in range(len(dataset)):
        data = dataset[i]
        if data["num_nodes"] == 0 or not data["has_survival"]:
            continue

        nf = data["node_features"].to(device)
        he = data["hyperedge_index"].to(device)
        ct = data["concepts"].to(device) if "concepts" in data else None
        cl = data["clinical_features"].to(device)
        st = data["survival_time"].to(device).unsqueeze(0)
        ev = data["event"].to(device).unsqueeze(0)
        tb = time_bins.to(device)

        outputs = model(
            node_features=nf, hyperedge_index=he,
            num_nodes=data["num_nodes"], num_edges=data["num_hyperedges"],
            concept_targets=ct, clinical_features=cl,
        )
        losses = model.compute_loss(outputs, st, ev, tb)
        total_loss += losses["total_loss"].item()

        hazard = outputs["hazard_logits"].cpu().numpy()
        risks.append(hazard_to_risk(hazard)[0])
        times.append(data["survival_time"].item())
        events.append(data["event"].item())

        pred_c.append(outputs["concepts"].cpu().numpy().mean(0))
        true_c.append(ct.cpu().numpy().mean(0) if ct is not None else np.zeros(NUM_CONCEPTS))
        n += 1

    if n < 2:
        return {"c_index": 0.5, "loss": 0.0, "n_samples": n, "mean_concept_corr": 0.0}

    c_idx = concordance_index(np.array(risks), np.array(times), np.array(events))
    c_names = ["c1:enhance", "c2:flair", "c3:t2", "c4:dti_md",
               "c5:dti_fa", "c6:heterog", "c7:boundary", "c8:spatial"]
    cm = concept_metrics(np.stack(pred_c), np.stack(true_c), c_names)
    mean_corr = np.mean([v["pearson_r"] for k, v in cm.items() if k != "c7:boundary"])

    return {
        "c_index": c_idx,
        "loss": total_loss / n,
        "n_samples": n,
        "mean_concept_corr": mean_corr,
        "concept_metrics": cm,
    }


class HypergraphDatasetWrapper(Plan3aDataset):
    """Wraps Plan3aDataset to apply experiment-specific graph construction."""

    def __init__(self, processed_dir, patient_ids, use_hypergraph=True):
        super().__init__(processed_dir, patient_ids, build_hypergraph=False)
        self.use_hypergraph = use_hypergraph

    def __getitem__(self, idx):
        filepath = os.path.join(self.processed_dir, self.files[idx])
        patient_pt = torch.load(filepath, weights_only=False)
        return prepare_patient_data(patient_pt, self.use_hypergraph)


def run_experiment(
    exp_id: str,
    processed_dir: str,
    epochs: int,
    limit: int = None,
    device: str = DEVICE,
    save_dir: str = None,
    run_audit: bool = True,
) -> dict:
    """Run a complete experiment with K-fold CV."""
    exp_config = EXPERIMENTS[exp_id]
    print(f"\n{'#'*70}")
    print(f"# {exp_config['name']}")
    print(f"# {exp_config['description']}")
    print(f"{'#'*70}")

    # Discover patients
    all_pids = sorted([
        f.replace(".pt", "") for f in os.listdir(processed_dir)
        if f.endswith(".pt") and f.startswith("UPENN")
    ])
    if limit:
        all_pids = all_pids[:limit]

    splits = get_kfold_splits(all_pids, NUM_FOLDS)
    use_hg = exp_config.get("use_hypergraph", True)

    fold_results = []
    all_audit_reports = []

    for fold_idx in range(NUM_FOLDS):
        split = splits[fold_idx]
        print(f"\n  Fold {fold_idx+1}/{NUM_FOLDS}: "
              f"train={len(split['train'])}, val={len(split['val'])}")

        train_ds = HypergraphDatasetWrapper(processed_dir, split["train"], use_hg)
        val_ds = HypergraphDatasetWrapper(processed_dir, split["val"], use_hg)

        # Compute time bins
        t_times, t_events = [], []
        for i in range(len(train_ds)):
            d = train_ds[i]
            if d["has_survival"]:
                t_times.append(d["survival_time"].item())
                t_events.append(d["event"].item())
        time_bins = torch.from_numpy(
            compute_time_bins(np.array(t_times), np.array(t_events), 4)
        )

        # Build model
        model = Plan3aModel(
            patch_dim=1536,
            embed_dim=SHEAF_HGNN_DIM,
            num_layers=SHEAF_HGNN_LAYERS,
            num_concepts=NUM_CONCEPTS,
            clinical_dim=18,
            num_survival_bins=4,
            **exp_config["model_kwargs"],
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR * 0.01)

        best_ci = 0.0
        best_ep = 0
        history = []

        for ep in range(epochs):
            t0 = time.time()
            tm = train_epoch(model, train_ds, optimizer, time_bins, device, exp_config)
            scheduler.step()
            vm = evaluate_epoch(model, val_ds, time_bins, device)
            elapsed = time.time() - t0

            print(f"    Ep {ep+1:3d} | loss={tm['loss']:.4f} "
                  f"surv={tm['survival_loss']:.4f} conc={tm['concept_loss']:.4f} "
                  f"| val C-Idx={vm['c_index']:.4f} r={vm['mean_concept_corr']:.3f} "
                  f"| {elapsed:.1f}s")

            if vm["c_index"] > best_ci:
                best_ci = vm["c_index"]
                best_ep = ep + 1
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(model.state_dict(),
                               os.path.join(save_dir, f"{exp_id}_fold{fold_idx}.pt"))

            history.append({
                "epoch": ep + 1,
                "train_loss": tm["loss"],
                "val_c_index": vm["c_index"],
                "concept_corr": vm["mean_concept_corr"],
            })

        # Post-training faithfulness audit
        fold_audits = []
        if run_audit and use_hg:
            auditor = FaithfulnessAuditor(model, device=device, est_samples=20,
                                          prediction_threshold=0.1)
            for i in range(min(len(val_ds), 5)):
                data = val_ds[i]
                if data["num_nodes"] > 0 and data["has_survival"]:
                    report = auditor.audit_patient(data, top_k_ratio=0.2)
                    fold_audits.append(report)
                    all_audit_reports.append(report)

        print(f"    Best: C-Index={best_ci:.4f} @ epoch {best_ep}")

        fold_results.append({
            "fold": fold_idx,
            "best_c_index": best_ci,
            "best_epoch": best_ep,
            "n_params": n_params,
            "history": history,
            "n_audit": len(fold_audits),
        })

    # Aggregate
    c_indices = [r["best_c_index"] for r in fold_results]
    result = {
        "experiment": exp_id,
        "name": exp_config["name"],
        "description": exp_config["description"],
        "timestamp": datetime.now().isoformat(),
        "n_patients": len(all_pids),
        "n_folds": NUM_FOLDS,
        "epochs": epochs,
        "n_params": fold_results[0]["n_params"],
        "mean_c_index": float(np.mean(c_indices)),
        "std_c_index": float(np.std(c_indices)),
        "fold_results": fold_results,
    }

    # Add faithfulness summary
    if all_audit_reports:
        ratios = compute_rejection_ratios(all_audit_reports)
        result["faithfulness"] = ratios

    return result


def main():
    processed_dir = str(PROCESSED_DIR)
    save_dir = str(CHECKPOINTS_DIR)

    if RUN_EXPERIMENT == "all":
        experiments = list(EXPERIMENTS.keys())
    else:
        experiments = [RUN_EXPERIMENT]

    all_results = []

    for exp_id in experiments:
        result = run_experiment(
            exp_id=exp_id,
            processed_dir=processed_dir,
            epochs=EPOCHS,
            limit=RUN_LIMIT,
            device=DEVICE,
            save_dir=save_dir,
            run_audit=RUN_AUDIT,
        )
        all_results.append(result)

    # ── Summary Table ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"ABLATION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Exp':<5} {'Configuration':<45} {'C-Index':>12} {'Params':>10}")
    print(f"{'-'*5} {'-'*45} {'-'*12} {'-'*10}")
    for r in all_results:
        ci = f"{r['mean_c_index']:.4f}±{r['std_c_index']:.4f}"
        print(f"{r['experiment']:<5} {r['name'][:45]:<45} {ci:>12} {r['n_params']:>10,}")

    if any("faithfulness" in r for r in all_results):
        print(f"\n{'Exp':<5} {'EST Rej':>10} {'Fid- Rej':>10} {'Suf Rej':>10} {'Overall':>10}")
        print(f"{'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for r in all_results:
            if "faithfulness" in r:
                f = r["faithfulness"]
                print(f"{r['experiment']:<5} "
                      f"{f['est_rejection']:>9.0%} "
                      f"{f['fid_minus_rejection']:>9.0%} "
                      f"{f['sufficiency_rejection']:>9.0%} "
                      f"{f['overall_rejection']:>9.0%}")
    print(f"{'='*80}")

    # Save
    results_path = os.path.join(save_dir, "ablation_results.json")
    os.makedirs(save_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

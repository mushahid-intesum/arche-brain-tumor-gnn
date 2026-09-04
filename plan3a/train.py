"""
Training Script for Plan 3a — Hypergraph Concept Bottleneck GNN.

Runs K-fold cross-validation with:
  - Per-epoch training loop (batch_size=1 due to variable graph sizes)
  - Gradient accumulation for effective larger batch
  - C-Index evaluation on validation set
  - Concept accuracy tracking
  - Best model checkpointing

Configuration:
    All settings are in plan3a/config.py:
      EPOCHS, TRAIN_LIMIT, TRAIN_FOLD, DEVICE, etc.

Usage:
    python -m plan3a.train
"""
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plan3a.config import (
    PROCESSED_DIR, BATCH_SIZE, GRAD_ACCUM_STEPS,
    LR, WEIGHT_DECAY, EPOCHS, NUM_FOLDS, DEVICE,
    SHEAF_HGNN_DIM, SHEAF_HGNN_LAYERS, NUM_CONCEPTS,
    TRAIN_LIMIT, TRAIN_FOLD, CHECKPOINTS_DIR,
)
from plan3a.data.dataset import Plan3aDataset, get_kfold_splits
from plan3a.model.full_model import Plan3aModel
from plan3a.eval.task_metrics import (
    concordance_index, concept_metrics,
    hazard_to_risk, compute_time_bins,
)


def train_one_epoch(
    model: nn.Module,
    dataset: Plan3aDataset,
    optimizer: torch.optim.Optimizer,
    time_bins: torch.Tensor,
    device: str,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
) -> dict:
    """Train for one epoch. Returns average losses."""
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_surv_loss = 0.0
    total_conc_loss = 0.0
    n_samples = 0
    n_with_survival = 0

    for i in range(len(dataset)):
        data = dataset[i]

        # Skip patients with no patches or no survival data
        if data["num_nodes"] == 0:
            continue
        if not data["has_survival"]:
            continue

        # Move tensors to device
        node_features = data["node_features"].to(device)
        hyperedge_index = data["hyperedge_index"].to(device)
        concepts_target = data["concepts"].to(device)
        clinical = data["clinical_features"].to(device)
        surv_time = data["survival_time"].to(device).unsqueeze(0)
        event = data["event"].to(device).unsqueeze(0)
        t_bins = time_bins.to(device)

        # Forward
        outputs = model(
            node_features=node_features,
            hyperedge_index=hyperedge_index,
            num_nodes=data["num_nodes"],
            num_edges=data["num_hyperedges"],
            concept_targets=concepts_target,
            clinical_features=clinical,
        )

        # Compute loss
        losses = model.compute_loss(outputs, surv_time, event, t_bins)
        loss = losses["total_loss"] / grad_accum_steps

        # Backward
        loss.backward()

        # Gradient accumulation
        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataset):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += losses["total_loss"].item()
        total_surv_loss += losses["survival_loss"].item()
        total_conc_loss += losses["concept_loss"].item()
        n_samples += 1
        n_with_survival += 1

    return {
        "loss": total_loss / max(n_samples, 1),
        "survival_loss": total_surv_loss / max(n_with_survival, 1),
        "concept_loss": total_conc_loss / max(n_samples, 1),
        "n_samples": n_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: Plan3aDataset,
    time_bins: torch.Tensor,
    device: str,
) -> dict:
    """Evaluate on validation set. Returns C-Index and concept metrics."""
    model.eval()

    all_risks = []
    all_times = []
    all_events = []
    all_pred_concepts = []
    all_true_concepts = []
    total_loss = 0.0
    n_samples = 0

    for i in range(len(dataset)):
        data = dataset[i]

        if data["num_nodes"] == 0:
            continue
        if not data["has_survival"]:
            continue

        node_features = data["node_features"].to(device)
        hyperedge_index = data["hyperedge_index"].to(device)
        concepts_target = data["concepts"].to(device)
        clinical = data["clinical_features"].to(device)
        surv_time = data["survival_time"].to(device).unsqueeze(0)
        event = data["event"].to(device).unsqueeze(0)
        t_bins = time_bins.to(device)

        outputs = model(
            node_features=node_features,
            hyperedge_index=hyperedge_index,
            num_nodes=data["num_nodes"],
            num_edges=data["num_hyperedges"],
            concept_targets=concepts_target,
            clinical_features=clinical,
        )

        losses = model.compute_loss(outputs, surv_time, event, t_bins)
        total_loss += losses["total_loss"].item()

        # Collect for C-Index
        hazard = outputs["hazard_logits"].cpu().numpy()
        risk = hazard_to_risk(hazard)
        all_risks.append(risk[0])
        all_times.append(data["survival_time"].item())
        all_events.append(data["event"].item())

        # Collect concept predictions (mean over nodes)
        pred_c = outputs["concepts"].cpu().numpy().mean(axis=0)
        true_c = concepts_target.cpu().numpy().mean(axis=0)
        all_pred_concepts.append(pred_c)
        all_true_concepts.append(true_c)

        n_samples += 1

    if n_samples == 0:
        return {"c_index": 0.5, "loss": 0.0, "n_samples": 0}

    # C-Index
    risks = np.array(all_risks)
    times = np.array(all_times)
    events = np.array(all_events)
    c_index = concordance_index(risks, times, events)

    # Concept metrics
    pred_concepts = np.stack(all_pred_concepts)
    true_concepts = np.stack(all_true_concepts)
    concept_names = [
        "c1:enhance", "c2:flair", "c3:t2", "c4:dti_md",
        "c5:dti_fa", "c6:heterog", "c7:boundary", "c8:spatial"
    ]
    c_metrics = concept_metrics(pred_concepts, true_concepts, concept_names)

    # Mean concept correlation
    mean_corr = np.mean([
        v["pearson_r"] for k, v in c_metrics.items() if k != "c7:boundary"
    ])

    return {
        "c_index": c_index,
        "loss": total_loss / n_samples,
        "n_samples": n_samples,
        "concept_metrics": c_metrics,
        "mean_concept_corr": mean_corr,
    }


def train_fold(
    fold: int,
    train_ids: list,
    val_ids: list,
    processed_dir: str,
    epochs: int,
    device: str,
    save_dir: str,
) -> dict:
    """Train and evaluate one fold."""
    print(f"\n{'='*60}")
    print(f"Fold {fold + 1}/{NUM_FOLDS}")
    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val:   {len(val_ids)} patients")
    print(f"{'='*60}")

    # Datasets
    train_ds = Plan3aDataset(processed_dir, train_ids, build_hypergraph=True)
    val_ds = Plan3aDataset(processed_dir, val_ids, build_hypergraph=True)

    # Compute time bins from training survival times
    train_times = []
    train_events = []
    for i in range(len(train_ds)):
        d = train_ds[i]
        if d["has_survival"]:
            train_times.append(d["survival_time"].item())
            train_events.append(d["event"].item())
    time_bins = torch.from_numpy(
        compute_time_bins(np.array(train_times), np.array(train_events), num_bins=4)
    )
    print(f"  Time bins: {time_bins.numpy()}")

    # Model
    model = Plan3aModel(
        patch_dim=1536,
        embed_dim=SHEAF_HGNN_DIM,
        num_layers=SHEAF_HGNN_LAYERS,
        num_concepts=NUM_CONCEPTS,
        clinical_dim=18,
        num_survival_bins=4,
        use_hecrl=True,
        residual_bypass=False,
        use_fusion=True,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR * 0.01)

    best_c_index = 0.0
    best_epoch = 0
    history = []

    for epoch in range(epochs):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_ds, optimizer, time_bins, device
        )
        scheduler.step()

        # Evaluate
        val_metrics = evaluate(model, val_ds, time_bins, device)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch+1:3d}/{epochs} "
            f"| train_loss={train_metrics['loss']:.4f} "
            f"surv={train_metrics['survival_loss']:.4f} "
            f"conc={train_metrics['concept_loss']:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} "
            f"C-Index={val_metrics['c_index']:.4f} "
            f"concept_r={val_metrics['mean_concept_corr']:.3f} "
            f"| lr={lr:.2e} "
            f"| {elapsed:.1f}s"
        )

        # Track best
        if val_metrics["c_index"] > best_c_index:
            best_c_index = val_metrics["c_index"]
            best_epoch = epoch + 1
            # Save best model
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"best_fold{fold}.pt"),
            )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "survival_loss": train_metrics["survival_loss"],
            "concept_loss": train_metrics["concept_loss"],
            "val_loss": val_metrics["loss"],
            "c_index": val_metrics["c_index"],
            "mean_concept_corr": val_metrics["mean_concept_corr"],
        })

    print(f"\n  Best C-Index: {best_c_index:.4f} (epoch {best_epoch})")

    return {
        "fold": fold,
        "best_c_index": best_c_index,
        "best_epoch": best_epoch,
        "history": history,
    }


def main():
    processed_dir = str(PROCESSED_DIR)
    save_dir = str(CHECKPOINTS_DIR)

    print("Plan 3a: Hypergraph Concept Bottleneck GNN Training")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Data: {processed_dir}")

    # Discover patients
    all_files = sorted([
        f.replace(".pt", "") for f in os.listdir(processed_dir)
        if f.endswith(".pt") and f.startswith("UPENN")
    ])
    if TRAIN_LIMIT:
        all_files = all_files[:TRAIN_LIMIT]
    print(f"  Patients: {len(all_files)}")

    # K-fold splits
    splits = get_kfold_splits(all_files, NUM_FOLDS)

    # Run folds
    fold_results = []
    folds_to_run = [TRAIN_FOLD] if TRAIN_FOLD is not None else range(NUM_FOLDS)

    for fold_idx in folds_to_run:
        split = splits[fold_idx]
        result = train_fold(
            fold=fold_idx,
            train_ids=split["train"],
            val_ids=split["val"],
            processed_dir=processed_dir,
            epochs=EPOCHS,
            device=DEVICE,
            save_dir=save_dir,
        )
        fold_results.append(result)

    # Summary
    c_indices = [r["best_c_index"] for r in fold_results]
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: C-Index = {r['best_c_index']:.4f} (epoch {r['best_epoch']})")
    if len(c_indices) > 1:
        print(f"  Mean C-Index: {np.mean(c_indices):.4f} ± {np.std(c_indices):.4f}")
    print(f"{'='*60}")

    # Save results
    results_path = os.path.join(save_dir, "results.json")
    os.makedirs(save_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(fold_results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()

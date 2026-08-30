"""
Unified Pipeline Orchestrator
Prediction → Classification → Segmentation → GNN

Each phase is independently trainable via its own module.
This script chains them end-to-end.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import time

from config import SHARED, PREDICTION, CLASSIFICATION, SEGMENTATION, GNN, PIPELINE, ensure_dirs

import prediction
import classification
import segmentation
import gnn


# ══════════════════════════════════════════════════════════════════════
# Phase 1: Binary Prediction (Brain MRI ND-5)
# ══════════════════════════════════════════════════════════════════════

def run_prediction():
    """Train binary tumor detector and return the model + test metrics."""
    print("\n" + "=" * 70)
    print("PHASE 1: BINARY PREDICTION")
    print("=" * 70)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = prediction.create_dataloaders(
        PREDICTION["data_root"], PREDICTION["batch_size"],
        PREDICTION["num_workers"], PREDICTION["val_split"],
    )
    print(f"Data: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")

    model = prediction.build_binary_classifier()
    prediction.train_binary(model, train_loader, val_loader)

    # Load best and evaluate
    model.load_state_dict(torch.load(str(PREDICTION["checkpoint"]), weights_only=True))
    criterion = nn.BCEWithLogitsLoss()
    result = prediction.evaluate(model, test_loader, criterion)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    print(f"\nPhase 1 Test: Acc={accuracy_score(result['labels'], result['preds']):.4f} "
          f"F1={f1_score(result['labels'], result['preds']):.4f} "
          f"AUC={roc_auc_score(result['labels'], result['probs']):.4f}")

    prediction.plot_confusion_matrix(result["labels"], result["preds"])

    return model, result, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════
# Phase 2: Multi-class Classification (Brain MRI ND-5)
# ══════════════════════════════════════════════════════════════════════

def run_classification(train_ds, val_ds, test_ds):
    """Train 3-class tumor classifier and generate GradCAM pseudo-masks."""
    print("\n" + "=" * 70)
    print("PHASE 2: MULTI-CLASS CLASSIFICATION")
    print("=" * 70)

    from torch.utils.data import DataLoader, Subset

    # Filter to tumor-only
    tumor_train_idx = [i for i in range(len(train_ds)) if train_ds.binary_labels[i].item() == 1]
    tumor_val_idx = [i for i in range(len(val_ds)) if val_ds.binary_labels[i].item() == 1]

    tumor_train_subset = Subset(train_ds, tumor_train_idx)
    tumor_val_subset = Subset(val_ds, tumor_val_idx)

    # Class weights
    tumor_class_counts = torch.zeros(3)
    for idx in tumor_train_idx:
        label = train_ds.multiclass_labels[idx].item()
        tumor_class_counts[label] += 1
    class_weights = (1.0 / tumor_class_counts)
    class_weights = class_weights / class_weights.sum() * 3.0

    tumor_train_loader = DataLoader(
        tumor_train_subset, batch_size=CLASSIFICATION["batch_size"],
        shuffle=True, num_workers=CLASSIFICATION["num_workers"], pin_memory=True,
    )
    tumor_val_loader = DataLoader(
        tumor_val_subset, batch_size=CLASSIFICATION["batch_size"],
        shuffle=False, num_workers=CLASSIFICATION["num_workers"], pin_memory=True,
    )
    print(f"Tumor data: train={len(tumor_train_subset)} | val={len(tumor_val_subset)}")

    model = classification.build_multiclass_classifier()
    classification.train_multiclass(model, tumor_train_loader, tumor_val_loader, class_weights)

    # Test
    model.load_state_dict(torch.load(str(CLASSIFICATION["checkpoint"]), weights_only=True))
    criterion = classification.FocalLoss(
        alpha=class_weights.to(SHARED["device"]), gamma=CLASSIFICATION["focal_gamma"],
    )

    tumor_test_idx = [i for i in range(len(test_ds)) if test_ds.binary_labels[i].item() == 1]
    tumor_test_subset = Subset(test_ds, tumor_test_idx)
    tumor_test_loader = DataLoader(
        tumor_test_subset, batch_size=CLASSIFICATION["batch_size"],
        shuffle=False, num_workers=CLASSIFICATION["num_workers"], pin_memory=True,
    )

    result = classification.evaluate(model, tumor_test_loader, criterion)

    from sklearn.metrics import classification_report
    print("\nPhase 2 Test:")
    print(classification_report(
        result["labels"], result["preds"],
        target_names=CLASSIFICATION["tumor_classes"],
    ))

    classification.plot_confusion_matrix(
        result["labels"], result["preds"],
        class_names=["Glioma", "Menin.", "Pituit."],
    )

    # Generate pseudo-masks
    classification.generate_pseudo_masks(
        model, CLASSIFICATION["data_root"], CLASSIFICATION["pseudo_mask_dir"],
    )

    return model, result


# ══════════════════════════════════════════════════════════════════════
# Phase 3: Multi-class Segmentation (BraTS 2023)
# ══════════════════════════════════════════════════════════════════════

def run_segmentation():
    """Train BraTS segmentation model and export for GNN."""
    print("\n" + "=" * 70)
    print("PHASE 3: BRATS SEGMENTATION")
    print("=" * 70)

    train_meta, val_meta, test_meta, train_loader, val_loader, test_loader = segmentation.prepare_data()

    model = segmentation.build_segmentation_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"DeepLabV3+ | Parameters: {total_params:,}")

    segmentation.train_segmentation(model, train_loader, val_loader)

    # Test
    model.load_state_dict(torch.load(str(SEGMENTATION["checkpoint"]), weights_only=True))
    metrics = segmentation.evaluate_segmentation(model, test_loader)

    print(f"\nPhase 3 Test ({len(metrics['preds'])} slices):")
    for c in [1, 2, 3]:
        name = SEGMENTATION["class_names"][c]
        print(f"  {name}: {metrics['per_class_dice'][c]:.4f} ± {metrics['per_class_std'][c]:.4f}")
    print(f"  BraTS: WT={metrics['brats_regions']['WT']:.4f} "
          f"TC={metrics['brats_regions']['TC']:.4f} "
          f"ET={metrics['brats_regions']['ET']:.4f}")

    segmentation.plot_confusion(metrics["preds"], metrics["gts"])
    segmentation.plot_test_results(test_meta, metrics["preds"], metrics["gts"])

    # Export for GNN
    segmentation.export_for_gnn(model, train_meta, val_meta, test_meta)

    return model, metrics, train_meta, val_meta, test_meta


# ══════════════════════════════════════════════════════════════════════
# Phase 4: 3D GNN Edge Prediction (BraTS outputs)
# ══════════════════════════════════════════════════════════════════════

def run_gnn():
    """Build 3D graphs, train OCN model, generate reasoning traces."""
    print("\n" + "=" * 70)
    print("PHASE 4: 3D GNN EDGE PREDICTION")
    print("=" * 70)

    train_graphs, val_graphs, test_graphs = gnn.build_all_graphs()
    gnn.plot_3d_graphs(train_graphs)

    model = gnn.EdgePredictor().to(SHARED["device"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"EdgePredictor | Parameters: {total_params:,}")

    gnn.train_gnn(model, train_graphs, val_graphs)

    # Test
    model.load_state_dict(torch.load(str(GNN["checkpoint"]), weights_only=True))
    sf_computer = gnn.StructuralFeatureComputer()
    test_metrics = gnn.evaluate_gnn(model, test_graphs, sf_computer)

    print(f"\nPhase 4 Test:")
    print(f"  AUC-ROC: {test_metrics['auc']:.4f}")
    print(f"  AP:      {test_metrics['ap']:.4f}")
    print(f"  Intra:   {test_metrics['intra_auc']:.4f}")
    print(f"  Inter:   {test_metrics['inter_auc']:.4f}")

    # Reasoning traces
    demo_graph = None
    for g in test_graphs:
        if g.edge_index.size(1) >= 10 and g.x.size(0) >= 5:
            demo_graph = g
            break
    if demo_graph is None and len(test_graphs) > 0:
        demo_graph = test_graphs[0]

    if demo_graph is not None and demo_graph.edge_index.size(1) >= 2:
        data = demo_graph.to(SHARED["device"])
        pos_ei = data.edge_index
        pos_sf, pos_cn = sf_computer.compute(demo_graph.edge_index, demo_graph.x.size(0), pos_ei)
        pos_src_t, pos_dst_t, pos_inter = gnn.get_edge_metadata(demo_graph, pos_ei)

        z, _, sv_attns = model.encode(data, return_attention=True)
        pos_pred = model.decode(
            z, pos_ei, pos_sf, pos_cn,
            pos_src_t.to(SHARED["device"]),
            pos_dst_t.to(SHARED["device"]),
            pos_inter.to(SHARED["device"]),
        )

        trace_gen = gnn.ReasoningTraceGenerator()
        traces = trace_gen.generate(
            demo_graph, z, pos_pred, pos_sf, pos_cn,
            pos_src_t, pos_dst_t, pos_inter, top_k=8,
        )

        print(f"\n--- Reasoning Traces for {demo_graph.case_id} ---")
        for t in traces:
            print(t)
            print()

        gnn.plot_results(demo_graph, pos_pred, test_metrics["tissue_pairs"],
                         test_metrics["all_labels"], test_metrics["all_scores"])

    return model, test_metrics


# ══════════════════════════════════════════════════════════════════════
# Full Pipeline
# ══════════════════════════════════════════════════════════════════════

def run_pipeline():
    """Execute the full end-to-end pipeline: Pred → Clas → Seg → GNN."""
    start = time.time()
    ensure_dirs()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          ARCHE — Brain Tumor Analysis Pipeline             ║")
    print("║     Prediction → Classification → Segmentation → GNN      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nDevice: {SHARED['device']}")

    # Phase 1 + 2: Brain MRI ND-5 dataset
    pred_model, pred_result, train_ds, val_ds, test_ds, _, _, _ = run_prediction()
    cls_model, cls_result = run_classification(train_ds, val_ds, test_ds)

    # Phase 3 + 4: BraTS dataset
    seg_model, seg_metrics, *_ = run_segmentation()
    gnn_model, gnn_metrics = run_gnn()

    # Summary
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    from sklearn.metrics import accuracy_score, roc_auc_score
    print(f"\n{'Phase':<25} {'Metric':<15} {'Value'}")
    print("-" * 55)
    print(f"{'1. Prediction':<25} {'Accuracy':<15} {accuracy_score(pred_result['labels'], pred_result['preds']):.4f}")
    print(f"{'1. Prediction':<25} {'AUC-ROC':<15} {roc_auc_score(pred_result['labels'], pred_result['probs']):.4f}")
    print(f"{'2. Classification':<25} {'Accuracy':<15} {cls_result['accuracy']:.4f}")
    print(f"{'3. Segmentation':<25} {'WT Dice':<15} {seg_metrics['brats_regions']['WT']:.4f}")
    print(f"{'3. Segmentation':<25} {'TC Dice':<15} {seg_metrics['brats_regions']['TC']:.4f}")
    print(f"{'3. Segmentation':<25} {'ET Dice':<15} {seg_metrics['brats_regions']['ET']:.4f}")
    print(f"{'4. GNN':<25} {'AUC-ROC':<15} {gnn_metrics['auc']:.4f}")
    print(f"{'4. GNN':<25} {'AP':<15} {gnn_metrics['ap']:.4f}")
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    run_pipeline()

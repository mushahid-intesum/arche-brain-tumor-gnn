import torch
import torch.nn as nn
import time

from config import SHARED, PREDICTION, CLASSIFICATION, SEGMENTATION, GNN, PIPELINE, ensure_dirs

import prediction
import classification
import segmentation

# TODO: Need to implement gnn pipeline


# ══════════════════════════════════════════════════════════════════════
# Phase 1: Binary Prediction (Brain MRI ND-5)
# ══════════════════════════════════════════════════════════════════════

def run_prediction():
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
    # TODO: Need to implement
    pass


# ══════════════════════════════════════════════════════════════════════
# Full Pipeline
# ══════════════════════════════════════════════════════════════════════

def run_pipeline():
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

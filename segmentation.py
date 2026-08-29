"""
Phase 3: Multi-class Brain Tumor Segmentation
DeepLabV3+ (EfficientNet-B4) with 4-channel MRI input → 4-class output.
Dataset: BraTS 2023 GLI (NIfTI volumes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import segmentation_models_pytorch as smp

from config import SHARED, SEGMENTATION


# ── Data Loading ──────────────────────────────────────────────────────

def discover_cases(data_dir):
    """Find all valid BraTS cases with 4 modalities + segmentation mask."""
    cases = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
    valid_cases = []
    for case_dir in cases:
        case_id = case_dir.name
        files = {
            "t1n": case_dir / f"{case_id}-t1n.nii",
            "t1c": case_dir / f"{case_id}-t1c.nii",
            "t2w": case_dir / f"{case_id}-t2w.nii",
            "t2f": case_dir / f"{case_id}-t2f.nii",
            "seg": case_dir / f"{case_id}-seg.nii",
        }
        if all(f.exists() for f in files.values()):
            valid_cases.append({"case_id": case_id, "files": files})
    return valid_cases


def load_volume(nii_path):
    """Load a NIfTI volume as float32 numpy array."""
    vol = nib.load(str(nii_path))
    return vol.get_fdata().astype(np.float32)


def zscore_normalize(volume):
    """Z-score normalize non-zero voxels (standard BraTS preprocessing)."""
    mask = volume > 0
    if mask.sum() == 0:
        return volume
    mean = volume[mask].mean()
    std = volume[mask].std()
    if std < 1e-8:
        return volume
    volume[mask] = (volume[mask] - mean) / std
    return volume


def extract_and_save_slices(case, cache_dir, config=None):
    """Extract 2D tumor slices from a case and save to disk."""
    config = config or SEGMENTATION
    seg_vol = load_volume(case["files"]["seg"])
    modality_vols = {}
    for mod in config["modalities"]:
        vol = load_volume(case["files"][mod])
        vol = zscore_normalize(vol)
        modality_vols[mod] = vol

    saved = []
    num_axial = seg_vol.shape[2]

    for s in range(num_axial):
        seg_slice = seg_vol[:, :, s]
        tumor_pixels = (seg_slice > 0).sum()
        if tumor_pixels < config["min_tumor_pixels"]:
            continue

        img_channels = []
        for mod in config["modalities"]:
            ch = modality_vols[mod][:, :, s]
            ch = cv2.resize(ch, (SHARED["img_size"], SHARED["img_size"]),
                            interpolation=cv2.INTER_LINEAR)
            img_channels.append(ch)

        seg_resized = cv2.resize(seg_slice, (SHARED["img_size"], SHARED["img_size"]),
                                 interpolation=cv2.INTER_NEAREST)

        img_4ch = np.stack(img_channels, axis=0).astype(np.float32)
        mask = seg_resized.astype(np.int64)

        fname = f"{case['case_id']}_s{s:03d}"
        img_path = cache_dir / f"{fname}_img.npy"
        mask_path = cache_dir / f"{fname}_mask.npy"
        np.save(str(img_path), img_4ch)
        np.save(str(mask_path), mask)

        saved.append({
            "img_path": str(img_path),
            "mask_path": str(mask_path),
            "case_id": case["case_id"],
            "slice_idx": s,
        })

    del modality_vols, seg_vol
    return saved


class BraTSSliceDataset(Dataset):
    """Lazy-loading BraTS slice dataset (reads .npy files on demand)."""

    def __init__(self, slice_meta_list, augment=False):
        self.meta = slice_meta_list
        self.augment = augment

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        m = self.meta[idx]
        image = np.load(m["img_path"]).copy()
        mask = np.load(m["mask_path"]).copy()

        if self.augment:
            if random.random() > 0.5:
                image = image[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if random.random() > 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[::-1, :].copy()
            k = random.randint(0, 3)
            if k > 0:
                image = np.rot90(image, k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()

        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()


def prepare_data(config=None):
    """Full data pipeline: discover → extract → split → dataloaders."""
    config = config or SEGMENTATION
    cache_dir = config["output_dir"] / "slice_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(config["data_root"])
    print(f"Discovered {len(cases)} valid BraTS cases")

    print("Extracting 2D slices to disk...")
    all_meta = []
    for i, case in enumerate(cases):
        case_meta = extract_and_save_slices(case, cache_dir, config)
        all_meta.extend(case_meta)
        if (i + 1) % 50 == 0 or i == 0 or i == len(cases) - 1:
            print(f"  Processed {i+1}/{len(cases)} cases | Total slices: {len(all_meta)}")

    print(f"Total slices extracted: {len(all_meta)}")

    # Patient-level split
    case_ids = [s["case_id"] for s in all_meta]
    unique_ids = sorted(set(case_ids))
    train_cases, temp_cases = train_test_split(
        unique_ids,
        test_size=config["val_ratio"] + config["test_ratio"],
        random_state=SHARED["seed"],
    )
    val_cases, test_cases = train_test_split(
        temp_cases,
        test_size=config["test_ratio"] / (config["val_ratio"] + config["test_ratio"]),
        random_state=SHARED["seed"],
    )

    train_set, val_set, test_set = set(train_cases), set(val_cases), set(test_cases)
    train_meta = [s for s in all_meta if s["case_id"] in train_set]
    val_meta = [s for s in all_meta if s["case_id"] in val_set]
    test_meta = [s for s in all_meta if s["case_id"] in test_set]

    print(f"Split: train={len(train_meta)}, val={len(val_meta)}, test={len(test_meta)}")

    train_ds = BraTSSliceDataset(train_meta, augment=True)
    val_ds = BraTSSliceDataset(val_meta, augment=False)
    test_ds = BraTSSliceDataset(test_meta, augment=False)

    kwargs = dict(num_workers=config["num_workers"], pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, **kwargs)

    return train_meta, val_meta, test_meta, train_loader, val_loader, test_loader


# ── Model ─────────────────────────────────────────────────────────────

def build_segmentation_model(config=None):
    """DeepLabV3+ with EfficientNet-B4, 4-channel input, 4-class output."""
    config = config or SEGMENTATION
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=4,
        classes=config["num_classes"],
    )

    # Expand first conv from 3→4 channels
    first_conv = model.encoder._conv_stem
    if first_conv.in_channels != 4:
        new_conv = nn.Conv2d(
            4, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = first_conv.weight
            new_conv.weight[:, 3] = first_conv.weight[:, 0]
        model.encoder._conv_stem = new_conv

    return model.to(SHARED["device"])


# ── Loss & Metrics ────────────────────────────────────────────────────

class DiceCELoss(nn.Module):
    """Combined Dice + CrossEntropy loss for multi-class segmentation."""

    def __init__(self, num_classes=4, dice_weight=1.0, ce_weight=1.0, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def dice_loss(self, pred_soft, target_onehot):
        dims = (0, 2, 3)
        intersection = (pred_soft * target_onehot).sum(dim=dims)
        cardinality = pred_soft.sum(dim=dims) + target_onehot.sum(dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class[1:].mean()  # exclude background

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pred_soft = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        d_loss = self.dice_loss(pred_soft, target_onehot)
        return self.ce_weight * ce_loss + self.dice_weight * d_loss


def compute_dice_per_class(pred_mask, gt_mask, num_classes=4, smooth=1e-5):
    """Per-class Dice scores (excludes background)."""
    dice = {}
    for c in range(1, num_classes):
        pred_c = (pred_mask == c).float()
        gt_c = (gt_mask == c).float()
        inter = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        dice[c] = ((2.0 * inter + smooth) / (union + smooth)).item()
    return dice


def compute_brats_regions(pred_mask, gt_mask, smooth=1e-5):
    """Official BraTS region Dice: ET, TC, WT."""
    def dice(p, g):
        inter = (p * g).sum()
        return ((2.0 * inter + smooth) / (p.sum() + g.sum() + smooth)).item()

    return {
        "ET": dice((pred_mask == 3).float(), (gt_mask == 3).float()),
        "TC": dice(((pred_mask == 1) | (pred_mask == 3)).float(),
                    ((gt_mask == 1) | (gt_mask == 3)).float()),
        "WT": dice((pred_mask > 0).float(), (gt_mask > 0).float()),
    }


# ── Training ──────────────────────────────────────────────────────────

def train_segmentation(model, train_loader, val_loader, config=None):
    """Full training loop with gradient accumulation and OneCycleLR."""
    config = config or SEGMENTATION
    criterion = DiceCELoss(num_classes=config["num_classes"])
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["lr"],
        steps_per_epoch=len(train_loader) // config["accum_steps"] + 1,
        epochs=config["epochs"],
    )

    best_val_dice = 0.0
    checkpoint_path = config["checkpoint"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {config['epochs']} epochs (effective batch={config['batch_size'] * config['accum_steps']})...")

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(SHARED["device"])
            masks = masks.to(SHARED["device"])
            logits = model(images)
            loss = criterion(logits, masks) / config["accum_steps"]
            loss.backward()

            if (batch_idx + 1) % config["accum_steps"] == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * config["accum_steps"]

        avg_loss = epoch_loss / len(train_loader)

        if (epoch + 1) % config["eval_every"] == 0 or epoch == 0:
            metrics = evaluate_segmentation(model, val_loader, criterion)
            overall_dice = np.mean(list(metrics["per_class_dice"].values()))

            print(
                f"Epoch {epoch+1:3d}/{config['epochs']} | "
                f"Loss: {avg_loss:.4f} | Val: {metrics['val_loss']:.4f} | "
                f"NCR: {metrics['per_class_dice'][1]:.3f} "
                f"ED: {metrics['per_class_dice'][2]:.3f} "
                f"ET: {metrics['per_class_dice'][3]:.3f} | "
                f"WT: {metrics['brats_regions']['WT']:.3f} "
                f"TC: {metrics['brats_regions']['TC']:.3f}"
            )

            if overall_dice > best_val_dice:
                best_val_dice = overall_dice
                torch.save(model.state_dict(), str(checkpoint_path))
                print(f"  -> Best model saved (mean Dice: {overall_dice:.4f})")

    print(f"Training complete. Best val mean Dice: {best_val_dice:.4f}")
    return best_val_dice


@torch.no_grad()
def evaluate_segmentation(model, loader, criterion=None):
    """Evaluate segmentation with per-class and BraTS region Dice."""
    model.eval()
    criterion = criterion or DiceCELoss(num_classes=SEGMENTATION["num_classes"])
    val_loss = 0.0
    dice_scores = {1: [], 2: [], 3: []}
    brats_scores = {"ET": [], "TC": [], "WT": []}
    all_preds, all_gts = [], []

    for images, masks in loader:
        images = images.to(SHARED["device"])
        masks = masks.to(SHARED["device"])
        logits = model(images)
        val_loss += criterion(logits, masks).item()
        preds = logits.argmax(dim=1)

        for b in range(preds.size(0)):
            dc = compute_dice_per_class(preds[b], masks[b])
            for c in dc:
                dice_scores[c].append(dc[c])
            br = compute_brats_regions(preds[b], masks[b])
            for k in br:
                brats_scores[k].append(br[k])
            all_preds.append(preds[b].cpu().numpy())
            all_gts.append(masks[b].cpu().numpy())

    return {
        "val_loss": val_loss / len(loader),
        "per_class_dice": {c: np.mean(dice_scores[c]) for c in dice_scores},
        "per_class_std": {c: np.std(dice_scores[c]) for c in dice_scores},
        "brats_regions": {k: np.mean(brats_scores[k]) for k in brats_scores},
        "preds": all_preds,
        "gts": all_gts,
    }


# ── Export ────────────────────────────────────────────────────────────

def export_for_gnn(model, train_meta, val_meta, test_meta, config=None):
    """Export predicted masks, GT masks, and raw slices for the GNN pipeline."""
    config = config or SEGMENTATION
    output = config["output_dir"]
    masks_dir = output / "masks"
    raw_dir = output / "raw_slices"
    gt_dir = output / "gt_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "case_ids": [], "slice_indices": [],
        "mask_files": [], "raw_files": [], "gt_files": [],
        "splits": [], "class_names": config["class_names"],
    }

    model.eval()
    count = 0
    with torch.no_grad():
        for split_name, split_meta in [("train", train_meta), ("val", val_meta), ("test", test_meta)]:
            for m in split_meta:
                img = np.load(m["img_path"])
                gt_mask = np.load(m["mask_path"])
                image = torch.from_numpy(img).unsqueeze(0).float().to(SHARED["device"])
                logits = model(image)
                pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                fname = f"{m['case_id']}_s{m['slice_idx']:03d}"
                np.save(str(masks_dir / f"{fname}_pred.npy"), pred)
                np.save(str(raw_dir / f"{fname}_raw.npy"), img)
                np.save(str(gt_dir / f"{fname}_gt.npy"), gt_mask.astype(np.uint8))

                metadata["case_ids"].append(m["case_id"])
                metadata["slice_indices"].append(m["slice_idx"])
                metadata["mask_files"].append(f"{fname}_pred.npy")
                metadata["raw_files"].append(f"{fname}_raw.npy")
                metadata["gt_files"].append(f"{fname}_gt.npy")
                metadata["splits"].append(split_name)
                count += 1

            print(f"  Exported {split_name}: {len(split_meta)} slices")

    torch.save(metadata, str(output / "metadata.pt"))
    print(f"Total exported: {count} slices → {output}")
    return metadata


# ── Visualization ─────────────────────────────────────────────────────

def plot_test_results(test_meta, preds, gts, n=6):
    """Plot GT vs predicted overlays for n random test slices."""
    indices = random.sample(range(len(preds)), min(n, len(preds)))
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))

    for col, idx in enumerate(indices):
        m = test_meta[idx]
        t1c = np.load(m["img_path"])[1]
        gt, pred = gts[idx], preds[idx]

        axes[0, col].imshow(t1c, cmap="gray")
        axes[0, col].set_title(f"T1c | {m['case_id'][-7:]}", fontsize=8)
        axes[0, col].axis("off")

        for row, data, label in [(1, gt, "GT"), (2, pred, "Pred")]:
            color = np.zeros((*data.shape, 3), dtype=np.float32)
            color[data == 1] = [1, 0, 0]
            color[data == 2] = [0, 1, 0]
            color[data == 3] = [1, 1, 0]
            axes[row, col].imshow(t1c, cmap="gray", alpha=0.5)
            axes[row, col].imshow(color, alpha=0.5)
            if row == 2:
                dc = compute_brats_regions(
                    torch.from_numpy(pred).long(),
                    torch.from_numpy(gt).long(),
                )
                axes[row, col].set_title(f"{label} | WT:{dc['WT']:.2f}", fontsize=8)
            else:
                axes[row, col].set_title(label, fontsize=8)
            axes[row, col].axis("off")

    plt.suptitle("Test: GT vs Predicted (R=NCR G=ED Y=ET)", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_confusion(preds, gts):
    """Plot 4-class segmentation confusion matrix."""
    flat_p = np.concatenate([p.flatten() for p in preds])
    flat_g = np.concatenate([g.flatten() for g in gts])
    cm = confusion_matrix(flat_g, flat_p, labels=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["BG", "NCR", "ED", "ET"]
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(4):
        for j in range(4):
            val = cm[i, j]
            text = f"{val:,}" if val < 1e6 else f"{val/1e6:.1f}M"
            ax.text(j, i, text, ha="center", va="center", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Segmentation Confusion Matrix")
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import ensure_dirs
    ensure_dirs()

    print(f"Device: {SHARED['device']}")

    # Data
    train_meta, val_meta, test_meta, train_loader, val_loader, test_loader = prepare_data()

    # Model
    model = build_segmentation_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"DeepLabV3+ | Parameters: {total_params:,}")

    # Sanity check
    model.eval()
    sample = torch.randn(1, 4, SHARED["img_size"], SHARED["img_size"]).to(SHARED["device"])
    with torch.no_grad():
        out = model(sample)
    model.train()
    print(f"Output shape: {out.shape}")
    del sample, out

    # Train
    train_segmentation(model, train_loader, val_loader)

    # Test
    model.load_state_dict(torch.load(str(SEGMENTATION["checkpoint"]), weights_only=True))
    metrics = evaluate_segmentation(model, test_loader)

    print(f"\nTest Results ({len(metrics['preds'])} slices):")
    for c in [1, 2, 3]:
        name = SEGMENTATION["class_names"][c]
        print(f"  {name}: {metrics['per_class_dice'][c]:.4f} ± {metrics['per_class_std'][c]:.4f}")
    print(f"  BraTS: WT={metrics['brats_regions']['WT']:.4f} "
          f"TC={metrics['brats_regions']['TC']:.4f} "
          f"ET={metrics['brats_regions']['ET']:.4f}")

    plot_confusion(metrics["preds"], metrics["gts"])
    plot_test_results(test_meta, metrics["preds"], metrics["gts"])

    # Export
    export_for_gnn(model, train_meta, val_meta, test_meta)

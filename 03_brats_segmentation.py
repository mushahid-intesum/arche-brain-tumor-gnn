import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


BRATS_CONFIG = {
    "data_dir": Path("BraTS"),
    "output_dir": Path("brats_outputs"),
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 42,
    "img_size": 224,
    "min_tumor_pixels": 50,
    "modalities": ["t1n", "t1c", "t2w", "t2f"],
    "num_classes": 4,
    "class_names": {0: "background", 1: "necrotic_core", 2: "edema", 3: "enhancing_tumor"},
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "batch_size": 8,
    "grad_accum_steps": 4,
    "epochs": 60,
    "lr": 1e-4,
    "weight_decay": 1e-4,
}

torch.manual_seed(BRATS_CONFIG["seed"])
np.random.seed(BRATS_CONFIG["seed"])
random.seed(BRATS_CONFIG["seed"])

print(f"Device: {BRATS_CONFIG['device']}")
BRATS_CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)


def discover_cases(data_dir):
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir()])
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

cases = discover_cases(BRATS_CONFIG["data_dir"])
print(f"Discovered {len(cases)} valid BraTS cases")


def load_volume(nii_path):
    vol = nib.load(str(nii_path))
    return vol.get_fdata().astype(np.float32)


def zscore_normalize(volume):
    mask = volume > 0
    if mask.sum() == 0:
        return volume
    mean = volume[mask].mean()
    std = volume[mask].std()
    if std < 1e-8:
        return volume
    volume[mask] = (volume[mask] - mean) / std
    return volume


def extract_tumor_slices(case, config):
    seg_vol = load_volume(case["files"]["seg"])
    modality_vols = {}
    for mod in config["modalities"]:
        vol = load_volume(case["files"][mod])
        vol = zscore_normalize(vol)
        modality_vols[mod] = vol

    slices = []
    num_axial = seg_vol.shape[2]

    for s in range(num_axial):
        seg_slice = seg_vol[:, :, s]
        tumor_pixels = (seg_slice > 0).sum()

        if tumor_pixels < config["min_tumor_pixels"]:
            continue

        img_channels = []
        for mod in config["modalities"]:
            ch = modality_vols[mod][:, :, s]
            ch = cv2.resize(ch, (config["img_size"], config["img_size"]),
                            interpolation=cv2.INTER_LINEAR)
            img_channels.append(ch)

        seg_resized = cv2.resize(seg_slice, (config["img_size"], config["img_size"]),
                                 interpolation=cv2.INTER_NEAREST)

        img_4ch = np.stack(img_channels, axis=0)

        slices.append({
            "image": img_4ch,
            "mask": seg_resized.astype(np.int64),
            "case_id": case["case_id"],
            "slice_idx": s,
        })

    return slices


print("Extracting 2D slices from all volumes...")
all_slices = []
case_ids_per_slice = []

for i, case in enumerate(cases):
    case_slices = extract_tumor_slices(case, BRATS_CONFIG)
    all_slices.extend(case_slices)
    case_ids_per_slice.extend([case["case_id"]] * len(case_slices))

    if (i + 1) % 50 == 0 or i == 0 or i == len(cases) - 1:
        print(f"  Processed {i+1}/{len(cases)} cases | Total slices so far: {len(all_slices)}")

print(f"\nTotal slices extracted: {len(all_slices)}")

label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
for s in all_slices:
    for lbl in range(4):
        label_counts[lbl] += (s["mask"] == lbl).sum()
total_px = sum(label_counts.values())
print("Label distribution (pixels):")
for lbl, count in label_counts.items():
    name = BRATS_CONFIG["class_names"][lbl]
    print(f"  {lbl} ({name}): {count:,} ({100*count/total_px:.1f}%)")


unique_case_ids = sorted(set(case_ids_per_slice))
train_cases, temp_cases = train_test_split(
    unique_case_ids,
    test_size=BRATS_CONFIG["val_ratio"] + BRATS_CONFIG["test_ratio"],
    random_state=BRATS_CONFIG["seed"],
)
val_cases, test_cases = train_test_split(
    temp_cases,
    test_size=BRATS_CONFIG["test_ratio"] / (BRATS_CONFIG["val_ratio"] + BRATS_CONFIG["test_ratio"]),
    random_state=BRATS_CONFIG["seed"],
)

train_cases_set = set(train_cases)
val_cases_set = set(val_cases)
test_cases_set = set(test_cases)

train_slices = [s for s in all_slices if s["case_id"] in train_cases_set]
val_slices = [s for s in all_slices if s["case_id"] in val_cases_set]
test_slices = [s for s in all_slices if s["case_id"] in test_cases_set]

print(f"\nPatient-level split:")
print(f"  Train: {len(train_cases)} patients, {len(train_slices)} slices")
print(f"  Val:   {len(val_cases)} patients, {len(val_slices)} slices")
print(f"  Test:  {len(test_cases)} patients, {len(test_slices)} slices")


class BraTSSliceDataset(Dataset):
    def __init__(self, slices_list, augment=False):
        self.slices = slices_list
        self.augment = augment

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        s = self.slices[idx]
        image = s["image"].copy()
        mask = s["mask"].copy()

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


train_dataset = BraTSSliceDataset(train_slices, augment=True)
val_dataset = BraTSSliceDataset(val_slices, augment=False)
test_dataset = BraTSSliceDataset(test_slices, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BRATS_CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BRATS_CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BRATS_CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

print(f"\nDataLoaders created:")
print(f"  Train: {len(train_loader)} batches (batch_size={BRATS_CONFIG['batch_size']})")
print(f"  Val:   {len(val_loader)} batches")
print(f"  Test:  {len(test_loader)} batches")


fig, axes = plt.subplots(3, 6, figsize=(24, 12))
sample_indices = random.sample(range(len(train_slices)), min(6, len(train_slices)))

for col, idx in enumerate(sample_indices):
    s = train_slices[idx]
    t1c = s["image"][1]
    mask = s["mask"]

    axes[0, col].imshow(t1c, cmap="gray")
    axes[0, col].set_title(f"T1c | {s['case_id'][-7:]}\nslice {s['slice_idx']}", fontsize=8)
    axes[0, col].axis("off")

    color_mask = np.zeros((*mask.shape, 3), dtype=np.float32)
    color_mask[mask == 1] = [1.0, 0.0, 0.0]
    color_mask[mask == 2] = [0.0, 1.0, 0.0]
    color_mask[mask == 3] = [1.0, 1.0, 0.0]
    axes[1, col].imshow(color_mask)
    axes[1, col].set_title("Mask (R=NCR G=ED Y=ET)", fontsize=8)
    axes[1, col].axis("off")

    axes[2, col].imshow(t1c, cmap="gray", alpha=0.6)
    axes[2, col].imshow(color_mask, alpha=0.4)
    axes[2, col].set_title("Overlay", fontsize=8)
    axes[2, col].axis("off")

plt.suptitle("BraTS 2023: T1c + Multi-class Segmentation Masks", fontsize=14)
plt.tight_layout()
plt.show()

print("\n=== Phase 1 Complete: Data Loading & Preprocessing ===")



import segmentation_models_pytorch as smp
from torch import optim

model = smp.DeepLabV3Plus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=4,
    classes=BRATS_CONFIG["num_classes"],
)

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

model = model.to(BRATS_CONFIG["device"])
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"DeepLabV3+ (EfficientNet-B4) | Total: {total_params:,} | Trainable: {trainable_params:,}")

sample_input = torch.randn(1, 4, BRATS_CONFIG["img_size"], BRATS_CONFIG["img_size"]).to(BRATS_CONFIG["device"])
with torch.no_grad():
    sample_out = model(sample_input)
print(f"Model output shape: {sample_out.shape} (expected: [1, {BRATS_CONFIG['num_classes']}, {BRATS_CONFIG['img_size']}, {BRATS_CONFIG['img_size']}])")
del sample_input, sample_out



class DiceCELoss(nn.Module):
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
        return 1.0 - dice_per_class[1:].mean()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pred_soft = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        d_loss = self.dice_loss(pred_soft, target_onehot)
        return self.ce_weight * ce_loss + self.dice_weight * d_loss


def compute_dice_per_class(pred_mask, gt_mask, num_classes=4, smooth=1e-5):
    dice_scores = {}
    for c in range(1, num_classes):
        pred_c = (pred_mask == c).float()
        gt_c = (gt_mask == c).float()
        intersection = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        dice_scores[c] = ((2.0 * intersection + smooth) / (union + smooth)).item()
    return dice_scores


def compute_brats_regions(pred_mask, gt_mask, smooth=1e-5):
    def dice(p, g):
        inter = (p * g).sum()
        return ((2.0 * inter + smooth) / (p.sum() + g.sum() + smooth)).item()

    et_pred = (pred_mask == 3).float()
    et_gt = (gt_mask == 3).float()

    tc_pred = ((pred_mask == 1) | (pred_mask == 3)).float()
    tc_gt = ((gt_mask == 1) | (gt_mask == 3)).float()

    wt_pred = (pred_mask > 0).float()
    wt_gt = (gt_mask > 0).float()

    return {
        "ET": dice(et_pred, et_gt),
        "TC": dice(tc_pred, tc_gt),
        "WT": dice(wt_pred, wt_gt),
    }


criterion = DiceCELoss(num_classes=BRATS_CONFIG["num_classes"])
optimizer = optim.AdamW(
    model.parameters(),
    lr=BRATS_CONFIG["lr"],
    weight_decay=BRATS_CONFIG["weight_decay"],
)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=BRATS_CONFIG["lr"],
    steps_per_epoch=len(train_loader) // BRATS_CONFIG["grad_accum_steps"] + 1,
    epochs=BRATS_CONFIG["epochs"],
)

best_val_dice = 0.0
best_model_path = BRATS_CONFIG["output_dir"] / "best_model.pt"

print(f"\nStarting training for {BRATS_CONFIG['epochs']} epochs...")
print(f"Effective batch size: {BRATS_CONFIG['batch_size'] * BRATS_CONFIG['grad_accum_steps']}")

for epoch in range(BRATS_CONFIG["epochs"]):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(BRATS_CONFIG["device"])
        masks = masks.to(BRATS_CONFIG["device"])

        logits = model(images)
        loss = criterion(logits, masks) / BRATS_CONFIG["grad_accum_steps"]
        loss.backward()

        if (batch_idx + 1) % BRATS_CONFIG["grad_accum_steps"] == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * BRATS_CONFIG["grad_accum_steps"]

    avg_train_loss = epoch_loss / len(train_loader)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        model.eval()
        val_loss = 0.0
        val_dice = {1: [], 2: [], 3: []}
        val_brats = {"ET": [], "TC": [], "WT": []}

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(BRATS_CONFIG["device"])
                masks = masks.to(BRATS_CONFIG["device"])

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                for b in range(preds.size(0)):
                    dc = compute_dice_per_class(preds[b], masks[b])
                    for c in dc:
                        val_dice[c].append(dc[c])
                    br = compute_brats_regions(preds[b], masks[b])
                    for k in br:
                        val_brats[k].append(br[k])

        avg_val_loss = val_loss / len(val_loader)
        mean_dice = {c: np.mean(val_dice[c]) for c in val_dice}
        mean_brats = {k: np.mean(val_brats[k]) for k in val_brats}
        overall_dice = np.mean(list(mean_dice.values()))

        print(
            f"Epoch {epoch+1:3d}/{BRATS_CONFIG['epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"NCR: {mean_dice[1]:.3f} ED: {mean_dice[2]:.3f} ET: {mean_dice[3]:.3f} | "
            f"WT: {mean_brats['WT']:.3f} TC: {mean_brats['TC']:.3f}"
        )

        if overall_dice > best_val_dice:
            best_val_dice = overall_dice
            torch.save(model.state_dict(), str(best_model_path))
            print(f"  -> New best model saved (mean Dice: {overall_dice:.4f})")

print(f"\nTraining complete. Best val mean Dice: {best_val_dice:.4f}")
print(f"Best model saved to: {best_model_path}")



print("\n=== Phase 4: Test Evaluation ===")

model.load_state_dict(torch.load(str(best_model_path), weights_only=True))
model.eval()

test_dice = {1: [], 2: [], 3: []}
test_brats = {"ET": [], "TC": [], "WT": []}
all_preds = []
all_gts = []

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(BRATS_CONFIG["device"])
        masks = masks.to(BRATS_CONFIG["device"])

        logits = model(images)
        preds = logits.argmax(dim=1)

        for b in range(preds.size(0)):
            dc = compute_dice_per_class(preds[b], masks[b])
            for c in dc:
                test_dice[c].append(dc[c])
            br = compute_brats_regions(preds[b], masks[b])
            for k in br:
                test_brats[k].append(br[k])
            all_preds.append(preds[b].cpu().numpy())
            all_gts.append(masks[b].cpu().numpy())

mean_test_dice = {c: np.mean(test_dice[c]) for c in test_dice}
std_test_dice = {c: np.std(test_dice[c]) for c in test_dice}
mean_test_brats = {k: np.mean(test_brats[k]) for k in test_brats}

print(f"\nTest Results ({len(all_preds)} slices):")
print(f"  Per-class Dice:")
for c in [1, 2, 3]:
    name = BRATS_CONFIG["class_names"][c]
    print(f"    {name}: {mean_test_dice[c]:.4f} ± {std_test_dice[c]:.4f}")
print(f"  BraTS Region Dice:")
for k in ["WT", "TC", "ET"]:
    print(f"    {k}: {mean_test_brats[k]:.4f}")

from sklearn.metrics import confusion_matrix

all_preds_flat = np.concatenate([p.flatten() for p in all_preds])
all_gts_flat = np.concatenate([g.flatten() for g in all_gts])
cm = confusion_matrix(all_gts_flat, all_preds_flat, labels=[0, 1, 2, 3])

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap="Blues")
class_labels = ["BG", "NCR", "ED", "ET"]
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(class_labels)
ax.set_yticklabels(class_labels)
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

fig, axes = plt.subplots(3, 6, figsize=(24, 12))
viz_indices = random.sample(range(len(all_preds)), min(6, len(all_preds)))

for col, idx in enumerate(viz_indices):
    s = test_slices[idx]
    t1c = s["image"][1]
    gt = all_gts[idx]
    pred = all_preds[idx]

    axes[0, col].imshow(t1c, cmap="gray")
    axes[0, col].set_title(f"T1c | {s['case_id'][-7:]}", fontsize=8)
    axes[0, col].axis("off")

    gt_color = np.zeros((*gt.shape, 3), dtype=np.float32)
    gt_color[gt == 1] = [1.0, 0.0, 0.0]
    gt_color[gt == 2] = [0.0, 1.0, 0.0]
    gt_color[gt == 3] = [1.0, 1.0, 0.0]
    axes[1, col].imshow(t1c, cmap="gray", alpha=0.5)
    axes[1, col].imshow(gt_color, alpha=0.5)
    axes[1, col].set_title("Ground Truth", fontsize=8)
    axes[1, col].axis("off")

    pred_color = np.zeros((*pred.shape, 3), dtype=np.float32)
    pred_color[pred == 1] = [1.0, 0.0, 0.0]
    pred_color[pred == 2] = [0.0, 1.0, 0.0]
    pred_color[pred == 3] = [1.0, 1.0, 0.0]
    axes[2, col].imshow(t1c, cmap="gray", alpha=0.5)
    axes[2, col].imshow(pred_color, alpha=0.5)
    dc = compute_brats_regions(
        torch.from_numpy(pred).long(),
        torch.from_numpy(gt).long(),
    )
    axes[2, col].set_title(f"Pred | WT:{dc['WT']:.2f} TC:{dc['TC']:.2f}", fontsize=8)
    axes[2, col].axis("off")

plt.suptitle("Test Set: Ground Truth vs Predicted (R=NCR G=ED Y=ET)", fontsize=14)
plt.tight_layout()
plt.show()



print("\n=== Phase 5: Export for GNN Pipeline ===")

masks_dir = BRATS_CONFIG["output_dir"] / "masks"
raw_dir = BRATS_CONFIG["output_dir"] / "raw_slices"
gt_dir = BRATS_CONFIG["output_dir"] / "gt_masks"
masks_dir.mkdir(parents=True, exist_ok=True)
raw_dir.mkdir(parents=True, exist_ok=True)
gt_dir.mkdir(parents=True, exist_ok=True)

export_metadata = {
    "case_ids": [],
    "slice_indices": [],
    "mask_files": [],
    "raw_files": [],
    "gt_files": [],
    "splits": [],
    "class_names": BRATS_CONFIG["class_names"],
}

model.eval()
all_export_slices = [
    ("train", train_slices),
    ("val", val_slices),
    ("test", test_slices),
]

export_count = 0
with torch.no_grad():
    for split_name, split_slices in all_export_slices:
        for i, s in enumerate(split_slices):
            image = torch.from_numpy(s["image"]).unsqueeze(0).float().to(BRATS_CONFIG["device"])
            logits = model(image)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            case_id = s["case_id"]
            slice_idx = s["slice_idx"]
            fname = f"{case_id}_s{slice_idx:03d}"

            mask_path = masks_dir / f"{fname}_pred.npy"
            raw_path = raw_dir / f"{fname}_raw.npy"
            gt_path = gt_dir / f"{fname}_gt.npy"

            np.save(str(mask_path), pred)
            np.save(str(raw_path), s["image"])
            np.save(str(gt_path), s["mask"].astype(np.uint8))

            export_metadata["case_ids"].append(case_id)
            export_metadata["slice_indices"].append(slice_idx)
            export_metadata["mask_files"].append(f"{fname}_pred.npy")
            export_metadata["raw_files"].append(f"{fname}_raw.npy")
            export_metadata["gt_files"].append(f"{fname}_gt.npy")
            export_metadata["splits"].append(split_name)
            export_count += 1

        print(f"  Exported {split_name}: {len(split_slices)} slices")

torch.save(export_metadata, str(BRATS_CONFIG["output_dir"] / "metadata.pt"))

print(f"\nTotal exported: {export_count} slices")
print(f"Output directory: {BRATS_CONFIG['output_dir']}")
print(f"  masks/     - predicted multi-class masks (.npy)")
print(f"  gt_masks/  - ground-truth masks (.npy)")
print(f"  raw_slices/- 4-channel raw MRI slices (.npy)")
print(f"  metadata.pt - case IDs, slice indices, splits")

print("\n=== BraTS Segmentation Pipeline Complete ===")


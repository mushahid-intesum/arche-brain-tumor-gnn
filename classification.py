"""
Phase 2: Multi-class Tumor Classification
ConvNeXt-Base 3-class classifier (glioma, meningioma, pituitary) + GradCAM pseudo-masks.
Dataset: Brain MRI ND-5 (tumor-only subset)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision import models
from torchvision.models import ConvNeXt_Base_Weights
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import os

from config import SHARED, CLASSIFICATION
from prediction import (
    BrainMRIDataset, TransformSubset, load_image,
    get_transforms, create_dataloaders, denormalize,
)


# ── Loss ──────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced classification."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ── Model ─────────────────────────────────────────────────────────────

class DualPoolConvNeXt(nn.Module):
    """ConvNeXt with avg+max dual pooling for 3-class tumor classification."""

    def __init__(self, base, feat_dim):
        super().__init__()
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Linear(feat_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 3),
        )

    def forward(self, x):
        x = self.features(x)
        avg_out = self.avgpool(x).flatten(1)
        max_out = self.maxpool(x).flatten(1)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.head(x)


def build_multiclass_classifier(device=None):
    """Build DualPoolConvNeXt for 3-class tumor type classification."""
    device = device or SHARED["device"]
    backbone = models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    in_features = backbone.classifier[2].in_features
    backbone.classifier = nn.Identity()
    model = DualPoolConvNeXt(backbone, in_features)
    return model.to(device)


# ── GradCAM ───────────────────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Map for pseudo-mask generation."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate CAM heatmaps for a batch of inputs."""
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        for i in range(len(target_class)):
            one_hot[i, target_class[i]] = 1.0
        output.backward(gradient=one_hot)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=(SHARED["img_size"], SHARED["img_size"]),
            mode="bilinear", align_corners=False,
        )
        cam = cam.squeeze(1)
        for i in range(cam.size(0)):
            c = cam[i]
            if c.max() > 0:
                cam[i] = (c - c.min()) / (c.max() - c.min())
        return cam.cpu().numpy()


# ── Training ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler, accum_steps=1):
    """Single training epoch for multiclass classification."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()
    for step, (imgs, _, multi_labels) in enumerate(loader):
        imgs = imgs.to(SHARED["device"])
        labels = multi_labels.to(SHARED["device"])
        out = model(imgs)
        loss = criterion(out, labels) / accum_steps
        loss.backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accum_steps * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluate multiclass classification on a loader."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for imgs, _, multi_labels in loader:
        imgs = imgs.to(SHARED["device"])
        labels = multi_labels.to(SHARED["device"])
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(multi_labels.numpy())
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
    }


def train_multiclass(model, train_loader, val_loader, class_weights, config=None):
    """Full training loop with freeze/unfreeze and focal loss."""
    config = config or CLASSIFICATION
    criterion = FocalLoss(
        alpha=class_weights.to(SHARED["device"]),
        gamma=config["focal_gamma"],
    )

    # Phase A: frozen backbone
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["lr"],
        steps_per_epoch=(len(train_loader) + config["accum_steps"] - 1) // config["accum_steps"],
        epochs=config["freeze_epochs"],
    )

    best_val_acc = 0.0
    for epoch in range(config["epochs"]):
        if epoch == config["freeze_epochs"]:
            # Phase B: unfreeze backbone
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config["lr"] * 0.1, weight_decay=config["weight_decay"],
            )
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=config["lr"] * 0.1,
                steps_per_epoch=(len(train_loader) + config["accum_steps"] - 1) // config["accum_steps"],
                epochs=config["epochs"] - config["freeze_epochs"],
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            accum_steps=config["accum_steps"],
        )
        val_result = evaluate(model, val_loader, criterion)

        print(
            f"[Classification] Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_result['loss']:.4f} Acc: {val_result['accuracy']:.4f}"
        )

        if val_result["accuracy"] > best_val_acc:
            best_val_acc = val_result["accuracy"]
            config["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(config["checkpoint"]))

    print(f"Best Classification Val Accuracy: {best_val_acc:.4f}")
    return best_val_acc


def predict_class(model, image_tensor):
    """Run multiclass prediction on a single image tensor. Returns dict."""
    model.eval()
    with torch.no_grad():
        out = model(image_tensor)
        probs = F.softmax(out, dim=1).squeeze()
        pred_idx = probs.argmax().item()
    return {
        "tumor_type": CLASSIFICATION["idx_to_class"][pred_idx],
        "confidence": probs[pred_idx].item(),
        "probs": {CLASSIFICATION["idx_to_class"][i]: probs[i].item() for i in range(3)},
    }


def generate_pseudo_masks(model, data_root, mask_dir, config=None):
    """Generate GradCAM-based pseudo segmentation masks."""
    config = config or CLASSIFICATION
    _, val_transform = get_transforms()

    model.load_state_dict(torch.load(str(config["checkpoint"]), weights_only=True))
    model.eval()
    gradcam = GradCAM(model, model.features[-1][-1])

    for cls_name in config["tumor_classes"]:
        os.makedirs(str(mask_dir / cls_name), exist_ok=True)

    no_aug_dataset = BrainMRIDataset(data_root / "Training", transform=val_transform)
    tumor_indices = [i for i in range(len(no_aug_dataset)) if no_aug_dataset.binary_labels[i].item() == 1]
    tumor_subset = Subset(no_aug_dataset, tumor_indices)
    loader = DataLoader(tumor_subset, batch_size=16, shuffle=False, num_workers=config["num_workers"])

    count = 0
    for batch_idx, (imgs, _, multi_labels) in enumerate(loader):
        imgs_gpu = imgs.to(SHARED["device"])
        cams = gradcam.generate(imgs_gpu)

        for i in range(len(imgs)):
            global_idx = tumor_indices[batch_idx * 16 + i]
            img_path = no_aug_dataset.get_path(global_idx)
            class_name = config["idx_to_class"][multi_labels[i].item()]

            cam_uint8 = (cams[i] * 255).astype(np.uint8)
            _, binary_mask = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            out_path = mask_dir / class_name / (img_path.stem + ".png")
            cv2.imwrite(str(out_path), binary_mask)
            count += 1

    print(f"Generated {count} pseudo-masks to {mask_dir}")
    return count


# ── Visualization ─────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, class_names, title="Classification Confusion Matrix"):
    """Plot NxN confusion matrix."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import ensure_dirs
    ensure_dirs()

    print(f"Device: {SHARED['device']}")

    # Data — reuse prediction dataloaders, then filter to tumor-only
    train_ds, val_ds, test_ds, _, _, test_loader = create_dataloaders(
        CLASSIFICATION["data_root"], CLASSIFICATION["batch_size"],
        CLASSIFICATION["num_workers"], CLASSIFICATION["val_split"],
    )

    tumor_train_idx = [i for i in range(len(train_ds)) if train_ds.binary_labels[i].item() == 1]
    tumor_val_idx = [i for i in range(len(val_ds)) if val_ds.binary_labels[i].item() == 1]

    tumor_train_subset = Subset(train_ds, tumor_train_idx)
    tumor_val_subset = Subset(val_ds, tumor_val_idx)

    # Compute class weights
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

    print(f"Tumor train: {len(tumor_train_subset)} | Tumor val: {len(tumor_val_subset)}")

    # Model
    model = build_multiclass_classifier()

    # Train
    train_multiclass(model, tumor_train_loader, tumor_val_loader, class_weights)

    # Test
    model.load_state_dict(torch.load(str(CLASSIFICATION["checkpoint"]), weights_only=True))
    criterion = FocalLoss(alpha=class_weights.to(SHARED["device"]), gamma=CLASSIFICATION["focal_gamma"])

    tumor_test_idx = [i for i in range(len(test_ds)) if test_ds.binary_labels[i].item() == 1]
    tumor_test_subset = Subset(test_ds, tumor_test_idx)
    tumor_test_loader = DataLoader(
        tumor_test_subset, batch_size=CLASSIFICATION["batch_size"],
        shuffle=False, num_workers=CLASSIFICATION["num_workers"], pin_memory=True,
    )

    result = evaluate(model, tumor_test_loader, criterion)
    print("\nClassification Test Results:")
    print(classification_report(
        result["labels"], result["preds"],
        target_names=CLASSIFICATION["tumor_classes"],
    ))
    plot_confusion_matrix(
        result["labels"], result["preds"],
        class_names=["Glioma", "Menin.", "Pituit."],
    )

    # GradCAM pseudo-masks
    generate_pseudo_masks(model, CLASSIFICATION["data_root"], CLASSIFICATION["pseudo_mask_dir"])

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision import models
from torchvision.models import ConvNeXt_Base_Weights
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
import random

from config import SHARED, PREDICTION


# ── Data ──────────────────────────────────────────────────────────────

class BrainMRIDataset(Dataset):
    CLASS_TO_IDX = {
        "glioma_tumor": 0,
        "meningioma_tumor": 1,
        "pituitary_tumor": 2,
        "no_tumor": 3,
    }
    IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.binary_labels = []
        self.multiclass_labels = []

        for class_name, class_idx in self.CLASS_TO_IDX.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    binary_label = 0 if class_name == "no_tumor" else 1
                    self.samples.append(img_path)
                    self.binary_labels.append(binary_label)
                    self.multiclass_labels.append(class_idx)

        self.binary_labels = torch.tensor(self.binary_labels, dtype=torch.long)
        self.multiclass_labels = torch.tensor(self.multiclass_labels, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = load_image(img_path)
        if self.transform:
            img = self.transform(img)
        return img, self.binary_labels[idx], self.multiclass_labels[idx]

    def get_path(self, idx):
        return self.samples[idx]


class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        self.dataset.transform = self.transform
        return self.dataset[real_idx]

    def get_path(self, idx):
        return self.dataset.get_path(self.indices[idx])

    @property
    def binary_labels(self):
        return self.dataset.binary_labels[self.indices]

    @property
    def multiclass_labels(self):
        return self.dataset.multiclass_labels[self.indices]


def load_image(img_path):
    img = Image.open(img_path)
    if img.mode == "I":
        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
        img = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    elif img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img


def get_transforms():
    train_transform = T.Compose([
        T.RandomResizedCrop(SHARED["img_size"], scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(mean=SHARED["imagenet_mean"], std=SHARED["imagenet_std"]),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    val_transform = T.Compose([
        T.Resize((SHARED["img_size"], SHARED["img_size"])),
        T.ToTensor(),
        T.Normalize(mean=SHARED["imagenet_mean"], std=SHARED["imagenet_std"]),
    ])
    return train_transform, val_transform


def create_dataloaders(data_root, batch_size, num_workers, val_split):
    train_transform, val_transform = get_transforms()

    full_train = BrainMRIDataset(data_root / "Training", transform=None)
    train_idx, val_idx = train_test_split(
        range(len(full_train)),
        test_size=val_split,
        stratify=full_train.multiclass_labels.numpy(),
        random_state=SHARED["seed"],
    )

    train_ds = TransformSubset(full_train, train_idx, train_transform)
    val_ds = TransformSubset(full_train, val_idx, val_transform)
    test_ds = BrainMRIDataset(data_root / "Testing", transform=val_transform)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def denormalize(tensor):
    mean = torch.tensor(SHARED["imagenet_mean"])
    std = torch.tensor(SHARED["imagenet_std"])
    return tensor * std[:, None, None] + mean[:, None, None]


# ── Model ─────────────────────────────────────────────────────────────

def build_binary_classifier(device=None):
    device = device or SHARED["device"]
    backbone = models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    in_features = backbone.classifier[2].in_features
    backbone.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.LayerNorm(in_features),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
    )
    return backbone.to(device)


# ── Training ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler, accum_steps=1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()
    for step, (imgs, binary_labels, _) in enumerate(loader):
        imgs = imgs.to(SHARED["device"])
        labels = binary_labels.float().to(SHARED["device"])
        out = model(imgs).squeeze(1)
        loss = criterion(out, labels) / accum_steps
        loss.backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accum_steps * imgs.size(0)
        preds = (torch.sigmoid(out) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for imgs, binary_labels, _ in loader:
        imgs = imgs.to(SHARED["device"])
        labels = binary_labels.float().to(SHARED["device"])
        out = model(imgs).squeeze(1)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(binary_labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
    }


def train_binary(model, train_loader, val_loader, config=None):
    config = config or PREDICTION
    criterion = nn.BCEWithLogitsLoss()

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
            f"[Prediction] Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_result['loss']:.4f} Acc: {val_result['accuracy']:.4f}"
        )

        if val_result["accuracy"] > best_val_acc:
            best_val_acc = val_result["accuracy"]
            config["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(config["checkpoint"]))

    print(f"Best Prediction Val Accuracy: {best_val_acc:.4f}")
    return best_val_acc


def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        out = model(image_tensor).squeeze()
        prob = torch.sigmoid(out).item()
    return {"has_tumor": prob > 0.5, "confidence": prob}


# ── Visualization ─────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, title="Binary Confusion Matrix"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Tumor", "Tumor"])
    ax.set_yticklabels(["No Tumor", "Tumor"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
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

    # Data
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = create_dataloaders(
        PREDICTION["data_root"], PREDICTION["batch_size"],
        PREDICTION["num_workers"], PREDICTION["val_split"],
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Model
    model = build_binary_classifier()

    # Train
    train_binary(model, train_loader, val_loader)

    # Test
    model.load_state_dict(torch.load(str(PREDICTION["checkpoint"]), weights_only=True))
    criterion = nn.BCEWithLogitsLoss()
    result = evaluate(model, test_loader, criterion)

    print("\nPrediction Test Results:")
    print(f"  Accuracy:  {accuracy_score(result['labels'], result['preds']):.4f}")
    print(f"  Precision: {precision_score(result['labels'], result['preds']):.4f}")
    print(f"  Recall:    {recall_score(result['labels'], result['preds']):.4f}")
    print(f"  F1:        {f1_score(result['labels'], result['preds']):.4f}")
    print(f"  AUC-ROC:   {roc_auc_score(result['labels'], result['probs']):.4f}")

    plot_confusion_matrix(result["labels"], result["preds"])

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision import models
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import cv2
import segmentation_models_pytorch as smp
import os
import random

CONFIG = {
    "data_root": Path("Brain MRI ND-5 Dataset/tumordata"),
    "img_size": 224,
    "batch_size": 32,
    "num_workers": 4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 42,
    "val_split": 0.15,
    "phase1": {
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 15,
        "freeze_epochs": 3,
        "checkpoint": "checkpoints/phase1_binary.pth",
    },
    "phase2": {
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 15,
        "freeze_epochs": 3,
        "checkpoint": "checkpoints/phase2_multiclass.pth",
    },
    "phase3": {
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 20,
        "checkpoint": "checkpoints/phase3_unet.pth",
    },
    "class_to_idx": {
        "glioma_tumor": 0,
        "meningioma_tumor": 1,
        "pituitary_tumor": 2,
        "no_tumor": 3,
    },
    "idx_to_class": {
        0: "glioma_tumor",
        1: "meningioma_tumor",
        2: "pituitary_tumor",
        3: "no_tumor",
    },
    "tumor_classes": ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"],
    "pseudo_mask_dir": Path("pseudo_masks"),
    "pipeline_output_dir": Path("pipeline_outputs"),
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

os.makedirs("checkpoints", exist_ok=True)
os.makedirs(str(CONFIG["pseudo_mask_dir"]), exist_ok=True)
os.makedirs(str(CONFIG["pipeline_output_dir"]), exist_ok=True)

print(f"Device: {CONFIG['device']}")

class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.binary_labels = []
        self.multiclass_labels = []

        for class_name, class_idx in CONFIG["class_to_idx"].items():
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
        img = Image.open(img_path)

        if img.mode == "I":
            arr = np.array(img, dtype=np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
            img = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
        elif img.mode == "RGBA":
            background = Image.new("RGB", img.size, (0, 0, 0))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.binary_labels[idx], self.multiclass_labels[idx]

    def get_path(self, idx):
        return self.samples[idx]


train_transform = T.Compose([
    T.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_train_dataset = BrainMRIDataset(
    CONFIG["data_root"] / "Training", transform=None
)

train_indices, val_indices = train_test_split(
    range(len(full_train_dataset)),
    test_size=CONFIG["val_split"],
    stratify=full_train_dataset.multiclass_labels.numpy(),
    random_state=CONFIG["seed"],
)


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


train_dataset = TransformSubset(full_train_dataset, train_indices, train_transform)
val_dataset = TransformSubset(full_train_dataset, val_indices, val_transform)

test_dataset = BrainMRIDataset(
    CONFIG["data_root"] / "Testing", transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")


def print_class_distribution(dataset, name):
    labels = dataset.multiclass_labels.numpy()
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n{name} Distribution:")
    for u, c in zip(unique, counts):
        class_name = CONFIG["idx_to_class"][u]
        print(f"  {class_name:20s}: {c:5d} ({c/len(labels)*100:.1f}%)")
    binary = dataset.binary_labels.numpy()
    tumor_count = binary.sum()
    print(f"  {'tumor':20s}: {tumor_count:5d} ({tumor_count/len(binary)*100:.1f}%)")
    print(f"  {'no_tumor':20s}: {len(binary)-tumor_count:5d} ({(len(binary)-tumor_count)/len(binary)*100:.1f}%)")

print_class_distribution(train_dataset, "Train")
print_class_distribution(val_dataset, "Validation")
print_class_distribution(test_dataset, "Test")

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

def denormalize(tensor):
    return tensor * STD[:, None, None] + MEAN[:, None, None]

fig, axes = plt.subplots(2, 8, figsize=(20, 5))
batch_imgs, batch_binary, batch_multi = next(iter(train_loader))
for i in range(min(16, len(batch_imgs))):
    ax = axes[i // 8, i % 8]
    img = denormalize(batch_imgs[i]).permute(1, 2, 0).clamp(0, 1).numpy()
    ax.imshow(img)
    class_name = CONFIG["idx_to_class"][batch_multi[i].item()]
    tumor_str = "T" if batch_binary[i].item() == 1 else "N"
    ax.set_title(f"{tumor_str}: {class_name[:4]}", fontsize=8)
    ax.axis("off")
plt.suptitle("Sample Training Batch", fontsize=14)
plt.tight_layout()
plt.show()


def build_binary_classifier():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model.to(CONFIG["device"])

def train_one_epoch_binary(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, binary_labels, _ in loader:
        imgs = imgs.to(CONFIG["device"])
        labels = binary_labels.float().to(CONFIG["device"])
        optimizer.zero_grad()
        out = model(imgs).squeeze(1)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(out) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def eval_binary(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for imgs, binary_labels, _ in loader:
        imgs = imgs.to(CONFIG["device"])
        labels = binary_labels.float().to(CONFIG["device"])
        out = model(imgs).squeeze(1)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(out) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

binary_model = build_binary_classifier()
criterion_binary = nn.BCEWithLogitsLoss()

for param in binary_model.features.parameters():
    param.requires_grad = False

optimizer_binary = optim.AdamW(
    filter(lambda p: p.requires_grad, binary_model.parameters()),
    lr=CONFIG["phase1"]["lr"],
    weight_decay=CONFIG["phase1"]["weight_decay"],
)
scheduler_binary = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_binary, T_max=CONFIG["phase1"]["epochs"]
)

best_val_acc = 0.0
for epoch in range(CONFIG["phase1"]["epochs"]):
    if epoch == CONFIG["phase1"]["freeze_epochs"]:
        for param in binary_model.features.parameters():
            param.requires_grad = True
        optimizer_binary = optim.AdamW(
            binary_model.parameters(),
            lr=CONFIG["phase1"]["lr"] * 0.1,
            weight_decay=CONFIG["phase1"]["weight_decay"],
        )
        scheduler_binary = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_binary,
            T_max=CONFIG["phase1"]["epochs"] - CONFIG["phase1"]["freeze_epochs"],
        )

    train_loss, train_acc = train_one_epoch_binary(
        binary_model, train_loader, criterion_binary, optimizer_binary
    )
    val_loss, val_acc = eval_binary(binary_model, val_loader, criterion_binary)
    scheduler_binary.step()

    print(
        f"[Phase1] Epoch {epoch+1}/{CONFIG['phase1']['epochs']} | "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(binary_model.state_dict(), CONFIG["phase1"]["checkpoint"])

print(f"Best Phase1 Val Accuracy: {best_val_acc:.4f}")



binary_model.load_state_dict(torch.load(CONFIG["phase1"]["checkpoint"], weights_only=True))
binary_model.eval()

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for imgs, binary_labels, _ in test_loader:
        imgs = imgs.to(CONFIG["device"])
        out = binary_model(imgs).squeeze(1)
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(binary_labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

print("Phase 1 - Binary Classification (Test Set)")
print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
print(f"Recall:    {recall_score(all_labels, all_preds):.4f}")
print(f"F1:        {f1_score(all_labels, all_preds):.4f}")
print(f"AUC-ROC:   {roc_auc_score(all_labels, all_probs):.4f}")

cm = confusion_matrix(all_labels, all_preds)
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
ax.set_title("Phase 1: Binary Classification Confusion Matrix")
plt.colorbar(im)
plt.tight_layout()
plt.show()



tumor_train_indices = [i for i in range(len(train_dataset)) if train_dataset.binary_labels[i].item() == 1]
tumor_val_indices = [i for i in range(len(val_dataset)) if val_dataset.binary_labels[i].item() == 1]

tumor_train_subset = Subset(train_dataset, tumor_train_indices)
tumor_val_subset = Subset(val_dataset, tumor_val_indices)

tumor_class_counts = torch.zeros(3)
for idx in tumor_train_indices:
    label = train_dataset.multiclass_labels[idx].item()
    tumor_class_counts[label] += 1
tumor_class_weights = (1.0 / tumor_class_counts)
tumor_class_weights = tumor_class_weights / tumor_class_weights.sum() * 3.0

tumor_train_loader = DataLoader(
    tumor_train_subset, batch_size=CONFIG["batch_size"],
    shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True,
)
tumor_val_loader = DataLoader(
    tumor_val_subset, batch_size=CONFIG["batch_size"],
    shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True,
)

def build_multiclass_classifier():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    return model.to(CONFIG["device"])

def train_one_epoch_multi(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, _, multi_labels in loader:
        imgs = imgs.to(CONFIG["device"])
        labels = multi_labels.to(CONFIG["device"])
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def eval_multi(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for imgs, _, multi_labels in loader:
        imgs = imgs.to(CONFIG["device"])
        labels = multi_labels.to(CONFIG["device"])
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

multi_model = build_multiclass_classifier()
criterion_multi = nn.CrossEntropyLoss(
    weight=tumor_class_weights.to(CONFIG["device"])
)

for param in multi_model.features.parameters():
    param.requires_grad = False

optimizer_multi = optim.AdamW(
    filter(lambda p: p.requires_grad, multi_model.parameters()),
    lr=CONFIG["phase2"]["lr"],
    weight_decay=CONFIG["phase2"]["weight_decay"],
)
scheduler_multi = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_multi, T_max=CONFIG["phase2"]["epochs"]
)

best_val_acc_multi = 0.0
for epoch in range(CONFIG["phase2"]["epochs"]):
    if epoch == CONFIG["phase2"]["freeze_epochs"]:
        for param in multi_model.features.parameters():
            param.requires_grad = True
        optimizer_multi = optim.AdamW(
            multi_model.parameters(),
            lr=CONFIG["phase2"]["lr"] * 0.1,
            weight_decay=CONFIG["phase2"]["weight_decay"],
        )
        scheduler_multi = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_multi,
            T_max=CONFIG["phase2"]["epochs"] - CONFIG["phase2"]["freeze_epochs"],
        )

    train_loss, train_acc = train_one_epoch_multi(
        multi_model, tumor_train_loader, criterion_multi, optimizer_multi
    )
    val_loss, val_acc = eval_multi(multi_model, tumor_val_loader, criterion_multi)
    scheduler_multi.step()

    print(
        f"[Phase2] Epoch {epoch+1}/{CONFIG['phase2']['epochs']} | "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc_multi:
        best_val_acc_multi = val_acc
        torch.save(multi_model.state_dict(), CONFIG["phase2"]["checkpoint"])

print(f"Best Phase2 Val Accuracy: {best_val_acc_multi:.4f}")



multi_model.load_state_dict(torch.load(CONFIG["phase2"]["checkpoint"], weights_only=True))
multi_model.eval()

tumor_test_indices = [i for i in range(len(test_dataset)) if test_dataset.binary_labels[i].item() == 1]
tumor_test_subset = Subset(test_dataset, tumor_test_indices)
tumor_test_loader = DataLoader(
    tumor_test_subset, batch_size=CONFIG["batch_size"],
    shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True,
)

all_preds_m, all_labels_m = [], []
with torch.no_grad():
    for imgs, _, multi_labels in tumor_test_loader:
        imgs = imgs.to(CONFIG["device"])
        out = multi_model(imgs)
        all_preds_m.extend(out.argmax(1).cpu().numpy())
        all_labels_m.extend(multi_labels.numpy())

all_preds_m = np.array(all_preds_m)
all_labels_m = np.array(all_labels_m)

print("Phase 2 - Tumor Type Classification (Test Set)")
print(classification_report(
    all_labels_m, all_preds_m,
    target_names=CONFIG["tumor_classes"],
))

cm2 = confusion_matrix(all_labels_m, all_preds_m)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm2, cmap="Blues")
ticks = list(range(3))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
short_names = ["Glioma", "Menin.", "Pituit."]
ax.set_xticklabels(short_names)
ax.set_yticklabels(short_names)
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm2[i, j]), ha="center", va="center", fontsize=13)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Phase 2: Tumor Type Confusion Matrix")
plt.colorbar(im)
plt.tight_layout()
plt.show()



class GradCAM:
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
        cam = F.interpolate(cam, size=(CONFIG["img_size"], CONFIG["img_size"]), mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        for i in range(cam.size(0)):
            c = cam[i]
            if c.max() > 0:
                cam[i] = (c - c.min()) / (c.max() - c.min())
        return cam.cpu().numpy()

multi_model.load_state_dict(torch.load(CONFIG["phase2"]["checkpoint"], weights_only=True))
multi_model.eval()
gradcam = GradCAM(multi_model, multi_model.features[-1])

for cls_name in CONFIG["tumor_classes"]:
    os.makedirs(str(CONFIG["pseudo_mask_dir"] / cls_name), exist_ok=True)

no_aug_dataset = BrainMRIDataset(CONFIG["data_root"] / "Training", transform=val_transform)
tumor_mask_indices = [i for i in range(len(no_aug_dataset)) if no_aug_dataset.binary_labels[i].item() == 1]
tumor_mask_subset = Subset(no_aug_dataset, tumor_mask_indices)
tumor_mask_loader = DataLoader(tumor_mask_subset, batch_size=16, shuffle=False, num_workers=CONFIG["num_workers"])

sample_imgs_for_viz = []
sample_cams_for_viz = []
sample_masks_for_viz = []
generated_count = 0

for batch_idx, (imgs, _, multi_labels) in enumerate(tumor_mask_loader):
    imgs_gpu = imgs.to(CONFIG["device"])
    cams = gradcam.generate(imgs_gpu)

    for i in range(len(imgs)):
        global_idx = tumor_mask_indices[batch_idx * 16 + i]
        img_path = no_aug_dataset.get_path(global_idx)
        class_name = CONFIG["idx_to_class"][multi_labels[i].item()]

        cam = cams[i]
        cam_uint8 = (cam * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask_path = CONFIG["pseudo_mask_dir"] / class_name / (img_path.stem + ".png")
        cv2.imwrite(str(mask_path), binary_mask)
        generated_count += 1

        if len(sample_imgs_for_viz) < 12:
            sample_imgs_for_viz.append(denormalize(imgs[i]).permute(1, 2, 0).clamp(0, 1).numpy())
            sample_cams_for_viz.append(cam)
            sample_masks_for_viz.append(binary_mask)

print(f"Generated {generated_count} pseudo-masks")

fig, axes = plt.subplots(4, 9, figsize=(22, 10))
for i in range(min(4, len(sample_imgs_for_viz))):
    row = i
    axes[row, 0].imshow(sample_imgs_for_viz[i * 3])
    axes[row, 0].set_title("Image", fontsize=8)
    axes[row, 0].axis("off")
    axes[row, 1].imshow(sample_cams_for_viz[i * 3], cmap="jet")
    axes[row, 1].set_title("GradCAM", fontsize=8)
    axes[row, 1].axis("off")
    axes[row, 2].imshow(sample_masks_for_viz[i * 3], cmap="gray")
    axes[row, 2].set_title("Mask", fontsize=8)
    axes[row, 2].axis("off")
    if i * 3 + 1 < len(sample_imgs_for_viz):
        axes[row, 3].imshow(sample_imgs_for_viz[i * 3 + 1])
        axes[row, 3].axis("off")
        axes[row, 4].imshow(sample_cams_for_viz[i * 3 + 1], cmap="jet")
        axes[row, 4].axis("off")
        axes[row, 5].imshow(sample_masks_for_viz[i * 3 + 1], cmap="gray")
        axes[row, 5].axis("off")
    if i * 3 + 2 < len(sample_imgs_for_viz):
        axes[row, 6].imshow(sample_imgs_for_viz[i * 3 + 2])
        axes[row, 6].axis("off")
        axes[row, 7].imshow(sample_cams_for_viz[i * 3 + 2], cmap="jet")
        axes[row, 7].axis("off")
        axes[row, 8].imshow(sample_masks_for_viz[i * 3 + 2], cmap="gray")
        axes[row, 8].axis("off")
plt.suptitle("GradCAM Pseudo-Mask Generation", fontsize=14)
plt.tight_layout()
plt.show()



class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_dir, transform=None):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.transform = transform
        self.valid_pairs = []
        for p in image_paths:
            class_name = p.parent.name
            mask_path = mask_dir / class_name / (p.stem + ".png")
            if mask_path.exists():
                self.valid_pairs.append((p, mask_path))

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
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

        img = img.resize((CONFIG["img_size"], CONFIG["img_size"]))
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((CONFIG["img_size"], CONFIG["img_size"]))

        img_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(img)

        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        return img_tensor, mask_tensor

class DiceLoss(nn.Module):
    def forward(self, pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

tumor_image_paths = []
for cls in CONFIG["tumor_classes"]:
    cls_dir = CONFIG["data_root"] / "Training" / cls
    for p in sorted(cls_dir.iterdir()):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            tumor_image_paths.append(p)

seg_dataset = SegmentationDataset(tumor_image_paths, CONFIG["pseudo_mask_dir"])
seg_train_size = int(0.85 * len(seg_dataset))
seg_val_size = len(seg_dataset) - seg_train_size
seg_train_ds, seg_val_ds = torch.utils.data.random_split(
    seg_dataset, [seg_train_size, seg_val_size],
    generator=torch.Generator().manual_seed(CONFIG["seed"]),
)

seg_train_loader = DataLoader(
    seg_train_ds, batch_size=CONFIG["batch_size"],
    shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True,
)
seg_val_loader = DataLoader(
    seg_val_ds, batch_size=CONFIG["batch_size"],
    shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True,
)

unet = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
).to(CONFIG["device"])

bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()
optimizer_seg = optim.AdamW(
    unet.parameters(),
    lr=CONFIG["phase3"]["lr"],
    weight_decay=CONFIG["phase3"]["weight_decay"],
)
scheduler_seg = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_seg, T_max=CONFIG["phase3"]["epochs"]
)

def dice_score(pred, target, threshold=0.5):
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    return (2.0 * intersection) / (pred_bin.sum() + target.sum() + 1e-8)

best_dice = 0.0
for epoch in range(CONFIG["phase3"]["epochs"]):
    unet.train()
    epoch_loss = 0.0
    for imgs, masks in seg_train_loader:
        imgs = imgs.to(CONFIG["device"])
        masks = masks.to(CONFIG["device"])
        optimizer_seg.zero_grad()
        out = unet(imgs)
        loss = 0.5 * bce_loss(out, masks) + 0.5 * dice_loss(out, masks)
        loss.backward()
        optimizer_seg.step()
        epoch_loss += loss.item() * imgs.size(0)

    unet.eval()
    val_dice_total = 0.0
    val_count = 0
    with torch.no_grad():
        for imgs, masks in seg_val_loader:
            imgs = imgs.to(CONFIG["device"])
            masks = masks.to(CONFIG["device"])
            out = unet(imgs)
            val_dice_total += dice_score(out, masks).item() * imgs.size(0)
            val_count += imgs.size(0)

    avg_dice = val_dice_total / val_count
    scheduler_seg.step()

    print(
        f"[Phase3] Epoch {epoch+1}/{CONFIG['phase3']['epochs']} | "
        f"Loss: {epoch_loss/len(seg_train_ds):.4f} | Val Dice: {avg_dice:.4f}"
    )

    if avg_dice > best_dice:
        best_dice = avg_dice
        torch.save(unet.state_dict(), CONFIG["phase3"]["checkpoint"])

print(f"Best Phase3 Val Dice: {best_dice:.4f}")



unet.load_state_dict(torch.load(CONFIG["phase3"]["checkpoint"], weights_only=True))
unet.eval()

tumor_test_paths = []
for cls in CONFIG["tumor_classes"]:
    cls_dir = CONFIG["data_root"] / "Testing" / cls
    for p in sorted(cls_dir.iterdir()):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            tumor_test_paths.append(p)

test_seg_dataset = SegmentationDataset(tumor_test_paths, CONFIG["pseudo_mask_dir"])

if len(test_seg_dataset) == 0:
    no_aug_test = BrainMRIDataset(CONFIG["data_root"] / "Testing", transform=val_transform)
    test_tumor_idx = [i for i in range(len(no_aug_test)) if no_aug_test.binary_labels[i].item() == 1]
    sample_test_imgs = []
    for idx in test_tumor_idx[:16]:
        img, _, _ = no_aug_test[idx]
        sample_test_imgs.append(img)
    sample_test_imgs = torch.stack(sample_test_imgs).to(CONFIG["device"])
    with torch.no_grad():
        pred_masks = torch.sigmoid(unet(sample_test_imgs)).cpu()

    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    for i in range(min(4, len(sample_test_imgs))):
        for j in range(2):
            idx = i * 2 + j
            if idx >= len(sample_test_imgs):
                break
            img_viz = denormalize(sample_test_imgs[idx].cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
            axes[i, j * 3].imshow(img_viz)
            axes[i, j * 3].set_title("Image", fontsize=8)
            axes[i, j * 3].axis("off")
            axes[i, j * 3 + 1].imshow(pred_masks[idx, 0], cmap="gray")
            axes[i, j * 3 + 1].set_title("Pred Mask", fontsize=8)
            axes[i, j * 3 + 1].axis("off")
            overlay = img_viz.copy()
            mask_np = pred_masks[idx, 0].numpy()
            overlay[:, :, 0] = np.clip(overlay[:, :, 0] + mask_np * 0.3, 0, 1)
            axes[i, j * 3 + 2].imshow(overlay)
            axes[i, j * 3 + 2].set_title("Overlay", fontsize=8)
            axes[i, j * 3 + 2].axis("off")
    plt.suptitle("Phase 3: Segmentation on Test Images", fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    test_seg_loader = DataLoader(test_seg_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    total_dice, total_iou, count = 0.0, 0.0, 0
    sample_imgs_viz, sample_gt_viz, sample_pred_viz = [], [], []
    with torch.no_grad():
        for imgs, masks in test_seg_loader:
            imgs = imgs.to(CONFIG["device"])
            masks = masks.to(CONFIG["device"])
            out = unet(imgs)
            pred = (torch.sigmoid(out) > 0.5).float()
            for k in range(imgs.size(0)):
                inter = (pred[k] * masks[k]).sum()
                union = pred[k].sum() + masks[k].sum() - inter
                d = (2 * inter) / (pred[k].sum() + masks[k].sum() + 1e-8)
                iou = inter / (union + 1e-8)
                total_dice += d.item()
                total_iou += iou.item()
                count += 1
                if len(sample_imgs_viz) < 8:
                    sample_imgs_viz.append(denormalize(imgs[k].cpu()).permute(1, 2, 0).clamp(0, 1).numpy())
                    sample_gt_viz.append(masks[k, 0].cpu().numpy())
                    sample_pred_viz.append(pred[k, 0].cpu().numpy())

    print(f"Test Dice: {total_dice/count:.4f}")
    print(f"Test IoU:  {total_iou/count:.4f}")

    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    for i in range(min(4, len(sample_imgs_viz))):
        for j in range(2):
            idx = i * 2 + j
            if idx >= len(sample_imgs_viz):
                break
            axes[i, j*3].imshow(sample_imgs_viz[idx])
            axes[i, j*3].set_title("Image", fontsize=8)
            axes[i, j*3].axis("off")
            axes[i, j*3+1].imshow(sample_gt_viz[idx], cmap="gray")
            axes[i, j*3+1].set_title("GT Mask", fontsize=8)
            axes[i, j*3+1].axis("off")
            axes[i, j*3+2].imshow(sample_pred_viz[idx], cmap="gray")
            axes[i, j*3+2].set_title("Pred Mask", fontsize=8)
            axes[i, j*3+2].axis("off")
    plt.suptitle("Phase 3: Segmentation Evaluation", fontsize=14)
    plt.tight_layout()
    plt.show()



binary_model.load_state_dict(torch.load(CONFIG["phase1"]["checkpoint"], weights_only=True))
binary_model.eval()
multi_model.load_state_dict(torch.load(CONFIG["phase2"]["checkpoint"], weights_only=True))
multi_model.eval()
unet.load_state_dict(torch.load(CONFIG["phase3"]["checkpoint"], weights_only=True))
unet.eval()

def run_pipeline(img_path):
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

    img_tensor = val_transform(img).unsqueeze(0).to(CONFIG["device"])

    with torch.no_grad():
        binary_out = binary_model(img_tensor).squeeze()
        binary_prob = torch.sigmoid(binary_out).item()
        has_tumor = binary_prob > 0.5

        result = {
            "has_tumor": has_tumor,
            "tumor_confidence": binary_prob,
            "tumor_type": None,
            "type_confidence": None,
            "segmentation_mask": None,
        }

        if has_tumor:
            multi_out = multi_model(img_tensor)
            type_probs = F.softmax(multi_out, dim=1).squeeze()
            pred_type = type_probs.argmax().item()
            result["tumor_type"] = CONFIG["idx_to_class"][pred_type]
            result["type_confidence"] = type_probs[pred_type].item()

            seg_out = unet(img_tensor)
            mask = (torch.sigmoid(seg_out) > 0.5).float().squeeze().cpu()
            result["segmentation_mask"] = mask

    return result

test_paths = []
for cls in CONFIG["class_to_idx"].keys():
    cls_dir = CONFIG["data_root"] / "Testing" / cls
    paths = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    test_paths.extend(random.sample(paths, min(2, len(paths))))

random.shuffle(test_paths)
test_paths = test_paths[:8]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, path in enumerate(test_paths):
    ax = axes[i // 4, i % 4]
    result = run_pipeline(path)

    img = Image.open(path).convert("RGB").resize((CONFIG["img_size"], CONFIG["img_size"]))
    img_np = np.array(img) / 255.0

    if result["has_tumor"] and result["segmentation_mask"] is not None:
        overlay = img_np.copy()
        mask_np = result["segmentation_mask"].numpy()
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + mask_np * 0.4, 0, 1)
        ax.imshow(overlay)
        ax.set_title(
            f"{result['tumor_type']}\n"
            f"Conf: {result['type_confidence']:.2f}",
            fontsize=9, color="red",
        )
    else:
        ax.imshow(img_np)
        ax.set_title("No Tumor", fontsize=9, color="green")
    ax.axis("off")

plt.suptitle("Full Pipeline Inference Demo", fontsize=16)
plt.tight_layout()
plt.show()



os.makedirs(str(CONFIG["pipeline_output_dir"] / "masks"), exist_ok=True)

export_paths = []
export_types = []
export_confs = []

for cls in CONFIG["tumor_classes"]:
    cls_dir = CONFIG["data_root"] / "Testing" / cls
    for p in sorted(cls_dir.iterdir()):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            export_paths.append(p)

export_loader_dataset = BrainMRIDataset.__new__(BrainMRIDataset)
export_loader_dataset.samples = export_paths
export_loader_dataset.transform = val_transform
export_loader_dataset.binary_labels = torch.ones(len(export_paths), dtype=torch.long)
export_loader_dataset.multiclass_labels = torch.zeros(len(export_paths), dtype=torch.long)

for i, p in enumerate(export_paths):
    cls_name = p.parent.name
    export_loader_dataset.multiclass_labels[i] = CONFIG["class_to_idx"][cls_name]

export_loader = DataLoader(
    export_loader_dataset, batch_size=16, shuffle=False, num_workers=CONFIG["num_workers"],
)

all_masks = []
all_types = []
all_type_confs = []
all_paths_str = []

with torch.no_grad():
    for batch_idx, (imgs, _, multi_labels) in enumerate(export_loader):
        imgs_gpu = imgs.to(CONFIG["device"])

        multi_out = multi_model(imgs_gpu)
        type_probs = F.softmax(multi_out, dim=1)
        pred_types = type_probs.argmax(1)
        pred_confs = type_probs.max(1).values

        seg_out = unet(imgs_gpu)
        masks = (torch.sigmoid(seg_out) > 0.5).float().squeeze(1).cpu()

        for i in range(imgs.size(0)):
            global_idx = batch_idx * 16 + i
            mask_filename = f"mask_{global_idx:05d}.pt"
            torch.save(masks[i], str(CONFIG["pipeline_output_dir"] / "masks" / mask_filename))
            all_masks.append(mask_filename)
            all_types.append(pred_types[i].item())
            all_type_confs.append(pred_confs[i].item())
            all_paths_str.append(str(export_paths[global_idx]))

metadata = {
    "image_paths": all_paths_str,
    "mask_files": all_masks,
    "tumor_types": all_types,
    "type_confidences": all_type_confs,
    "idx_to_class": CONFIG["idx_to_class"],
}
torch.save(metadata, str(CONFIG["pipeline_output_dir"] / "metadata.pt"))

print(f"Exported {len(all_masks)} masks to {CONFIG['pipeline_output_dir']}")
print(f"Metadata saved to {CONFIG['pipeline_output_dir'] / 'metadata.pt'}")

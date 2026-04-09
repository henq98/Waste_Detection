"""
GEO5017 — Waste Detection Pipeline (Folder-Based Version)
===========================================================
Transfer Learning with ResNet-50 / ResNet-101 / EfficientNet-B3.
Includes YOLO-based bounding box annotation for the localization bonus.

EXPECTED FOLDER STRUCTURE (no CSV needed):
-------------------------------------------
dataset/
├── train/
│   ├── waste/          ← waste images for training
│   └── no_waste/       ← clean images for training
├── val/
│   ├── waste/
│   └── no_waste/
└── test/
    ├── waste/
    └── no_waste/

If your images are currently in year-based subfolders AND already split
into waste/no_waste sub-subfolders, use build_dataset_from_labeled_folders()
to reorganise them into the structure above.

If your images are ALREADY in the structure above, just set
DATASET_ROOT and run run_full_pipeline() directly.

For the leaderboard Top-100, point RAW_DATA_ROOT at the full 10k folder.
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import shutil
import random
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    average_precision_score,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, models, transforms

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# 1. CONFIGURATION — edit these paths and hyperparameters
# =============================================================================

# Root of the unlabelled 10k images (used ONLY for leaderboard inference)
RAW_DATA_ROOT = Path("UrbanWaste-images-10k-right")

# Root of your pre-labeled dataset (train/val/test with waste/no_waste folders)
DATASET_ROOT = Path("dataset")

# Where to save model checkpoints
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ── Leaderboard ───────────────────────────────────────────────────────────────
TOP_K = 100

# ── Class-imbalance strategy (pick ONE) ──────────────────────────────────────
USE_WEIGHTED_SAMPLER = True   # Recommended default
USE_POS_WEIGHT       = False  # Alternative: reweight the loss

if USE_WEIGHTED_SAMPLER and USE_POS_WEIGHT:
    raise ValueError("Enable either USE_WEIGHTED_SAMPLER or USE_POS_WEIGHT, not both.")

# ── Backbone choice ───────────────────────────────────────────────────────────
# Options: "resnet50" | "resnet101" | "efficientnet_b3"
BACKBONE = "resnet50"

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMG_SIZE    = 320
BATCH_SIZE  = 256
NUM_WORKERS = 0
persistent_workers=False # Keep 0 on Windows to avoid shared memory errors

# Phase 1 — classification head only
LR_HEAD      = 1e-3
EPOCHS_HEAD  = 30

# Phase 2 — head + ResNet layer4 unfrozen
LR_FINETUNE      = 5e-5
EPOCHS_FINETUNE  = 70

PATIENCE = 10   # Early stopping: epochs without val AP improvement

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_LABELS     = {"waste", "no_waste"}


# =============================================================================
# 2. OPTIONAL — reorganise year-based labeled folders into train/val/test
# =============================================================================

def build_dataset_from_labeled_folders(
    labeled_root: Path,
    dataset_root: Path,
    train_years: List[str],
    val_years: List[str],
    test_years: List[str],
    rebuild: bool = True,
) -> None:
    """
    Use this ONLY if your data looks like:

        labeled_root/
        ├── year_2016/
        │   ├── waste/
        │   └── no_waste/
        ├── year_2017/
        │   ├── waste/
        │   └── no_waste/
        ...

    It copies images into dataset/train|val|test/waste|no_waste/.

    If your data is ALREADY in dataset/train/waste etc., skip this function.
    """
    split_map: Dict[str, str] = {}
    for y in train_years:
        split_map[y] = "train"
    for y in val_years:
        split_map[y] = "val"
    for y in test_years:
        split_map[y] = "test"

    if rebuild and dataset_root.exists():
        shutil.rmtree(dataset_root)
        print(f"Deleted existing dataset folder: {dataset_root}")

    for split in ["train", "val", "test"]:
        for cls in VALID_LABELS:
            (dataset_root / split / cls).mkdir(parents=True, exist_ok=True)

    copied = 0
    for year_folder, split in split_map.items():
        year_path = labeled_root / year_folder
        if not year_path.exists():
            print(f"Warning: {year_path} not found — skipping.")
            continue
        for cls in VALID_LABELS:
            cls_path = year_path / cls
            if not cls_path.exists():
                continue
            for img in sorted(cls_path.iterdir()):
                if img.suffix.lower() in IMAGE_EXTENSIONS:
                    dest = dataset_root / split / cls / img.name
                    shutil.copy2(img, dest)
                    copied += 1

    print(f"Done: {copied} images copied into {dataset_root}.")
    _print_split_stats(dataset_root)


def _print_split_stats(dataset_root: Path) -> None:
    """Prints class counts for each split."""
    print(f"\nDataset statistics:")
    print(f"{'Split':<10} {'waste':>8} {'no_waste':>10} {'total':>8}")
    print("-" * 42)
    for split in ["train", "val", "test"]:
        w  = len(list((dataset_root / split / "waste").glob("*")))
        nw = len(list((dataset_root / split / "no_waste").glob("*")))
        print(f"{split:<10} {w:>8} {nw:>10} {w + nw:>8}")


# =============================================================================
# 3. DATA LOADERS & AUGMENTATION
# =============================================================================

def get_transforms(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE,scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def _validate_split(dataset_obj, split_name: str) -> None:
    class_names = dataset_obj.classes
    if set(class_names) != VALID_LABELS:
        raise ValueError(
            f"{split_name} must contain exactly 'waste' and 'no_waste'. "
            f"Found: {class_names}"
        )
    counts = np.bincount(dataset_obj.targets, minlength=len(class_names))
    stats  = dict(zip(class_names, counts.tolist()))
    print(f"{split_name.title()} split class counts: {stats}")
    if split_name == "train" and np.any(counts == 0):
        raise ValueError("Training split is missing one class.")


def get_dataloaders(
    dataset_root: Path,
    use_weighted_sampler: bool = USE_WEIGHTED_SAMPLER,
) -> dict:
    """Creates DataLoaders for train, val, and test splits."""
    datasets_dict = {
        split: datasets.ImageFolder(
            root=str(dataset_root / split),
            transform=get_transforms(split),
        )
        for split in ["train", "val", "test"]
    }

    for split, ds in datasets_dict.items():
        _validate_split(ds, split)

    train_ds       = datasets_dict["train"]
    class_to_idx   = train_ds.class_to_idx
    pos_class_idx  = class_to_idx["waste"]
    neg_class_idx  = class_to_idx["no_waste"]

    train_kwargs: dict = {
        "dataset": train_ds,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
    }

    if use_weighted_sampler:
        class_counts   = np.bincount(train_ds.targets, minlength=len(train_ds.classes)).astype(float)
        class_weights  = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in train_ds.targets]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_kwargs["sampler"] = sampler
        print("Training loader uses WeightedRandomSampler.")
    else:
        train_kwargs["shuffle"] = True
        print("Training loader uses standard shuffled sampling.")

    loaders = {
        "train": DataLoader(**train_kwargs),
        "val": DataLoader(datasets_dict["val"],  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
        "test": DataLoader(datasets_dict["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
        "class_names":       train_ds.classes,
        "class_to_idx":      class_to_idx,
        "positive_class_idx": pos_class_idx,
        "negative_class_idx": neg_class_idx,
    }

    print(f"Class names (index order): {loaders['class_names']}")
    print(f"Train batches : {len(loaders['train'])}")
    print(f"Val   batches : {len(loaders['val'])}")
    print(f"Test  batches : {len(loaders['test'])}")
    return loaders


def show_batch(loader: DataLoader, class_names: list, n: int = 8) -> None:
    """Displays n images from one training batch after augmentation."""
    images, labels = next(iter(loader))
    images = images[:n]
    labels = labels[:n]

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    images_disp = (images * std + mean).clamp(0, 1)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images_disp[i].permute(1, 2, 0).numpy())
        ax.set_title(class_names[labels[i].item()], fontsize=10)
        ax.axis("off")
    plt.suptitle("Sample training batch (after augmentation)", y=1.02)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 4. MODEL
# =============================================================================

def build_model(freeze_backbone: bool = True, backbone: str = BACKBONE) -> nn.Module:
    """
    Builds a binary classification model using the chosen backbone.

    Supported backbones:
        "resnet50"        — fastest, good baseline
        "resnet101"       — stronger features, ~2x slower
        "efficientnet_b3" — best accuracy/speed tradeoff

    All backbones use:
        Custom head: Linear(in→256) → ReLU → Dropout(0.5) → Linear(256→1)
    """
    if backbone == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze_backbone:
            for param in base.parameters():
                param.requires_grad = False
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(256, 1),
        )

    elif backbone == "resnet101":
        base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        if freeze_backbone:
            for param in base.parameters():
                param.requires_grad = False
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
        )

    elif backbone == "efficientnet_b3":
        base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in base.parameters():
                param.requires_grad = False
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
        )

    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose resnet50, resnet101, or efficientnet_b3.")

    base = base.to(DEVICE)
    total     = sum(p.numel() for p in base.parameters())
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    print(f"Backbone            : {backbone}")
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return base


def unfreeze_last_block(model: nn.Module, backbone: str = BACKBONE) -> nn.Module:
    """
    Unfreezes the last feature block + head for Phase 2 fine-tuning.
    Works for ResNet (layer4) and EfficientNet (features[-1]).
    """
    for param in model.parameters():
        param.requires_grad = False

    if backbone in ("resnet50", "resnet101"):
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
    elif backbone == "efficientnet_b3":
        for param in model.features[-1].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after unfreezing last block: {trainable:,}")
    return model


# =============================================================================
# 5. TRAINING UTILITIES
# =============================================================================

def compute_class_weight(dataset_root: Path) -> torch.Tensor:
    n_waste    = len(list((dataset_root / "train" / "waste").glob("*")))
    n_no_waste = len(list((dataset_root / "train" / "no_waste").glob("*")))
    if n_waste == 0:
        raise ValueError("train/waste folder is empty — cannot compute pos_weight.")
    pos_weight = n_no_waste / n_waste
    print(f"BCEWithLogitsLoss pos_weight: {pos_weight:.2f}  (waste={n_waste}, no_waste={n_no_waste})")
    return torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)


def _to_binary(labels: torch.Tensor, positive_index: int) -> torch.Tensor:
    return (labels == positive_index).float()


def _class_indices_from_binary(
    binary_preds: np.ndarray,
    positive_index: int,
    negative_index: int,
) -> np.ndarray:
    return np.where(binary_preds == 1, positive_index, negative_index)


def precision_at_k(y_true_binary: np.ndarray, y_scores: np.ndarray, k: int = TOP_K) -> float:
    if len(y_true_binary) == 0:
        return float("nan")
    effective_k = min(k, len(y_true_binary))
    top_idx = np.argsort(-y_scores)[:effective_k]
    return float(y_true_binary[top_idx].sum() / effective_k)


def average_precision_safe(y_true_binary: np.ndarray, y_scores: np.ndarray) -> float:
    if y_true_binary.sum() == 0 or (1 - y_true_binary).sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true_binary, y_scores))


def train_one_epoch(model, loader, criterion, optimizer, positive_index: int):
    model.train()
    running_loss = correct = total = 0

    for images, labels in loader:
        images        = images.to(DEVICE)
        binary_labels = _to_binary(labels, positive_index).unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, binary_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds    = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == binary_labels).sum().item()
        total   += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, positive_index: int, top_k: int = TOP_K) -> dict:
    model.eval()
    running_loss = correct = total = 0
    all_scores, all_true = [], []

    for images, labels in loader:
        images        = images.to(DEVICE)
        binary_labels = _to_binary(labels, positive_index).unsqueeze(1).to(DEVICE)
        logits = model(images)
        probs  = torch.sigmoid(logits)

        running_loss += criterion(logits, binary_labels).item() * images.size(0)
        preds    = (probs >= 0.5).float()
        correct += (preds == binary_labels).sum().item()
        total   += images.size(0)

        all_scores.extend(probs.squeeze(1).cpu().numpy())
        all_true.extend(binary_labels.squeeze(1).cpu().numpy())

    y_scores = np.asarray(all_scores, dtype=float)
    y_true   = np.asarray(all_true,   dtype=int)

    return {
        "loss":     running_loss / total,
        "acc":      correct / total,
        "ap":       average_precision_safe(y_true, y_scores),
        "p_at_100": precision_at_k(y_true, y_scores, top_k),
    }


def train(
    model,
    loaders,
    num_epochs,
    lr,
    checkpoint_name,
    patience=PATIENCE,
    pos_weight=None,
    top_k: int = TOP_K,
) -> dict:
    """Full training loop with early stopping on validation Average Precision."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    history = {k: [] for k in [
        "train_loss", "val_loss", "train_acc", "val_acc",
        "train_ap", "val_ap", "val_p_at_100",
    ]}

    best_val_ap   = float("-inf")
    best_ckpt     = CHECKPOINT_DIR / f"{checkpoint_name}.pth"
    no_improve    = 0
    positive_index = loaders["positive_class_idx"]

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, positive_index)
        val_m   = evaluate(model, loaders["val"],   criterion, positive_index, top_k)
        train_m = evaluate(model, loaders["train"], criterion, positive_index, top_k)
        scheduler.step(val_m["loss"])

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_m["loss"])
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_m["acc"])
        history["train_ap"].append(train_m["ap"])
        history["val_ap"].append(val_m["ap"])
        history["val_p_at_100"].append(val_m["p_at_100"])

        ap_str = "n/a" if np.isnan(val_m["ap"]) else f"{val_m['ap']:.4f}"
        print(
            f"Epoch [{epoch:>2}/{num_epochs}]  "
            f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  "
            f"Val Loss: {val_m['loss']:.4f}  Val Acc: {val_m['acc']:.4f}  "
            f"Val AP: {ap_str}  Val P@{top_k}: {val_m['p_at_100']:.4f}"
        )

        if np.isnan(val_m["ap"]):
            raise ValueError(
                "Validation AP is undefined — ensure val/ has both waste and no_waste images."
            )

        if val_m["ap"] > best_val_ap:
            best_val_ap = val_m["ap"]
            no_improve  = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ New best model saved (val_ap={val_m['ap']:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping triggered after {epoch} epochs.")
                break

    print(f"\nTraining complete. Best val AP: {best_val_ap:.4f}")
    print(f"Best model checkpoint: {best_ckpt}")
    return history


# =============================================================================
# 6. VISUALISATION
# =============================================================================

def plot_learning_curves(history_p1: dict, history_p2: dict = None) -> None:
    """
    Plots learning curves with train and test lines for each metric.
    Panel 1: Loss (train vs val)
    Panel 2: Average Precision (train vs test)
    Panel 3: P@100 (test only)
    """
    if history_p2 is not None:
        history    = {k: history_p1[k] + history_p2[k] for k in history_p1}
        phase1_end = len(history_p1["train_loss"])
    else:
        history    = history_p1
        phase1_end = None

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def _vline(ax):
        if phase1_end:
            ax.axvline(phase1_end, color="grey", linestyle=":", label="Phase 2 starts")

    # Panel 1: Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train loss", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val loss",   linewidth=2, linestyle="--")
    _vline(ax)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss curves")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 2: Average Precision — train vs val (learning curve)
    ax = axes[1]
    ax.plot(epochs, history["train_ap"], label="Train AP", linewidth=2)
    ax.plot(epochs, history["val_ap"],   label="Val AP",   linewidth=2, linestyle="--")
    _vline(ax)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Average Precision")
    ax.set_title("Learning curve — Average Precision")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 3: Val P@100 over training
    ax = axes[2]
    ax.plot(epochs, history["val_p_at_100"], label="Val P@100", linewidth=2, color="tab:green")
    _vline(ax)
    ax.set_xlabel("Epoch"); ax.set_ylabel("P@100")
    ax.set_title("Val P@100 over training")
    ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Learning curves — Waste Detection ({BACKBONE})", fontsize=14)
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150, bbox_inches="tight")
    print("Learning curves saved to learning_curves.png")
    plt.show()


# =============================================================================
# 7. TEST SET EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list,
    top_k: int = TOP_K,
) -> None:
    model.eval()
    positive_index = class_names.index("waste")
    negative_index = class_names.index("no_waste")

    all_labels_idx, all_probs = [], []
    for images, labels in test_loader:
        images = images.to(DEVICE)
        probs  = torch.sigmoid(model(images)).squeeze(1)
        all_labels_idx.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_labels_idx = np.array(all_labels_idx)
    all_probs      = np.array(all_probs)
    binary_labels  = (all_labels_idx == positive_index).astype(int)
    binary_preds   = (all_probs >= 0.5).astype(int)
    class_preds    = _class_indices_from_binary(binary_preds, positive_index, negative_index)

    print("=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(classification_report(all_labels_idx, class_preds, target_names=class_names))

    test_ap    = average_precision_safe(binary_labels, all_probs)
    test_p_at_k = precision_at_k(binary_labels, all_probs, top_k)
    print(f"Test Average Precision : {test_ap:.4f}" if not np.isnan(test_ap) else "Test AP: n/a")
    print(f"Test P@{min(top_k, len(binary_labels))}          : {test_p_at_k:.4f}")

    cm   = confusion_matrix(all_labels_idx, class_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

    if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
        print("ROC AUC undefined — test set contains only one class.")
        return

    fpr, tpr, _ = roc_curve(binary_labels, all_probs)
    roc_auc     = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Test Set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# 8. LEADERBOARD — TOP-100 INFERENCE (runs on full unlabelled dataset)
# =============================================================================

class UnlabelledImageDataset(Dataset):
    """Loads all images from a directory tree without labels."""
    def __init__(self, root: Path, transform):
        self.paths     = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), str(self.paths[idx])


@torch.no_grad()
def generate_top100(
    model: nn.Module,
    raw_root: Path,
    output_dir: Path = Path("top100"),
    top_k: int = TOP_K,
) -> None:
    """
    Runs inference over all images in raw_root, ranks by waste confidence,
    and copies the top-k images to output_dir.
    Also writes a ranked CSV (all_scores.csv) for analysis.
    """
    if not raw_root.exists():
        print(f"WARNING: RAW_DATA_ROOT '{raw_root}' not found — skipping Top-100.")
        print("Set RAW_DATA_ROOT at the top of the script to your 10k image folder.")
        return

    model.eval()

    # Remove stale output dir to avoid permission errors from open files
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = UnlabelledImageDataset(raw_root, get_transforms("test"))
    if len(dataset) == 0:
        print(f"WARNING: No images found in '{raw_root}' — check the path.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Running inference on {len(dataset):,} images...")
    results = []

    for images, paths in loader:
        images = images.to(DEVICE)
        scores = torch.sigmoid(model(images)).squeeze(1)
        for score, path in zip(scores.cpu().numpy(), paths):
            results.append((float(score), path))

    results.sort(key=lambda x: x[0], reverse=True)

    # Save full ranked CSV
    scores_csv = output_dir / "all_scores.csv"
    with open(scores_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "score", "filename", "relative_path"])
        for rank, (score, path) in enumerate(results, start=1):
            try:
                rel_path = Path(path).relative_to(raw_root).as_posix()
            except ValueError:
                rel_path = path
            writer.writerow([rank, f"{score:.6f}", Path(path).name, rel_path])
    print(f"All scores saved to {scores_csv}")

    # Copy top-k images
    print(f"\nTop {top_k} detections:")
    print(f"{'Rank':<6} {'Score':<10} {'Filename'}")
    print("-" * 80)
    for rank, (score, path) in enumerate(results[:top_k], start=1):
        fname = Path(path).name
        dest  = output_dir / f"{rank:03d}_{fname}"
        shutil.copy2(path, dest)
        print(f"{rank:<6} {score:<10.4f} {fname}")

    print(f"\nTop {top_k} images saved to: {output_dir}")


# =============================================================================
# 9. YOLO LOCALIZATION BONUS — bounding box annotation (+5% extra credit)
# =============================================================================

# YOLO COCO classes that overlap with urban waste
YOLO_WASTE_CLASSES = {
    "bottle", "cup", "bowl", "handbag", "backpack", "suitcase",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "cell phone", "book", "vase",
    "banana", "apple", "orange", "carrot", "hot dog", "pizza",
    "donut", "cake", "sandwich",
}

# Objects that commonly cause FALSE POSITIVES in streetview waste detection.
# Bicycles especially trigger the ResNet because they appear in cluttered scenes.
# Shown in ORANGE so you can quickly spot and review suspicious detections.
YOLO_FP_CLASSES = {
    "bicycle", "motorcycle", "car", "truck", "bus", "train",
    "person", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse",
}


def annotate_with_yolo(
    input_dir: Path = Path("top100"),
    output_dir: Path = Path("top100_annotated"),
    yolo_model_size: str = "yolov8m.pt",
    conf_threshold: float = 0.25,
) -> None:
    """
    Runs YOLOv8 on all images in input_dir and draws bounding boxes.

    Box colours:
        RED    = waste-relevant COCO class (bottle, suitcase, etc.)
        ORANGE = likely false positive trigger (bicycle, person, car, etc.)
        GREY   = other detected objects

    Saves annotated images + a summary CSV (with waste + FP counts) to output_dir.

    Install requirement: pip install ultralytics opencv-python
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Run: pip install ultralytics opencv-python")
        return

    try:
        import cv2
    except ImportError:
        print("Run: pip install opencv-python")
        return

    if not input_dir.exists():
        print(f"'{input_dir}' not found — run generate_top100() first.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO model: {yolo_model_size}  (downloads on first run)")
    yolo = YOLO(yolo_model_size)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Annotating {len(image_paths)} images...")
    print("  RED = waste object | ORANGE = false positive trigger | GREY = other")
    summary = []

    for img_path in image_paths:
        img_cv = cv2.imread(str(img_path))
        if img_cv is None:
            continue

        results    = yolo(img_path, conf=conf_threshold, verbose=False)
        detections = results[0].boxes
        waste_count = 0
        fp_count    = 0

        if detections is not None and len(detections) > 0:
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_score = float(box.conf[0])
                label      = yolo.names[int(box.cls[0])]
                is_waste   = label.lower() in YOLO_WASTE_CLASSES
                is_fp      = label.lower() in YOLO_FP_CLASSES

                # RED = waste, ORANGE = FP trigger, GREY = other
                if is_waste:
                    color, thickness = (0, 0, 220), 2      # red (BGR)
                    waste_count += 1
                elif is_fp:
                    color, thickness = (0, 165, 255), 2    # orange (BGR)
                    fp_count += 1
                else:
                    color, thickness = (180, 180, 180), 1  # grey

                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(img_cv, f"{label} {conf_score:.2f}",
                            (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        cv2.imwrite(str(output_dir / img_path.name), img_cv)
        summary.append((img_path.name, waste_count, fp_count))

    summary_csv = output_dir / "yolo_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "waste_detections", "fp_class_detections"])
        writer.writerows(summary)

    waste_found = sum(1 for _, w, _ in summary if w > 0)
    fp_found    = sum(1 for _, _, fp in summary if fp > 0)
    print(f"\nYOLO annotation complete!")
    print(f"  Images with waste objects (red)   : {waste_found} / {len(image_paths)}")
    print(f"  Images with FP triggers (orange)  : {fp_found} / {len(image_paths)}")
    print(f"  Annotated images saved to         : {output_dir}")
    print(f"  Summary CSV                       : {summary_csv}")
    print("\n  Tip: images with ONLY orange boxes and no red = likely false positives")


# =============================================================================
# 10. FULL PIPELINE — end-to-end run
# =============================================================================

def run_full_pipeline() -> None:
    """
    Runs the complete pipeline:
        1.  Dataset statistics
        2.  Data loaders
        3.  Phase 1 — train classification head (backbone frozen)
        4.  Phase 2 — fine-tune head + last block
        5.  Training curves
        6.  Test set evaluation
        7.  Top-100 leaderboard submission
        8.  YOLO bounding box annotation (localization bonus)
    """
    print("STEP 1: Dataset statistics")
    _print_split_stats(DATASET_ROOT)

    print("\nSTEP 2: Creating data loaders...")
    loaders    = get_dataloaders(DATASET_ROOT, use_weighted_sampler=USE_WEIGHTED_SAMPLER)
    pos_weight = compute_class_weight(DATASET_ROOT) if USE_POS_WEIGHT else None

    print("\nSTEP 3: Phase 1 — training classification head...")
    model = build_model(freeze_backbone=True, backbone=BACKBONE)
    history_p1 = train(
        model=model, loaders=loaders,
        num_epochs=EPOCHS_HEAD, lr=LR_HEAD,
        checkpoint_name="phase1_best",
        patience=PATIENCE, pos_weight=pos_weight, top_k=TOP_K,
    )

    print("\nSTEP 4: Phase 2 — fine-tuning last block...")
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "phase1_best.pth", map_location=DEVICE))
    model = unfreeze_last_block(model, backbone=BACKBONE)
    history_p2 = train(
        model=model, loaders=loaders,
        num_epochs=EPOCHS_FINETUNE, lr=LR_FINETUNE,
        checkpoint_name="phase2_best",
        patience=PATIENCE, pos_weight=pos_weight, top_k=TOP_K,
    )

    print("\nSTEP 5: Plotting learning curves...")
    plot_learning_curves(history_p1, history_p2)

    print("\nSTEP 6: Evaluating on test set...")
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "phase2_best.pth", map_location=DEVICE))
    evaluate_test_set(model, loaders["test"], loaders["class_names"], top_k=TOP_K)

    print("\nSTEP 7: Generating leaderboard Top-100...")
    generate_top100(model=model, raw_root=RAW_DATA_ROOT, top_k=TOP_K)

    print("\nSTEP 8: YOLO bounding box annotation (localization bonus)...")
    annotate_with_yolo(
        input_dir=Path("top100"),
        output_dir=Path("top100_annotated"),
        yolo_model_size="yolov8m.pt",
        conf_threshold=0.25,
    )

    print("\nPipeline complete!")
    print("Outputs:")
    print("  learning_curves.png    — loss / AP (train vs val) / val P@100")
    print("  confusion_matrix.png   — test set confusion matrix")
    print("  roc_curve.png          — ROC curve with AUC")
    print("  top100/                — top 100 predicted waste images")
    print("  top100_annotated/      — same images with YOLO bounding boxes")
    print("  checkpoints/           — saved model weights")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_full_pipeline()

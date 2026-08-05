"""
Shared data utilities for CMMD breast cancer subtype classification.
(PyTorch version)
Provides:
    - CMMDDataset       : PyTorch Dataset for loading mammogram PNGs + labels
    - get_dataloaders   : stratified 70/15/15 train/val/test DataLoaders
    - get_class_weights : inverse-frequency weights for imbalanced classes
    - plot_training_curves : loss & accuracy over epochs
    - evaluate_model    : confusion matrix, classification report, saves to results/
Label mapping (5 classes):
    0 = Benign
    1 = Luminal A
    2 = Luminal B
    3 = HER2-enriched
    4 = triple negative
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from PIL import Image
# ── Label mapping ────────────────────────────────────────────────────────
LABEL_MAP = {
    "Benign": 0,
    "Luminal A": 1,
    "Luminal B": 2,
    "HER2-enriched": 3,
    "triple negative": 4,
}
CLASS_NAMES = list(LABEL_MAP.keys())
NUM_CLASSES = len(CLASS_NAMES)
# ── Paths (relative to project root) ────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "processed_data.csv")
DEFAULT_IMG_DIR = os.path.join(PROJECT_ROOT, "processed_images")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
# ── Dataset ──────────────────────────────────────────────────────────────
def crop_breast(arr, threshold_ratio=0.05, margin=10):
    """Crop black background out of mammogram to focus on breast tissue."""
    mask = arr > (arr.max() * threshold_ratio)
    if not np.any(mask):
        return arr
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    h, w = arr.shape
    ymin, ymax = max(0, ymin - margin), min(h - 1, ymax + margin)
    xmin, xmax = max(0, xmin - margin), min(w - 1, xmax + margin)
    return arr[ymin:ymax + 1, xmin:xmax + 1]


class CMMDDataset(Dataset):
    """
    PyTorch Dataset for CMMD mammography images.
    Reads a DataFrame, loads 16-bit grayscale PNGs,
    crops background, converts to 3-channel RGB, and applies transforms.
    """
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.targets = [LABEL_MAP[lbl] for lbl in self.df["label"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_filename"])
        img = Image.open(img_path).convert("I")
        arr = np.array(img, dtype=np.float32)

        # Crop background
        arr = crop_breast(arr)

        # Min-Max normalize breast tissue
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)
        arr = arr.astype(np.uint8)

        img = Image.fromarray(arr, mode="L").convert("RGB")
        if self.transform:
            img = self.transform(img)

        target = self.targets[idx]
        return img, target


# ── Dataloaders ──────────────────────────────────────────────────────────
def get_dataloaders(
    csv_path=DEFAULT_CSV,
    img_dir=DEFAULT_IMG_DIR,
    train_transform=None,
    val_transform=None,
    batch_size=32,
    seed=42,
    num_workers=0,
    use_sampler=True,
):
    """
    Create train/val/test DataLoaders with stratified 70/15/15 split.
    Uses WeightedRandomSampler for balanced batch training if use_sampler=True.
    """
    df = pd.read_csv(csv_path)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"]
    )
    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")
    print(f"Train label counts: {dict(Counter(train_df['label']))}")

    train_ds = CMMDDataset(train_df, img_dir, transform=train_transform)
    val_ds = CMMDDataset(val_df, img_dir, transform=val_transform)
    test_ds = CMMDDataset(test_df, img_dir, transform=val_transform)

    # Compute sampler / class weights
    weights = get_class_weights(train_ds.targets)

    if use_sampler:
        sample_weights = [weights[t].item() for t in train_ds.targets]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, weights
# ── Class weights ────────────────────────────────────────────────────────
def get_class_weights(targets):
    """
    Compute inverse-frequency class weights for imbalanced data.
    Args:
        targets : list of integer class labels
    Returns:
        torch.FloatTensor of shape (NUM_CLASSES,)
    """
    counts = Counter(targets)
    total = len(targets)
    weights = []
    for c in range(NUM_CLASSES):
        w = total / (NUM_CLASSES * counts.get(c, 1))
        weights.append(w)
    weights = torch.FloatTensor(weights)
    print(f"Class weights: {dict(zip(CLASS_NAMES, weights.tolist()))}")
    return weights
# ── Training curves ──────────────────────────────────────────────────────
def plot_training_curves(history, save_path=None):
    """
    Plot training/validation loss and accuracy curves.
    Args:
        history : dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path : optional path to save the figure
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Loss
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy over Epochs")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")
    plt.show()
# ── Evaluation ───────────────────────────────────────────────────────────
def evaluate_model(model, test_loader, device, model_name="model", save_dir=RESULTS_DIR):
    """
    Evaluate a trained model on the test set.
    Prints classification report, plots confusion matrix,
    and saves results to save_dir.
    Args:
        model       : trained PyTorch model
        test_loader : DataLoader for test set
        device      : torch device
        model_name  : string name used for file naming
        save_dir    : directory to save results
    Returns:
        accuracy (float), report (str)
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # Accuracy
    acc = accuracy_score(all_labels, all_preds) * 100
    print(f"\n{'='*50}")
    print(f"  {model_name} — Test Accuracy: {acc:.2f}%")
    print(f"{'='*50}\n")
    # Classification report
    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, digits=4
    )
    print(report)
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    # Save
    os.makedirs(save_dir, exist_ok=True)
    cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"Saved confusion matrix to {cm_path}")
    plt.show()
    # Save report as text
    report_path = os.path.join(save_dir, f"{model_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {acc:.2f}%\n\n")
        f.write(report)
    print(f"Saved report to {report_path}")
    return acc, report
"""
ResNet101 — Breast Cancer Subtype Classification
=================================================

Train a ResNet101 model (pretrained on ImageNet) to classify CMMD mammograms
into 5 subtypes: Benign, Luminal A, Luminal B, HER2-enriched, Triple Negative.

Optimizations:
  - Automatic breast region cropping (strips out ~85% blank background space)
  - Balanced batch sampling via WeightedRandomSampler
  - Differential learning rates: 1e-5 for backbone (layer3 + layer4), 3e-4 for FC head
  - CosineAnnealingLR scheduler & AdamW optimizer
  - Label smoothing CrossEntropyLoss

Usage:
    python src/train_resnet101.py
"""

import sys
import os

# Add src directory and project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data_utils import (
    get_dataloaders, plot_training_curves,
    evaluate_model, CLASS_NAMES, NUM_CLASSES, MODELS_DIR, RESULTS_DIR,
)

# ── Config ───────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR_HEAD = 3e-4
LR_BACKBONE = 1e-5
SEED = 42
MODEL_NAME = "resnet101"


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Step 1: Data Augmentation & Loading ──────────────────────────────────
    print("\n" + "=" * 50)
    print("Step 1: Loading Data (with Breast Cropping & Balanced Sampler)")
    print("=" * 50)

    # Training augmentations
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Validation/Test — no augmentation, just resize + normalize
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load data (stratified 70/15/15 with WeightedRandomSampler for balanced batches)
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=BATCH_SIZE,
        seed=SEED,
        num_workers=0,
        use_sampler=True,
    )

    # ── Step 2: Visualize Sample Images ─────────────────────────────────────
    print("\n" + "=" * 50)
    print("Step 2: Saving Sample Cropped Images")
    print("=" * 50)

    images, labels = next(iter(train_loader))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            break
        # Undo normalization for display
        img = images[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(CLASS_NAMES[labels[i]], fontsize=12)
        ax.axis("off")
    plt.suptitle("Sample Cropped Training Mammograms", fontsize=16)
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sample_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_sample_images.png")
    plt.savefig(sample_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sample images to {sample_path}")

    # ── Step 3: Define ResNet101 Model ───────────────────────────────────────
    print("\n" + "=" * 50)
    print("Step 3: Building ResNet101 Model with Differential Fine-Tuning")
    print("=" * 50)

    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)

    # Freeze conv1, bn1, layer1, layer2; unfreeze layer3 and layer4 for fine-tuning
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    # Classifier head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES),
    )

    model = model.to(device)

    # Differential parameter groups
    backbone_params = list(model.layer3.parameters()) + list(model.layer4.parameters())
    head_params = list(model.fc.parameters())

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,} (layer3 + layer4 + FC head)")

    # ── Step 4: Optimizer, Loss & Scheduler ─────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ── Step 5: Training Loop ───────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Step 5: Training")
    print("=" * 50)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_cmmd.pkl")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{EPOCHS}] Train", leave=False)
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_pbar.set_postfix(
                loss=f"{running_loss / total:.4f}",
                acc=f"{correct / total * 100:.2f}%"
            )

        train_loss = running_loss / total
        train_acc = correct / total * 100

        # ── Validate ──
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch:02d}/{EPOCHS}] Val  ", leave=False)
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                val_pbar.set_postfix(
                    loss=f"{running_loss / total:.4f}",
                    acc=f"{correct / total * 100:.2f}%"
                )

        val_loss = running_loss / total
        val_acc = correct / total * 100

        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%"
            + (" \u2605" if val_acc >= best_val_acc else "")
        )

    print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")

    # ── Step 6: Plot Training Curves ────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Step 6: Training Curves")
    print("=" * 50)

    curves_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_training_curves.png")
    plot_training_curves(history, save_path=curves_path)

    # ── Step 7: Evaluate on Test Set ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Step 7: Evaluation on Test Set")
    print("=" * 50)

    # Load best checkpoint
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.to(device)

    # Evaluate
    accuracy, report = evaluate_model(
        model, test_loader, device,
        model_name=MODEL_NAME,
        save_dir=RESULTS_DIR,
    )

    # ── Step 8: Per-Class Accuracy Bar Chart ─────────────────────────────────
    print("\n" + "=" * 50)
    print("Step 8: Per-Class Accuracy")
    print("=" * 50)

    from sklearn.metrics import confusion_matrix as cm_func

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = cm_func(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(CLASS_NAMES, per_class_acc,
                  color=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("ResNet101 \u2014 Per-Class Test Accuracy")
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", fontsize=11)
    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_per_class_accuracy.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {bar_path}")

    print("\n" + "=" * 50)
    print("Done! All results saved to results/ folder.")
    print("=" * 50)


if __name__ == "__main__":
    main()

"""
SceneNet Training Script v2 — With class-weighted loss for imbalanced styles.

Improvements over v1:
  - Inverse-frequency class weighting for style head (modern=3617 vs bohemian=39)
  - Differential learning rates: lower LR for backbone, higher for heads
  - Gradient clipping to stabilize training
  - Early stopping with patience
  - Saves class weight tensors in checkpoint for reproducibility

Expected training time: ~60 min on Apple MPS (20 epochs).
Output: models/scene_net.pth
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.scene_net import (
    SceneNet, SceneNetLoss, compute_class_weights,
    ROOM_TYPES, STYLES, LIGHTING_LEVELS, CONDITIONS,
)
from training.scene_dataset import SceneNetDataset


def train():
    # ─── Config ───
    EPOCHS = 20
    BATCH_SIZE = 32
    BACKBONE_LR = 1e-4     # Lower LR for pretrained backbone
    HEAD_LR = 5e-4          # Higher LR for randomly initialized heads
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 0         # Set to 0 for MPS compatibility
    IMAGE_SIZE = 224
    PATIENCE = 6            # Early stopping patience
    GRAD_CLIP = 1.0
    SAVE_PATH = PROJECT_ROOT / "models" / "scene_net.pth"

    # ─── Device ───
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ─── Data ───
    print("\n📦 Loading datasets...")
    train_ds = SceneNetDataset(str(PROJECT_ROOT), split="train", image_size=IMAGE_SIZE, augment=True)
    val_ds = SceneNetDataset(str(PROJECT_ROOT), split="val", image_size=IMAGE_SIZE, augment=False)

    if len(train_ds) == 0:
        print("❌ No training data found! Check your data/datasets/ directory.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    print(f"Train: {len(train_ds)} samples ({len(train_loader)} batches)")
    print(f"Val:   {len(val_ds)} samples ({len(val_loader)} batches)")

    # ─── Compute Class Weights ───
    print("\n⚖️  Computing class weights for imbalanced data...")
    style_weights = compute_class_weights(train_ds, "style", len(STYLES)).to(device)
    room_weights = compute_class_weights(train_ds, "room_type", len(ROOM_TYPES)).to(device)

    print("  Style weights:")
    for i, s in enumerate(STYLES):
        print(f"    {s:15s}: {style_weights[i]:.2f}")
    print("  Room weights:")
    for i, r in enumerate(ROOM_TYPES):
        print(f"    {r:15s}: {room_weights[i]:.2f}")

    # ─── Model ───
    print("\n🧠 Building SceneNet...")
    model = SceneNet(pretrained=True).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    # ─── Loss with Class Weights ───
    criterion = SceneNetLoss(
        weight_room=1.0,
        weight_style=2.0,       # Bump style weight since it's hardest
        weight_light=0.8,
        weight_palette=0.5,
        weight_condition=0.8,
        style_class_weights=style_weights,
        room_class_weights=room_weights,
    )

    # ─── Differential Learning Rates ───
    backbone_params = list(model.features.parameters())
    head_params = (
        list(model.room_type_head.parameters()) +
        list(model.style_head.parameters()) +
        list(model.lighting_head.parameters()) +
        list(model.palette_head.parameters()) +
        list(model.condition_head.parameters())
    )

    optimizer = AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ─── Training Loop ───
    best_val_acc = 0.0
    patience_counter = 0
    print(f"\n🚀 Training for {EPOCHS} epochs (patience={PATIENCE})...\n")

    for epoch in range(EPOCHS):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = {"room_type": 0, "style": 0, "lighting": 0, "condition": 0}
        train_total = {"room_type": 0, "style": 0, "lighting": 0, "condition": 0}

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            targets = {
                "room_type": batch["room_type"].to(device),
                "style": batch["style"].to(device),
                "lighting": batch["lighting"].to(device),
                "palette": batch["palette"].to(device),
                "condition": batch["condition"].to(device),
            }

            optimizer.zero_grad()
            predictions = model(images)
            loss, loss_dict = criterion(predictions, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            train_loss += loss.item()

            # Track accuracy per head
            for key in ["room_type", "style", "lighting", "condition"]:
                mask = targets[key] != -1
                if mask.any():
                    preds = predictions[key][mask].argmax(dim=1)
                    correct = (preds == targets[key][mask]).sum().item()
                    train_correct[key] += correct
                    train_total[key] += mask.sum().item()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = {"room_type": 0, "style": 0, "lighting": 0, "condition": 0}
        val_total = {"room_type": 0, "style": 0, "lighting": 0, "condition": 0}

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = {
                    "room_type": batch["room_type"].to(device),
                    "style": batch["style"].to(device),
                    "lighting": batch["lighting"].to(device),
                    "palette": batch["palette"].to(device),
                    "condition": batch["condition"].to(device),
                }

                predictions = model(images)
                loss, _ = criterion(predictions, targets)
                val_loss += loss.item()

                for key in ["room_type", "style", "lighting", "condition"]:
                    mask = targets[key] != -1
                    if mask.any():
                        preds = predictions[key][mask].argmax(dim=1)
                        correct = (preds == targets[key][mask]).sum().item()
                        val_correct[key] += correct
                        val_total[key] += mask.sum().item()

        # ── Metrics ──
        scheduler.step()
        elapsed = time.time() - t0

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / max(len(val_loader), 1)

        train_accs = {k: (train_correct[k] / train_total[k] * 100 if train_total[k] > 0 else 0) for k in train_correct}
        val_accs = {k: (val_correct[k] / val_total[k] * 100 if val_total[k] > 0 else 0) for k in val_correct}

        # Combined accuracy (average of room_type + style as primary metric)
        combined_val_acc = (val_accs["room_type"] + val_accs["style"]) / 2

        # ── Logging ──
        print(f"Epoch {epoch+1:2d}/{EPOCHS} ({elapsed:.0f}s) | "
              f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
              f"Room: {train_accs['room_type']:.0f}%/{val_accs['room_type']:.0f}% | "
              f"Style: {train_accs['style']:.0f}%/{val_accs['style']:.0f}% | "
              f"Light: {train_accs['lighting']:.0f}%/{val_accs['lighting']:.0f}% | "
              f"Cond: {train_accs['condition']:.0f}%/{val_accs['condition']:.0f}%")

        # ── Save Best ──
        if combined_val_acc > best_val_acc:
            best_val_acc = combined_val_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_acc": {
                    "room_type": val_accs["room_type"],
                    "style": val_accs["style"],
                    "lighting": val_accs["lighting"],
                    "condition": val_accs["condition"],
                    "combined": combined_val_acc,
                },
                "class_weights": {
                    "style": style_weights.cpu(),
                    "room_type": room_weights.cpu(),
                },
                "config": {
                    "image_size": IMAGE_SIZE,
                    "num_room_types": len(ROOM_TYPES),
                    "num_styles": len(STYLES),
                    "num_lighting": len(LIGHTING_LEVELS),
                    "num_conditions": len(CONDITIONS),
                },
            }, SAVE_PATH)
            print(f"  ✅ Best model saved! Combined val acc: {combined_val_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break

    print(f"\n🎉 Training complete! Best combined val accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {SAVE_PATH}")

    # ── Final inference benchmark ──
    print("\n⚡ Benchmarking inference speed...")
    # Load best model for benchmark
    checkpoint = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fake_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(5):
        _ = model.infer(fake_img, device=str(device))
    
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        result = model.infer(fake_img, device=str(device))
        times.append((time.perf_counter() - t0) * 1000)
    
    print(f"Inference: {np.mean(times):.1f}ms avg (±{np.std(times):.1f}ms)")
    print(f"  → {1000/np.mean(times):.0f} FPS real-time capability")

    # Test on a real image
    print("\n📸 Testing on a real image...")
    import glob
    test_images = glob.glob(str(PROJECT_ROOT / "data" / "datasets" / "raw" / "*.jpg"))[:3]
    if test_images:
        import cv2
        for img_path in test_images:
            img = cv2.imread(img_path)
            if img is not None:
                result = model.infer(img, device=str(device))
                fname = Path(img_path).name
                print(f"  {fname}: {result.room_type} ({result.room_type_confidence:.0%}), "
                      f"style={result.style} ({result.style_confidence:.0%}), "
                      f"light={result.lighting}")


if __name__ == "__main__":
    train()

# ============================================================
# 1. IMPORTS
# ============================================================

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 2

IMAGE_SIZE = 32
NUM_CLASSES = 10

MODEL_PATH = "best_model.pth"


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("SYSTEM INFORMATION")
print("=" * 60)

print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
print(f"Device          : {device}")

if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory      : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print("=" * 60)


# ============================================================
# 5. DATA TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


val_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


# ============================================================
# 6. DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

val_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=val_transform
)

print(f"Training samples   : {len(train_dataset)}")
print(f"Validation samples : {len(val_dataset)}")


# ============================================================
# 7. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 8. MODEL
# ============================================================

class CNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 4 * 4, 256),

            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = CNN(NUM_CLASSES).to(device)

print("\n" + "=" * 60)
print("MODEL")
print("=" * 60)

print(model)


# ============================================================
# 9. LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 10. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# 11. LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# 12. TRAIN FUNCTION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Clear gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 13. VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 14. TRAINING LOOP
# ============================================================

print("\n" + "=" * 60)
print("START TRAINING")
print("=" * 60)

best_val_accuracy = 0.0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}


for epoch in range(EPOCHS):

    start_time = time.time()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loss, val_accuracy = validate(
        model,
        val_loader,
        criterion
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step()

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    epoch_time = time.time() - start_time

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nEpoch [{epoch + 1:02d}/{EPOCHS}] "
        f"| Time: {epoch_time:.1f}s"
    )

    print(
        f"Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss:   {val_loss:.4f} "
        f"| Val Acc:   {val_accuracy * 100:.2f}%"
    )

    print(
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    # --------------------------------------------------------
    # Best Model Checkpoint
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
                "val_loss": val_loss,
            },
            MODEL_PATH
        )

        print(
            f"✓ Best model saved "
            f"(Val Acc: {val_accuracy * 100:.2f}%)"
        )


# ============================================================
# 15. FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(f"Model saved to: {MODEL_PATH}")


# ============================================================
# 16. LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("\nBest model loaded successfully.")


# ============================================================
# 17. FINAL VALIDATION
# ============================================================

final_loss, final_accuracy = validate(
    model,
    val_loader,
    criterion
)

print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

print(f"Loss     : {final_loss:.4f}")
print(f"Accuracy : {final_accuracy * 100:.2f}%")

print("=" * 60)

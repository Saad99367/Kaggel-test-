# ============================================================
# 1. IMPORTS
# ============================================================

import time
import random
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001

MODEL_PATH = "iris_model.pth"


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
print("SYSTEM")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 60)


# ============================================================
# 5. LOAD IRIS DATASET
# ============================================================

print("\nLoading Iris dataset...")

iris = load_iris()

X = iris.data
y = iris.target

print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Classes:", len(np.unique(y)))


# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)


# ============================================================
# 7. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# ============================================================
# 8. CONVERT TO PYTORCH TENSORS
# ============================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.long
)


# ============================================================
# 9. DATASETS
# ============================================================

train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)


# ============================================================
# 10. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 11. MODEL
# ============================================================

class IrisModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(4, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 3)
        )

    def forward(self, x):

        return self.network(x)


model = IrisModel().to(device)

print("\nMODEL")
print("=" * 60)

print(model)


# ============================================================
# 12. LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 14. TRAIN FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.size(0)

    accuracy = correct / total

    return total_loss / len(train_loader), accuracy


# ============================================================
# 15. VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in val_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.size(0)

    accuracy = correct / total

    return total_loss / len(val_loader), accuracy


# ============================================================
# 16. TRAINING
# ============================================================

print("\n")
print("=" * 60)
print("START TRAINING")
print("=" * 60)

best_accuracy = 0

start_time = time.time()


for epoch in range(EPOCHS):

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc = validate()

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"| "
        f"Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_acc * 100:.2f}% "
        f"| "
        f"Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_acc * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save Best Model
    # --------------------------------------------------------

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "accuracy": val_acc,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_
            },
            MODEL_PATH
        )

        print(
            f"   ✓ Best model saved "
            f"({val_acc * 100:.2f}%)"
        )


# ============================================================
# 17. FINAL RESULTS
# ============================================================

training_time = time.time() - start_time

print("\n")
print("=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Training Time: "
    f"{training_time:.2f} seconds"
)

print(
    f"Model: {MODEL_PATH}"
)

print("=" * 60)

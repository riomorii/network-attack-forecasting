import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================
# CONFIG
# ============================================

BATCH_SIZE = 256
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LEARNING_RATE = 0.001
EPOCHS = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================
# LOAD DATA
# ============================================

X_train = np.load(
    "data/multiclass_3class_X_train.npy"
)

y_train = np.load(
    "data/multiclass_3class_y_train.npy"
)

X_val = np.load(
    "data/multiclass_3class_X_val.npy"
)

y_val = np.load(
    "data/multiclass_3class_y_val.npy"
)

print("===============================================")
print("MULTICLASS LSTM WORLD MODEL")
print("===============================================")

print("Train:", X_train.shape)
print("Val:  ", X_val.shape)

print("Device:", DEVICE)

# ============================================
# TORCH DATASETS
# ============================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.long
)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

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

# ============================================
# MODEL
# ============================================

class MulticlassLSTM(nn.Module):

    def __init__(
        self,
        input_size=15,
        hidden_size=64,
        num_layers=1,
        num_classes=3
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Last timestep
        last_hidden = output[:, -1, :]

        logits = self.classifier(
            last_hidden
        )

        return logits


model = MulticlassLSTM(
    input_size=15,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=3
).to(DEVICE)

# ============================================
# LOSS + OPTIMIZER
# ============================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================
# TRAINING
# ============================================

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(
            logits,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item() *
            X_batch.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        train_correct += (
            (predictions == y_batch)
            .sum()
            .item()
        )

        train_total += y_batch.size(0)

    train_loss /= train_total

    train_accuracy = (
        train_correct /
        train_total
    )

    # ========================================
    # VALIDATION
    # ========================================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch
            )

            val_loss += (
                loss.item() *
                X_batch.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            val_correct += (
                (predictions == y_batch)
                .sum()
                .item()
            )

            val_total += y_batch.size(0)

    val_loss /= val_total

    val_accuracy = (
        val_correct /
        val_total
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

# ============================================
# SAVE
# ============================================

torch.save(
    model.state_dict(),
    "models/multiclass_lstm.pth"
)

print("\n===============================================")
print("MODEL SAVED")
print("===============================================")

print(
    "models/multiclass_lstm.pth"
)
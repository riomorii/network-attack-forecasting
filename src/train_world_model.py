import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. Load prepared network-state sequences
# ============================================================

X_train = np.load("data/state_sequences_X_train.npy")
y_train = np.load("data/state_sequences_y_train.npy")

X_val = np.load("data/state_sequences_X_val.npy")
y_val = np.load("data/state_sequences_y_val.npy")

print("Training input shape :", X_train.shape)
print("Training target shape:", y_train.shape)

print("Validation input shape :", X_val.shape)
print("Validation target shape:", y_val.shape)


# ============================================================
# 2. Convert NumPy arrays to PyTorch tensors
# ============================================================

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)


# ============================================================
# 3. Create DataLoaders
# ============================================================

BATCH_SIZE = 256

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

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
# 4. Define Network-State World Model
# ============================================================

class NetworkStateWorldModel(nn.Module):

    def __init__(self, input_size=15, hidden_size=64, output_size=15):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        # Use the final time step
        last_hidden = output[:, -1, :]

        prediction = self.fc(last_hidden)

        return prediction


# ============================================================
# 5. Select CPU/GPU
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 6. Create model
# ============================================================

model = NetworkStateWorldModel(
    input_size=15,
    hidden_size=64,
    output_size=15
)

model = model.to(device)


# ============================================================
# 7. Loss and optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 8. Training
# ============================================================

EPOCHS = 10

print("\nStarting World Model training...\n")

for epoch in range(EPOCHS):

    # -------------------------
    # Training
    # -------------------------

    model.train()

    total_train_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        prediction = model(X_batch)

        loss = criterion(
            prediction,
            y_batch
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss / len(train_loader)
    )


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            prediction = model(X_batch)

            loss = criterion(
                prediction,
                y_batch
            )

            total_val_loss += loss.item()

    average_val_loss = (
        total_val_loss / len(val_loader)
    )


    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Train Loss: {average_train_loss:.6f} "
        f"- Val Loss: {average_val_loss:.6f}"
    )


# ============================================================
# 9. Save model
# ============================================================

torch.save(
    model.state_dict(),
    "models/network_state_world_model.pth"
)

print("\nTraining complete.")

print(
    "Model saved to: "
    "models/network_state_world_model.pth"
)
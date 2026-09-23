import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading training data...")

X_train = np.load("data/X_train_scaled.npy")

print("Training shape:", X_train.shape)


# ============================================================
# 2. CREATE INPUT AND TARGET
# ============================================================
#
# Input:
#   flows 0 -> 9
#
# Target:
#   flow 10
#
# Example:
#
# [flow1, flow2, ..., flow10] -> flow11
#
# The model learns to predict the next network state.
# ============================================================

X = X_train[:, :-1, :]
Y = X_train[:, -1, :]

print()
print("Input shape:", X.shape)
print("Target shape:", Y.shape)


# ============================================================
# 3. CONVERT TO PYTORCH TENSORS
# ============================================================

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True
)


# ============================================================
# 4. DEFINE WORLD MODEL
# ============================================================

class WorldModel(nn.Module):

    def __init__(self, input_size=32, hidden_size=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.output = nn.Linear(
            hidden_size,
            input_size
        )

    def forward(self, x):

        lstm_output, _ = self.lstm(x)

        last_output = lstm_output[:, -1, :]

        prediction = self.output(last_output)

        return prediction


# ============================================================
# 5. CREATE MODEL
# ============================================================

print()
print("Creating World Model...")

model = WorldModel(
    input_size=32,
    hidden_size=64
)

print(model)


# ============================================================
# 6. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 7. TRAIN
# ============================================================

EPOCHS = 5

print()
print("Starting training...")
print("=" * 60)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch_X, batch_Y in loader:

        optimizer.zero_grad()

        prediction = model(batch_X)

        loss = criterion(
            prediction,
            batch_Y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.6f}"
    )


# ============================================================
# 8. SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "models/world_model.pth"
)

print()
print("=" * 60)
print("WORLD MODEL TRAINING COMPLETE")
print("=" * 60)

print()
print("Model saved to:")
print("models/world_model.pth")
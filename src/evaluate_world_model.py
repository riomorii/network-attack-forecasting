import numpy as np
import torch
import torch.nn as nn


# ============================================================
# WORLD MODEL
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
# LOAD TEST DATA
# ============================================================

print("Loading test data...")

X_test = np.load("data/X_test_scaled.npy")

print("Test shape:", X_test.shape)


# ============================================================
# CREATE INPUT / TARGET
# ============================================================

X = X_test[:, :-1, :]
Y = X_test[:, -1, :]

print()
print("Input shape:", X.shape)
print("Target shape:", Y.shape)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading World Model...")

model = WorldModel()

model.load_state_dict(
    torch.load(
        "models/world_model.pth",
        map_location="cpu"
    )
)

model.eval()

print("Model loaded successfully!")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

print()
print("Generating predictions...")

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)

with torch.no_grad():

    predictions = model(X_tensor).numpy()


# ============================================================
# CALCULATE ERRORS
# ============================================================

errors = np.mean(
    (predictions - Y) ** 2,
    axis=1
)

mae = np.mean(
    np.abs(predictions - Y)
)

mse = np.mean(
    (predictions - Y) ** 2
)

rmse = np.sqrt(mse)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("WORLD MODEL EVALUATION")
print("=" * 60)

print()
print("Mean Absolute Error (MAE):", f"{mae:.6f}")
print("Mean Squared Error (MSE):", f"{mse:.6f}")
print("Root Mean Squared Error (RMSE):", f"{rmse:.6f}")

print()
print("Average sample prediction error:", f"{np.mean(errors):.6f}")
print("Minimum sample error:", f"{np.min(errors):.6f}")
print("Maximum sample error:", f"{np.max(errors):.6f}")

print()
print("=" * 60)
print("WORLD MODEL TEST COMPLETE")
print("=" * 60)
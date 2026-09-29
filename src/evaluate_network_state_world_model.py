import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. Load test data
# ============================================================

X_test = np.load(
    "data/state_sequences_X_test.npy"
)

y_test = np.load(
    "data/state_sequences_y_test.npy"
)

print("Test input shape :", X_test.shape)
print("Test target shape:", y_test.shape)


# ============================================================
# 2. Define the same World Model architecture
# ============================================================

class NetworkStateWorldModel(nn.Module):

    def __init__(
        self,
        input_size=15,
        hidden_size=64,
        output_size=15
    ):

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

        last_hidden = output[:, -1, :]

        prediction = self.fc(last_hidden)

        return prediction


# ============================================================
# 3. Load trained model
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = NetworkStateWorldModel()

model.load_state_dict(
    torch.load(
        "models/network_state_world_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ============================================================
# 4. Convert test data to tensors
# ============================================================

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(device)


# ============================================================
# 5. Generate predictions
# ============================================================

with torch.no_grad():

    predictions = model(
        X_test_tensor
    ).cpu().numpy()


# ============================================================
# 6. Calculate evaluation metrics
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)


print("\n========================================")
print("NETWORK-STATE WORLD MODEL EVALUATION")
print("========================================")

print(
    f"MAE  : {mae:.6f}"
)

print(
    f"MSE  : {mse:.6f}"
)

print(
    f"RMSE : {rmse:.6f}"
)

print("========================================")


# ============================================================
# 7. Save predictions
# ============================================================

np.save(
    "data/network_state_predictions.npy",
    predictions
)

print(
    "\nPredictions saved to:"
    " data/network_state_predictions.npy"
)
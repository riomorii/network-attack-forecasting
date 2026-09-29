import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# Configuration
# ============================================================

FORECAST_STEPS = 5

MODEL_PATH = "models/network_state_world_model.pth"

X_TEST_PATH = "data/state_sequences_X_test.npy"
Y_TEST_PATH = "data/state_sequences_y_test.npy"


# ============================================================
# World Model
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

        return self.fc(last_hidden)


# ============================================================
# Load test data
# ============================================================

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# Load model
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = NetworkStateWorldModel()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# ============================================================
# Evaluate autoregressive rollout
# ============================================================

all_predictions = []

all_targets = []


for sample_index in range(len(X_test) - FORECAST_STEPS):

    current_sequence = X_test[sample_index].copy()

    predictions = []
    targets = []

    for step in range(FORECAST_STEPS):

        input_tensor = torch.tensor(
            current_sequence,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            predicted_state = model(
                input_tensor
            ).cpu().numpy()[0]

        predictions.append(
            predicted_state
        )

        # The actual future target
        target_state = y_test[
            sample_index + step
        ]

        targets.append(
            target_state
        )

        # Feed prediction back
        current_sequence = np.concatenate(
            [
                current_sequence[1:],
                predicted_state.reshape(1, -1)
            ],
            axis=0
        )

    all_predictions.append(predictions)
    all_targets.append(targets)


all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)


# ============================================================
# Print shapes
# ============================================================

print("\nPrediction shape:")
print(all_predictions.shape)

print("\nTarget shape:")
print(all_targets.shape)


# ============================================================
# Evaluate each forecast horizon
# ============================================================

print("\n========================================")
print("MULTI-STEP WORLD MODEL EVALUATION")
print("========================================")

for step in range(FORECAST_STEPS):

    predictions = all_predictions[:, step, :]
    targets = all_targets[:, step, :]

    mae = mean_absolute_error(
        targets,
        predictions
    )

    mse = mean_squared_error(
        targets,
        predictions
    )

    rmse = np.sqrt(mse)

    print(
        f"Horizon +{step + 1}: "
        f"MAE={mae:.6f} "
        f"RMSE={rmse:.6f}"
    )


print("========================================")


# ============================================================
# Save rollout evaluation data
# ============================================================

np.save(
    "data/multi_step_predictions.npy",
    all_predictions
)

np.save(
    "data/multi_step_targets.npy",
    all_targets
)

print(
    "\nSaved:"
)

print(
    "data/multi_step_predictions.npy"
)

print(
    "data/multi_step_targets.npy"
)
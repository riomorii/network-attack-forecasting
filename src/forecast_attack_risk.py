import numpy as np
import torch
import torch.nn as nn
import joblib


# ==================================================
# Configuration
# ==================================================

FORECAST_STEPS = 5

WORLD_MODEL_PATH = "models/network_state_world_model.pth"
RISK_MODEL_PATH = "models/attack_risk_model.pkl"

X_TEST_PATH = "data/state_sequences_X_test.npy"


# ==================================================
# World Model
# ==================================================

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


# ==================================================
# Load data
# ==================================================

X_test = np.load(X_TEST_PATH)

print("========================================")
print("ATTACK RISK FORECAST")
print("========================================")

print("Test sequences:", X_test.shape)


# ==================================================
# Device
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ==================================================
# Load World Model
# ==================================================

world_model = NetworkStateWorldModel()

world_model.load_state_dict(
    torch.load(
        WORLD_MODEL_PATH,
        map_location=device
    )
)

world_model = world_model.to(device)
world_model.eval()


# ==================================================
# Load Attack Risk Model
# ==================================================

risk_model = joblib.load(
    RISK_MODEL_PATH
)


# ==================================================
# Select one test sequence
# ==================================================

sequence_index = 0

current_sequence = X_test[
    sequence_index
].copy()

print(
    "\nStarting sequence shape:",
    current_sequence.shape
)


# ==================================================
# Forecast future states
# ==================================================

future_states = []

attack_risks = []


for step in range(FORECAST_STEPS):

    input_tensor = torch.tensor(
        current_sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(device)


    # ----------------------------------------------
    # Predict next network state
    # ----------------------------------------------

    with torch.no_grad():

        predicted_state = (
            world_model(input_tensor)
            .cpu()
            .numpy()[0]
        )


    # ----------------------------------------------
    # Predict attack probability
    # ----------------------------------------------

    attack_probability = risk_model.predict_proba(
        predicted_state.reshape(1, -1)
    )[0, 1]


    future_states.append(
        predicted_state
    )

    attack_risks.append(
        attack_probability
    )


    # ----------------------------------------------
    # Feed prediction back into World Model
    # ----------------------------------------------

    current_sequence = np.concatenate(
        [
            current_sequence[1:],
            predicted_state.reshape(1, -1)
        ],
        axis=0
    )


# ==================================================
# Print results
# ==================================================

print("\n========================================")
print("5-STEP ATTACK RISK FORECAST")
print("========================================")

for i, risk in enumerate(attack_risks):

    print(
        f"Future +{i+1}: "
        f"Attack Risk = {risk * 100:.2f}%"
    )


# ==================================================
# Save results
# ==================================================

future_states = np.array(
    future_states
)

attack_risks = np.array(
    attack_risks
)

np.save(
    "data/forecast_future_states.npy",
    future_states
)

np.save(
    "data/forecast_attack_risks.npy",
    attack_risks
)


print("\n========================================")
print("FORECAST COMPLETE")
print("========================================")

print(
    "Future states shape:",
    future_states.shape
)

print(
    "Attack risks shape:",
    attack_risks.shape
)

print("\nSaved:")
print("data/forecast_future_states.npy")
print("data/forecast_attack_risks.npy")
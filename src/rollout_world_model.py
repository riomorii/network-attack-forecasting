import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 1. Configuration
# ============================================================

WINDOW_SIZE = 5
FORECAST_STEPS = 5

MODEL_PATH = "models/network_state_world_model.pth"
TEST_DATA_PATH = "data/state_sequences_X_test.npy"

OUTPUT_PATH = "data/world_model_rollout.npy"


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
# 3. Load test data
# ============================================================

X_test = np.load(
    TEST_DATA_PATH
)

print("Test data shape:", X_test.shape)


# ============================================================
# 4. Load model
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


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
# 5. Select one test sequence
# ============================================================

sequence_index = 0

current_sequence = X_test[
    sequence_index
].copy()

print(
    "\nInitial sequence shape:",
    current_sequence.shape
)


# ============================================================
# 6. Autoregressive rollout
# ============================================================

predicted_states = []


print("\n========================================")
print("WORLD MODEL AUTOREGRESSIVE ROLLOUT")
print("========================================")


for step in range(FORECAST_STEPS):

    # Convert current sequence to tensor
    input_tensor = torch.tensor(
        current_sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    input_tensor = input_tensor.to(device)


    # Predict next network state
    with torch.no_grad():

        predicted_state = model(
            input_tensor
        ).cpu().numpy()[0]


    # Store prediction
    predicted_states.append(
        predicted_state
    )


    print(
        f"Forecast step {step + 1}: "
        f"state shape = {predicted_state.shape}"
    )


    # --------------------------------------------------------
    # Feed prediction back into the sequence
    # --------------------------------------------------------

    current_sequence = np.concatenate(
        [
            current_sequence[1:],
            predicted_state.reshape(1, -1)
        ],
        axis=0
    )


# ============================================================
# 7. Convert predictions to NumPy array
# ============================================================

predicted_states = np.array(
    predicted_states
)


# ============================================================
# 8. Save rollout
# ============================================================

np.save(
    OUTPUT_PATH,
    predicted_states
)


# ============================================================
# 9. Display results
# ============================================================

print("\n========================================")
print("ROLLOUT COMPLETE")
print("========================================")

print(
    "Predicted rollout shape:",
    predicted_states.shape
)

print(
    "Saved to:",
    OUTPUT_PATH
)

print("\nPredicted future states:")

for i, state in enumerate(
    predicted_states
):

    print(
        f"\nFuture state +{i + 1}:"
    )

    print(
        np.round(
            state,
            4
        )
    )
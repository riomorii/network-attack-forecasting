import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 1. LSTM model
# ============================================================

class LSTMModel(nn.Module):

    def __init__(self, input_size=32, hidden_size=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):

        lstm_output, (hidden, cell) = self.lstm(x)

        last_output = lstm_output[:, -1, :]

        prediction = self.output(last_output)

        return prediction.squeeze(1)


# ============================================================
# 2. Load trained model
# ============================================================

print("Loading trained LSTM model...")

model = LSTMModel()

model.load_state_dict(
    torch.load(
        "models/lstm_model.pth",
        map_location="cpu"
    )
)

model.eval()

print("Model loaded successfully!")


# ============================================================
# 3. Load test data
# ============================================================

print()
print("Loading test data...")

X_test = np.load("data/X_test_scaled.npy")
y_test = np.load("data/y_test.npy")

print("Test data shape:", X_test.shape)


# ============================================================
# 4. Select a starting sequence
# ============================================================

start_index = 0

current_window = X_test[start_index].copy()

print()
print("Starting forecast from test sequence:", start_index)


# ============================================================
# 5. Multi-step forecasting
# ============================================================

FORECAST_STEPS = 10

forecast_probabilities = []

print()
print("=" * 55)
print("MULTI-STEP ATTACK FORECAST")
print("=" * 55)


for step in range(FORECAST_STEPS):

    # Convert current window to tensor
    input_tensor = torch.tensor(
        current_window,
        dtype=torch.float32
    ).unsqueeze(0)

    # Model prediction
    with torch.no_grad():

        logit = model(input_tensor)

        probability = torch.sigmoid(logit).item()

    forecast_probabilities.append(probability)

    print(
        f"Future flow {step + 1}: "
        f"PortScan probability = {probability:.4f}"
    )

    # --------------------------------------------------------
    # Move window forward
    #
    # IMPORTANT:
    # We do not invent a future feature vector.
    # For this first prototype we use the next observed
    # test sequence as the next window.
    # --------------------------------------------------------

    next_index = start_index + step + 1

    if next_index < len(X_test):

        current_window = X_test[next_index].copy()


# ============================================================
# 6. Summary
# ============================================================

print()
print("=" * 55)
print("FORECAST SUMMARY")
print("=" * 55)

for i, probability in enumerate(
    forecast_probabilities,
    start=1
):

    print(
        f"Step {i}: "
        f"{probability * 100:.2f}% PortScan risk"
    )
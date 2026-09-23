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

        lstm_output, _ = self.lstm(x)

        last_output = lstm_output[:, -1, :]

        return self.output(last_output).squeeze(1)


# ============================================================
# 2. Load model
# ============================================================

print("Loading trained LSTM...")

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

X_test = np.load("data/X_test_scaled.npy")
y_test = np.load("data/y_test.npy")

print()
print("Test windows:", len(X_test))


# ============================================================
# 4. Generate probabilities
# ============================================================

print()
print("Generating probabilities...")

probabilities = []

with torch.no_grad():

    for i in range(len(X_test)):

        window = torch.tensor(
            X_test[i],
            dtype=torch.float32
        ).unsqueeze(0)

        logit = model(window)

        probability = torch.sigmoid(logit).item()

        probabilities.append(probability)

probabilities = np.array(probabilities)


# ============================================================
# 5. Strict warning settings
# ============================================================

WARNING_THRESHOLD = 0.80

CONSECUTIVE_WARNINGS = 2

print()
print("Warning threshold:", WARNING_THRESHOLD)
print(
    "Required consecutive warnings:",
    CONSECUTIVE_WARNINGS
)


# ============================================================
# 6. Find BENIGN → PortScan transitions
# ============================================================

transitions = []

for i in range(1, len(y_test)):

    if y_test[i - 1] == 0 and y_test[i] == 1:

        transitions.append(i)


print()
print("=" * 60)
print("STRICT EARLY-WARNING ANALYSIS")
print("=" * 60)

print(
    "BENIGN → PortScan transitions:",
    len(transitions)
)


# ============================================================
# 7. Analyze transitions
# ============================================================

valid_warnings = []

for transition in transitions:

    warning_index = None

    # Only search in the BENIGN period immediately
    # before the attack transition.
    start = max(
        0,
        transition - 20
    )

    end = transition

    for i in range(
        start,
        end - CONSECUTIVE_WARNINGS + 1
    ):

        # All required windows must still be BENIGN
        labels_are_benign = np.all(
            y_test[
                i:i + CONSECUTIVE_WARNINGS
            ] == 0
        )

        # All probabilities must cross threshold
        probabilities_high = np.all(
            probabilities[
                i:i + CONSECUTIVE_WARNINGS
            ] >= WARNING_THRESHOLD
        )

        if labels_are_benign and probabilities_high:

            warning_index = i
            break

    if warning_index is not None:

        lead_time = (
            transition - warning_index
        )

        valid_warnings.append(
            lead_time
        )


# ============================================================
# 8. Results
# ============================================================

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(
    "Total transitions:",
    len(transitions)
)

print(
    "Valid early warnings:",
    len(valid_warnings)
)

if len(transitions) > 0:

    detection_rate = (
        len(valid_warnings)
        / len(transitions)
    )

    print(
        "Early-warning detection rate:",
        f"{detection_rate * 100:.2f}%"
    )


if len(valid_warnings) > 0:

    print()
    print(
        "Average lead time:",
        f"{np.mean(valid_warnings):.2f}",
        "flow steps"
    )

    print(
        "Maximum lead time:",
        np.max(valid_warnings),
        "flow steps"
    )

    print(
        "Minimum lead time:",
        np.min(valid_warnings),
        "flow steps"
    )

else:

    print()
    print(
        "No valid early warnings were found "
        "under the strict criteria."
    )
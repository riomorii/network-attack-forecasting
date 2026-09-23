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
# 5. Find BENIGN → PortScan transitions
# ============================================================

transitions = []

for i in range(1, len(y_test)):

    previous_label = y_test[i - 1]
    current_label = y_test[i]

    if previous_label == 0 and current_label == 1:

        transitions.append(i)


print()
print("=" * 60)
print("BENIGN → PORTSCAN TRANSITIONS")
print("=" * 60)

print(
    "Number of transitions found:",
    len(transitions)
)


# ============================================================
# 6. Analyze each transition
# ============================================================

WARNING_THRESHOLD = 0.80

lead_times = []

for transition in transitions:

    warning_index = None

    # Search backwards up to 20 flow steps
    start = max(0, transition - 20)

    for i in range(start, transition):

        if probabilities[i] >= WARNING_THRESHOLD:

            warning_index = i
            break

    if warning_index is not None:

        lead_time = transition - warning_index
        lead_times.append(lead_time)


# ============================================================
# 7. Results
# ============================================================

print()

if len(transitions) == 0:

    print(
        "No BENIGN → PortScan transitions were found."
    )

else:

    print(
        "Transitions with a warning:",
        len(lead_times),
        "/",
        len(transitions)
    )

    if len(lead_times) > 0:

        print()
        print(
            "Average warning lead:",
            f"{np.mean(lead_times):.2f}",
            "flow steps"
        )

        print(
            "Maximum warning lead:",
            f"{np.max(lead_times)}",
            "flow steps"
        )

        print(
            "Minimum warning lead:",
            f"{np.min(lead_times)}",
            "flow steps"
        )

    else:

        print(
            "No warning was raised before the detected transitions."
        )


# ============================================================
# 8. Show first 10 transitions
# ============================================================

print()
print("=" * 60)
print("FIRST TRANSITIONS")
print("=" * 60)

for transition in transitions[:10]:

    print()
    print(
        "Transition at step:",
        transition
    )

    start = max(0, transition - 3)
    end = min(len(y_test), transition + 3)

    for i in range(start, end):

        label = (
            "PortScan"
            if y_test[i] == 1
            else "BENIGN"
        )

        print(
            f"  Step {i:6d} | "
            f"{label:8s} | "
            f"Risk: {probabilities[i] * 100:6.2f}%"
        )
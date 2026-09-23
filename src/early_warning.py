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
print("Generating attack probabilities...")

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
# 5. Choose a warning threshold
# ============================================================

WARNING_THRESHOLD = 0.80

print()
print("Warning threshold:", WARNING_THRESHOLD)


# ============================================================
# 6. Find attack transitions
# ============================================================

attack_indices = np.where(y_test == 1)[0]

if len(attack_indices) == 0:

    print("No PortScan samples found in test data.")

else:

    first_attack = attack_indices[0]

    print()
    print("First PortScan test index:", first_attack)

    # Find the first warning at or before the first attack
    warning_indices = np.where(
        probabilities[:first_attack + 1]
        >= WARNING_THRESHOLD
    )[0]

    if len(warning_indices) > 0:

        warning_index = warning_indices[0]

        lead_steps = first_attack - warning_index

        print()
        print("=" * 55)
        print("EARLY WARNING RESULT")
        print("=" * 55)

        print(
            "Warning raised at flow step:",
            warning_index
        )

        print(
            "PortScan appears at flow step:",
            first_attack
        )

        print(
            "Lead time:",
            lead_steps,
            "flow steps"
        )

        print(
            "Warning probability:",
            f"{probabilities[warning_index] * 100:.2f}%"
        )

    else:

        print()
        print(
            "No warning was raised before the first "
            "PortScan in this test sequence."
        )


# ============================================================
# 7. Show probability around the transition
# ============================================================

print()
print("=" * 55)
print("PROBABILITY AROUND FIRST ATTACK")
print("=" * 55)

start = max(0, first_attack - 10)
end = min(len(y_test), first_attack + 11)

for i in range(start, end):

    label = (
        "PortScan"
        if y_test[i] == 1
        else "BENIGN"
    )

    print(
        f"Step {i:6d} | "
        f"{label:8s} | "
        f"Risk: {probabilities[i] * 100:6.2f}%"
    )
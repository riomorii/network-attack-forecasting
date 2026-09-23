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

        return self.output(last_output)


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "Average Packet Size",
    "packets_per_second",
    "bytes_per_second",
    "fwd_bwd_packet_ratio",
    "fwd_bwd_byte_ratio",
    "syn_ack_ratio",
    "rst_ratio",
    "packet_size_range",
    "iat_variability"
]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading test data...")

X_test = np.load("data/X_test_scaled.npy")

X = X_test[:, :-1, :]
Y = X_test[:, -1, :]

print("Input:", X.shape)
print("Target:", Y.shape)


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
# PREDICTIONS
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
# FEATURE ERROR
# ============================================================

absolute_errors = np.abs(predictions - Y)

mean_feature_error = np.mean(
    absolute_errors,
    axis=0
)

max_feature_error = np.max(
    absolute_errors,
    axis=0
)


# ============================================================
# SORT FEATURES BY ERROR
# ============================================================

ranking = np.argsort(
    mean_feature_error
)[::-1]


print()
print("=" * 70)
print("FEATURE-LEVEL WORLD MODEL ERROR")
print("=" * 70)

print()

for rank, feature_index in enumerate(ranking):

    name = feature_names[feature_index]

    mean_error = mean_feature_error[feature_index]

    max_error = max_feature_error[feature_index]

    print(
        f"{rank + 1:2d}. "
        f"{name:30s} "
        f"Mean Error: {mean_error:.6f} "
        f"Max Error: {max_error:.6f}"
    )


# ============================================================
# WORST SAMPLE
# ============================================================

sample_errors = np.mean(
    (predictions - Y) ** 2,
    axis=1
)

worst_index = np.argmax(sample_errors)

print()
print("=" * 70)
print("WORST TEST SAMPLE")
print("=" * 70)

print()
print("Sample index:", worst_index)
print("Sample MSE:", sample_errors[worst_index])

print()
print("Actual state:")
print(Y[worst_index])

print()
print("Predicted state:")
print(predictions[worst_index])
import numpy as np


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


print("Loading training data...")

X = np.load("data/X_train_scaled.npy")

print("Shape:", X.shape)

# Flatten all flows so we can inspect the distribution
X_flat = X.reshape(-1, X.shape[-1])


print()
print("=" * 80)
print("FEATURE DISTRIBUTION")
print("=" * 80)

for i, name in enumerate(feature_names):

    values = X_flat[:, i]

    print()
    print(f"{i + 1:2d}. {name}")

    print(f"    Mean   : {np.mean(values):.4f}")
    print(f"    Std    : {np.std(values):.4f}")
    print(f"    Min    : {np.min(values):.4f}")
    print(f"    Max    : {np.max(values):.4f}")
    print(f"    Median : {np.median(values):.4f}")
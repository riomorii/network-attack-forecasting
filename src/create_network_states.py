import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD FEATURE-ENGINEERED FLOW DATA
# ============================================================

print("Loading feature-engineered flow data...")

df = pd.read_csv("data/portscan_features.csv")

print("Original shape:", df.shape)


# ============================================================
# 2. REMOVE INVALID NEGATIVE-DURATION FLOWS
# ============================================================

invalid = df["Flow Duration"] < 0

print()
print("Removing invalid negative-duration flows:", invalid.sum())

df = df[~invalid].copy()

print("Shape after cleaning:", df.shape)


# ============================================================
# 3. CREATE SAFE FLOW FEATURES
# ============================================================
#
# Instead of using the original ratio features directly,
# create bounded versions.
#
# forward / (backward + 1)
#
# This prevents enormous values when backward traffic = 0.
# ============================================================

df["safe_packet_ratio"] = (
    df["Total Fwd Packets"]
    /
    (df["Total Backward Packets"] + 1.0)
)

df["safe_byte_ratio"] = (
    df["Total Length of Fwd Packets"]
    /
    (df["Total Length of Bwd Packets"] + 1.0)
)


# ============================================================
# 4. FEATURES USED TO BUILD NETWORK STATE
# ============================================================

features = [
    "Flow Duration",
    "Flow Packets/s",
    "Flow Bytes/s",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "SYN Flag Count",
    "ACK Flag Count",
    "RST Flag Count",
    "FIN Flag Count",
    "Average Packet Size",
    "Destination Port",
    "safe_packet_ratio",
    "safe_byte_ratio",
    "Flow IAT Std",
    "iat_variability",
    "packets_per_second",
    "bytes_per_second"
]


# ============================================================
# 5. CHECK FEATURES
# ============================================================

missing = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing:

    print()
    print("ERROR: Missing features:")

    for feature in missing:
        print("-", feature)

    raise SystemExit


# ============================================================
# 6. EXTRACT FLOW DATA
# ============================================================

X = df[features].to_numpy(dtype=np.float64)

print()
print("Selected flow features:", len(features))
print("Flow data shape:", X.shape)


# ============================================================
# 7. CHECK FOR INVALID VALUES
# ============================================================

if np.isnan(X).any():

    print("ERROR: NaN values detected.")
    raise SystemExit

if np.isinf(X).any():

    print("ERROR: Infinite values detected.")
    raise SystemExit


# ============================================================
# 8. CREATE NETWORK STATES
# ============================================================
#
# Every 10 consecutive flows become one network state.
#
# ============================================================

WINDOW = 10

states = []

print()
print("Creating network states...")


for i in range(0, len(X), WINDOW):

    block = X[i:i + WINDOW]

    # Ignore incomplete final block
    if len(block) < WINDOW:
        break

    state = [

        # Traffic volume
        np.mean(block[:, 17]),
        np.std(block[:, 17]),

        np.mean(block[:, 18]),
        np.std(block[:, 18]),

        # Flow behaviour
        np.mean(block[:, 0]),

        # Packet behaviour
        np.mean(block[:, 11]),

        # TCP behaviour
        np.mean(block[:, 7]),
        np.mean(block[:, 8]),
        np.mean(block[:, 9]),
        np.mean(block[:, 10]),

        # Port behaviour
        len(np.unique(block[:, 12])),

        # Directional behaviour
        np.mean(block[:, 13]),
        np.mean(block[:, 14]),

        # Timing behaviour
        np.mean(block[:, 15]),
        np.mean(block[:, 16])
    ]

    states.append(state)


states = np.array(states, dtype=np.float32)


# ============================================================
# 9. SAVE
# ============================================================

np.save(
    "data/network_states_train.npy",
    states
)


# ============================================================
# 10. RESULTS
# ============================================================

print()
print("=" * 60)
print("NETWORK STATE CREATION COMPLETE")
print("=" * 60)

print()
print("Number of network states:", len(states))
print("Features per state:", states.shape[1])
print("State shape:", states.shape)

print()
print("Saved:")
print("data/network_states_train.npy")


# ============================================================
# 11. FEATURE NAMES
# ============================================================

state_names = [
    "Average packets/sec",
    "Packet-rate variability",
    "Average bytes/sec",
    "Byte-rate variability",
    "Average flow duration",
    "Average packet size",
    "SYN activity",
    "ACK activity",
    "RST activity",
    "FIN activity",
    "Unique destination ports",
    "Safe forward/backward packet ratio",
    "Safe forward/backward byte ratio",
    "IAT standard deviation",
    "IAT variability"
]


# ============================================================
# 12. FIRST 3 STATES
# ============================================================

print()
print("=" * 60)
print("FIRST 3 NETWORK STATES")
print("=" * 60)

for i in range(min(3, len(states))):

    print()
    print("State", i + 1)

    for j, name in enumerate(state_names):

        print(
            f"  {name:40s}: {states[i, j]:.4f}"
        )
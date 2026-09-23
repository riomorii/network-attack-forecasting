import numpy as np


# ============================================================
# LOAD NETWORK STATES
# ============================================================

states = np.load("data/network_states_train.npy")

print("Network state shape:", states.shape)


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = [
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
    "Forward/backward packet ratio",
    "Forward/backward byte ratio",
    "IAT standard deviation",
    "IAT variability"
]


# ============================================================
# STATISTICS
# ============================================================

print()
print("=" * 75)
print("NETWORK STATE FEATURE STATISTICS")
print("=" * 75)

for i, name in enumerate(feature_names):

    values = states[:, i]

    print()
    print(name)

    print(f"  Mean   : {np.mean(values):.4f}")
    print(f"  Std    : {np.std(values):.4f}")
    print(f"  Min    : {np.min(values):.4f}")
    print(f"  Median : {np.median(values):.4f}")
    print(f"  Max    : {np.max(values):.4f}")


# ============================================================
# CHECK FOR INVALID VALUES
# ============================================================

print()
print("=" * 75)
print("INVALID VALUE CHECK")
print("=" * 75)

print("NaN values :", np.isnan(states).sum())
print("Inf values :", np.isinf(states).sum())


# ============================================================
# CHECK EXTREME VALUES
# ============================================================

print()
print("=" * 75)
print("EXTREME VALUE CHECK")
print("=" * 75)

for i, name in enumerate(feature_names):

    values = states[:, i]

    q99 = np.percentile(values, 99)
    q999 = np.percentile(values, 99.9)

    print()
    print(name)
    print(f"  99th percentile  : {q99:.4f}")
    print(f"  99.9th percentile: {q999:.4f}")
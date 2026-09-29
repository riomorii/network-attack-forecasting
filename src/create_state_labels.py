import numpy as np
import pandas as pd

DATA_PATH = "data/portscan_features.csv"

WINDOW_SIZE = 10

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# Keep exactly the same cleaning rule used
# when creating network states.
df = df[df["Flow Duration"] >= 0].reset_index(drop=True)

print("After cleaning:", df.shape)

# Convert labels into binary values
df["attack"] = (
    df["Label"].astype(str).str.strip().str.lower() != "benign"
).astype(int)

# Number of complete states
num_states = len(df) // WINDOW_SIZE

df = df.iloc[:num_states * WINDOW_SIZE]

# Reshape attack labels into groups of 10 flows
attack_values = df["attack"].values.reshape(num_states, WINDOW_SIZE)

# Fraction of attack flows in each network state
attack_ratio = attack_values.mean(axis=1)

# Majority label:
# 0 = mostly benign
# 1 = mostly attack
state_label = (attack_ratio >= 0.5).astype(int)

print("\n========================================")
print("NETWORK STATE LABELS")
print("========================================")

print("Number of network states:", num_states)

print("\nAttack ratio examples:")

for i in range(min(10, num_states)):
    print(
        f"State {i+1}: "
        f"attack_ratio={attack_ratio[i]:.2f}, "
        f"label={state_label[i]}"
    )

print("\nState label distribution:")
print("Benign states:", np.sum(state_label == 0))
print("Attack states:", np.sum(state_label == 1))

print("\nAttack ratio statistics:")
print("Minimum :", attack_ratio.min())
print("Maximum :", attack_ratio.max())
print("Mean    :", attack_ratio.mean())
print("Median  :", np.median(attack_ratio))

# Save both continuous attack ratio and binary label
np.save(
    "data/network_state_attack_ratio.npy",
    attack_ratio
)

np.save(
    "data/network_state_labels.npy",
    state_label
)

print("\nSaved:")
print("data/network_state_attack_ratio.npy")
print("data/network_state_labels.npy")

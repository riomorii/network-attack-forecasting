import numpy as np


# ============================================================
# LOAD NETWORK STATES
# ============================================================

states = np.load("data/network_states_train.npy")

print("Total network states:", len(states))
print("State shape:", states.shape)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(states)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = states[:train_end]
val = states[train_end:val_end]
test = states[val_end:]


# ============================================================
# SAVE
# ============================================================

np.save("data/network_states_train_split.npy", train)
np.save("data/network_states_val.npy", val)
np.save("data/network_states_test.npy", test)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("NETWORK STATE SPLIT")
print("=" * 60)

print()
print("Training states   :", train.shape)
print("Validation states :", val.shape)
print("Test states       :", test.shape)

print()
print("Saved:")
print("data/network_states_train_split.npy")
print("data/network_states_val.npy")
print("data/network_states_test.npy")
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

train = np.load("data/network_states_train_split.npy")
val = np.load("data/network_states_val.npy")
test = np.load("data/network_states_test.npy")

print("Training shape  :", train.shape)
print("Validation shape:", val.shape)
print("Test shape      :", test.shape)

scaler = StandardScaler()

print()
print("Fitting StandardScaler on training states...")

# IMPORTANT:
# Fit only on training data to avoid data leakage.
scaler.fit(train)

train_scaled = scaler.transform(train)
val_scaled = scaler.transform(val)
test_scaled = scaler.transform(test)

# Save scaled datasets
np.save(
    "data/network_states_train_scaled.npy",
    train_scaled.astype(np.float32)
)

np.save(
    "data/network_states_val_scaled.npy",
    val_scaled.astype(np.float32)
)

np.save(
    "data/network_states_test_scaled.npy",
    test_scaled.astype(np.float32)
)

# Save scaler for later inference
joblib.dump(
    scaler,
    "models/network_state_scaler.pkl"
)

print()
print("=" * 60)
print("NETWORK STATE SCALING COMPLETE")
print("=" * 60)

print()
print("Training scaled shape   :", train_scaled.shape)
print("Validation scaled shape :", val_scaled.shape)
print("Test scaled shape       :", test_scaled.shape)

print()
print("Training mean:")
print(np.mean(train_scaled, axis=0))

print()
print("Training standard deviation:")
print(np.std(train_scaled, axis=0))

print()
print("Saved:")
print("data/network_states_train_scaled.npy")
print("data/network_states_val_scaled.npy")
print("data/network_states_test_scaled.npy")
print("models/network_state_scaler.pkl")
import numpy as np


WINDOW_SIZE = 5


def create_sequences(data, window_size):
    X = []
    y = []

    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# Load scaled network states
train = np.load("data/network_states_train_scaled.npy")
val = np.load("data/network_states_val_scaled.npy")
test = np.load("data/network_states_test_scaled.npy")

print("Original shapes:")
print("Train:", train.shape)
print("Validation:", val.shape)
print("Test:", test.shape)

# Create temporal sequences
X_train, y_train = create_sequences(train, WINDOW_SIZE)
X_val, y_val = create_sequences(val, WINDOW_SIZE)
X_test, y_test = create_sequences(test, WINDOW_SIZE)

print()
print("Sequence shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

# Save
np.save("data/state_sequences_X_train.npy", X_train)
np.save("data/state_sequences_y_train.npy", y_train)

np.save("data/state_sequences_X_val.npy", X_val)
np.save("data/state_sequences_y_val.npy", y_val)

np.save("data/state_sequences_X_test.npy", X_test)
np.save("data/state_sequences_y_test.npy", y_test)

print()
print("=" * 60)
print("STATE SEQUENCE CREATION COMPLETE")
print("=" * 60)

print()
print("Window size:", WINDOW_SIZE)
print("Features per state:", train.shape[1])

print()
print("Saved files:")
print("data/state_sequences_X_train.npy")
print("data/state_sequences_y_train.npy")
print("data/state_sequences_X_val.npy")
print("data/state_sequences_y_val.npy")
print("data/state_sequences_X_test.npy")
print("data/state_sequences_y_test.npy")
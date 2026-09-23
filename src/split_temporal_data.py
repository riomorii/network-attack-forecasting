import numpy as np

# --------------------------------
# 1. Load temporal windows
# --------------------------------

print("Loading temporal windows...")

X = np.load("data/X_windows.npy")
y = np.load("data/y_windows.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)


# --------------------------------
# 2. Calculate split points
# --------------------------------

n = len(X)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

print()
print("Total windows:", n)
print("Training end:", train_end)
print("Validation end:", val_end)


# --------------------------------
# 3. Chronological split
# --------------------------------

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]


# --------------------------------
# 4. Display split sizes
# --------------------------------

print()
print("TRAIN:")
print("X:", X_train.shape)
print("y:", y_train.shape)

print()
print("VALIDATION:")
print("X:", X_val.shape)
print("y:", y_val.shape)

print()
print("TEST:")
print("X:", X_test.shape)
print("y:", y_test.shape)


# --------------------------------
# 5. Display class distribution
# --------------------------------

print()
print("Training labels:")
print("BENIGN:", np.sum(y_train == 0))
print("PortScan:", np.sum(y_train == 1))

print()
print("Validation labels:")
print("BENIGN:", np.sum(y_val == 0))
print("PortScan:", np.sum(y_val == 1))

print()
print("Test labels:")
print("BENIGN:", np.sum(y_test == 0))
print("PortScan:", np.sum(y_test == 1))


# --------------------------------
# 6. Save splits
# --------------------------------

np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)

np.save("data/X_val.npy", X_val)
np.save("data/y_val.npy", y_val)

np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)

print()
print("Temporal splits saved successfully!")

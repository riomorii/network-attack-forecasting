import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# --------------------------------
# 1. Load temporal splits
# --------------------------------

print("Loading temporal splits...")

X_train = np.load("data/X_train.npy")
X_val = np.load("data/X_val.npy")
X_test = np.load("data/X_test.npy")

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)


# --------------------------------
# 2. Create scaler
# --------------------------------

scaler = StandardScaler()


# --------------------------------
# 3. Reshape training data
# --------------------------------

# Current shape:
# (samples, time_steps, features)

n_train, time_steps, n_features = X_train.shape

X_train_2d = X_train.reshape(
    -1,
    n_features
)


# --------------------------------
# 4. Fit scaler ONLY on training
# --------------------------------

print()
print("Fitting scaler on training data...")

scaler.fit(X_train_2d)


# --------------------------------
# 5. Transform all datasets
# --------------------------------

def scale_data(X):

    n_samples, time_steps, n_features = X.shape

    X_2d = X.reshape(-1, n_features)

    X_scaled = scaler.transform(X_2d)

    return X_scaled.reshape(
        n_samples,
        time_steps,
        n_features
    )


print("Scaling training data...")
X_train_scaled = scale_data(X_train)

print("Scaling validation data...")
X_val_scaled = scale_data(X_val)

print("Scaling test data...")
X_test_scaled = scale_data(X_test)


# --------------------------------
# 6. Check results
# --------------------------------

print()
print("Scaled shapes:")

print("Training:", X_train_scaled.shape)
print("Validation:", X_val_scaled.shape)
print("Test:", X_test_scaled.shape)


# --------------------------------
# 7. Save scaled data
# --------------------------------

np.save("data/X_train_scaled.npy", X_train_scaled)
np.save("data/X_val_scaled.npy", X_val_scaled)
np.save("data/X_test_scaled.npy", X_test_scaled)

# Save scaler for future inference
joblib.dump(
    scaler,
    "models/feature_scaler.pkl"
)

print()
print("Scaled datasets saved successfully!")

print()
print("Scaler saved to:")
print("models/feature_scaler.pkl")
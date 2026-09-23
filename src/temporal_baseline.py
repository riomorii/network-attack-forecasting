import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# --------------------------------
# 1. Load scaled temporal data
# --------------------------------

print("Loading temporal data...")

X_train = np.load("data/X_train_scaled.npy")
y_train = np.load("data/y_train.npy")

X_val = np.load("data/X_val_scaled.npy")
y_val = np.load("data/y_val.npy")

X_test = np.load("data/X_test_scaled.npy")
y_test = np.load("data/y_test.npy")


print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# --------------------------------
# 2. Flatten temporal windows
# --------------------------------

# Logistic Regression cannot directly
# process a 3D sequence.

# Convert:
# (samples, 10, 32)
#
# into:
# (samples, 320)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


print()
print("Flattened training:", X_train_flat.shape)
print("Flattened validation:", X_val_flat.shape)
print("Flattened test:", X_test_flat.shape)


# --------------------------------
# 3. Train Logistic Regression
# --------------------------------

print()
print("Training temporal Logistic Regression...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_flat, y_train)

print("Training complete!")


# --------------------------------
# 4. Validation
# --------------------------------

y_val_pred = model.predict(X_val_flat)

print()
print("========== VALIDATION ==========")

print(
    classification_report(
        y_val,
        y_val_pred,
        target_names=["BENIGN", "PortScan"]
    )
)


# --------------------------------
# 5. Test
# --------------------------------

y_test_pred = model.predict(X_test_flat)

accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print()
print("========== TEST ==========")

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["BENIGN", "PortScan"]
    )
)

print("Test accuracy:", accuracy)


# --------------------------------
# 6. Confusion matrix
# --------------------------------

print()
print("========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_test,
        y_test_pred
    )
)
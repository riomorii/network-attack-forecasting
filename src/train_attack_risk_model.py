import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

TRAIN_X = "data/network_states_train_scaled.npy"
VAL_X = "data/network_states_val_scaled.npy"
TEST_X = "data/network_states_test_scaled.npy"

LABELS = "data/network_state_labels.npy"

# --------------------------------------------------
# Load data
# --------------------------------------------------

X_train = np.load(TRAIN_X)
X_val = np.load(VAL_X)
X_test = np.load(TEST_X)

all_labels = np.load(LABELS)

# Same 70/15/15 split used for network states
n_train = len(X_train)
n_val = len(X_val)
n_test = len(X_test)

y_train = all_labels[:n_train]
y_val = all_labels[n_train:n_train + n_val]
y_test = all_labels[n_train + n_val:n_train + n_val + n_test]

print("========================================")
print("ATTACK RISK MODEL")
print("========================================")

print("Training shape  :", X_train.shape)
print("Validation shape:", X_val.shape)
print("Testing shape   :", X_test.shape)

print("\nTraining labels:")
print("Benign:", np.sum(y_train == 0))
print("Attack:", np.sum(y_train == 1))

print("\nValidation labels:")
print("Benign:", np.sum(y_val == 0))
print("Attack:", np.sum(y_val == 1))

print("\nTesting labels:")
print("Benign:", np.sum(y_test == 0))
print("Attack:", np.sum(y_test == 1))

# --------------------------------------------------
# Train Logistic Regression
# --------------------------------------------------

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# Validation
# --------------------------------------------------

val_predictions = model.predict(X_val)
val_probabilities = model.predict_proba(X_val)[:, 1]

print("\n========================================")
print("VALIDATION RESULTS")
print("========================================")

print("Accuracy :", accuracy_score(y_val, val_predictions))
print("Precision:", precision_score(y_val, val_predictions))
print("Recall   :", recall_score(y_val, val_predictions))
print("F1 Score :", f1_score(y_val, val_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_predictions))

# --------------------------------------------------
# Test
# --------------------------------------------------

test_predictions = model.predict(X_test)
test_probabilities = model.predict_proba(X_test)[:, 1]

print("\n========================================")
print("TEST RESULTS")
print("========================================")

print("Accuracy :", accuracy_score(y_test, test_predictions))
print("Precision:", precision_score(y_test, test_predictions))
print("Recall   :", recall_score(y_test, test_predictions))
print("F1 Score :", f1_score(y_test, test_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    target_names=["BENIGN", "ATTACK"]
))

# --------------------------------------------------
# Save model and probabilities
# --------------------------------------------------

import joblib

joblib.dump(
    model,
    "models/attack_risk_model.pkl"
)

np.save(
    "data/attack_risk_test_probabilities.npy",
    test_probabilities
)

print("\n========================================")
print("SAVED")
print("========================================")

print("models/attack_risk_model.pkl")
print("data/attack_risk_test_probabilities.npy")
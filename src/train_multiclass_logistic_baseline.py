import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# ============================================
# CONFIG
# ============================================

MAX_ITER = 1000
RANDOM_STATE = 42

# ============================================
# LOAD DATA
# ============================================

X_train = np.load(
    "data/multiclass_3class_X_train.npy"
)

y_train = np.load(
    "data/multiclass_3class_y_train.npy"
)

X_val = np.load(
    "data/multiclass_3class_X_val.npy"
)

y_val = np.load(
    "data/multiclass_3class_y_val.npy"
)

X_test = np.load(
    "data/multiclass_3class_X_test.npy"
)

y_test = np.load(
    "data/multiclass_3class_y_test.npy"
)

print("===============================================")
print("MULTICLASS LOGISTIC REGRESSION BASELINE")
print("===============================================")

print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)

# ============================================
# FLATTEN TEMPORAL WINDOWS
# ============================================

# 5 states × 15 features = 75 features

X_train_flat = X_train.reshape(
    X_train.shape[0],
    -1
)

X_val_flat = X_val.reshape(
    X_val.shape[0],
    -1
)

X_test_flat = X_test.reshape(
    X_test.shape[0],
    -1
)

print("\nFlattened:")
print("Train:", X_train_flat.shape)
print("Val:  ", X_val_flat.shape)
print("Test: ", X_test_flat.shape)

# ============================================
# SCALE
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_flat
)

X_val_scaled = scaler.transform(
    X_val_flat
)

X_test_scaled = scaler.transform(
    X_test_flat
)

# ============================================
# MODEL
# ============================================

model = LogisticRegression(
    max_iter=MAX_ITER,
    random_state=RANDOM_STATE
)

print("\nTraining Logistic Regression...")

model.fit(
    X_train_scaled,
    y_train
)

print("Training complete.")

# ============================================
# VALIDATION
# ============================================

val_predictions = model.predict(
    X_val_scaled
)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_balanced_accuracy = balanced_accuracy_score(
    y_val,
    val_predictions
)

val_precision, val_recall, val_f1, _ = (
    precision_recall_fscore_support(
        y_val,
        val_predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )
)

print("\n===============================================")
print("VALIDATION RESULTS")
print("===============================================")

print(
    f"Accuracy:          {val_accuracy:.4f}"
)

print(
    f"Balanced Accuracy: {val_balanced_accuracy:.4f}"
)

print(
    f"Macro F1:          {val_f1.mean():.4f}"
)

# ============================================
# TEST
# ============================================

test_predictions = model.predict(
    X_test_scaled
)

test_probabilities = model.predict_proba(
    X_test_scaled
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    test_predictions
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_test,
        test_predictions,
        labels=[0, 1, 2],
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1, 2]
)

# ============================================
# RESULTS
# ============================================

class_names = [
    "BENIGN",
    "DOS_DDOS",
    "RECONNAISSANCE"
]

print("\n===============================================")
print("TEST RESULTS")
print("===============================================")

print(
    f"Accuracy:          {test_accuracy:.4f}"
)

print(
    f"Balanced Accuracy: {test_balanced_accuracy:.4f}"
)

print(
    f"Macro Precision:   {precision.mean():.4f}"
)

print(
    f"Macro Recall:      {recall.mean():.4f}"
)

print(
    f"Macro F1:          {f1.mean():.4f}"
)

print("\n===============================================")
print("PER-CLASS RESULTS")
print("===============================================")

for i, name in enumerate(class_names):

    print(
        f"{name:18s} "
        f"Precision={precision[i]:.4f} "
        f"Recall={recall[i]:.4f} "
        f"F1={f1[i]:.4f} "
        f"Support={support[i]}"
    )

print("\n===============================================")
print("CONFUSION MATRIX")
print("===============================================")

print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(
    "                BENIGN  DOS_DDOS  RECON"
)

for i, row in enumerate(cm):

    print(
        f"{class_names[i]:15s} "
        f"{row[0]:7d} "
        f"{row[1]:9d} "
        f"{row[2]:7d}"
    )

print("\n===============================================")
print("CLASSIFICATION REPORT")
print("===============================================")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

# ============================================
# SAVE
# ============================================

joblib.dump(
    scaler,
    "models/multiclass_logistic_scaler.pkl"
)

joblib.dump(
    model,
    "models/multiclass_logistic_baseline.pkl"
)

np.save(
    "data/multiclass_logistic_test_predictions.npy",
    test_predictions
)

np.save(
    "data/multiclass_logistic_test_probabilities.npy",
    test_probabilities
)

print("\n===============================================")
print("SAVED")
print("===============================================")

print(
    "models/multiclass_logistic_scaler.pkl"
)

print(
    "models/multiclass_logistic_baseline.pkl"
)

print(
    "data/multiclass_logistic_test_predictions.npy"
)

print(
    "data/multiclass_logistic_test_probabilities.npy"
)
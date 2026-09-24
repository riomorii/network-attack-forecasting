import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from world_model import MultiTaskWorldModel


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "multitask_world_model_final.pth"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "multitask_world_model_final_scaler.pkl"
)

X_TEST_FILE = os.path.join(
    DATA_DIR,
    "multiclass_3class_X_test.npy"
)

Y_TEST_FILE = os.path.join(
    DATA_DIR,
    "multiclass_3class_y_test.npy"
)

ROLLOUT_FILE = os.path.join(
    DATA_DIR,
    "multitask_rollout_results.csv"
)

PERSISTENCE_FILE = os.path.join(
    DATA_DIR,
    "persistence_rollout_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    DATA_DIR,
    "final_ml_summary.csv"
)

OUTPUT_PER_CLASS = os.path.join(
    DATA_DIR,
    "final_ml_per_class.csv"
)

OUTPUT_COMPARISON = os.path.join(
    DATA_DIR,
    "final_world_model_vs_persistence.csv"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

INPUT_SIZE = 15
HIDDEN_SIZE = 64
OUTPUT_SIZE = 15
NUM_CLASSES = 3

CLASS_NAMES = [
    "BENIGN",
    "DOS_DDOS",
    "RECONNAISSANCE",
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FINAL ML EVALUATION REPORT")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(
    f"Device: {device}"
)


model = MultiTaskWorldModel(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    num_classes=NUM_CLASSES
)


checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=False
)


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict,
    strict=True
)

model.to(device)
model.eval()


scaler = joblib.load(
    SCALER_FILE
)


print(
    f"Loaded model: {MODEL_FILE}"
)

print(
    f"Loaded scaler: {SCALER_FILE}"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test = np.load(
    X_TEST_FILE
)

y_test = np.load(
    Y_TEST_FILE
)


print(
    f"\nX_test shape: {X_test.shape}"
)

print(
    f"y_test shape: {y_test.shape}"
)


# ============================================================
# FINAL ORDINARY TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


# Scale every test sequence using the saved training scaler.

flat = X_test.reshape(
    -1,
    INPUT_SIZE
)

flat = np.nan_to_num(
    flat,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

scaled = scaler.transform(
    flat
)

scaled = scaled.reshape(
    X_test.shape
).astype(
    np.float32
)


# Convert to tensor.

x_tensor = torch.tensor(
    scaled,
    dtype=torch.float32,
    device=device
)


# Run inference.

with torch.no_grad():

    _, class_logits = model(
        x_tensor
    )

    probabilities = F.softmax(
        class_logits,
        dim=1
    )

    predictions = torch.argmax(
        probabilities,
        dim=1
    )


predictions = (
    predictions
    .cpu()
    .numpy()
)


# Metrics.

test_accuracy = accuracy_score(
    y_test,
    predictions
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions
)

test_macro_precision = precision_score(
    y_test,
    predictions,
    labels=[0, 1, 2],
    average="macro",
    zero_division=0
)

test_macro_recall = recall_score(
    y_test,
    predictions,
    labels=[0, 1, 2],
    average="macro",
    zero_division=0
)

test_macro_f1 = f1_score(
    y_test,
    predictions,
    labels=[0, 1, 2],
    average="macro",
    zero_division=0
)


print(
    f"\nAccuracy:          {test_accuracy:.4f}"
)

print(
    f"Balanced Accuracy: {test_balanced_accuracy:.4f}"
)

print(
    f"Macro Precision:   {test_macro_precision:.4f}"
)

print(
    f"Macro Recall:      {test_macro_recall:.4f}"
)

print(
    f"Macro F1:          {test_macro_f1:.4f}"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)


per_class_rows = []

for class_name in CLASS_NAMES:

    row = report[class_name]

    per_class_rows.append({
        "class": class_name,
        "precision": row["precision"],
        "recall": row["recall"],
        "f1": row["f1-score"],
        "support": int(row["support"]),
    })


per_class_df = pd.DataFrame(
    per_class_rows
)


print("\n" + "=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

print(
    per_class_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# LOAD ROLLOUT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LOADING ROLLOUT RESULTS")
print("=" * 70)


rollout_df = pd.read_csv(
    ROLLOUT_FILE
)

persistence_df = pd.read_csv(
    PERSISTENCE_FILE
)


print(
    f"World Model rollout rows: "
    f"{len(rollout_df)}"
)

print(
    f"Persistence rows: "
    f"{len(persistence_df)}"
)


# ============================================================
# VERIFY HORIZONS
# ============================================================

expected_horizons = [1, 2, 3, 4, 5]


if rollout_df["horizon"].tolist() != expected_horizons:

    raise ValueError(
        "Unexpected World Model rollout horizons."
    )


if persistence_df["horizon"].tolist() != expected_horizons:

    raise ValueError(
        "Unexpected persistence horizons."
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

comparison_df = pd.DataFrame({
    "horizon": expected_horizons,

    "world_model_accuracy": (
        rollout_df["accuracy"]
    ),

    "world_model_balanced_accuracy": (
        rollout_df["balanced_accuracy"]
    ),

    "world_model_macro_precision": (
        rollout_df["macro_precision"]
    ),

    "world_model_macro_recall": (
        rollout_df["macro_recall"]
    ),

    "world_model_macro_f1": (
        rollout_df["macro_f1"]
    ),

    "persistence_accuracy": (
        persistence_df["accuracy"]
    ),

    "persistence_balanced_accuracy": (
        persistence_df["balanced_accuracy"]
    ),

    "persistence_macro_precision": (
        persistence_df["macro_precision"]
    ),

    "persistence_macro_recall": (
        persistence_df["macro_recall"]
    ),

    "persistence_macro_f1": (
        persistence_df["macro_f1"]
    ),
})


print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

summary_rows = [

    {
        "section": "ordinary_test",
        "metric": "accuracy",
        "value": test_accuracy
    },

    {
        "section": "ordinary_test",
        "metric": "balanced_accuracy",
        "value": test_balanced_accuracy
    },

    {
        "section": "ordinary_test",
        "metric": "macro_precision",
        "value": test_macro_precision
    },

    {
        "section": "ordinary_test",
        "metric": "macro_recall",
        "value": test_macro_recall
    },

    {
        "section": "ordinary_test",
        "metric": "macro_f1",
        "value": test_macro_f1
    },

]


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE ALL REPORTS
# ============================================================

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

per_class_df.to_csv(
    OUTPUT_PER_CLASS,
    index=False
)

comparison_df.to_csv(
    OUTPUT_COMPARISON,
    index=False
)


# ============================================================
# FINAL DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("FINAL ML FILES")
print("=" * 70)

print(
    OUTPUT_SUMMARY
)

print(
    OUTPUT_PER_CLASS
)

print(
    OUTPUT_COMPARISON
)


print("\n" + "=" * 70)
print("FINAL ML EVALUATION COMPLETED")
print("=" * 70)

print(
    "No training was performed."
)
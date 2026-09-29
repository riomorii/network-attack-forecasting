import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("===============================================")
print("CORRECT TEMPORAL ATTACK-ONSET FORECAST")
print("===============================================")

WINDOW_SIZE = 5
HORIZONS = [1, 2, 3, 5]

states = np.load(
    "data/multiclass_network_states.npy"
)

labels = np.load(
    "data/multiclass_state_labels.npy",
    allow_pickle=True
)

metadata = pd.read_csv(
    "data/multiclass_state_metadata.csv"
)

print("States:", states.shape)
print("Labels:", labels.shape)

# ------------------------------------------------
# Build chronological splits FIRST
# ------------------------------------------------

train_X = []
train_labels = []

val_X = []
val_labels = []

test_X = []
test_labels = []

for scenario, group in metadata.groupby(
    "source_file",
    sort=False
):

    indices = group.index.to_numpy()

    scenario_states = states[indices]
    scenario_labels = labels[indices]

    n = len(scenario_labels)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # We need enough future states for the largest horizon.
    max_horizon = max(HORIZONS)

    # Each observation uses the current state only.
    # We keep the chronological position intact.
    for i in range(
        WINDOW_SIZE - 1,
        n - max_horizon
    ):

        current_state = scenario_states[
            i
        ]

        current_label = str(
            scenario_labels[i]
        )

        future_labels = [
            str(x)
            for x in scenario_labels[
                i + 1:i + 1 + max_horizon
            ]
        ]

        sample = (
            current_state,
            current_label,
            future_labels
        )

        # Assign according to ORIGINAL scenario position.
        if i < train_end:
            train_X.append(sample)

        elif i < val_end:
            val_X.append(sample)

        else:
            test_X.append(sample)


print("\nOriginal chronological samples:")
print("Train:", len(train_X))
print("Val:  ", len(val_X))
print("Test: ", len(test_X))


# ------------------------------------------------
# Convert samples
# ------------------------------------------------

def unpack(samples):

    X = np.asarray(
        [x[0] for x in samples]
    )

    current = np.asarray(
        [x[1] for x in samples]
    )

    future = np.asarray(
        [x[2] for x in samples]
    )

    return X, current, future


X_train_all, current_train, future_train = unpack(
    train_X
)

X_val_all, current_val, future_val = unpack(
    val_X
)

X_test_all, current_test, future_test = unpack(
    test_X
)


# ------------------------------------------------
# Keep ONLY currently benign states
# ------------------------------------------------

train_mask = (
    current_train == "BENIGN"
)

val_mask = (
    current_val == "BENIGN"
)

test_mask = (
    current_test == "BENIGN"
)

X_train = X_train_all[train_mask]
future_train = future_train[train_mask]

X_val = X_val_all[val_mask]
future_val = future_val[val_mask]

X_test = X_test_all[test_mask]
future_test = future_test[test_mask]

print("\nCurrently benign states:")
print("Train:", len(X_train))
print("Val:  ", len(X_val))
print("Test: ", len(X_test))


# ------------------------------------------------
# Evaluate horizons
# ------------------------------------------------

results = []

for horizon in HORIZONS:

    print("\n===============================================")
    print(f"HORIZON +{horizon} STATE(S)")
    print("===============================================")

    def make_target(future):

        targets = []

        for row in future:

            attack = any(
                label != "BENIGN"
                for label in row[:horizon]
            )

            targets.append(
                int(attack)
            )

        return np.asarray(targets)


    y_train = make_target(
        future_train
    )

    y_val = make_target(
        future_val
    )

    y_test = make_target(
        future_test
    )

    print(
        "Train positives:",
        int(y_train.sum()),
        "/",
        len(y_train)
    )

    print(
        "Val positives:",
        int(y_val.sum()),
        "/",
        len(y_val)
    )

    print(
        "Test positives:",
        int(y_test.sum()),
        "/",
        len(y_test)
    )

    # ------------------------------------------------
    # Scale using TRAIN ONLY
    # ------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ------------------------------------------------
    # Train classifier
    # ------------------------------------------------

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # ------------------------------------------------
    # Predictions
    # ------------------------------------------------

    test_pred = model.predict(
        X_test_scaled
    )

    # ------------------------------------------------
    # Metrics
    # ------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        test_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        test_pred
    )

    precision = precision_score(
        y_test,
        test_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=[0, 1]
    )

    print("\nResults:")

    print(
        f"Accuracy:          {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy: {balanced_accuracy:.4f}"
    )

    print(
        f"Precision:         {precision:.4f}"
    )

    print(
        f"Recall:            {recall:.4f}"
    )

    print(
        f"F1:                {f1:.4f}"
    )

    print("\nConfusion Matrix")
    print(
        "Rows = Actual, Columns = Predicted"
    )
    print(cm)

    results.append({
        "horizon": horizon,
        "test_samples": len(y_test),
        "test_positive": int(y_test.sum()),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


# ------------------------------------------------
# Save
# ------------------------------------------------

results_df = pd.DataFrame(results)

print("\n===============================================")
print("FINAL RESULTS")
print("===============================================")

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    "data/correct_onset_forecast_results.csv",
    index=False
)

print(
    "\nSaved:"
    " data/correct_onset_forecast_results.csv"
)
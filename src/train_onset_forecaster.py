import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

print("===============================================")
print("ATTACK-ONSET FORECASTER")
print("===============================================")

# ------------------------------------------------
# 1. Load network states and labels
# ------------------------------------------------

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
# 2. Build sequences within each scenario
# ------------------------------------------------

WINDOW_SIZE = 5

X_list = []
current_labels = []
future_labels = []
scenario_list = []

for scenario, group in metadata.groupby(
    "source_file",
    sort=False
):

    indices = group.index.to_numpy()

    scenario_states = states[indices]
    scenario_labels = labels[indices]

    n = len(scenario_labels)

    for i in range(n - WINDOW_SIZE - 5):

        # Last state in current observation window
        current_state = scenario_states[
            i + WINDOW_SIZE - 1
        ]

        current_label = str(
            scenario_labels[
                i + WINDOW_SIZE - 1
            ]
        )

        # Future labels
        future = scenario_labels[
            i + WINDOW_SIZE :
            i + WINDOW_SIZE + 5
        ]

        X_list.append(current_state)
        current_labels.append(current_label)
        future_labels.append(
            [str(x) for x in future]
        )
        scenario_list.append(scenario)

X = np.asarray(X_list)
current_labels = np.asarray(current_labels)
future_labels = np.asarray(future_labels)
scenario_list = np.asarray(scenario_list)

print("Forecast samples:", X.shape)

# ------------------------------------------------
# 3. Keep ONLY currently benign states
# ------------------------------------------------

benign_mask = (
    current_labels == "BENIGN"
)

X = X[benign_mask]
future_labels = future_labels[benign_mask]
scenario_list = scenario_list[benign_mask]

print(
    "Currently benign samples:",
    len(X)
)

# ------------------------------------------------
# 4. Create temporal train/val/test split
#    separately inside every scenario
# ------------------------------------------------

train_indices = []
val_indices = []
test_indices = []

for scenario in np.unique(scenario_list):

    idx = np.where(
        scenario_list == scenario
    )[0]

    n = len(idx)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_indices.extend(
        idx[:train_end]
    )

    val_indices.extend(
        idx[train_end:val_end]
    )

    test_indices.extend(
        idx[val_end:]
    )

train_indices = np.asarray(train_indices)
val_indices = np.asarray(val_indices)
test_indices = np.asarray(test_indices)

print("\nSplit sizes:")
print("Train:", len(train_indices))
print("Val:  ", len(val_indices))
print("Test: ", len(test_indices))

# ------------------------------------------------
# 5. Evaluate multiple forecast horizons
# ------------------------------------------------

horizons = [1, 2, 3, 5]

results = []

for horizon in horizons:

    print("\n===============================================")
    print(f"FORECAST HORIZON: +{horizon} STATE(S)")
    print("===============================================")

    # --------------------------------------------
    # Target:
    # Will an attack occur within next H states?
    # --------------------------------------------

    def make_target(indices):

        targets = []

        for idx in indices:

            future = future_labels[idx]

            within_horizon = future[:horizon]

            attack = any(
                x != "BENIGN"
                for x in within_horizon
            )

            targets.append(
                int(attack)
            )

        return np.asarray(targets)

    y_train = make_target(train_indices)
    y_val = make_target(val_indices)
    y_test = make_target(test_indices)

    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]

    print(
        "Train attack rate:",
        y_train.mean()
    )

    print(
        "Val attack rate:",
        y_val.mean()
    )

    print(
        "Test attack rate:",
        y_test.mean()
    )

    # --------------------------------------------
    # Scale using TRAIN only
    # --------------------------------------------

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

    # --------------------------------------------
    # Logistic Regression
    # --------------------------------------------

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # Validation
    val_pred = model.predict(
        X_val_scaled
    )

    # Test
    test_pred = model.predict(
        X_test_scaled
    )

    # --------------------------------------------
    # Metrics
    # --------------------------------------------

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
        test_pred
    )

    print("\nTest Results")

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
    print(cm)

    results.append({
        "horizon": horizon,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


# ------------------------------------------------
# 6. Save results
# ------------------------------------------------

results_df = pd.DataFrame(results)

print("\n===============================================")
print("FINAL ONSET FORECAST RESULTS")
print("===============================================")

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    "data/onset_forecast_results.csv",
    index=False
)

print(
    "\nSaved:"
    " data/onset_forecast_results.csv"
)
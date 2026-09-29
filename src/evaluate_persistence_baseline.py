import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix
)


# ==================================================
# Configuration
# ==================================================

FORECAST_STEPS = 5

X_TEST_PATH = "data/state_sequences_X_test.npy"
STATE_LABELS_PATH = "data/network_state_labels.npy"
RISK_MODEL_PATH = "models/attack_risk_model.pkl"


# ==================================================
# Load data
# ==================================================

X_test = np.load(X_TEST_PATH)

all_state_labels = np.load(
    STATE_LABELS_PATH
)

risk_model = joblib.load(
    RISK_MODEL_PATH
)


print("========================================")
print("PERSISTENCE BASELINE")
print("========================================")

print("X_test shape:", X_test.shape)


# ==================================================
# Reconstruct test labels
# ==================================================

TRAIN_STATES = 20024
VAL_STATES = 4291

test_state_labels = all_state_labels[
    TRAIN_STATES + VAL_STATES:
]

print(
    "Test labels shape:",
    test_state_labels.shape
)


# ==================================================
# Evaluate
# ==================================================

num_samples = len(X_test) - FORECAST_STEPS

all_probabilities = []
all_targets = []


for sample_index in range(num_samples):

    # ----------------------------------------------
    # Current observed state
    #
    # The last state in the history is the current
    # state available to the baseline.
    # ----------------------------------------------

    current_state = X_test[
        sample_index,
        -1,
        :
    ]

    # Current attack probability
    current_probability = risk_model.predict_proba(
        current_state.reshape(1, -1)
    )[0, 1]


    sample_probabilities = []
    sample_targets = []


    for step in range(FORECAST_STEPS):

        # Persistence assumption:
        # current probability remains unchanged.

        sample_probabilities.append(
            current_probability
        )


        # Actual future state:
        #
        # history has 5 states:
        # S0 S1 S2 S3 S4
        #
        # +1 = S5
        # +2 = S6
        # ...

        target_index = (
            sample_index
            + 5
            + step
        )

        sample_targets.append(
            test_state_labels[target_index]
        )


    all_probabilities.append(
        sample_probabilities
    )

    all_targets.append(
        sample_targets
    )


# ==================================================
# Convert arrays
# ==================================================

all_probabilities = np.array(
    all_probabilities
)

all_targets = np.array(
    all_targets
)


print("\nPrediction shape:")
print(all_probabilities.shape)

print("Target shape:")
print(all_targets.shape)


# ==================================================
# Evaluate each horizon
# ==================================================

print("\n========================================")
print("PERSISTENCE BASELINE RESULTS")
print("========================================")


for step in range(FORECAST_STEPS):

    probabilities = all_probabilities[
        :, step
    ]

    targets = all_targets[
        :, step
    ]


    predictions = (
        probabilities >= 0.50
    ).astype(int)


    accuracy = accuracy_score(
        targets,
        predictions
    )

    precision = precision_score(
        targets,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        targets,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        targets,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        targets,
        probabilities
    )

    brier = brier_score_loss(
        targets,
        probabilities
    )

    cm = confusion_matrix(
        targets,
        predictions
    )


    print(
        f"\nHorizon +{step + 1}"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {auc:.4f}"
    )

    print(
        f"Brier    : {brier:.4f}"
    )

    print("Confusion Matrix:")
    print(cm)


# ==================================================
# Save
# ==================================================

np.save(
    "data/persistence_forecast_risks.npy",
    all_probabilities
)

np.save(
    "data/persistence_forecast_targets.npy",
    all_targets
)


print("\n========================================")
print("BASELINE EVALUATION COMPLETE")
print("========================================")

print("Saved:")
print(
    "data/persistence_forecast_risks.npy"
)

print(
    "data/persistence_forecast_targets.npy"
)
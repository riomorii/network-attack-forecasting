import numpy as np
import torch
import torch.nn as nn
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

WORLD_MODEL_PATH = "models/network_state_world_model.pth"
RISK_MODEL_PATH = "models/attack_risk_model.pkl"

X_TEST_PATH = "data/state_sequences_X_test.npy"
STATE_LABELS_PATH = "data/network_state_labels.npy"


# ==================================================
# Network-State World Model
# ==================================================

class NetworkStateWorldModel(nn.Module):

    def __init__(
        self,
        input_size=15,
        hidden_size=64,
        output_size=15
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_hidden = output[:, -1, :]

        return self.fc(last_hidden)


# ==================================================
# Load test sequences
# ==================================================

X_test = np.load(X_TEST_PATH)

# These are the actual binary labels for ALL
# network states.
all_state_labels = np.load(
    STATE_LABELS_PATH
)

print("========================================")
print("ATTACK FORECAST EVALUATION")
print("========================================")

print("X_test shape:", X_test.shape)

print(
    "All state labels shape:",
    all_state_labels.shape
)


# ==================================================
# Reconstruct the same temporal split
# ==================================================

# These are the numbers of states created earlier:
TRAIN_STATES = 20024
VAL_STATES = 4291
TEST_STATES = 4291

test_state_labels = all_state_labels[
    TRAIN_STATES + VAL_STATES:
]

print(
    "Test state labels shape:",
    test_state_labels.shape
)

print(
    "Test benign states:",
    np.sum(test_state_labels == 0)
)

print(
    "Test attack states:",
    np.sum(test_state_labels == 1)
)


# ==================================================
# Device
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ==================================================
# Load World Model
# ==================================================

world_model = NetworkStateWorldModel()

world_model.load_state_dict(
    torch.load(
        WORLD_MODEL_PATH,
        map_location=device
    )
)

world_model = world_model.to(device)

world_model.eval()


# ==================================================
# Load Attack Risk Model
# ==================================================

risk_model = joblib.load(
    RISK_MODEL_PATH
)


# ==================================================
# Forecast all test sequences
# ==================================================

# X_test contains 4286 sequences.
#
# Each sequence contains 5 states.
#
# The first future state is state index 5.
#
# Therefore we can evaluate:
#
# sample 0:
#   +1 -> test_state_labels[5]
#   +2 -> test_state_labels[6]
#   +3 -> test_state_labels[7]
#   +4 -> test_state_labels[8]
#   +5 -> test_state_labels[9]
#
# sample 1:
#   +1 -> test_state_labels[6]
#   ...
#
# We stop 5 steps before the end.

num_samples = len(X_test) - FORECAST_STEPS

print(
    "\nEvaluating",
    num_samples,
    "test sequences..."
)


all_risks = []
all_targets = []


for sample_index in range(num_samples):

    current_sequence = X_test[
        sample_index
    ].copy()

    sample_risks = []
    sample_targets = []


    for step in range(FORECAST_STEPS):

        # ------------------------------------------
        # Predict next network state
        # ------------------------------------------

        input_tensor = torch.tensor(
            current_sequence,
            dtype=torch.float32
        ).unsqueeze(0).to(device)


        with torch.no_grad():

            predicted_state = (
                world_model(input_tensor)
                .cpu()
                .numpy()[0]
            )


        # ------------------------------------------
        # Convert predicted state into
        # attack probability
        # ------------------------------------------

        attack_probability = risk_model.predict_proba(
            predicted_state.reshape(1, -1)
        )[0, 1]


        sample_risks.append(
            attack_probability
        )


        # ------------------------------------------
        # ACTUAL future label
        # ------------------------------------------

        target_index = (
            sample_index
            + 5
            + step
        )

        actual_label = test_state_labels[
            target_index
        ]

        sample_targets.append(
            actual_label
        )


        # ------------------------------------------
        # Feed predicted state back into
        # World Model
        # ------------------------------------------

        current_sequence = np.concatenate(
            [
                current_sequence[1:],
                predicted_state.reshape(1, -1)
            ],
            axis=0
        )


    all_risks.append(
        sample_risks
    )

    all_targets.append(
        sample_targets
    )


# ==================================================
# Convert to arrays
# ==================================================

all_risks = np.array(
    all_risks
)

all_targets = np.array(
    all_targets
)


print("\nPrediction shape:")
print(all_risks.shape)

print("Target shape:")
print(all_targets.shape)


# ==================================================
# Evaluate each horizon
# ==================================================

print("\n========================================")
print("MULTI-STEP ATTACK FORECAST RESULTS")
print("========================================")


for step in range(FORECAST_STEPS):

    probabilities = all_risks[
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
# Save results
# ==================================================

np.save(
    "data/all_attack_forecast_risks.npy",
    all_risks
)

np.save(
    "data/all_attack_forecast_targets.npy",
    all_targets
)


print("\n========================================")
print("EVALUATION COMPLETE")
print("========================================")

print("Saved:")
print(
    "data/all_attack_forecast_risks.npy"
)

print(
    "data/all_attack_forecast_targets.npy"
)
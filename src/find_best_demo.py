import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
MODEL_DIR = "models"

X_TEST_FILE = os.path.join(
    DATA_DIR,
    "multiclass_3class_X_test.npy"
)

Y_TEST_FILE = os.path.join(
    DATA_DIR,
    "multiclass_3class_y_test.npy"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "multitask_world_model_scaler.pkl"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "multitask_world_model_best.pth"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "best_demo_candidates.csv"
)

INPUT_SIZE = 15
HIDDEN_SIZE = 64
OUTPUT_SIZE = 15
NUM_CLASSES = 3

BATCH_SIZE = 512

CLASS_NAMES = [
    "BENIGN",
    "DOS_DDOS",
    "RECONNAISSANCE",
]


# ============================================================
# MODEL
# ============================================================

class MultiTaskWorldModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_classes
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.state_head = nn.Linear(
            hidden_size,
            output_size
        )

        self.class_head = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        lstm_output, _ = self.lstm(x)

        last_hidden = lstm_output[:, -1, :]

        future_state = self.state_head(
            last_hidden
        )

        class_logits = self.class_head(
            last_hidden
        )

        return future_state, class_logits


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("SEARCHING FOR AN INFORMATIVE SIH DEMO SAMPLE")
print("=" * 70)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


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
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

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

X_test = np.load(
    X_TEST_FILE
)

y_test = np.load(
    Y_TEST_FILE
)


print(
    f"X_test: {X_test.shape}"
)

print(
    f"y_test: {y_test.shape}"
)


# ============================================================
# HELPERS
# ============================================================

@torch.no_grad()
def predict_batch(
    raw_batch
):

    n = raw_batch.shape[0]

    flat = raw_batch.reshape(
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
        n,
        raw_batch.shape[1],
        INPUT_SIZE
    ).astype(
        np.float32
    )

    x = torch.tensor(
        scaled,
        dtype=torch.float32,
        device=device
    )

    _, logits = model(x)

    probabilities = F.softmax(
        logits,
        dim=1
    )

    probabilities = probabilities.cpu().numpy()

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    return predictions, probabilities


def apply_intervention(
    batch,
    intervention
):

    modified = batch.copy()


    if intervention == "Rate limiting":

        modified[:, :, 0] *= 0.50
        modified[:, :, 1] *= 0.50
        modified[:, :, 2] *= 0.50
        modified[:, :, 3] *= 0.50


    elif intervention == "SYN filtering":

        modified[:, :, 6] *= 0.25


    elif intervention == "Port-scan blocking":

        modified[:, :, 10] *= 0.50


    elif intervention == "Combined defense":

        modified[:, :, 0] *= 0.50
        modified[:, :, 1] *= 0.50
        modified[:, :, 2] *= 0.50
        modified[:, :, 3] *= 0.50

        modified[:, :, 6] *= 0.25

        modified[:, :, 10] *= 0.50


    return modified


# ============================================================
# FIND BASELINE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 — BASELINE PREDICTIONS")
print("=" * 70)


all_predictions = []
all_probabilities = []


for start in range(
    0,
    len(X_test),
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        len(X_test)
    )

    pred, probs = predict_batch(
        X_test[start:end]
    )

    all_predictions.append(
        pred
    )

    all_probabilities.append(
        probs
    )


predictions = np.concatenate(
    all_predictions
)

probabilities = np.concatenate(
    all_probabilities
)


print(
    f"Predictions calculated: "
    f"{len(predictions)}"
)


# ============================================================
# FIND ATTACK CANDIDATES
# ============================================================

candidate_mask = (
    (predictions == 1)
    | (predictions == 2)
)


candidate_indices = np.where(
    candidate_mask
)[0]


print(
    f"Attack-predicted candidates: "
    f"{len(candidate_indices)}"
)


# ============================================================
# DEFENSE SEARCH
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — TESTING DEFENSE INTERVENTIONS")
print("=" * 70)


interventions = [
    "Rate limiting",
    "SYN filtering",
    "Port-scan blocking",
    "Combined defense",
]


records = []


candidate_X = X_test[
    candidate_indices
]


candidate_true = y_test[
    candidate_indices
]


candidate_base_probs = probabilities[
    candidate_indices
]


candidate_base_pred = predictions[
    candidate_indices
]


for intervention in interventions:

    print(
        f"\nTesting: {intervention}"
    )


    modified = apply_intervention(
        candidate_X,
        intervention
    )


    intervention_probs_list = []
    intervention_pred_list = []


    for start in range(
        0,
        len(modified),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(modified)
        )


        pred, probs = predict_batch(
            modified[start:end]
        )


        intervention_pred_list.append(
            pred
        )

        intervention_probs_list.append(
            probs
        )


    intervention_predictions = np.concatenate(
        intervention_pred_list
    )

    intervention_probabilities = np.concatenate(
        intervention_probs_list
    )


    for i in range(
        len(candidate_indices)
    ):

        original_class = int(
            candidate_base_pred[i]
        )

        original_attack_probability = (
            candidate_base_probs[i][
                original_class
            ]
        )

        new_attack_probability = (
            intervention_probabilities[i][
                original_class
            ]
        )


        reduction = (
            original_attack_probability
            - new_attack_probability
        )


        changed_class = (
            int(
                intervention_predictions[i]
            )
            != original_class
        )


        records.append({

            "sample_index": int(
                candidate_indices[i]
            ),

            "true_class": CLASS_NAMES[
                int(candidate_true[i])
            ],

            "baseline_class": CLASS_NAMES[
                original_class
            ],

            "intervention": intervention,

            "baseline_attack_probability": (
                original_attack_probability
            ),

            "post_intervention_attack_probability": (
                new_attack_probability
            ),

            "probability_reduction": reduction,

            "baseline_confidence": (
                candidate_base_probs[i].max()
            ),

            "post_intervention_class": CLASS_NAMES[
                int(intervention_predictions[i])
            ],

            "class_changed": changed_class,
        })


results = pd.DataFrame(
    records
)


# ============================================================
# RANK DEMO CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — FINDING BEST DEMO CASES")
print("=" * 70)


# Prefer cases where:
#   1. model initially predicts an attack
#   2. confidence is not essentially 100%
#   3. intervention reduces attack probability
#
# We give priority to class-changing interventions.

results["priority"] = (
    results["probability_reduction"]
    + results["class_changed"].astype(float)
)


best = results.sort_values(
    [
        "class_changed",
        "probability_reduction"
    ],
    ascending=[
        False,
        False
    ]
).head(20)


# ============================================================
# DISPLAY
# ============================================================

print("\nTop demo candidates:\n")

display_columns = [
    "sample_index",
    "true_class",
    "baseline_class",
    "intervention",
    "baseline_attack_probability",
    "post_intervention_attack_probability",
    "probability_reduction",
    "baseline_confidence",
    "post_intervention_class",
    "class_changed",
]


print(
    best[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)


# ============================================================
# RECOMMENDED SAMPLE
# ============================================================

if len(best) > 0:

    recommended = best.iloc[0]

    print("\n" + "=" * 70)
    print("RECOMMENDED SIH DEMO SAMPLE")
    print("=" * 70)

    print(
        f"Sample index: "
        f"{int(recommended['sample_index'])}"
    )

    print(
        f"True class: "
        f"{recommended['true_class']}"
    )

    print(
        f"Baseline prediction: "
        f"{recommended['baseline_class']}"
    )

    print(
        f"Intervention: "
        f"{recommended['intervention']}"
    )

    print(
        f"Baseline attack probability: "
        f"{recommended['baseline_attack_probability'] * 100:.3f}%"
    )

    print(
        f"Post-intervention attack probability: "
        f"{recommended['post_intervention_attack_probability'] * 100:.3f}%"
    )

    print(
        f"Probability reduction: "
        f"{recommended['probability_reduction'] * 100:.3f} percentage points"
    )

    print(
        f"Post-intervention class: "
        f"{recommended['post_intervention_class']}"
    )

else:

    print(
        "\nNo attack-predicted candidates were found."
    )


print(
    "\nSearch completed successfully."
)
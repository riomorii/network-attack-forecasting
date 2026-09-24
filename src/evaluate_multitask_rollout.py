import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = 5
HORIZON = 5

INPUT_SIZE = 15
HIDDEN_SIZE = 64
NUM_CLASSES = 3

DEVICE = torch.device("cpu")

LABEL_TO_ID = {
    "BENIGN": 0,
    "DOS_DDOS": 1,
    "RECONNAISSANCE": 2,
}

ID_TO_LABEL = {
    0: "BENIGN",
    1: "DOS_DDOS",
    2: "RECONNAISSANCE",
}


# ============================================================
# MODEL
# ============================================================

class MultiTaskWorldModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )

        self.state_head = nn.Linear(
            HIDDEN_SIZE,
            INPUT_SIZE
        )

        self.class_head = nn.Linear(
            HIDDEN_SIZE,
            NUM_CLASSES
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        h = output[:, -1, :]

        next_state = self.state_head(h)

        class_logits = self.class_head(h)

        return next_state, class_logits


# ============================================================
# START
# ============================================================

print("=" * 55)
print("MULTI-TASK WORLD MODEL — MULTI-STEP ROLLOUT")
print("=" * 55)

print("Device:", DEVICE)


# ============================================================
# LOAD MODEL
# ============================================================

model = MultiTaskWorldModel().to(DEVICE)

checkpoint_path = (
    "models/multitask_world_model_best.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True,
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()

print(
    "Loaded:",
    checkpoint_path
)


# ============================================================
# LOAD DATA
# ============================================================

X_test = np.load(
    "data/multiclass_3class_X_test.npy"
)

y_test = np.load(
    "data/multiclass_3class_y_test.npy"
)

network_states = np.load(
    "data/multiclass_network_states.npy"
)

state_labels_raw = np.load(
    "data/multiclass_state_labels.npy",
    allow_pickle=True
)

metadata = pd.read_csv(
    "data/multiclass_state_metadata.csv"
)


print()
print(
    "Test input shape:",
    X_test.shape
)

print(
    "Test labels shape:",
    y_test.shape
)

print(
    "Network states:",
    network_states.shape
)

print(
    "State labels:",
    state_labels_raw.shape
)


# ============================================================
# VERIFY DATA LENGTHS
# ============================================================

if len(network_states) != len(state_labels_raw):

    raise ValueError(
        "Network states and state labels have different lengths."
    )

if len(network_states) != len(metadata):

    raise ValueError(
        "Network states and metadata have different lengths."
    )

print(
    "Data lengths verified."
)


# ============================================================
# CONVERT STRING LABELS TO NUMERIC IDs
# ============================================================

state_labels = np.array(
    [
        LABEL_TO_ID.get(
            str(label).strip(),
            -1
        )
        for label in state_labels_raw
    ],
    dtype=np.int64
)


print()
print(
    "Converted state labels:"
)

unique_labels, counts = np.unique(
    state_labels,
    return_counts=True
)

for label, count in zip(
    unique_labels,
    counts
):

    if label in ID_TO_LABEL:

        name = ID_TO_LABEL[label]

    else:

        name = f"UNKNOWN_{label}"

    print(
        f"  {label} = {name}: {count}"
    )


# ============================================================
# CHECK FOR UNKNOWN LABELS
# ============================================================

unknown_count = np.sum(
    state_labels == -1
)

if unknown_count > 0:

    print()
    print(
        "WARNING:",
        unknown_count,
        "states have unknown labels."
    )


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(
    "models/multitask_world_model_scaler.pkl"
)

print()
print(
    "Loaded scaler."
)


# ============================================================
# SCENARIO COLUMN
# ============================================================

if "source_file" not in metadata.columns:

    raise ValueError(
        "metadata does not contain 'source_file'. "
        f"Available columns: {list(metadata.columns)}"
    )

scenario_names = metadata[
    "source_file"
].values


# ============================================================
# BUILD TEST ROLLOUT DATASET
#
# Each example contains:
#
# History:
#   t-4, t-3, t-2, t-1, t
#
# Targets:
#   t+1, t+2, t+3, t+4, t+5
#
# We never cross scenario boundaries.
# ============================================================

allowed_classes = {
    0,
    1,
    2
}

rollout_inputs = []

future_true_states = []

future_true_labels = []


print()
print("=" * 55)
print("BUILDING ROLLOUT DATASET")
print("=" * 55)


unique_scenarios = metadata[
    "source_file"
].unique()


for scenario in unique_scenarios:

    indices = np.where(
        scenario_names == scenario
    )[0]

    states = network_states[
        indices
    ]

    labels = state_labels[
        indices
    ]

    n = len(states)

    # Same split used during training:
    # 70% train
    # 15% validation
    # 15% test

    train_end = int(
        n * 0.70
    )

    val_end = int(
        n * 0.85
    )

    # Held-out test section
    test_states = states[
        val_end:
    ]

    test_labels = labels[
        val_end:
    ]

    # Need 5 history states + 5 future states
    max_start = (
        len(test_states)
        - HORIZON
    )

    for i in range(
        WINDOW_SIZE,
        max_start
    ):

        current_label = int(
            test_labels[i]
        )

        if current_label not in allowed_classes:

            continue

        history = test_states[
            i - WINDOW_SIZE:i
        ]

        true_states = []

        true_labels = []

        valid = True

        for h in range(
            1,
            HORIZON + 1
        ):

            future_index = i + h

            future_label = int(
                test_labels[
                    future_index
                ]
            )

            if future_label not in allowed_classes:

                valid = False

                break

            true_states.append(
                test_states[
                    future_index
                ]
            )

            true_labels.append(
                future_label
            )

        if not valid:

            continue

        rollout_inputs.append(
            history
        )

        future_true_states.append(
            true_states
        )

        future_true_labels.append(
            true_labels
        )


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

rollout_inputs = np.array(
    rollout_inputs,
    dtype=np.float32
)

future_true_states = np.array(
    future_true_states,
    dtype=np.float32
)

future_true_labels = np.array(
    future_true_labels,
    dtype=np.int64
)


print()
print(
    "Rollout dataset:"
)

print(
    "Starting sequences:",
    rollout_inputs.shape
)

print(
    "Future states:",
    future_true_states.shape
)

print(
    "Future labels:",
    future_true_labels.shape
)


# ============================================================
# VALIDATE SHAPES
# ============================================================

if len(rollout_inputs) == 0:

    raise ValueError(
        "No rollout samples were created."
    )

if future_true_states.shape[1] != HORIZON:

    raise ValueError(
        "Future state horizon is incorrect."
    )

if future_true_labels.shape[1] != HORIZON:

    raise ValueError(
        "Future label horizon is incorrect."
    )


# ============================================================
# SCALE INITIAL HISTORY
# ============================================================

N = len(
    rollout_inputs
)

rollout_scaled = scaler.transform(
    rollout_inputs.reshape(
        -1,
        INPUT_SIZE
    )
).reshape(
    N,
    WINDOW_SIZE,
    INPUT_SIZE
)


# ============================================================
# SCALE TRUE FUTURE STATES
# ============================================================

future_true_states_scaled = scaler.transform(
    future_true_states.reshape(
        -1,
        INPUT_SIZE
    )
).reshape(
    N,
    HORIZON,
    INPUT_SIZE
)


# ============================================================
# INITIAL MODEL INPUT
# ============================================================

current_sequence = torch.tensor(
    rollout_scaled,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# AUTOREGRESSIVE ROLLOUT
#
# +1:
#   real history -> prediction
#
# +2:
#   predicted +1 becomes part of input
#
# +3:
#   predicted +2 becomes part of input
#
# etc.
# ============================================================

print()
print("=" * 55)
print("AUTOREGRESSIVE FORECAST")
print("=" * 55)


predicted_states = []

predicted_classes = []

predicted_probabilities = []


with torch.no_grad():

    for step in range(
        HORIZON
    ):

        next_state, class_logits = model(
            current_sequence
        )

        probabilities = torch.softmax(
            class_logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        )

        predicted_states.append(
            next_state.cpu().numpy()
        )

        predicted_classes.append(
            predicted_class.cpu().numpy()
        )

        predicted_probabilities.append(
            probabilities.cpu().numpy()
        )

        # Feed predicted state back
        # into the next timestep.

        current_sequence = torch.cat(
            [
                current_sequence[:, 1:, :],
                next_state.unsqueeze(1)
            ],
            dim=1
        )


# ============================================================
# STACK RESULTS
# ============================================================

predicted_states = np.stack(
    predicted_states,
    axis=1
)

predicted_classes = np.stack(
    predicted_classes,
    axis=1
)

predicted_probabilities = np.stack(
    predicted_probabilities,
    axis=1
)


print()
print(
    "Predicted states:",
    predicted_states.shape
)

print(
    "Predicted classes:",
    predicted_classes.shape
)

print(
    "Predicted probabilities:",
    predicted_probabilities.shape
)


# ============================================================
# EVALUATE EACH HORIZON
# ============================================================

print()
print("=" * 55)
print("MULTI-STEP RESULTS")
print("=" * 55)


results = []


for step in range(
    HORIZON
):

    horizon = step + 1

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    y_true = future_true_labels[
        :,
        step
    ]

    y_pred = predicted_classes[
        :,
        step
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # STATE PREDICTION ERROR
    # --------------------------------------------------------

    state_error = (
        predicted_states[
            :,
            step,
            :
        ]
        -
        future_true_states_scaled[
            :,
            step,
            :
        ]
    )

    state_mse = np.mean(
        state_error ** 2
    )

    state_rmse = np.sqrt(
        state_mse
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print(
        f"+{horizon} state"
    )

    print(
        f"  State MSE:       {state_mse:.6f}"
    )

    print(
        f"  State RMSE:      {state_rmse:.6f}"
    )

    print(
        f"  Accuracy:        {accuracy:.4f}"
    )

    print(
        f"  Balanced Acc:    {balanced_acc:.4f}"
    )

    print(
        f"  Macro Precision: {macro_precision:.4f}"
    )

    print(
        f"  Macro Recall:    {macro_recall:.4f}"
    )

    print(
        f"  Macro F1:        {macro_f1:.4f}"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
            2
        ]
    )

    print(
        "  Confusion Matrix:"
    )

    print(cm)

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    results.append(
        {
            "horizon": horizon,
            "state_mse": state_mse,
            "state_rmse": state_rmse,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    "data/multitask_rollout_results.csv",
    index=False
)


np.save(
    "data/multitask_rollout_predicted_states.npy",
    predicted_states
)


np.save(
    "data/multitask_rollout_predicted_classes.npy",
    predicted_classes
)


np.save(
    "data/multitask_rollout_probabilities.npy",
    predicted_probabilities
)


np.save(
    "data/multitask_rollout_true_states.npy",
    future_true_states_scaled
)


np.save(
    "data/multitask_rollout_true_classes.npy",
    future_true_labels
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 55)
print("ROLLOUT SUMMARY")
print("=" * 55)

print()

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print()
print("=" * 55)
print("SAVED FILES")
print("=" * 55)

print(
    "data/multitask_rollout_results.csv"
)

print(
    "data/multitask_rollout_predicted_states.npy"
)

print(
    "data/multitask_rollout_predicted_classes.npy"
)

print(
    "data/multitask_rollout_probabilities.npy"
)

print(
    "data/multitask_rollout_true_states.npy"
)

print(
    "data/multitask_rollout_true_classes.npy"
)

print()
print(
    "Rollout evaluation completed successfully."
)
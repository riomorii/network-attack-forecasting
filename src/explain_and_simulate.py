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
    "explainability_intervention_results.csv"
)

INPUT_SIZE = 15
HIDDEN_SIZE = 64
OUTPUT_SIZE = 15
NUM_CLASSES = 3

# ============================================================
# IMPORTANT:
# This is the informative sample discovered by find_best_demo.py
# ============================================================

SAMPLE_INDEX = 4406

CLASS_NAMES = [
    "BENIGN",
    "DOS_DDOS",
    "RECONNAISSANCE",
]

FEATURE_NAMES = [
    "Avg Flow Packets/s",
    "Std Flow Packets/s",
    "Avg Flow Bytes/s",
    "Std Flow Bytes/s",
    "Avg Flow Duration",
    "Avg Average Packet Size",
    "Avg SYN Flag Count",
    "Avg ACK Flag Count",
    "Avg RST Flag Count",
    "Avg FIN Flag Count",
    "Unique Destination Port Count",
    "Mean Safe Packet Ratio",
    "Mean Safe Byte Ratio",
    "Mean Flow IAT Std",
    "Mean IAT Variability",
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# WORLD MODEL
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

        # Last observed hidden state
        last_hidden = lstm_output[:, -1, :]

        # Future continuous network state
        future_state = self.state_head(
            last_hidden
        )

        # Future attack class
        class_logits = self.class_head(
            last_hidden
        )

        return future_state, class_logits


# ============================================================
# LOAD SAVED MODEL
# ============================================================

def load_model():

    print("\nLoading saved World Model...")

    model = MultiTaskWorldModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
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

    model.to(DEVICE)

    model.eval()

    print(
        f"Loaded: {MODEL_FILE}"
    )

    return model


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict(
    model,
    scaled_sequence
):

    x = torch.tensor(
        scaled_sequence,
        dtype=torch.float32,
        device=DEVICE
    )

    if x.ndim == 2:

        x = x.unsqueeze(0)


    future_state, class_logits = model(x)


    probabilities = F.softmax(
        class_logits,
        dim=1
    )


    predicted_class = torch.argmax(
        probabilities,
        dim=1
    )


    return (
        future_state.cpu().numpy(),
        predicted_class.cpu().numpy(),
        probabilities.cpu().numpy()
    )


# ============================================================
# SCALE RAW SEQUENCE
# ============================================================

def scale_sequence(
    raw_sequence,
    scaler
):

    flat = raw_sequence.reshape(
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

    return scaled.reshape(
        raw_sequence.shape
    ).astype(
        np.float32
    )


# ============================================================
# PRINT RAW FEATURE VALUES
# ============================================================

def print_sample_features(
    raw_sequence
):

    print(
        "\nObserved 5-state history:"
    )

    print(
        "Rows = historical network states"
    )

    print(
        "Columns = network-state features\n"
    )


    for t in range(
        raw_sequence.shape[0]
    ):

        print(
            f"\nHistory state {t + 1}:"
        )

        for j, feature in enumerate(
            FEATURE_NAMES
        ):

            value = raw_sequence[
                t,
                j
            ]

            print(
                f"  {feature:32s}: {value:.6f}"
            )


# ============================================================
# INTERVENTIONS
# ============================================================

def apply_intervention(
    sequence,
    intervention
):

    modified = sequence.copy()


    if intervention == "No intervention":

        pass


    elif intervention == "Rate limiting":

        # Hypothetical feature-space intervention:
        # reduce traffic-rate-related features by 50%.

        modified[:, 0] *= 0.50
        modified[:, 1] *= 0.50
        modified[:, 2] *= 0.50
        modified[:, 3] *= 0.50


    elif intervention == "SYN filtering":

        # Hypothetical 75% reduction in SYN activity.

        modified[:, 6] *= 0.25


    elif intervention == "Port-scan blocking":

        # Hypothetical 50% reduction in unique destination
        # ports observed in the traffic representation.

        modified[:, 10] *= 0.50


    elif intervention == "Combined defense":

        # Combine the three hypothetical interventions.

        modified[:, 0] *= 0.50
        modified[:, 1] *= 0.50
        modified[:, 2] *= 0.50
        modified[:, 3] *= 0.50

        modified[:, 6] *= 0.25

        modified[:, 10] *= 0.50


    else:

        raise ValueError(
            f"Unknown intervention: {intervention}"
        )


    return modified


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("WORLD MODEL — EXPLAINABILITY + DEFENSE WHAT-IF")
print("=" * 70)

print(
    f"Device: {DEVICE}"
)

print(
    f"Selected sample: {SAMPLE_INDEX}"
)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    X_TEST_FILE,
    Y_TEST_FILE,
    SCALER_FILE,
    MODEL_FILE,
]

for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

X_test = np.load(
    X_TEST_FILE
)

y_test = np.load(
    Y_TEST_FILE
)

scaler = joblib.load(
    SCALER_FILE
)

model = load_model()


print(
    f"\nX_test shape: {X_test.shape}"
)

print(
    f"y_test shape: {y_test.shape}"
)


# ============================================================
# VERIFY SAMPLE
# ============================================================

if SAMPLE_INDEX < 0 or SAMPLE_INDEX >= len(X_test):

    raise IndexError(
        f"Sample index {SAMPLE_INDEX} is outside "
        f"the test set of size {len(X_test)}."
    )


raw_sequence = X_test[
    SAMPLE_INDEX
].copy()


true_class = int(
    y_test[SAMPLE_INDEX]
)


if true_class not in [0, 1, 2]:

    raise ValueError(
        f"Unexpected test label: {true_class}"
    )


# ============================================================
# SAMPLE DETAILS
# ============================================================

print("\n" + "=" * 70)
print("SELECTED TEST SAMPLE")
print("=" * 70)

print(
    f"Sample index: {SAMPLE_INDEX}"
)

print(
    f"True next-state class: "
    f"{CLASS_NAMES[true_class]}"
)

print(
    f"Input sequence shape: "
    f"{raw_sequence.shape}"
)


# ============================================================
# BASELINE PREDICTION
# ============================================================

scaled_sequence = scale_sequence(
    raw_sequence,
    scaler
)


_, baseline_pred, baseline_probs = predict(
    model,
    scaled_sequence
)


baseline_pred = int(
    baseline_pred[0]
)

baseline_probs = baseline_probs[0]


print("\n" + "=" * 70)
print("BASELINE PREDICTION")
print("=" * 70)

print(
    f"True class:       "
    f"{CLASS_NAMES[true_class]}"
)

print(
    f"Predicted class:  "
    f"{CLASS_NAMES[baseline_pred]}"
)

print("\nClass probabilities:")

for i, name in enumerate(
    CLASS_NAMES
):

    print(
        f"  {name:18s}: "
        f"{baseline_probs[i] * 100:9.3f}%"
    )


# ============================================================
# PRINT RAW FEATURE VALUES
# ============================================================

print("\n" + "=" * 70)
print("NETWORK STATE FEATURES")
print("=" * 70)

print_sample_features(
    raw_sequence
)


# ============================================================
# FEATURE OCCLUSION EXPLAINABILITY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE OCCLUSION EXPLAINABILITY")
print("=" * 70)

print(
    "\nMethod:"
)

print(
    "One feature is replaced by its scaler-training mean "
    "across all 5 history states."
)

print(
    "Importance is the change in the probability of the "
    "original predicted class."
)


importance_rows = []


for feature_idx, feature_name in enumerate(
    FEATURE_NAMES
):

    modified_scaled = scale_sequence(
        raw_sequence,
        scaler
    )


    # StandardScaler maps the training mean to approximately zero.

    modified_scaled[
        :,
        feature_idx
    ] = 0.0


    _, _, modified_probs = predict(
        model,
        modified_scaled
    )


    modified_probs = modified_probs[0]


    original_probability = (
        baseline_probs[
            baseline_pred
        ]
    )


    occluded_probability = (
        modified_probs[
            baseline_pred
        ]
    )


    probability_change = (
        original_probability
        - occluded_probability
    )


    importance = abs(
        probability_change
    )


    importance_rows.append({

        "feature": feature_name,

        "original_probability": (
            original_probability
        ),

        "occluded_probability": (
            occluded_probability
        ),

        "probability_change": (
            probability_change
        ),

        "absolute_importance": (
            importance
        ),
    })


importance_df = pd.DataFrame(
    importance_rows
)


importance_df = importance_df.sort_values(
    "absolute_importance",
    ascending=False
)


print(
    "\nFeature importance ranking:"
)


for rank, row in enumerate(
    importance_df.itertuples(
        index=False
    ),
    start=1
):

    print(
        f"{rank:2d}. "
        f"{row.feature:34s} "
        f"importance={row.absolute_importance:.4f} "
        f"change={row.probability_change:+.4f}"
    )


# ============================================================
# DEFENSE WHAT-IF SIMULATION
# ============================================================

interventions = [
    "No intervention",
    "Rate limiting",
    "SYN filtering",
    "Port-scan blocking",
    "Combined defense",
]


print("\n" + "=" * 70)
print("DEFENSE WHAT-IF SIMULATION")
print("=" * 70)

print(
    "\nThese are hypothetical feature-space interventions."
)

print(
    "They do NOT represent guaranteed real-world firewall outcomes."
)


simulation_rows = []


for intervention in interventions:

    modified_raw = apply_intervention(
        raw_sequence,
        intervention
    )


    modified_scaled = scale_sequence(
        modified_raw,
        scaler
    )


    _, prediction, probabilities = predict(
        model,
        modified_scaled
    )


    prediction = int(
        prediction[0]
    )

    probabilities = probabilities[0]


    baseline_attack_probability = (
        baseline_probs[
            baseline_pred
        ]
    )


    post_attack_probability = (
        probabilities[
            baseline_pred
        ]
    )


    reduction = (
        baseline_attack_probability
        - post_attack_probability
    )


    print(
        f"\n{intervention}"
    )

    print(
        f"  Predicted class: "
        f"{CLASS_NAMES[prediction]}"
    )

    print(
        f"  BENIGN:          "
        f"{probabilities[0] * 100:9.3f}%"
    )

    print(
        f"  DOS_DDOS:        "
        f"{probabilities[1] * 100:9.3f}%"
    )

    print(
        f"  RECONNAISSANCE:  "
        f"{probabilities[2] * 100:9.3f}%"
    )

    print(
        f"  Original-class probability reduction: "
        f"{reduction * 100:9.3f} percentage points"
    )


    simulation_rows.append({

        "sample_index": SAMPLE_INDEX,

        "intervention": intervention,

        "true_class": CLASS_NAMES[
            true_class
        ],

        "baseline_class": CLASS_NAMES[
            baseline_pred
        ],

        "predicted_class": CLASS_NAMES[
            prediction
        ],

        "benign_probability": probabilities[0],

        "dos_ddos_probability": probabilities[1],

        "reconnaissance_probability": probabilities[2],

        "original_class_probability": (
            post_attack_probability
        ),

        "probability_reduction": (
            reduction
        ),
    })


# ============================================================
# SAVE SIMULATION RESULTS
# ============================================================

simulation_df = pd.DataFrame(
    simulation_rows
)


simulation_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SIH DEMO SUMMARY")
print("=" * 70)

print(
    f"Sample: {SAMPLE_INDEX}"
)

print(
    f"True class: {CLASS_NAMES[true_class]}"
)

print(
    f"Baseline prediction: {CLASS_NAMES[baseline_pred]}"
)

print(
    f"Baseline confidence: "
    f"{baseline_probs[baseline_pred] * 100:.3f}%"
)


portscan_row = simulation_df[
    simulation_df["intervention"]
    == "Port-scan blocking"
]


if len(portscan_row) == 1:

    row = portscan_row.iloc[0]

    print(
        "\nPort-scan blocking what-if:"
    )

    print(
        f"  Original-class probability: "
        f"{row['original_class_probability'] * 100:.3f}%"
    )

    print(
        f"  Probability reduction: "
        f"{row['probability_reduction'] * 100:.3f} percentage points"
    )

    print(
        f"  Post-intervention prediction: "
        f"{row['predicted_class']}"
    )


print("\n" + "=" * 70)
print("SAVED RESULTS")
print("=" * 70)

print(
    OUTPUT_FILE
)

print(
    "\nExplainability + intervention completed successfully."
)
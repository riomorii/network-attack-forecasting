import os

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from world_model import MultiTaskWorldModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = (
    "models/multitask_world_model_final.pth"
)

SCALER_FILE = (
    "models/multitask_world_model_final_scaler.pkl"
)

INPUT_SIZE = 15
HIDDEN_SIZE = 64
OUTPUT_SIZE = 15
NUM_CLASSES = 3

HISTORY = 5

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
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_FILE}"
        )


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


    return model


# ============================================================
# LOAD SCALER
# ============================================================

def load_scaler():

    if not os.path.exists(SCALER_FILE):

        raise FileNotFoundError(
            f"Scaler file not found:\n{SCALER_FILE}"
        )


    scaler = joblib.load(
        SCALER_FILE
    )


    return scaler


# ============================================================
# SCALE INPUT
# ============================================================

def scale_input(
    network_states,
    scaler
):
    """
    Convert raw 5 x 15 network states into
    StandardScaler-normalized values.
    """

    array = np.asarray(
        network_states,
        dtype=np.float32
    )


    if array.shape != (
        HISTORY,
        INPUT_SIZE
    ):

        raise ValueError(
            "\nExpected input shape "
            f"({HISTORY}, {INPUT_SIZE}) "
            f"but received {array.shape}."
        )


    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    scaled = scaler.transform(
        array
    )


    return scaled.astype(
        np.float32
    )


# ============================================================
# PREDICT NEXT STATE
# ============================================================

@torch.no_grad()
def predict_next_state(
    model,
    scaler,
    network_states
):
    """
    Predict the next network state and attack class.

    Parameters
    ----------
    network_states : array-like, shape (5, 15)
        Five historical network states.

    Returns
    -------
    result : dict
        Contains:
        - future_state
        - predicted_class_id
        - predicted_class
        - probabilities
    """

    scaled_input = scale_input(
        network_states,
        scaler
    )


    tensor = torch.tensor(
        scaled_input,
        dtype=torch.float32,
        device=DEVICE
    )


    tensor = tensor.unsqueeze(0)


    future_state, class_logits = model(
        tensor
    )


    probabilities = F.softmax(
        class_logits,
        dim=1
    )


    predicted_class_id = int(
        torch.argmax(
            probabilities,
            dim=1
        ).item()
    )


    future_state = (
        future_state
        .cpu()
        .numpy()[0]
    )


    probabilities = (
        probabilities
        .cpu()
        .numpy()[0]
    )


    return {
        "future_state": future_state,

        "predicted_class_id": predicted_class_id,

        "predicted_class": CLASS_NAMES[
            predicted_class_id
        ],

        "probabilities": {
            "BENIGN": float(
                probabilities[0]
            ),

            "DOS_DDOS": float(
                probabilities[1]
            ),

            "RECONNAISSANCE": float(
                probabilities[2]
            ),
        }
    }


# ============================================================
# MULTI-STEP FORECAST
# ============================================================

@torch.no_grad()
def forecast_multiple_steps(
    model,
    scaler,
    network_states,
    steps=5
):
    """
    Perform autoregressive forecasting.

    Each predicted future state is fed back into
    the model to produce the next prediction.

    Parameters
    ----------
    network_states : array-like, shape (5, 15)
        Initial historical states.

    steps : int
        Number of future states to predict.

    Returns
    -------
    results : list of dictionaries
    """

    if steps < 1:

        raise ValueError(
            "steps must be at least 1."
        )


    if steps > 20:

        raise ValueError(
            "steps should not exceed 20 "
            "for this demo."
        )


    current_states = np.asarray(
        network_states,
        dtype=np.float32
    ).copy()


    results = []


    for horizon in range(
        1,
        steps + 1
    ):

        result = predict_next_state(
            model,
            scaler,
            current_states
        )


        results.append(
            {
                "horizon": horizon,

                "predicted_class": (
                    result["predicted_class"]
                ),

                "predicted_class_id": (
                    result["predicted_class_id"]
                ),

                "BENIGN_probability": (
                    result["probabilities"]["BENIGN"]
                ),

                "DOS_DDOS_probability": (
                    result["probabilities"]["DOS_DDOS"]
                ),

                "RECONNAISSANCE_probability": (
                    result[
                        "probabilities"
                    ]["RECONNAISSANCE"]
                ),

                "future_state": (
                    result["future_state"]
                ),
            }
        )


        # Feed predicted future state back into
        # the input history.

        next_state = (
            result["future_state"]
            .reshape(
                1,
                INPUT_SIZE
            )
        )


        current_states = np.concatenate(
            [
                current_states[1:],
                next_state
            ],
            axis=0
        )


    return results


# ============================================================
# DEMO / COMMAND-LINE TEST
# ============================================================

def main():

    print("=" * 70)
    print("WORLD MODEL — CLEAN INFERENCE TEST")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    model = load_model()

    scaler = load_scaler()


    print(
        f"Loaded model: {MODEL_FILE}"
    )

    print(
        f"Loaded scaler: {SCALER_FILE}"
    )


    # --------------------------------------------------------
    # Load existing test data
    # --------------------------------------------------------

    X_TEST_FILE = (
        "data/multiclass_3class_X_test.npy"
    )

    Y_TEST_FILE = (
        "data/multiclass_3class_y_test.npy"
    )


    if not os.path.exists(
        X_TEST_FILE
    ):

        raise FileNotFoundError(
            f"Test input file not found:\n"
            f"{X_TEST_FILE}"
        )


    if not os.path.exists(
        Y_TEST_FILE
    ):

        raise FileNotFoundError(
            f"Test label file not found:\n"
            f"{Y_TEST_FILE}"
        )


    X_test = np.load(
        X_TEST_FILE
    )

    y_test = np.load(
        Y_TEST_FILE
    )


    # --------------------------------------------------------
    # Use the same informative sample that we identified
    # earlier for the SIH demonstration.
    # --------------------------------------------------------

    SAMPLE_INDEX = 4406


    if SAMPLE_INDEX >= len(X_test):

        raise IndexError(
            f"Sample {SAMPLE_INDEX} is outside "
            f"the test dataset."
        )


    sample = X_test[
        SAMPLE_INDEX
    ]


    true_class_id = int(
        y_test[SAMPLE_INDEX]
    )


    # --------------------------------------------------------
    # One-step prediction
    # --------------------------------------------------------

    result = predict_next_state(
        model,
        scaler,
        sample
    )


    print("\n" + "=" * 70)
    print("ONE-STEP FORECAST")
    print("=" * 70)


    print(
        f"Sample index: {SAMPLE_INDEX}"
    )


    print(
        f"True class: "
        f"{CLASS_NAMES[true_class_id]}"
    )


    print(
        f"Predicted class: "
        f"{result['predicted_class']}"
    )


    print("\nProbabilities:")


    for class_name in CLASS_NAMES:

        probability = (
            result[
                "probabilities"
            ][class_name]
        )


        print(
            f"  {class_name:18s}: "
            f"{probability * 100:8.3f}%"
        )


    # --------------------------------------------------------
    # Five-step autoregressive forecast
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FIVE-STEP AUTOREGRESSIVE FORECAST")
    print("=" * 70)


    forecast = forecast_multiple_steps(
        model,
        scaler,
        sample,
        steps=5
    )


    for item in forecast:

        print(
            f"\n+{item['horizon']} state"
        )

        print(
            f"  Predicted class: "
            f"{item['predicted_class']}"
        )

        print(
            f"  BENIGN: "
            f"{item['BENIGN_probability'] * 100:8.3f}%"
        )

        print(
            f"  DOS_DDOS: "
            f"{item['DOS_DDOS_probability'] * 100:8.3f}%"
        )

        print(
            f"  RECONNAISSANCE: "
            f"{item['RECONNAISSANCE_probability'] * 100:8.3f}%"
        )


    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLEAN INFERENCE TEST SUCCESSFUL")
    print("=" * 70)

    print(
        "The dashboard can use predict_next_state()."
    )

    print(
        "The dashboard can use forecast_multiple_steps() "
        "for +1 to +5 forecasting."
    )

    print(
        "No training was performed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
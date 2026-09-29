import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from world_model import MultiTaskWorldModel


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
X_TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "multiclass_3class_X_test.npy"
)

Y_TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "multiclass_3class_y_test.npy"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "multitask_world_model_final.pth"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "multitask_world_model_final_scaler.pkl"
)

OUTPUT_ALL = os.path.join(
    PROJECT_ROOT,
    "data",
    "final_test_predictions.csv"
)

OUTPUT_FP = os.path.join(
    PROJECT_ROOT,
    "data",
    "false_positive_samples.csv"
)


CLASS_NAMES = {
    0: "BENIGN",
    1: "DOS_DDOS",
    2: "RECONNAISSANCE",
}


def normalize_labels(y):
    y = np.asarray(y)

    if np.issubdtype(y.dtype, np.number):
        return y.astype(int)

    mapping = {
        "BENIGN": 0,
        "DOS_DDOS": 1,
        "RECONNAISSANCE": 2,
    }

    result = []

    for value in y:
        value = str(value).strip()

        if value not in mapping:
            raise ValueError(
                f"Unknown label: {value}"
            )

        result.append(mapping[value])

    return np.asarray(result, dtype=int)


def load_model():
    model = MultiTaskWorldModel(
        input_size=15,
        hidden_size=64,
        num_classes=3,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


def get_class_logits(output):
    if isinstance(output, (tuple, list)):
        return output[1]

    if isinstance(output, dict):
        for key in [
            "class_logits",
            "logits",
            "class_output",
        ]:
            if key in output:
                return output[key]

    raise RuntimeError(
        "Could not find classification logits."
    )


def main():

    print("=" * 70)
    print("CORRECTED FALSE POSITIVE ANALYSIS")
    print("=" * 70)

    # -----------------------------------------------------
    # Load test data
    # -----------------------------------------------------
    X_test = np.load(X_TEST_PATH)

    y_test_raw = np.load(
        Y_TEST_PATH,
        allow_pickle=True
    )

    y_test = normalize_labels(y_test_raw)

    print(f"Raw X_test shape: {X_test.shape}")
    print(f"y_test shape:     {y_test.shape}")

    if X_test.shape != (42458, 5, 15):
        raise ValueError(
            f"Unexpected X_test shape: {X_test.shape}"
        )

    # -----------------------------------------------------
    # Load SAME scaler used by final evaluation
    # -----------------------------------------------------
    scaler = joblib.load(SCALER_PATH)

    print("Loaded scaler:")
    print(SCALER_PATH)

    # -----------------------------------------------------
    # IMPORTANT:
    # Apply scaler feature-wise to the 15 features.
    # -----------------------------------------------------
    original_shape = X_test.shape

    X_flat = X_test.reshape(-1, 15)

    X_scaled_flat = scaler.transform(X_flat)

    X_test_scaled = X_scaled_flat.reshape(
        original_shape
    )

    print(
        f"Scaled X_test shape: {X_test_scaled.shape}"
    )

    # -----------------------------------------------------
    # Load saved final model
    # -----------------------------------------------------
    model = load_model()

    print("Loaded model:")
    print(MODEL_PATH)

    print("No training performed.")

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------
    X_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )

    with torch.no_grad():

        output = model(X_tensor)

        logits = get_class_logits(output)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

    y_pred = predictions.cpu().numpy()

    probs = probabilities.cpu().numpy()

    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------
    correct = int(
        np.sum(y_pred == y_test)
    )

    total = len(y_test)

    wrong = total - correct

    accuracy = correct / total

    print()
    print("=" * 70)
    print("FINAL ACCURACY RE-CHECK")
    print("=" * 70)

    print(
        f"Total test samples : {total}"
    )

    print(
        f"Correct predictions: {correct}"
    )

    print(
        f"Wrong predictions  : {wrong}"
    )

    print(
        f"Accuracy           : {accuracy:.4f}"
    )

    print(
        f"Accuracy (%)       : {accuracy * 100:.2f}%"
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2]
    )

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print("Rows    = Actual")
    print("Columns = Predicted")
    print()

    print(
        "                 BENIGN   DOS_DDOS   RECON"
    )

    print(
        f"BENIGN          {cm[0,0]:7d}"
        f" {cm[0,1]:10d}"
        f" {cm[0,2]:8d}"
    )

    print(
        f"DOS_DDOS        {cm[1,0]:7d}"
        f" {cm[1,1]:10d}"
        f" {cm[1,2]:8d}"
    )

    print(
        f"RECONNAISSANCE  {cm[2,0]:7d}"
        f" {cm[2,1]:10d}"
        f" {cm[2,2]:8d}"
    )

    # -----------------------------------------------------
    # Attack false positives
    # Actual BENIGN → Predicted attack
    # -----------------------------------------------------
    benign_to_dos = int(cm[0, 1])
    benign_to_recon = int(cm[0, 2])

    total_fp = (
        benign_to_dos +
        benign_to_recon
    )

    fp_rate = total_fp / total

    print()
    print("=" * 70)
    print("ATTACK FALSE POSITIVES")
    print("=" * 70)

    print(
        f"BENIGN -> DOS_DDOS       : {benign_to_dos}"
    )

    print(
        f"BENIGN -> RECONNAISSANCE : {benign_to_recon}"
    )

    print(
        f"Total attack false positives: {total_fp}"
    )

    print(
        f"False positive rate: "
        f"{fp_rate * 100:.4f}%"
    )

    # -----------------------------------------------------
    # Extract FP rows
    # -----------------------------------------------------
    fp_mask = (
        (y_test == 0) &
        (y_pred != 0)
    )

    fp_indices = np.where(fp_mask)[0]

    fp_rows = []

    for idx in fp_indices:

        pred_class = int(y_pred[idx])

        fp_rows.append({
            "test_sample_index": int(idx),
            "actual_class": "BENIGN",
            "predicted_class":
                CLASS_NAMES[pred_class],
            "prob_benign":
                float(probs[idx, 0]),
            "prob_dos_ddos":
                float(probs[idx, 1]),
            "prob_reconnaissance":
                float(probs[idx, 2]),
            "prediction_confidence":
                float(np.max(probs[idx])),
        })

    fp_df = pd.DataFrame(fp_rows)

    # -----------------------------------------------------
    # Save ALL predictions
    # -----------------------------------------------------
    all_rows = []

    for idx in range(total):

        actual = int(y_test[idx])
        predicted = int(y_pred[idx])

        all_rows.append({
            "test_sample_index": int(idx),
            "actual_class":
                CLASS_NAMES[actual],
            "predicted_class":
                CLASS_NAMES[predicted],
            "prob_benign":
                float(probs[idx, 0]),
            "prob_dos_ddos":
                float(probs[idx, 1]),
            "prob_reconnaissance":
                float(probs[idx, 2]),
            "correct":
                bool(actual == predicted),
        })

    all_df = pd.DataFrame(all_rows)

    all_df.to_csv(
        OUTPUT_ALL,
        index=False
    )

    fp_df.to_csv(
        OUTPUT_FP,
        index=False
    )

    # -----------------------------------------------------
    # Show false positives
    # -----------------------------------------------------
    print()
    print("=" * 70)
    print("FIRST FALSE POSITIVE SAMPLES")
    print("=" * 70)

    if fp_df.empty:
        print("No attack false positives found.")
    else:
        print(
            fp_df.head(20).to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(OUTPUT_ALL)
    print(OUTPUT_FP)

    print()
    print(
        "CORRECTED FALSE POSITIVE ANALYSIS COMPLETED."
    )


if __name__ == "__main__":
    main()
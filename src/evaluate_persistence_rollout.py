import os
import bisect
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data"

METADATA_FILE = os.path.join(
    DATA_DIR,
    "multiclass_state_metadata.csv"
)

STATE_LABELS_FILE = os.path.join(
    DATA_DIR,
    "multiclass_state_labels.npy"
)

ROLLOUT_TRUE_FILE = os.path.join(
    DATA_DIR,
    "multitask_rollout_true_classes.npy"
)

OUTPUT_CSV = os.path.join(
    DATA_DIR,
    "persistence_rollout_results.csv"
)

OUTPUT_PRED = os.path.join(
    DATA_DIR,
    "persistence_rollout_predicted_classes.npy"
)

HISTORY = 5
FUTURE = 5

CLASS_IDS = [0, 1, 2]

CLASS_NAMES = [
    "BENIGN",
    "DOS_DDOS",
    "RECONNAISSANCE",
]

LABEL_TO_ID = {
    "BENIGN": 0,
    "DOS_DDOS": 1,
    "RECONNAISSANCE": 2,
}


# ============================================================
# HEADER
# ============================================================

print("=" * 63)
print("PERSISTENCE BASELINE — MATCHING SAVED WORLD MODEL ROLLOUT")
print("=" * 63)


# ============================================================
# LOAD DATA
# ============================================================

metadata = pd.read_csv(METADATA_FILE)

state_labels_raw = np.load(
    STATE_LABELS_FILE,
    allow_pickle=True
)

saved_true_future = np.load(
    ROLLOUT_TRUE_FILE
)


print(f"\nMetadata shape:      {metadata.shape}")
print(f"State labels shape:  {state_labels_raw.shape}")
print(f"Saved rollout truth: {saved_true_future.shape}")


if saved_true_future.ndim != 2:
    raise ValueError(
        f"Expected rollout truth to be 2D, got "
        f"{saved_true_future.shape}"
    )

if saved_true_future.shape[1] != FUTURE:
    raise ValueError(
        f"Expected {FUTURE} future steps, got "
        f"{saved_true_future.shape[1]}"
    )


TARGET_SAMPLES = saved_true_future.shape[0]


# ============================================================
# CONVERT RAW STATE LABELS
# ============================================================

state_labels = np.array([
    LABEL_TO_ID.get(str(x).strip(), -1)
    for x in state_labels_raw
], dtype=np.int64)


print("\nState labels:")

for class_id, class_name in enumerate(CLASS_NAMES):
    print(
        f"  {class_id} = {class_name}: "
        f"{np.sum(state_labels == class_id)}"
    )

print(
    f"  -1 = UNKNOWN: "
    f"{np.sum(state_labels == -1)}"
)


# ============================================================
# BUILD TEST REGIONS BY SCENARIO
# ============================================================

source_files = metadata["source_file"].astype(str)

unique_sources = list(
    dict.fromkeys(source_files.tolist())
)


print("\n" + "=" * 63)
print("BUILDING ALL POSSIBLE TEST ROLLOUT WINDOWS")
print("=" * 63)


candidate_future = []
candidate_current = []
candidate_source = []


for source in unique_sources:

    mask = source_files == source

    indices = np.where(mask)[0]

    scenario_labels = state_labels[indices]

    n_states = len(scenario_labels)

    # Number of sequence samples when history = 5.
    n_sequences = n_states - HISTORY

    if n_sequences <= 0:
        continue


    # Same temporal sequence split:
    train_end = int(n_sequences * 0.70)
    val_end = int(n_sequences * 0.85)

    test_start = val_end
    test_end = n_sequences


    # Each sequence start s has:
    #
    # history:
    #   s ... s+4
    #
    # current:
    #   s+4
    #
    # future:
    #   s+5 ... s+9
    #
    # We generate ALL possible test rollout windows here.
    # We will later match them against the SAVED World Model
    # rollout rows.


    max_start = min(
        test_end,
        n_states - HISTORY - FUTURE + 1
    )


    for s in range(
        test_start,
        max_start
    ):

        current_index = s + HISTORY - 1

        future_start = s + HISTORY

        future_end = future_start + FUTURE


        current_label = scenario_labels[
            current_index
        ]

        future = scenario_labels[
            future_start:future_end
        ]


        if len(future) != FUTURE:
            continue


        candidate_current.append(
            int(current_label)
        )

        candidate_future.append(
            tuple(
                int(x) for x in future
            )
        )

        candidate_source.append(
            source
        )


print(
    f"Total candidate windows: "
    f"{len(candidate_future)}"
)


# ============================================================
# INDEX WINDOWS BY THEIR FIVE FUTURE LABELS
# ============================================================

print("\nIndexing candidate windows...")


positions = {}


for idx, future_tuple in enumerate(candidate_future):

    positions.setdefault(
        future_tuple,
        []
    ).append(idx)


# ============================================================
# MATCH THE SAVED WORLD MODEL WINDOWS
# ============================================================

print("\n" + "=" * 63)
print("MATCHING SAVED WORLD MODEL ROLLOUT WINDOWS")
print("=" * 63)


matched_indices = []

search_position = 0


for row_number in range(TARGET_SAMPLES):

    target_tuple = tuple(
        int(x)
        for x in saved_true_future[row_number]
    )


    possible_positions = positions.get(
        target_tuple,
        []
    )


    # Find first candidate at or after the current search position.
    pos = bisect.bisect_left(
        possible_positions,
        search_position
    )


    if pos >= len(possible_positions):

        raise ValueError(
            "\nCould not align saved World Model rollout "
            f"row {row_number}.\n\n"
            f"Target future sequence: {target_tuple}\n"
            f"Search position: {search_position}\n\n"
            "The persistence baseline cannot safely continue."
        )


    selected = possible_positions[pos]

    matched_indices.append(
        selected
    )

    search_position = selected + 1


matched_indices = np.asarray(
    matched_indices,
    dtype=np.int64
)


print(
    f"Matched rows: {len(matched_indices)}"
)


# ============================================================
# VERIFY EXACT MATCH
# ============================================================

matched_future = np.asarray([
    candidate_future[i]
    for i in matched_indices
], dtype=np.int64)


if not np.array_equal(
    matched_future,
    saved_true_future
):

    raise ValueError(
        "\nFINAL ALIGNMENT CHECK FAILED.\n"
        "The matched future-label sequences are not identical "
        "to the World Model rollout truth."
    )


print(
    "Exact future-label alignment verified."
)


# ============================================================
# GET CURRENT STATES
# ============================================================

current_labels = np.asarray([
    candidate_current[i]
    for i in matched_indices
], dtype=np.int64)


sources = [
    candidate_source[i]
    for i in matched_indices
]


print(
    f"Current labels: {current_labels.shape}"
)


# ============================================================
# CHECK CURRENT LABELS
# ============================================================

unknown_current = np.sum(
    ~np.isin(current_labels, CLASS_IDS)
)


print(
    f"Unknown current labels: {unknown_current}"
)


if unknown_current > 0:

    print(
        "\nWARNING:"
        " Some starting states are outside the three "
        "trained classes."
    )


# ============================================================
# BUILD PERSISTENCE PREDICTIONS
# ============================================================

print("\n" + "=" * 63)
print("BUILDING PERSISTENCE PREDICTIONS")
print("=" * 63)


# Persistence assumption:
#
# Current state remains unchanged.
#
# Example:
#
# current = DOS_DDOS
#
# +1 = DOS_DDOS
# +2 = DOS_DDOS
# +3 = DOS_DDOS
# +4 = DOS_DDOS
# +5 = DOS_DDOS


predicted_classes = np.repeat(
    current_labels[:, None],
    FUTURE,
    axis=1
)


print(
    f"Prediction shape: {predicted_classes.shape}"
)


# ============================================================
# EVALUATE
# ============================================================

print("\n" + "=" * 63)
print("PERSISTENCE RESULTS")
print("=" * 63)


results = []


for h in range(FUTURE):

    horizon = h + 1

    y_true = saved_true_future[:, h]

    y_pred = predicted_classes[:, h]


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
        labels=CLASS_IDS,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_IDS
    )


    print(f"\n+{horizon} state")

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

    print("  Confusion Matrix:")

    print(cm)


    results.append({
        "horizon": horizon,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    })


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)


results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


np.save(
    OUTPUT_PRED,
    predicted_classes
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 63)
print("PERSISTENCE SUMMARY")
print("=" * 63)


print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 63)
print("SAVED FILES")
print("=" * 63)

print(OUTPUT_CSV)
print(OUTPUT_PRED)


print(
    "\nPersistence evaluation completed successfully."
)
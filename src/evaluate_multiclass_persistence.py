import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# ============================================
# LOAD
# ============================================

X_test = np.load(
    "data/multiclass_3class_X_test.npy"
)

y_test = np.load(
    "data/multiclass_3class_y_test.npy"
)

print("===============================================")
print("MULTICLASS PERSISTENCE BASELINE")
print("===============================================")

print("X test:", X_test.shape)
print("y test:", y_test.shape)

# ============================================
# PERSISTENCE PREDICTION
# ============================================

# We do not actually need the feature values here.
#
# The sequence target is the state immediately
# following the 5-state input window.
#
# Persistence prediction:
# use the label of the CURRENT state.
#
# The label is not stored inside X, so we reconstruct
# it from the original multiclass labels.

original_test_labels = np.load(
    "data/multiclass_y_test.npy",
    allow_pickle=True
)

# The current state for each sequence is the state
# immediately before its target.
#
# The test sequence creation started from the original
# test state sequence, so we need the corresponding
# current-state labels.

# ============================================
# RECONSTRUCT CURRENT STATE LABELS
# ============================================

# The original multiclass sequence generation used:
#
# X[i] = states[i:i+5]
# y[i] = states[i+5]
#
# Therefore current-state label = labels[i+4].
#
# However, multiclass_y_test contains only target labels.
# We therefore use the test scenario metadata to reconstruct
# the correct state labels.

test_scenarios = np.load(
    "data/multiclass_test_scenarios.npy",
    allow_pickle=True
)

metadata = __import__("pandas").read_csv(
    "data/multiclass_state_metadata.csv"
)

all_labels = np.load(
    "data/multiclass_state_labels.npy",
    allow_pickle=True
)

# Build state index sequences exactly as before.

scenario_column = "source_file"

current_labels = []

for scenario in test_scenarios:

    # This reconstruction is handled below using a
    # sequential pointer for each scenario.
    pass

# ============================================
# EASIER APPROACH
# ============================================

# Recreate all temporal sequences and the original
# target/current labels.

current_list = []
target_list = []
scenario_list = []

for scenario, group in metadata.groupby(
    scenario_column,
    sort=False
):

    indices = group.index.to_numpy()

    scenario_labels = all_labels[indices]

    n = len(scenario_labels)

    if n <= 5:
        continue

    for i in range(n - 5):

        current_label = str(
            scenario_labels[i + 4]
        )

        target_label = str(
            scenario_labels[i + 5]
        )

        current_list.append(
            current_label
        )

        target_list.append(
            target_label
        )

        scenario_list.append(
            scenario
        )

# ============================================
# APPLY SAME TEMPORAL SPLIT
# ============================================

current_list = np.array(
    current_list
)

target_list = np.array(
    target_list
)

scenario_list = np.array(
    scenario_list
)

test_indices = []

for scenario in np.unique(scenario_list):

    idx = np.where(
        scenario_list == scenario
    )[0]

    n = len(idx)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    test_indices.extend(
        idx[val_end:]
    )

test_indices = np.array(
    test_indices
)

current_test = current_list[
    test_indices
]

target_test = target_list[
    test_indices
]

# ============================================
# KEEP ONLY 3 CLASSES
# ============================================

mapping = {
    "BENIGN": 0,
    "DOS_DDOS": 1,
    "RECONNAISSANCE": 2
}

valid = (
    np.isin(
        current_test,
        list(mapping.keys())
    )
    &
    np.isin(
        target_test,
        list(mapping.keys())
    )
)

current_test = current_test[valid]
target_test = target_test[valid]

current_encoded = np.array(
    [mapping[x] for x in current_test]
)

target_encoded = np.array(
    [mapping[x] for x in target_test]
)

# ============================================
# PERSISTENCE PREDICTION
# ============================================

predictions = current_encoded

# ============================================
# METRICS
# ============================================

accuracy = accuracy_score(
    target_encoded,
    predictions
)

balanced_accuracy = balanced_accuracy_score(
    target_encoded,
    predictions
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        target_encoded,
        predictions,
        labels=[0, 1, 2],
        zero_division=0
    )
)

cm = confusion_matrix(
    target_encoded,
    predictions,
    labels=[0, 1, 2]
)

class_names = [
    "BENIGN",
    "DOS_DDOS",
    "RECONNAISSANCE"
]

# ============================================
# OUTPUT
# ============================================

print("\n===============================================")
print("RESULTS")
print("===============================================")

print(
    f"Accuracy:          {accuracy:.4f}"
)

print(
    f"Balanced Accuracy: {balanced_accuracy:.4f}"
)

print(
    f"Macro Precision:   {precision.mean():.4f}"
)

print(
    f"Macro Recall:      {recall.mean():.4f}"
)

print(
    f"Macro F1:          {f1.mean():.4f}"
)

print("\n===============================================")
print("PER-CLASS RESULTS")
print("===============================================")

for i, name in enumerate(class_names):

    print(
        f"{name:18s} "
        f"Precision={precision[i]:.4f} "
        f"Recall={recall[i]:.4f} "
        f"F1={f1[i]:.4f} "
        f"Support={support[i]}"
    )

print("\n===============================================")
print("CONFUSION MATRIX")
print("===============================================")

print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(
    "                BENIGN  DOS_DDOS  RECON"
)

for i, row in enumerate(cm):

    print(
        f"{class_names[i]:15s} "
        f"{row[0]:7d} "
        f"{row[1]:9d} "
        f"{row[2]:7d}"
    )

print("\n===============================================")
print("CLASSIFICATION REPORT")
print("===============================================")

print(
    classification_report(
        target_encoded,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)
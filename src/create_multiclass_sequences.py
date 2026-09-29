import numpy as np
import pandas as pd
from pathlib import Path

# ============================================
# CONFIG
# ============================================

STATE_FILE = Path("data/multiclass_network_states.npy")
LABEL_FILE = Path("data/multiclass_state_labels.npy")
META_FILE = Path("data/multiclass_state_metadata.csv")

WINDOW_SIZE = 5

# ============================================
# LOAD
# ============================================

states = np.load(STATE_FILE)
labels = np.load(LABEL_FILE)
metadata = pd.read_csv(META_FILE)

print("===============================================")
print("MULTI-SCENARIO SEQUENCE CREATION")
print("===============================================")

print(f"States shape: {states.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Metadata shape: {metadata.shape}")

# ============================================
# CREATE SEQUENCES WITHOUT CROSSING SCENARIOS
# ============================================

X_list = []
y_list = []
scenario_list = []

# The metadata contains the original scenario/file.
# We process each scenario independently so that
# a sequence never crosses from one CSV into another.

scenario_column = None

for candidate in ["source_file", "scenario", "file", "source"]:
    if candidate in metadata.columns:
        scenario_column = candidate
        break

if scenario_column is None:
    raise ValueError(
        "Could not find scenario/source column in metadata. "
        f"Available columns: {list(metadata.columns)}"
    )

print(f"\nUsing scenario column: {scenario_column}")

for scenario, group in metadata.groupby(scenario_column, sort=False):

    indices = group.index.to_numpy()

    # Metadata rows should correspond to state rows.
    scenario_states = states[indices]
    scenario_labels = labels[indices]

    if len(scenario_states) <= WINDOW_SIZE:
        continue

    scenario_sequences = 0

    for i in range(len(scenario_states) - WINDOW_SIZE):

        # Previous WINDOW_SIZE states
        X_list.append(
            scenario_states[i:i + WINDOW_SIZE]
        )

        # Next state label
        y_list.append(
            scenario_labels[i + WINDOW_SIZE]
        )

        scenario_list.append(scenario)

        scenario_sequences += 1

    print(
        f"{str(scenario)[:55]:55s} "
        f"states={len(scenario_states):6d} "
        f"sequences={scenario_sequences:6d}"
    )

# ============================================
# CONVERT
# ============================================

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)

scenario_array = np.array(scenario_list)

print("\n===============================================")
print("SEQUENCE SUMMARY")
print("===============================================")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ============================================
# TEMPORAL TRAIN / VAL / TEST SPLIT
# ============================================

# IMPORTANT:
# Split each scenario chronologically.
#
# 70% -> train
# 15% -> validation
# 15% -> test
#
# This prevents future traffic from leaking into training.

train_indices = []
val_indices = []
test_indices = []

for scenario in np.unique(scenario_array):

    idx = np.where(scenario_array == scenario)[0]

    n = len(idx)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_indices.extend(idx[:train_end])
    val_indices.extend(idx[train_end:val_end])
    test_indices.extend(idx[val_end:])

train_indices = np.array(train_indices)
val_indices = np.array(val_indices)
test_indices = np.array(test_indices)

X_train = X[train_indices]
y_train = y[train_indices]

X_val = X[val_indices]
y_val = y[val_indices]

X_test = X[test_indices]
y_test = y[test_indices]

# ============================================
# SAVE
# ============================================

np.save("data/multiclass_X_train.npy", X_train)
np.save("data/multiclass_y_train.npy", y_train)

np.save("data/multiclass_X_val.npy", X_val)
np.save("data/multiclass_y_val.npy", y_val)

np.save("data/multiclass_X_test.npy", X_test)
np.save("data/multiclass_y_test.npy", y_test)

np.save("data/multiclass_test_scenarios.npy", scenario_array[test_indices])

# ============================================
# REPORT
# ============================================

print("\n===============================================")
print("FINAL SPLIT")
print("===============================================")

print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Val:   X={X_val.shape}, y={y_val.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")

print("\nTrain labels:")
print(pd.Series(y_train).value_counts())

print("\nValidation labels:")
print(pd.Series(y_val).value_counts())

print("\nTest labels:")
print(pd.Series(y_test).value_counts())

print("\n===============================================")
print("SAVED")
print("===============================================")

print("data/multiclass_X_train.npy")
print("data/multiclass_y_train.npy")
print("data/multiclass_X_val.npy")
print("data/multiclass_y_val.npy")
print("data/multiclass_X_test.npy")
print("data/multiclass_y_test.npy")
print("data/multiclass_test_scenarios.npy")
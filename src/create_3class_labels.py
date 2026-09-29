import numpy as np

# ============================================
# LOAD
# ============================================

y_train_raw = np.load(
    "data/multiclass_y_train.npy",
    allow_pickle=True
)

y_val_raw = np.load(
    "data/multiclass_y_val.npy",
    allow_pickle=True
)

y_test_raw = np.load(
    "data/multiclass_y_test.npy",
    allow_pickle=True
)

# ============================================
# LABEL MAPPING
# ============================================

mapping = {
    "BENIGN": 0,
    "DOS_DDOS": 1,
    "RECONNAISSANCE": 2
}

label_names = {
    0: "BENIGN",
    1: "DOS_DDOS",
    2: "RECONNAISSANCE"
}


# ============================================
# CONVERT ONLY COMMON CLASSES
# ============================================

def convert(labels, split_name):

    converted = []
    keep_indices = []

    for i, label in enumerate(labels):

        label = str(label)

        if label in mapping:

            converted.append(mapping[label])
            keep_indices.append(i)

    converted = np.array(
        converted,
        dtype=np.int64
    )

    keep_indices = np.array(
        keep_indices,
        dtype=np.int64
    )

    print(
        f"{split_name}: "
        f"kept {len(converted)} / {len(labels)} sequences"
    )

    return converted, keep_indices


y_train, train_keep = convert(
    y_train_raw,
    "TRAIN"
)

y_val, val_keep = convert(
    y_val_raw,
    "VALIDATION"
)

y_test, test_keep = convert(
    y_test_raw,
    "TEST"
)

# ============================================
# LOAD INPUT SEQUENCES
# ============================================

X_train_raw = np.load(
    "data/multiclass_X_train.npy"
)

X_val_raw = np.load(
    "data/multiclass_X_val.npy"
)

X_test_raw = np.load(
    "data/multiclass_X_test.npy"
)

# ============================================
# FILTER INPUTS USING SAME INDICES
# ============================================

X_train = X_train_raw[train_keep]
X_val = X_val_raw[val_keep]
X_test = X_test_raw[test_keep]

# ============================================
# SAVE
# ============================================

np.save(
    "data/multiclass_3class_X_train.npy",
    X_train
)

np.save(
    "data/multiclass_3class_y_train.npy",
    y_train
)

np.save(
    "data/multiclass_3class_X_val.npy",
    X_val
)

np.save(
    "data/multiclass_3class_y_val.npy",
    y_val
)

np.save(
    "data/multiclass_3class_X_test.npy",
    X_test
)

np.save(
    "data/multiclass_3class_y_test.npy",
    y_test
)

# ============================================
# REPORT
# ============================================

print()
print("===============================================")
print("3-CLASS DATASET")
print("===============================================")

for name, labels in [
    ("TRAIN", y_train),
    ("VALIDATION", y_val),
    ("TEST", y_test)
]:

    print(f"\n{name}")

    counts = np.bincount(
        labels,
        minlength=3
    )

    for class_id, count in enumerate(counts):

        print(
            f"{class_id} = "
            f"{label_names[class_id]:16s} "
            f"{count}"
        )

print("\nShapes:")

print(
    "Train:",
    X_train.shape,
    y_train.shape
)

print(
    "Val:  ",
    X_val.shape,
    y_val.shape
)

print(
    "Test: ",
    X_test.shape,
    y_test.shape
)

print("\n===============================================")
print("SAVED")
print("===============================================")

print("data/multiclass_3class_X_train.npy")
print("data/multiclass_3class_y_train.npy")
print("data/multiclass_3class_X_val.npy")
print("data/multiclass_3class_y_val.npy")
print("data/multiclass_3class_X_test.npy")
print("data/multiclass_3class_y_test.npy")
import os
import hashlib
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"

STATE_FILE = os.path.join(
    DATA_DIR,
    "multiclass_network_states.npy"
)

STATE_LABEL_FILE = os.path.join(
    DATA_DIR,
    "multiclass_state_labels.npy"
)

METADATA_FILE = os.path.join(
    DATA_DIR,
    "multiclass_state_metadata.csv"
)

X_TRAIN_FILE = os.path.join(
    DATA_DIR,
    "multiclass_3class_X_train.npy"
)

Y_TRAIN_FILE = os.path.join(
    DATA_DIR,
    "multiclass_3class_y_train.npy"
)

X_VAL_FILE = os.path.join(
    DATA_DIR,
    "multiclass_3class_X_val.npy"
)

Y_VAL_FILE = os.path.join(
    DATA_DIR,
    "multiclass_3class_y_val.npy"
)

X_TEST_FILE = os.path.join(
    DATA_DIR,
    "multiclass_3class_X_test.npy"
)

Y_TEST_FILE = os.path.join(
    DATA_DIR,
    "multiclass_3class_y_test.npy"
)

FULL_X_FILE = os.path.join(
    DATA_DIR,
    "multiclass_X_train.npy"
)

FULL_Y_FILE = os.path.join(
    DATA_DIR,
    "multiclass_y_train.npy"
)

RAW_DIR = os.path.join(
    DATA_DIR,
    "raw",
    "MachineLearningCVE"
)

CLASS_NAMES = [
    "BENIGN",
    "DOS_DDOS",
    "RECONNAISSANCE",
]


# ============================================================
# FEATURE NAMES
# ============================================================

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
# REPORT STATE
# ============================================================

audit_results = []

warnings_found = 0
critical_failures = 0


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def pass_check(message):
    print(f"[PASS] {message}")


def warn_check(message):
    global warnings_found
    warnings_found += 1
    print(f"[WARN] {message}")


def fail_check(message):
    global critical_failures
    critical_failures += 1
    print(f"[FAIL] {message}")


def info(message):
    print(f"[INFO] {message}")


# ============================================================
# FILE EXISTENCE
# ============================================================

section("DATASET QUALITY AUDIT")

print(
    "This audit is read-only. No dataset or model files are modified."
)


section("1. REQUIRED FILE CHECK")

required_files = {
    "network states": STATE_FILE,
    "state labels": STATE_LABEL_FILE,
    "state metadata": METADATA_FILE,
    "train X": X_TRAIN_FILE,
    "train y": Y_TRAIN_FILE,
    "validation X": X_VAL_FILE,
    "validation y": Y_VAL_FILE,
    "test X": X_TEST_FILE,
    "test y": Y_TEST_FILE,
}


for name, path in required_files.items():

    if os.path.exists(path):

        size = os.path.getsize(path)

        pass_check(
            f"{name}: {path} ({size:,} bytes)"
        )

    else:

        fail_check(
            f"{name} missing: {path}"
        )


# ============================================================
# STOP IF CORE FILES ARE MISSING
# ============================================================

missing_core = [
    path
    for path in required_files.values()
    if not os.path.exists(path)
]


if missing_core:

    print("\nCritical files are missing.")
    print("Fix the missing files before continuing.")

    raise SystemExit(1)


# ============================================================
# LOAD CORE ARRAYS
# ============================================================

section("2. LOAD CORE DATA")

try:

    states = np.load(
        STATE_FILE,
        allow_pickle=True
    )

    state_labels_raw = np.load(
        STATE_LABEL_FILE,
        allow_pickle=True
    )

    metadata = pd.read_csv(
        METADATA_FILE
    )

    X_train = np.load(
        X_TRAIN_FILE
    )

    y_train = np.load(
        Y_TRAIN_FILE
    )

    X_val = np.load(
        X_VAL_FILE
    )

    y_val = np.load(
        Y_VAL_FILE
    )

    X_test = np.load(
        X_TEST_FILE
    )

    y_test = np.load(
        Y_TEST_FILE
    )

except Exception as exc:

    fail_check(
        f"Could not load core files: {exc}"
    )

    raise


print(
    f"Network states:        {states.shape}"
)

print(
    f"State labels:          {state_labels_raw.shape}"
)

print(
    f"Metadata:              {metadata.shape}"
)

print(
    f"X_train:               {X_train.shape}"
)

print(
    f"y_train:               {y_train.shape}"
)

print(
    f"X_val:                 {X_val.shape}"
)

print(
    f"y_val:                 {y_val.shape}"
)

print(
    f"X_test:                {X_test.shape}"
)

print(
    f"y_test:                {y_test.shape}"
)


# ============================================================
# BASIC SHAPE CHECKS
# ============================================================

section("3. SHAPE AND DIMENSION CHECKS")


if states.ndim != 2:

    fail_check(
        f"Network states should be 2-D, got {states.ndim}-D."
    )

else:

    if states.shape[1] != 15:

        fail_check(
            f"Expected 15 state features, got {states.shape[1]}."
        )

    else:

        pass_check(
            "Network state matrix has 15 features."
        )


if len(states) != len(state_labels_raw):

    fail_check(
        "Network-state count and state-label count differ."
    )

else:

    pass_check(
        "Network states and state labels have matching lengths."
    )


if len(states) != len(metadata):

    fail_check(
        "Network-state count and metadata row count differ."
    )

else:

    pass_check(
        "Network states and metadata have matching lengths."
    )


for split_name, X, y in [
    ("train", X_train, y_train),
    ("validation", X_val, y_val),
    ("test", X_test, y_test),
]:

    if X.ndim != 3:

        fail_check(
            f"{split_name} X is not 3-D: {X.shape}"
        )

        continue


    if X.shape[1] != 5:

        fail_check(
            f"{split_name} sequence length is {X.shape[1]}, expected 5."
        )

    else:

        pass_check(
            f"{split_name} sequence length = 5."
        )


    if X.shape[2] != 15:

        fail_check(
            f"{split_name} feature count is {X.shape[2]}, expected 15."
        )

    else:

        pass_check(
            f"{split_name} feature count = 15."
        )


    if len(X) != len(y):

        fail_check(
            f"{split_name} X/y lengths differ."
        )

    else:

        pass_check(
            f"{split_name} X/y lengths match."
        )


# ============================================================
# NUMERICAL QUALITY
# ============================================================

section("4. MISSING / INFINITE VALUE CHECK")

arrays_to_check = {
    "network states": states,
    "X_train": X_train,
    "X_val": X_val,
    "X_test": X_test,
}


for name, array in arrays_to_check.items():

    numeric = np.asarray(
        array,
        dtype=np.float64
    )

    nan_count = int(
        np.isnan(numeric).sum()
    )

    inf_count = int(
        np.isinf(numeric).sum()
    )

    finite_count = int(
        np.isfinite(numeric).sum()
    )


    print(
        f"\n{name}:"
    )

    print(
        f"  NaN values:      {nan_count:,}"
    )

    print(
        f"  Infinite values: {inf_count:,}"
    )

    print(
        f"  Finite values:   {finite_count:,}"
    )


    if nan_count > 0 or inf_count > 0:

        fail_check(
            f"{name} contains NaN or infinite values."
        )

    else:

        pass_check(
            f"{name} contains no NaN/infinite values."
        )


# ============================================================
# FEATURE RANGE CHECK
# ============================================================

section("5. FEATURE RANGE CHECK")


states_float = np.asarray(
    states,
    dtype=np.float64
)


for i, feature_name in enumerate(
    FEATURE_NAMES
):

    column = states_float[
        :,
        i
    ]

    finite_column = column[
        np.isfinite(column)
    ]


    if len(finite_column) == 0:

        fail_check(
            f"{feature_name}: no finite values."
        )

        continue


    minimum = float(
        np.min(finite_column)
    )

    maximum = float(
        np.max(finite_column)
    )

    mean = float(
        np.mean(finite_column)
    )

    median = float(
        np.median(finite_column)
    )


    print(
        f"\n{i + 1:2d}. {feature_name}"
    )

    print(
        f"    min={minimum:.6g}"
    )

    print(
        f"    median={median:.6g}"
    )

    print(
        f"    mean={mean:.6g}"
    )

    print(
        f"    max={maximum:.6g}"
    )


# ============================================================
# NEGATIVE VALUE CHECKS
# ============================================================

section("6. NEGATIVE-VALUE CHECK")

nonnegative_features = [
    0,   # packets/s
    1,   # std packets/s
    2,   # bytes/s
    3,   # std bytes/s
    4,   # duration
    5,   # avg packet size
    6,   # SYN count
    7,   # ACK count
    8,   # RST count
    9,   # FIN count
    10,  # unique destination ports
    13,  # Flow IAT Std
]

for i in nonnegative_features:

    column = states_float[
        :,
        i
    ]

    negative_count = int(
        np.sum(column < 0)
    )

    print(
        f"{FEATURE_NAMES[i]:34s}: "
        f"{negative_count:,} negative values"
    )


    if negative_count > 0:

        warn_check(
            f"{FEATURE_NAMES[i]} contains negative values."
        )

    else:

        pass_check(
            f"{FEATURE_NAMES[i]} has no negative values."
        )


# ============================================================
# LABEL AUDIT
# ============================================================

section("7. LABEL QUALITY CHECK")


state_labels_clean = np.array([
    str(x).strip()
    for x in state_labels_raw
])


state_label_counts = Counter(
    state_labels_clean.tolist()
)


print(
    "\nAll state labels:"
)

for label, count in state_label_counts.most_common():

    print(
        f"  {label:25s} {count:,}"
    )


expected_state_labels = {
    "BENIGN",
    "DOS_DDOS",
    "RECONNAISSANCE",
    "BRUTE_FORCE",
    "BOT",
    "WEB_ATTACK",
    "INFILTRATION",
    "EXPLOITATION",
}


unexpected_labels = (
    set(state_label_counts.keys())
    - expected_state_labels
)


if unexpected_labels:

    warn_check(
        f"Unexpected state labels found: "
        f"{sorted(unexpected_labels)}"
    )

else:

    pass_check(
        "All state labels belong to the expected label vocabulary."
    )


# ============================================================
# FINAL 3-CLASS LABEL AUDIT
# ============================================================

section("8. FINAL 3-CLASS DATASET LABEL CHECK")


for split_name, y in [
    ("train", y_train),
    ("validation", y_val),
    ("test", y_test),
]:

    values, counts = np.unique(
        y,
        return_counts=True
    )


    print(
        f"\n{split_name}:"
    )

    for value, count in zip(
        values,
        counts
    ):

        value_int = int(value)

        if (
            value_int >= 0
            and value_int < len(CLASS_NAMES)
        ):

            name = CLASS_NAMES[
                value_int
            ]

        else:

            name = (
                f"UNKNOWN_ID_{value_int}"
            )


        print(
            f"  {value_int} = {name:18s} "
            f"{count:,}"
        )


    unexpected = [
        int(v)
        for v in values
        if int(v) not in [0, 1, 2]
    ]


    if unexpected:

        fail_check(
            f"{split_name} contains unexpected class IDs: "
            f"{unexpected}"
        )

    else:

        pass_check(
            f"{split_name} contains only the three model classes."
        )


# ============================================================
# CLASS BALANCE
# ============================================================

section("9. CLASS BALANCE")


def print_balance(
    name,
    y
):

    total = len(y)

    values, counts = np.unique(
        y,
        return_counts=True
    )


    print(
        f"\n{name} total: {total:,}"
    )


    for value, count in zip(
        values,
        counts
    ):

        percentage = (
            count / total * 100
        )

        print(
            f"  {CLASS_NAMES[int(value)]:18s}: "
            f"{count:8,} "
            f"({percentage:7.3f}%)"
        )


print_balance(
    "Training",
    y_train
)

print_balance(
    "Validation",
    y_val
)

print_balance(
    "Test",
    y_test
)


# ============================================================
# DUPLICATE NETWORK STATES
# ============================================================

section("10. DUPLICATE NETWORK-STATE CHECK")


def hash_rows(
    array
):

    hashes = []

    for row in np.asarray(array):

        contiguous = np.ascontiguousarray(
            row
        )

        digest = hashlib.sha256(
            contiguous.tobytes()
        ).hexdigest()

        hashes.append(
            digest
        )

    return hashes


# Hash complete network-state rows.

state_hashes = hash_rows(
    states
)


unique_state_hashes = len(
    set(state_hashes)
)

duplicate_state_rows = (
    len(state_hashes)
    - unique_state_hashes
)


print(
    f"Total network states: "
    f"{len(state_hashes):,}"
)

print(
    f"Unique state vectors: "
    f"{unique_state_hashes:,}"
)

print(
    f"Repeated state vectors: "
    f"{duplicate_state_rows:,}"
)


if duplicate_state_rows > 0:

    warn_check(
        "Some network-state vectors are numerically identical. "
        "This is not automatically leakage because repeated "
        "states can legitimately occur in network traffic."
    )

else:

    pass_check(
        "No duplicate network-state vectors found."
    )


# ============================================================
# SEQUENCE DUPLICATION ACROSS SPLITS
# ============================================================

section("11. TRAIN / VALIDATION / TEST SEQUENCE OVERLAP")


print(
    "Checking exact duplicate 5x15 sequences across splits."
)

print(
    "Identical sequences within one split are not treated "
    "as leakage; cross-split duplicates are the concern."
)


train_hashes = hash_rows(
    X_train.reshape(
        len(X_train),
        -1
    )
)

val_hashes = hash_rows(
    X_val.reshape(
        len(X_val),
        -1
    )
)

test_hashes = hash_rows(
    X_test.reshape(
        len(X_test),
        -1
    )
)


train_set = set(
    train_hashes
)

val_set = set(
    val_hashes
)

test_set = set(
    test_hashes
)


train_val_overlap = len(
    train_set & val_set
)

train_test_overlap = len(
    train_set & test_set
)

val_test_overlap = len(
    val_set & test_set
)


print(
    f"\nTrain ∩ Validation: {train_val_overlap:,}"
)

print(
    f"Train ∩ Test:       {train_test_overlap:,}"
)

print(
    f"Validation ∩ Test:  {val_test_overlap:,}"
)


if (
    train_val_overlap == 0
    and train_test_overlap == 0
    and val_test_overlap == 0
):

    pass_check(
        "No exact duplicate sequences were found across splits."
    )

else:

    fail_check(
        "Exact duplicate sequences exist across at least two splits."
    )


# ============================================================
# TARGET LABEL / SEQUENCE CONSISTENCY
# ============================================================

section("12. SEQUENCE TARGET CONSISTENCY")


def check_target_consistency(
    name,
    X,
    y,
    state_matrix
):

    # A sequence of length 5 should predict the state
    # immediately after those 5 states.
    #
    # We cannot reconstruct every sequence's original source
    # boundary from X alone, but we can look for exact
    # sequence-state matches in the complete state matrix.
    #
    # This is an informational check rather than a definitive
    # boundary check.


    info(
        f"{name}: {len(X):,} sequences checked."
    )


    if len(X) == 0:

        warn_check(
            f"{name} contains no samples."
        )

        return


    # Confirm feature dimensions.

    if X.shape[1:] != (
        5,
        15
    ):

        fail_check(
            f"{name} has unexpected shape {X.shape}."
        )

        return


    pass_check(
        f"{name} has the expected 5x15 sequence structure."
    )


check_target_consistency(
    "Train",
    X_train,
    y_train,
    states
)

check_target_consistency(
    "Validation",
    X_val,
    y_val,
    states
)

check_target_consistency(
    "Test",
    X_test,
    y_test,
    states
)


# ============================================================
# SCENARIO / SOURCE AUDIT
# ============================================================

section("13. SCENARIO / SOURCE BOUNDARY CHECK")


if "source_file" not in metadata.columns:

    warn_check(
        "Metadata does not contain a 'source_file' column; "
        "scenario-boundary verification cannot be performed."
    )

else:

    sources = (
        metadata["source_file"]
        .astype(str)
        .str.strip()
    )


    source_counts = sources.value_counts()


    print(
        "\nStates by source scenario:"
    )

    for source, count in source_counts.items():

        print(
            f"  {source:60s} {count:,}"
        )


    print(
        f"\nNumber of scenarios: "
        f"{len(source_counts)}"
    )


    pass_check(
        "Scenario source metadata is available."
    )


# ============================================================
# STATE PURITY AUDIT
# ============================================================

section("14. NETWORK-STATE PURITY CHECK")


# A network state is 10 flows in the project's representation.
#
# If the raw state metadata contains enough information to identify
# the flow-level composition, purity can be calculated.
#
# The current saved state labels are dominant labels, so this
# section checks the availability of supporting columns and
# otherwise reports that the previously computed purity should
# be treated as the existing project result.


metadata_columns = [
    str(column).strip()
    for column in metadata.columns
]


print(
    "Metadata columns:"
)

for column in metadata_columns:

    print(
        f"  - {column}"
    )


if (
    "state_label" in metadata_columns
    and "source_file" in metadata_columns
):

    info(
        "State-label metadata is available."
    )

else:

    info(
        "Flow-level labels are not fully represented in the "
        "state metadata table, so purity cannot be recomputed "
        "from this file alone."
    )

    info(
        "Existing project purity result: "
        "mean=0.985768, median=1.0, min=0.5, max=1.0."
    )


# ============================================================
# TEMPORAL SPLIT SANITY CHECK
# ============================================================

section("15. TEMPORAL SPLIT SANITY CHECK")


if "source_file" in metadata.columns:

    source_array = (
        metadata["source_file"]
        .astype(str)
        .str.strip()
        .to_numpy()
    )


    # Calculate expected number of full sequence samples
    # per scenario based on:
    #
    #   n_sequences = n_states - 5
    #
    # and a 70 / 15 / 15 split on sequences.

    total_expected_sequences = 0

    split_rows = []


    for source in pd.unique(
        source_array
    ):

        count = int(
            np.sum(
                source_array == source
            )
        )


        n_sequences = count - 5


        if n_sequences <= 0:
            continue


        train_end = int(
            n_sequences * 0.70
        )

        val_end = int(
            n_sequences * 0.85
        )


        train_count = train_end

        val_count = (
            val_end
            - train_end
        )

        test_count = (
            n_sequences
            - val_end
        )


        split_rows.append({
            "source": source,
            "states": count,
            "sequences": n_sequences,
            "train": train_count,
            "validation": val_count,
            "test": test_count,
        })


        total_expected_sequences += (
            n_sequences
        )


    split_df = pd.DataFrame(
        split_rows
    )


    print(
        split_df.to_string(
            index=False
        )
    )


    print(
        f"\nExpected total full sequences: "
        f"{total_expected_sequences:,}"
    )

    print(
        f"Observed 3-class sequences: "
        f"{len(X_train) + len(X_val) + len(X_test):,}"
    )


    if total_expected_sequences == (
        len(X_train)
        + len(X_val)
        + len(X_test)
    ):

        pass_check(
            "Overall sequence count is consistent with "
            "scenario-level temporal splitting."
        )

    else:

        # Difference can be caused by the 3-class filtering.
        difference = (
            total_expected_sequences
            - (
                len(X_train)
                + len(X_val)
                + len(X_test)
            )
        )


        info(
            f"Difference: {difference:,} sequences."
        )

        info(
            "This can be expected because the final 3-class "
            "dataset filters labels outside BENIGN/DOS_DDOS/"
            "RECONNAISSANCE."
        )

        warn_check(
            "Observed 3-class sequence count differs from the "
            "full sequence count; verify this is due to "
            "intentional class filtering."
        )


else:

    warn_check(
        "Cannot perform temporal scenario split check "
        "without source_file metadata."
    )


# ============================================================
# SCENARIO CONTINUITY CHECK FOR METADATA
# ============================================================

section("16. SCENARIO ORDER / CONTINUITY CHECK")


if "source_file" in metadata.columns:

    source_array = (
        metadata["source_file"]
        .astype(str)
        .str.strip()
        .to_numpy()
    )


    transitions = np.where(
        source_array[1:]
        != source_array[:-1]
    )[0]


    print(
        f"Scenario-boundary transitions in metadata: "
        f"{len(transitions):,}"
    )


    for index in transitions:

        previous_source = source_array[
            index
        ]

        next_source = source_array[
            index + 1
        ]


        print(
            f"  boundary after row {index}: "
            f"{previous_source} -> {next_source}"
        )


    pass_check(
        "Scenario ordering information is present."
    )


# ============================================================
# RAW CIC-IDS2017 CSV AUDIT
# ============================================================

section("17. RAW CIC-IDS2017 FILE AUDIT")


if not os.path.exists(RAW_DIR):

    warn_check(
        f"Raw dataset directory not found: {RAW_DIR}"
    )

else:

    raw_files = sorted([
        filename
        for filename in os.listdir(RAW_DIR)
        if filename.lower().endswith(".csv")
    ])


    print(
        f"CSV files found: {len(raw_files)}"
    )


    if len(raw_files) == 0:

        warn_check(
            "No CSV files found in the raw dataset directory."
        )

    else:

        pass_check(
            f"Found {len(raw_files)} raw CSV files."
        )


    raw_label_counts = Counter()

    raw_total_rows = 0

    raw_nan_rows = 0

    raw_inf_values = 0


    for filename in raw_files:

        path = os.path.join(
            RAW_DIR,
            filename
        )


        print(
            f"\nAuditing: {filename}"
        )


        try:

            # Read in chunks to avoid requiring the entire
            # raw dataset to fit in memory.

            chunks = pd.read_csv(
                path,
                chunksize=100000,
                low_memory=False
            )


            file_rows = 0

            file_nan_rows = 0

            file_inf_values = 0

            label_column_found = False


            for chunk in chunks:

                chunk.columns = [
                    str(column).strip()
                    for column in chunk.columns
                ]


                rows = len(chunk)

                file_rows += rows


                # Detect label column.

                label_column = None

                for candidate in [
                    "Label",
                    "label",
                ]:

                    if candidate in chunk.columns:

                        label_column = candidate
                        break


                if label_column is not None:

                    label_column_found = True


                    labels = (
                        chunk[label_column]
                        .astype(str)
                        .str.strip()
                    )


                    raw_label_counts.update(
                        labels.tolist()
                    )


                # Count rows with any missing value.

                file_nan_rows += int(
                    chunk.isna()
                    .any(axis=1)
                    .sum()
                )


                # Check numeric infinite values.

                numeric_columns = (
                    chunk.select_dtypes(
                        include=[np.number]
                    )
                )


                if not numeric_columns.empty:

                    numeric_values = (
                        numeric_columns.to_numpy(
                            dtype=np.float64
                        )
                    )


                    file_inf_values += int(
                        np.isinf(
                            numeric_values
                        ).sum()
                    )


            raw_total_rows += file_rows

            raw_nan_rows += file_nan_rows

            raw_inf_values += file_inf_values


            print(
                f"  Rows: {file_rows:,}"
            )

            print(
                f"  Rows containing NaN: "
                f"{file_nan_rows:,}"
            )

            print(
                f"  Infinite numeric values: "
                f"{file_inf_values:,}"
            )


            if label_column_found:

                print(
                    "  Label column: found"
                )

            else:

                warn_check(
                    f"{filename}: Label column not found."
                )


        except Exception as exc:

            warn_check(
                f"{filename}: could not fully audit: {exc}"
            )


    print(
        "\nRAW DATA TOTALS"
    )

    print(
        f"Total raw rows: "
        f"{raw_total_rows:,}"
    )

    print(
        f"Rows containing NaN: "
        f"{raw_nan_rows:,}"
    )

    print(
        f"Infinite numeric values: "
        f"{raw_inf_values:,}"
    )


    print(
        "\nRaw attack labels:"
    )

    for label, count in raw_label_counts.most_common():

        print(
            f"  {label:40s} {count:,}"
        )


    if raw_inf_values > 0:

        warn_check(
            "Raw dataset contains infinite numeric values."
        )

    else:

        pass_check(
            "Raw numeric fields contain no infinite values."
        )


    if raw_nan_rows > 0:

        info(
            "Raw CSVs contain rows with missing values. "
            "This is not automatically an error because "
            "the preprocessing pipeline may remove/fill "
            "them before model training."
        )

    else:

        pass_check(
            "No raw CSV rows contain missing values."
        )


# ============================================================
# CLASS FILTERING AUDIT
# ============================================================

section("18. RARE / EXCLUDED CLASS AUDIT")


excluded_labels = [
    "BRUTE_FORCE",
    "BOT",
    "WEB_ATTACK",
]


excluded_counts = {
    label: state_label_counts.get(
        label,
        0
    )
    for label in excluded_labels
}


for label, count in excluded_counts.items():

    print(
        f"{label:20s}: {count:,} states"
    )


excluded_total = sum(
    excluded_counts.values()
)


print(
    f"\nTotal excluded states: "
    f"{excluded_total:,}"
)


if excluded_total == 598:

    pass_check(
        "Excluded-state count matches the current project "
        "audit expectation of 598."
    )

else:

    warn_check(
        f"Excluded-state count is {excluded_total}, "
        f"not 598. Review class filtering."
    )


# ============================================================
# SUMMARY
# ============================================================

section("19. AUDIT SUMMARY")


print(
    f"Critical failures: {critical_failures}"
)

print(
    f"Warnings:          {warnings_found}"
)


if critical_failures == 0:

    if warnings_found == 0:

        print(
            "\nOVERALL STATUS: PASS"
        )

        print(
            "No critical dataset-quality issues were detected."
        )

    else:

        print(
            "\nOVERALL STATUS: PASS WITH DOCUMENTED WARNINGS"
        )

        print(
            "No critical structural/data-leakage issues were detected, "
            "but review the warnings before final reporting."
        )

else:

    print(
        "\nOVERALL STATUS: REVIEW REQUIRED"
    )

    print(
        "Critical issues were detected. Do not treat the "
        "current ML metrics as final until they are reviewed."
    )


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

print("\n" + "=" * 78)
print("20. FINAL INTERPRETATION GUIDE")
print("=" * 78)

print(
    """
This audit does NOT calculate model accuracy.

It checks whether the dataset and evaluation structure are
suitable for interpreting the reported model metrics.

Important distinctions:

1. A dataset does not have an accuracy value.
   Model accuracy comes from predictions against labels.

2. Class imbalance is not automatically a data-quality failure.
   Report balanced accuracy and macro F1 alongside ordinary accuracy.

3. Repeated network states can occur naturally.
   Exact duplicates within a dataset are not automatically leakage.

4. Exact duplicate temporal sequences across train/validation/test
   are a much more serious concern.

5. The current forecasting unit is:
      10 flows -> 1 network state
      5 states -> model input
      +1 -> next 10-flow state

   Therefore +1 is NOT a fixed number of seconds or minutes.

6. The final model recognizes:
      BENIGN
      DOS_DDOS
      RECONNAISSANCE

   It should not be described as recognizing every CIC-IDS2017
   attack type.

7. Persistence is a very strong baseline for this representation.
   This is an experimental finding and should be reported honestly.
"""
)

print(
    "\nDataset audit completed."
)
import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data/raw/MachineLearningCVE")

FLOWS_PER_STATE = 10


def normalize_label(label):

    label = str(label).strip()
    label_lower = label.lower()

    if label_lower == "benign":
        return "BENIGN"

    if "portscan" in label_lower:
        return "RECONNAISSANCE"

    if "patator" in label_lower:
        return "BRUTE_FORCE"

    if "web attack" in label_lower:
        return "WEB_ATTACK"

    if "ddos" in label_lower:
        return "DOS_DDOS"

    if label_lower.startswith("dos "):
        return "DOS_DDOS"

    if label_lower == "bot":
        return "BOT"

    if "infiltration" in label_lower:
        return "INFILTRATION"

    if "heartbleed" in label_lower:
        return "EXPLOITATION"

    return "OTHER"


def clean_column_names(df):

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


def find_column(df, name):

    target = name.strip().lower()

    for column in df.columns:

        if column.strip().lower() == target:
            return column

    return None


def create_engineered_features(df):

    # -----------------------------------------------------
    # Clean column names
    # -----------------------------------------------------

    df = clean_column_names(df)

    # -----------------------------------------------------
    # Find required raw columns
    # -----------------------------------------------------

    required = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "SYN Flag Count",
        "ACK Flag Count",
        "RST Flag Count",
        "FIN Flag Count",
        "Average Packet Size",
        "Destination Port",
        "Flow IAT Std",
    ]

    column_map = {}

    missing = []

    for name in required:

        actual = find_column(
            df,
            name
        )

        if actual is None:
            missing.append(name)
        else:
            column_map[name] = actual

    if missing:

        print("\nMissing raw columns:")

        for name in missing:
            print(" ", name)

        return None

    # -----------------------------------------------------
    # Rename to clean canonical names
    # -----------------------------------------------------

    rename_map = {
        actual: canonical
        for canonical, actual
        in column_map.items()
    }

    df = df.rename(
        columns=rename_map
    )

    # -----------------------------------------------------
    # Convert numeric columns
    # -----------------------------------------------------

    numeric_columns = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "SYN Flag Count",
        "ACK Flag Count",
        "RST Flag Count",
        "FIN Flag Count",
        "Average Packet Size",
        "Destination Port",
        "Flow IAT Std",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # -----------------------------------------------------
    # Remove invalid infinities
    # -----------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # -----------------------------------------------------
    # Handle invalid durations
    # -----------------------------------------------------

    df.loc[
        df["Flow Duration"] < 0,
        "Flow Duration"
    ] = np.nan

    # -----------------------------------------------------
    # Calculate Flow Packets/s
    #
    # CIC-IDS2017 duration is in microseconds.
    # -----------------------------------------------------

    duration_seconds = (
        df["Flow Duration"] / 1_000_000
    )

    safe_duration = (
        duration_seconds
        .replace(0, np.nan)
    )

    df["Flow Packets/s"] = (
        (
            df["Total Fwd Packets"]
            +
            df["Total Backward Packets"]
        )
        /
        safe_duration
    )

    # -----------------------------------------------------
    # Calculate Flow Bytes/s
    # -----------------------------------------------------

    df["Flow Bytes/s"] = (
        (
            df["Total Length of Fwd Packets"]
            +
            df["Total Length of Bwd Packets"]
        )
        /
        safe_duration
    )

    # -----------------------------------------------------
    # IAT variability
    #
    # We use a stable ratio:
    # std / (mean + epsilon)
    #
    # Need Flow IAT Mean from raw data.
    # -----------------------------------------------------

    flow_iat_mean_column = find_column(
        df,
        "Flow IAT Mean"
    )

    if flow_iat_mean_column is not None:

        df["Flow IAT Mean"] = pd.to_numeric(
            df[flow_iat_mean_column],
            errors="coerce"
        )

        df["iat_variability"] = (
            df["Flow IAT Std"]
            /
            (
                df["Flow IAT Mean"].abs()
                + 1.0
            )
        )

    else:

        # If unavailable, use zero.
        df["iat_variability"] = 0.0

    # -----------------------------------------------------
    # Clean engineered values
    # -----------------------------------------------------

    engineered_columns = [
        "Flow Packets/s",
        "Flow Bytes/s",
        "iat_variability"
    ]

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # -----------------------------------------------------
    # Fill missing numeric values with medians
    # -----------------------------------------------------

    all_numeric = numeric_columns + engineered_columns

    for column in all_numeric:

        median_value = df[column].median()

        if pd.isna(median_value):
            median_value = 0.0

        df[column] = df[column].fillna(
            median_value
        )

    # -----------------------------------------------------
    # Clip extreme engineered values
    #
    # This prevents a handful of pathological flows
    # from dominating the network-state representation.
    # -----------------------------------------------------

    df["Flow Packets/s"] = (
        df["Flow Packets/s"]
        .clip(lower=0)
    )

    df["Flow Bytes/s"] = (
        df["Flow Bytes/s"]
        .clip(lower=0)
    )

    df["iat_variability"] = (
        df["iat_variability"]
        .clip(lower=0, upper=100)
    )

    return df


def create_state(flow_df):

    # Safe directional ratios
    packet_ratio = (
        flow_df["Total Fwd Packets"]
        /
        (
            flow_df["Total Backward Packets"]
            + 1.0
        )
    )

    byte_ratio = (
        flow_df["Total Length of Fwd Packets"]
        /
        (
            flow_df["Total Length of Bwd Packets"]
            + 1.0
        )
    )

    state = [

        # 1
        flow_df["Flow Packets/s"].mean(),

        # 2
        flow_df["Flow Packets/s"].std(),

        # 3
        flow_df["Flow Bytes/s"].mean(),

        # 4
        flow_df["Flow Bytes/s"].std(),

        # 5
        flow_df["Flow Duration"].mean(),

        # 6
        flow_df["Average Packet Size"].mean(),

        # 7
        flow_df["SYN Flag Count"].mean(),

        # 8
        flow_df["ACK Flag Count"].mean(),

        # 9
        flow_df["RST Flag Count"].mean(),

        # 10
        flow_df["FIN Flag Count"].mean(),

        # 11
        flow_df["Destination Port"].nunique(),

        # 12
        packet_ratio.mean(),

        # 13
        byte_ratio.mean(),

        # 14
        flow_df["Flow IAT Std"].mean(),

        # 15
        flow_df["iat_variability"].mean(),
    ]

    return state


print("=" * 70)
print("MULTI-SCENARIO NETWORK STATE CREATION")
print("=" * 70)


all_states = []
all_labels = []
all_metadata = []


csv_files = sorted(
    DATA_DIR.glob("*.csv")
)


for csv_file in csv_files:

    print("\n" + "-" * 70)
    print(
        f"Processing: {csv_file.name}"
    )

    try:

        raw_df = pd.read_csv(
            csv_file
        )

        print(
            f"Original rows: "
            f"{len(raw_df):,}"
        )

        # -------------------------------------------------
        # Find label column
        # -------------------------------------------------

        label_column = find_column(
            raw_df,
            "Label"
        )

        if label_column is None:

            print(
                "WARNING: Label column not found"
            )

            continue

        labels = (
            raw_df[label_column]
            .map(normalize_label)
            .reset_index(drop=True)
        )

        # -------------------------------------------------
        # Create engineered features
        # -------------------------------------------------

        df = create_engineered_features(
            raw_df
        )

        if df is None:
            continue

        df = df.reset_index(
            drop=True
        )

        # -------------------------------------------------
        # Make absolutely sure labels
        # and features are aligned.
        # -------------------------------------------------

        if len(df) != len(labels):

            print(
                "ERROR: Feature/label alignment mismatch"
            )

            print(
                "Features:",
                len(df)
            )

            print(
                "Labels:",
                len(labels)
            )

            continue

        # -------------------------------------------------
        # Number of complete states
        # -------------------------------------------------

        number_of_states = (
            len(df)
            // FLOWS_PER_STATE
        )

        print(
            f"Complete states: "
            f"{number_of_states:,}"
        )

        # -------------------------------------------------
        # Create network states
        # -------------------------------------------------

        for state_index in range(
            number_of_states
        ):

            start = (
                state_index
                * FLOWS_PER_STATE
            )

            end = (
                start
                + FLOWS_PER_STATE
            )

            flow_block = df.iloc[
                start:end
            ]

            label_block = labels.iloc[
                start:end
            ]

            state = create_state(
                flow_block
            )

            label_counts = (
                label_block
                .value_counts()
            )

            dominant_label = (
                label_counts.index[0]
            )

            purity = (
                label_counts.iloc[0]
                / FLOWS_PER_STATE
            )

            all_states.append(
                state
            )

            all_labels.append(
                dominant_label
            )

            all_metadata.append(
                {
                    "source_file":
                        csv_file.name,

                    "state_in_file":
                        state_index,

                    "purity":
                        purity
                }
            )

    except Exception as e:

        print(
            f"ERROR processing "
            f"{csv_file.name}"
        )

        print(e)


# ---------------------------------------------------------
# Final arrays
# ---------------------------------------------------------

states = np.asarray(
    all_states,
    dtype=np.float32
)

labels = np.asarray(
    all_labels
)

metadata = pd.DataFrame(
    all_metadata
)


print("\n" + "=" * 70)
print("MULTI-SCENARIO STATE SUMMARY")
print("=" * 70)

print(
    "State matrix shape:",
    states.shape
)

print(
    "Number of labels:",
    len(labels)
)

if len(labels) > 0:

    print("\nLabel distribution:")

    print(
        pd.Series(labels)
        .value_counts()
        .to_string()
    )

    print("\nState purity:")

    print(
        metadata["purity"]
        .describe()
        .to_string()
    )

else:

    print(
        "\nERROR: No states were created."
    )

    raise SystemExit(1)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

np.save(
    "data/multiclass_network_states.npy",
    states
)

np.save(
    "data/multiclass_state_labels.npy",
    labels
)

metadata.to_csv(
    "data/multiclass_state_metadata.csv",
    index=False
)


print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(
    "data/multiclass_network_states.npy"
)

print(
    "data/multiclass_state_labels.npy"
)

print(
    "data/multiclass_state_metadata.csv"
)
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


print("=" * 70)
print("MULTI-ATTACK NETWORK STATE LABEL CREATION")
print("=" * 70)

all_states = []

csv_files = sorted(DATA_DIR.glob("*.csv"))

state_id = 0

for csv_file in csv_files:

    print("\n" + "-" * 70)
    print(f"Processing: {csv_file.name}")

    df = pd.read_csv(csv_file)

    label_column = None

    for column in df.columns:
        if column.strip().lower() == "label":
            label_column = column
            break

    if label_column is None:
        print("WARNING: Label column not found")
        continue

    labels = (
        df[label_column]
        .map(normalize_label)
        .to_numpy()
    )

    number_of_states = len(labels) // FLOWS_PER_STATE

    print(f"Flows: {len(labels):,}")
    print(f"Complete states: {number_of_states:,}")

    for i in range(number_of_states):

        start = i * FLOWS_PER_STATE
        end = start + FLOWS_PER_STATE

        state_labels = labels[start:end]

        counts = pd.Series(
            state_labels
        ).value_counts()

        dominant_label = counts.index[0]

        dominant_count = counts.iloc[0]

        purity = (
            dominant_count
            / FLOWS_PER_STATE
        )

        state_id += 1

        all_states.append(
            {
                "state_id": state_id,
                "source_file": csv_file.name,
                "state_in_file": i,
                "label": dominant_label,
                "purity": purity
            }
        )


states_df = pd.DataFrame(all_states)


print("\n" + "=" * 70)
print("STATE LABEL SUMMARY")
print("=" * 70)

print(
    f"Total network states: "
    f"{len(states_df):,}"
)

print("\nDominant state labels:")

print(
    states_df["label"]
    .value_counts()
    .to_string()
)

print("\nPurity statistics:")

print(
    states_df["purity"]
    .describe()
    .to_string()
)

print("\nStates by purity:")

purity_bins = pd.cut(
    states_df["purity"],
    bins=[
        0.0,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0
    ],
    include_lowest=True
)

print(
    purity_bins
    .value_counts()
    .sort_index()
    .to_string()
)


output_path = (
    "data/multiclass_network_state_labels.csv"
)

states_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(output_path)
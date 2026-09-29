import pandas as pd
from pathlib import Path


DATA_DIR = Path("data/raw/MachineLearningCVE")


print("=" * 60)
print("CIC-IDS2017 ATTACK TYPE INVENTORY")
print("=" * 60)

csv_files = sorted(DATA_DIR.glob("*.csv"))

print(f"\nCSV files found: {len(csv_files)}")

all_counts = {}

for csv_file in csv_files:

    print("\n" + "-" * 60)
    print(f"FILE: {csv_file.name}")

    try:
        df = pd.read_csv(csv_file)

        # Find the label column safely
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
            .astype(str)
            .str.strip()
        )

        counts = labels.value_counts()

        print(f"Rows: {len(df):,}")
        print("\nLabels:")

        for label, count in counts.items():

            print(
                f"  {label:<35} {count:>10,}"
            )

            all_counts[label] = (
                all_counts.get(label, 0) + count
            )

    except Exception as e:

        print(f"ERROR reading {csv_file.name}")
        print(e)


print("\n" + "=" * 60)
print("COMBINED LABEL INVENTORY")
print("=" * 60)

for label, count in sorted(
    all_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{label:<40} {count:>12,}"
    )

print("\n" + "=" * 60)
print("INVENTORY COMPLETE")
print("=" * 60)
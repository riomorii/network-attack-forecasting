import pandas as pd
from pathlib import Path


DATA_DIR = Path("data/raw/MachineLearningCVE")


def normalize_label(label):

    label = str(label).strip()

    # Normalise common encoding issue
    label_lower = label.lower()

    # Benign traffic
    if label_lower == "benign":
        return "BENIGN"

    # Reconnaissance
    if "portscan" in label_lower:
        return "RECONNAISSANCE"

    # Brute-force attacks
    if "patator" in label_lower:
        return "BRUTE_FORCE"

    # Web attacks
    if "web attack" in label_lower:
        return "WEB_ATTACK"

    # DoS / DDoS
    if "ddos" in label_lower:
        return "DOS_DDOS"

    if label_lower.startswith("dos "):
        return "DOS_DDOS"

    # Bot traffic
    if label_lower == "bot":
        return "BOT"

    # Infiltration
    if "infiltration" in label_lower:
        return "INFILTRATION"

    # Heartbleed
    if "heartbleed" in label_lower:
        return "EXPLOITATION"

    # Anything unexpected
    return "OTHER"


print("=" * 60)
print("NORMALIZED CIC-IDS2017 ATTACK FAMILIES")
print("=" * 60)

csv_files = sorted(DATA_DIR.glob("*.csv"))

total_counts = {}

for csv_file in csv_files:

    print("\n" + "-" * 60)
    print(csv_file.name)

    try:

        df = pd.read_csv(csv_file)

        label_column = None

        for column in df.columns:
            if column.strip().lower() == "label":
                label_column = column
                break

        if label_column is None:
            print("Label column not found")
            continue

        normalized = (
            df[label_column]
            .map(normalize_label)
        )

        counts = normalized.value_counts()

        for family, count in counts.items():

            print(
                f"{family:<20} {count:>10,}"
            )

            total_counts[family] = (
                total_counts.get(family, 0)
                + count
            )

    except Exception as e:

        print("ERROR:")
        print(e)


print("\n" + "=" * 60)
print("COMBINED NORMALIZED COUNTS")
print("=" * 60)

for family, count in sorted(
    total_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{family:<20} {count:>12,}"
    )

print("\n" + "=" * 60)
print("NORMALIZATION COMPLETE")
print("=" * 60)
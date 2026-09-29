import pandas as pd
import numpy as np


INPUT_PATH = (
    "data/multiclass_network_state_labels.csv"
)


print("=" * 70)
print("MULTI-ATTACK NETWORK STATE TRANSITION ANALYSIS")
print("=" * 70)


df = pd.read_csv(INPUT_PATH)


labels = sorted(
    df["label"].unique()
)

print("\nLabels found:")
for label in labels:
    print(" ", label)


# ---------------------------------------------------------
# Transition analysis
# ---------------------------------------------------------

transition_counts = pd.DataFrame(
    0,
    index=labels,
    columns=labels,
    dtype=int
)


transition_total = 0


# Process each capture independently
for source_file, group in df.groupby(
    "source_file",
    sort=False
):

    group = group.sort_values(
        "state_in_file"
    )

    state_labels = (
        group["label"]
        .to_numpy()
    )

    for i in range(
        len(state_labels) - 1
    ):

        current_label = state_labels[i]
        next_label = state_labels[i + 1]

        transition_counts.loc[
            current_label,
            next_label
        ] += 1

        transition_total += 1


print("\n" + "=" * 70)
print("TRANSITION COUNT MATRIX")
print("=" * 70)

print(
    transition_counts.to_string()
)


# ---------------------------------------------------------
# Non-self transitions
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("NON-SELF TRANSITIONS")
print("=" * 70)


transitions = []


for current_label in labels:

    for next_label in labels:

        count = transition_counts.loc[
            current_label,
            next_label
        ]

        if (
            current_label != next_label
            and count > 0
        ):

            transitions.append(
                (
                    current_label,
                    next_label,
                    int(count)
                )
            )


transitions.sort(
    key=lambda x: x[2],
    reverse=True
)


for current, next_label, count in transitions:

    print(
        f"{current:<20}"
        f" -> "
        f"{next_label:<20}"
        f"{count:>10,}"
    )


# ---------------------------------------------------------
# Transition probabilities
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP TRANSITIONS BY SOURCE LABEL")
print("=" * 70)


for current_label in labels:

    total = transition_counts.loc[
        current_label
    ].sum()

    if total == 0:
        continue

    print(
        f"\n{current_label}"
        f" (total outgoing transitions: {total:,})"
    )

    row = (
        transition_counts.loc[
            current_label
        ]
        .sort_values(
            ascending=False
        )
    )

    shown = 0

    for next_label, count in row.items():

        if count == 0:
            continue

        probability = (
            count / total
        )

        print(
            f"  -> "
            f"{next_label:<20}"
            f"{count:>10,}"
            f"  ({probability:.2%})"
        )

        shown += 1

        if shown >= 5:
            break


print("\n" + "=" * 70)
print("TRANSITION ANALYSIS COMPLETE")
print("=" * 70)
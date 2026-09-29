import pandas as pd


INPUT_PATH = (
    "data/multiclass_network_state_labels.csv"
)


print("=" * 70)
print("ATTACK ONSET ANALYSIS")
print("=" * 70)


df = pd.read_csv(INPUT_PATH)


attack_labels = {
    "BOT",
    "BRUTE_FORCE",
    "DOS_DDOS",
    "RECONNAISSANCE",
    "WEB_ATTACK"
}


onsets = []


for source_file, group in df.groupby(
    "source_file",
    sort=False
):

    group = group.sort_values(
        "state_in_file"
    ).reset_index(drop=True)

    labels = group["label"].tolist()

    for i in range(1, len(labels)):

        previous_label = labels[i - 1]
        current_label = labels[i]

        # Detect BENIGN -> ATTACK
        if (
            previous_label == "BENIGN"
            and current_label in attack_labels
        ):

            # Count consecutive benign states
            benign_run = 0

            j = i - 1

            while (
                j >= 0
                and labels[j] == "BENIGN"
            ):

                benign_run += 1
                j -= 1

            # Count consecutive attack states
            attack_run = 0

            j = i

            while (
                j < len(labels)
                and labels[j] == current_label
            ):

                attack_run += 1
                j += 1

            onsets.append(
                {
                    "source_file": source_file,
                    "state_in_file": int(
                        group.loc[
                            i,
                            "state_in_file"
                        ]
                    ),
                    "attack_family": current_label,
                    "benign_run_before": benign_run,
                    "attack_run_after": attack_run
                }
            )


onsets_df = pd.DataFrame(onsets)


print("\n" + "=" * 70)
print("ATTACK ONSET SUMMARY")
print("=" * 70)


print(
    f"\nTotal attack onsets: "
    f"{len(onsets_df):,}"
)


if len(onsets_df) > 0:

    print("\nOnsets by attack family:")

    print(
        onsets_df[
            "attack_family"
        ]
        .value_counts()
        .to_string()
    )


    print("\nAverage attack duration:")

    print(
        onsets_df.groupby(
            "attack_family"
        )["attack_run_after"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max"
            ]
        )
        .round(2)
        .to_string()
    )


    print("\nBenign states before attack:")

    print(
        onsets_df.groupby(
            "attack_family"
        )["benign_run_before"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max"
            ]
        )
        .round(2)
        .to_string()
    )


    print("\nFirst 20 attack onsets:")

    print(
        onsets_df.head(20)
        .to_string(index=False)
    )


output_path = (
    "data/attack_onsets.csv"
)

onsets_df.to_csv(
    output_path,
    index=False
)


print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(output_path)
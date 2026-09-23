import pandas as pd
import numpy as np


print("Loading feature-engineered data...")

df = pd.read_csv("data/portscan_features.csv")

print("Shape:", df.shape)


columns = [
    "Flow Packets/s",
    "Flow Bytes/s",
    "Flow Duration",
    "fwd_bwd_packet_ratio",
    "fwd_bwd_byte_ratio",
    "Flow IAT Std",
    "iat_variability",
    "packets_per_second",
    "bytes_per_second"
]


print()
print("=" * 75)
print("SUSPICIOUS FEATURE CHECK")
print("=" * 75)


for column in columns:

    values = pd.to_numeric(df[column], errors="coerce")

    print()
    print(column)

    print("  Negative values :", (values < 0).sum())
    print("  Zero values     :", (values == 0).sum())
    print("  NaN values      :", values.isna().sum())
    print("  Inf values      :", np.isinf(values).sum())

    print("  Min             :", values.min())
    print("  Median          :", values.median())
    print("  Max             :", values.max())


print()
print("=" * 75)
print("ROWS WITH NEGATIVE PACKET RATE")
print("=" * 75)

negative = df[df["Flow Packets/s"] < 0]

print("Number of rows:", len(negative))

if len(negative) > 0:

    print()

    print(
        negative[
            [
                "Flow Duration",
                "Total Fwd Packets",
                "Total Backward Packets",
                "Flow Packets/s",
                "packets_per_second"
            ]
        ].head(10)
    )


print()
print("=" * 75)
print("ROWS WITH EXTREME FORWARD/BACKWARD RATIOS")
print("=" * 75)

extreme = df[
    (df["fwd_bwd_packet_ratio"] > 1000)
    |
    (df["fwd_bwd_byte_ratio"] > 1000)
]

print("Number of rows:", len(extreme))

if len(extreme) > 0:

    print()

    print(
        extreme[
            [
                "Total Fwd Packets",
                "Total Backward Packets",
                "Total Length of Fwd Packets",
                "Total Length of Bwd Packets",
                "fwd_bwd_packet_ratio",
                "fwd_bwd_byte_ratio"
            ]
        ].head(10)
    )
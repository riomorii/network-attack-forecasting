import pandas as pd
import numpy as np

# --------------------------------
# 1. Load cleaned dataset
# --------------------------------

file_path = "data/portscan_clean.csv"

print("Loading cleaned dataset...")

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

print("Dataset loaded!")
print("Original shape:", df.shape)


# --------------------------------
# 2. Create behavioural features
# --------------------------------

print()
print("Creating behavioural features...")


# Avoid division by zero
epsilon = 1e-6


# Traffic volume
df["packets_per_second"] = (
    df["Total Fwd Packets"] + df["Total Backward Packets"]
) / (df["Flow Duration"] / 1_000_000 + epsilon)


df["bytes_per_second"] = (
    df["Total Length of Fwd Packets"] +
    df["Total Length of Bwd Packets"]
) / (df["Flow Duration"] / 1_000_000 + epsilon)


# Forward / backward packet ratio
df["fwd_bwd_packet_ratio"] = (
    df["Total Fwd Packets"] /
    (df["Total Backward Packets"] + epsilon)
)


# Forward / backward byte ratio
df["fwd_bwd_byte_ratio"] = (
    df["Total Length of Fwd Packets"] /
    (df["Total Length of Bwd Packets"] + epsilon)
)


# TCP SYN / ACK relationship
df["syn_ack_ratio"] = (
    df["SYN Flag Count"] /
    (df["ACK Flag Count"] + epsilon)
)


# TCP reset ratio
df["rst_ratio"] = (
    df["RST Flag Count"] /
    (
        df["SYN Flag Count"] +
        df["ACK Flag Count"] +
        df["RST Flag Count"] +
        epsilon
    )
)


# Packet-size behaviour
df["packet_size_range"] = (
    df["Max Packet Length"] -
    df["Min Packet Length"]
)


# IAT behaviour
df["iat_variability"] = (
    df["Flow IAT Std"] /
    (df["Flow IAT Mean"] + epsilon)
)


print("Feature engineering complete!")


# --------------------------------
# 3. Clean generated values
# --------------------------------

df = df.replace([np.inf, -np.inf], np.nan)

before = len(df)

df = df.dropna()

after = len(df)

print()
print("Rows before final cleaning:", before)
print("Rows after final cleaning:", after)
print("Rows removed:", before - after)


# --------------------------------
# 4. Display new features
# --------------------------------

new_features = [
    "packets_per_second",
    "bytes_per_second",
    "fwd_bwd_packet_ratio",
    "fwd_bwd_byte_ratio",
    "syn_ack_ratio",
    "rst_ratio",
    "packet_size_range",
    "iat_variability"
]

print()
print("New behavioural features:")

for feature in new_features:
    print("-", feature)


# --------------------------------
# 5. Save feature-engineered data
# --------------------------------

output_path = "data/portscan_features.csv"

df.to_csv(output_path, index=False)

print()
print("Feature-engineered dataset saved to:")
print(output_path)

print()
print("Final shape:", df.shape)
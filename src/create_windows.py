import pandas as pd
import numpy as np

# --------------------------------
# 1. Load feature-engineered data
# --------------------------------

file_path = "data/portscan_features.csv"

print("Loading feature-engineered dataset...")

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)


# --------------------------------
# 2. Convert labels to numbers
# --------------------------------

df["Label"] = df["Label"].map({
    "BENIGN": 0,
    "PortScan": 1
})


# --------------------------------
# 3. Select input features
# --------------------------------

feature_columns = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "Average Packet Size",

    # Our engineered features
    "packets_per_second",
    "bytes_per_second",
    "fwd_bwd_packet_ratio",
    "fwd_bwd_byte_ratio",
    "syn_ack_ratio",
    "rst_ratio",
    "packet_size_range",
    "iat_variability"
]


X = df[feature_columns].values
y = df["Label"].values


print()
print("Number of selected features:", len(feature_columns))


# --------------------------------
# 4. Create temporal windows
# --------------------------------

WINDOW_SIZE = 10

X_windows = []
y_windows = []

print()
print("Creating temporal windows...")


for i in range(len(X) - WINDOW_SIZE):

    window = X[i:i + WINDOW_SIZE]

    # Predict the state immediately after the window
    target = y[i + WINDOW_SIZE]

    X_windows.append(window)
    y_windows.append(target)


X_windows = np.array(X_windows)
y_windows = np.array(y_windows)


# --------------------------------
# 5. Display results
# --------------------------------

print()
print("Window creation complete!")

print("X shape:", X_windows.shape)
print("y shape:", y_windows.shape)

print()
print("Meaning of X shape:")
print("(number_of_windows, window_size, number_of_features)")

print()
print("Target distribution:")
print(pd.Series(y_windows).value_counts())


# --------------------------------
# 6. Save arrays
# --------------------------------

np.save("data/X_windows.npy", X_windows)
np.save("data/y_windows.npy", y_windows)

print()
print("Saved:")
print("data/X_windows.npy")
print("data/y_windows.npy")
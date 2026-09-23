import pandas as pd
import numpy as np

file_path = "data/raw/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

print("Loading PortScan dataset...")

df = pd.read_csv(file_path)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print()
print("Original shape:")
print(df.shape)

# Replace infinity values with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Count missing values
missing_before = df.isnull().sum().sum()

print()
print("Missing values before cleaning:")
print(missing_before)

# Remove rows containing missing values
df = df.dropna()

print()
print("Shape after cleaning:")
print(df.shape)

print()
print("Labels after cleaning:")
print(df["Label"].value_counts())

# Save cleaned dataset
output_path = "data/portscan_clean.csv"

df.to_csv(output_path, index=False)

print()
print("Cleaned dataset saved to:")
print(output_path)
import pandas as pd

file_path = "data/portscan_features.csv"

print("Loading feature-engineered dataset...")

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

print()
print("Dataset shape:")
print(df.shape)

print()
print("First 10 rows - selected temporal/network columns:")

columns_to_check = [
    "Flow Duration",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Destination Port",
    "Label"
]

print(df[columns_to_check].head(10))

print()
print("Label sequence for first 50 rows:")

print(df["Label"].head(50).to_list())

print()
print("Label counts:")

print(df["Label"].value_counts())
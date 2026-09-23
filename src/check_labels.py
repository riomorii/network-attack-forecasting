import pandas as pd

file_path = "data/raw/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

print("Reading PortScan dataset...")

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

print()
print("Dataset shape:")
print(df.shape)

print()
print("Labels:")
print(df["Label"].value_counts())

print()
print("Missing values:")
print(df.isnull().sum().sum())
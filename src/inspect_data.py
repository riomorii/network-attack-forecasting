import pandas as pd
file_path = "data/raw/MachineLearningCVE/Monday-workingHours.pcap_ISCX.csv"

df = pd.read_csv(file_path, nrows=5)

print("Dataset loaded successfully!")
print()
print("Number of columns:", len(df.columns))
print()
print("Column names:")
for column in df.columns:
    print(column)

print()
print("First 5 rows:")
print(df)
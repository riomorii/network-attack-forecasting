print("========== SIH DATA PIPELINE ==========")

print()
print("Dataset:")
print("CIC-IDS2017 PortScan")

print()
print("Original cleaned dataset:")
print("286,096 rows")
print("79 columns")

print()
print("Feature engineered dataset:")
print("286,063 rows")
print("87 columns")

print()
print("Selected model features:")
print("32")

print()
print("Temporal window:")
print("10 flows")

print()
print("Prediction target:")
print("Next-flow network state")

print()
print("Labels:")
print("0 = BENIGN")
print("1 = PortScan")

print()
print("Temporal split:")
print("70% Training")
print("15% Validation")
print("15% Testing")

print()
print("Scaling:")
print("StandardScaler")
print("Scaler fitted only on training data")

print()
print("Final training shape:")
print("(200237, 10, 32)")

print()
print("Final validation shape:")
print("(42908, 10, 32)")

print()
print("Final test shape:")
print("(42908, 10, 32)")

print()
print("========================================")
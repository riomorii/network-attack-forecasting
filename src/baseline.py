import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------
# 1. Load cleaned dataset
# --------------------------------

file_path = "data/portscan_clean.csv"

print("Loading cleaned dataset...")

df = pd.read_csv(file_path)

# Remove spaces from column names
df.columns = df.columns.str.strip()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------
# 2. Convert labels to numbers
# --------------------------------

# BENIGN = 0
# PortScan = 1

df["Label"] = df["Label"].map({
    "BENIGN": 0,
    "PortScan": 1
})

print()
print("Labels:")
print(df["Label"].value_counts())


# --------------------------------
# 3. Separate features and target
# --------------------------------

X = df.drop(columns=["Label"])
y = df["Label"]


# --------------------------------
# 4. Make sure features are numeric
# --------------------------------

X = X.apply(pd.to_numeric, errors="coerce")

# Remove rows containing invalid values
valid_rows = X.notna().all(axis=1)

X = X[valid_rows]
y = y[valid_rows]

print()
print("Usable rows:", len(X))


# --------------------------------
# 5. Split into training/testing
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------
# 6. Scale the features
# --------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------
# 7. Train Logistic Regression
# --------------------------------

model = LogisticRegression(
    max_iter=1000
)

print()
print("Training Logistic Regression...")

model.fit(X_train_scaled, y_train)

print("Training complete!")


# --------------------------------
# 8. Make predictions
# --------------------------------

y_pred = model.predict(X_test_scaled)


# --------------------------------
# 9. Evaluate model
# --------------------------------

print()
print("========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["BENIGN", "PortScan"]
    )
)


print("========== CONFUSION MATRIX ==========")

print(confusion_matrix(y_test, y_pred))
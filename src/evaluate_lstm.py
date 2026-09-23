import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# 1. Load test data
# ============================================================

print("Loading test data...")

X_test = np.load("data/X_test_scaled.npy")
y_test = np.load("data/y_test.npy")

print("Test data:", X_test.shape)
print("Test labels:", y_test.shape)


# ============================================================
# 2. Convert to PyTorch tensors
# ============================================================

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# ============================================================
# 3. Define the same LSTM architecture
# ============================================================

class LSTMModel(nn.Module):

    def __init__(self, input_size=32, hidden_size=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):

        lstm_output, (hidden, cell) = self.lstm(x)

        last_output = lstm_output[:, -1, :]

        prediction = self.output(last_output)

        return prediction.squeeze(1)


# ============================================================
# 4. Load trained model
# ============================================================

model = LSTMModel()

model.load_state_dict(
    torch.load(
        "models/lstm_model.pth",
        map_location="cpu"
    )
)

model.eval()

print()
print("Trained LSTM model loaded successfully!")


# ============================================================
# 5. Make predictions
# ============================================================

print()
print("Running predictions on test data...")

with torch.no_grad():

    logits = model(X_test)

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= 0.5
    ).int()


# Convert tensors to NumPy
y_true = y_test.numpy()
y_pred = predictions.numpy()


# ============================================================
# 6. Calculate accuracy
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print()
print("=" * 50)
print("FINAL LSTM TEST RESULTS")
print("=" * 50)

print()
print(
    f"Test Accuracy: {accuracy:.4f}"
)

print(
    f"Test Accuracy (%): {accuracy * 100:.2f}%"
)


# ============================================================
# 7. Classification report
# ============================================================

print()
print("Classification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "BENIGN",
            "PortScan"
        ]
    )
)


# ============================================================
# 8. Confusion matrix
# ============================================================

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# 1. Load data
# ============================================================

print("Loading training and validation data...")

X_train = np.load("data/X_train_scaled.npy")
y_train = np.load("data/y_train.npy")

X_val = np.load("data/X_val_scaled.npy")
y_val = np.load("data/y_val.npy")

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)


# ============================================================
# 2. Convert NumPy → PyTorch tensors
# ============================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)


# ============================================================
# 3. Create DataLoader
# ============================================================

BATCH_SIZE = 256

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 4. Define LSTM model
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

        # Take output from final time step
        last_output = lstm_output[:, -1, :]

        prediction = self.output(last_output)

        return prediction.squeeze(1)


# ============================================================
# 5. Create model
# ============================================================

model = LSTMModel()

print()
print("LSTM model created.")

print(model)


# ============================================================
# 6. Loss and optimizer
# ============================================================

loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 7. Training
# ============================================================

EPOCHS = 5

print()
print("Starting LSTM training...")
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)


for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0

    for X_batch, y_batch in train_loader:

        # Forward pass
        predictions = model(X_batch)

        # Calculate loss
        loss = loss_function(predictions, y_batch)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss / len(train_loader)
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    total_val_loss = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            predictions = model(X_batch)

            loss = loss_function(
                predictions,
                y_batch
            )

            total_val_loss += loss.item()

            probabilities = torch.sigmoid(predictions)

            predicted_labels = (
                probabilities >= 0.5
            ).int()

            all_predictions.extend(
                predicted_labels.numpy()
            )

            all_targets.extend(
                y_batch.numpy()
            )

    average_val_loss = (
        total_val_loss / len(val_loader)
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {average_train_loss:.4f}"
    )

    print(
        f"Validation Loss: {average_val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {accuracy:.4f}"
    )


# ============================================================
# 8. Final validation report
# ============================================================

print()
print("=" * 50)
print("FINAL VALIDATION RESULTS")
print("=" * 50)

print()

print(
    classification_report(
        all_targets,
        all_predictions,
        target_names=[
            "BENIGN",
            "PortScan"
        ]
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        all_targets,
        all_predictions
    )
)


# ============================================================
# 9. Save model
# ============================================================

model_path = "models/lstm_model.pth"

torch.save(
    model.state_dict(),
    model_path
)

print()
print("LSTM model saved to:")
print(model_path)
import numpy as np
import torch
import torch.nn as nn

print("Loading small training sample...")

# Load only a small sample for testing
X = np.load("data/X_train_scaled.npy")[:2000]
y = np.load("data/y_train.npy")[:2000]

# Convert NumPy arrays to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

print("Input shape:", X.shape)
print("Target shape:", y.shape)


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # LSTM processes the 10 time steps
        lstm_output, (hidden, cell) = self.lstm(x)

        # Use the final time step
        last_output = lstm_output[:, -1, :]

        # Convert to one prediction
        prediction = self.output(last_output)

        return prediction.squeeze(1)


# Our data has 32 features
model = LSTMModel(
    input_size=32,
    hidden_size=32
)

print()
print("LSTM model created!")

# Binary classification loss
loss_function = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Starting one test training step...")

# Forward pass
predictions = model(X)

# Calculate loss
loss = loss_function(predictions, y)

# Clear old gradients
optimizer.zero_grad()

# Backpropagation
loss.backward()

# Update model
optimizer.step()

print()
print("Test training successful!")
print("Loss:", loss.item())
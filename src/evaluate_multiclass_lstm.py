import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)

# ============================================
# CONFIG
# ============================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================
# MODEL
# ============================================

class MulticlassLSTM(nn.Module):

    def __init__(
        self,
        input_size=15,
        hidden_size=64,
        num_layers=1,
        num_classes=3
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_hidden = output[:, -1, :]

        return self.classifier(last_hidden)


# ============================================
# LOAD TEST DATA
# ============================================

X_test = np.load(
    "data/multiclass_3class_X_test.npy"
)

y_test = np.load(
    "data/multiclass_3class_y_test.npy"
)

print("===============================================")
print("MULTICLASS LSTM TEST EVALUATION")
print("===============================================")

print("Test X:", X_test.shape)
print("Test y:", y_test.shape)

# ============================================
# LOAD MODEL
# ============================================

model = MulticlassLSTM().to(DEVICE)

model.load_state_dict(
    torch.load(
        "models/multiclass_lstm.pth",
        map_location=DEVICE
    )
)

model.eval()

# ============================================
# PREDICTION
# ============================================

X_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():

    logits = model(X_tensor)

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predictions = torch.argmax(
        probabilities,
        dim=1
    )

predictions = predictions.cpu().numpy()
probabilities = probabilities.cpu().numpy()

# ============================================
# METRICS
# ============================================

accuracy = accuracy_score(
    y_test,
    predictions
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_test,
        predictions,
        labels=[0, 1, 2],
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1, 2]
)

# ============================================
# RESULTS
# ============================================

print("\n===============================================")
print("OVERALL RESULTS")
print("===============================================")

print(
    f"Accuracy:          {accuracy:.4f}"
)

print(
    f"Balanced Accuracy: {balanced_accuracy:.4f}"
)

print("\n===============================================")
print("PER-CLASS RESULTS")
print("===============================================")

class_names = [
    "BENIGN",
    "DOS_DDOS",
    "RECONNAISSANCE"
]

for i, name in enumerate(class_names):

    print(
        f"{name:18s} "
        f"Precision={precision[i]:.4f} "
        f"Recall={recall[i]:.4f} "
        f"F1={f1[i]:.4f} "
        f"Support={support[i]}"
    )

print("\n===============================================")
print("CONFUSION MATRIX")
print("===============================================")

print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(
    "                BENIGN  DOS_DDOS  RECON"
)

for i, row in enumerate(cm):

    print(
        f"{class_names[i]:15s} "
        f"{row[0]:7d} "
        f"{row[1]:9d} "
        f"{row[2]:7d}"
    )

print("\n===============================================")
print("CLASSIFICATION REPORT")
print("===============================================")

print(
    classification_report(
        y_test,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

# ============================================
# SAVE PREDICTIONS
# ============================================

np.save(
    "data/multiclass_test_predictions.npy",
    predictions
)

np.save(
    "data/multiclass_test_probabilities.npy",
    probabilities
)

print("===============================================")
print("SAVED")
print("===============================================")

print(
    "data/multiclass_test_predictions.npy"
)

print(
    "data/multiclass_test_probabilities.npy"
)
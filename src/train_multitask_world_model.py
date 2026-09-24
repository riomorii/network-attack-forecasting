import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

print("===============================================")
print("MULTI-TASK WORLD MODEL")
print("===============================================")

# ------------------------------------------------
# Configuration
# ------------------------------------------------

BATCH_SIZE = 256
HIDDEN_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

# Loss weighting
STATE_LOSS_WEIGHT = 1.0
CLASS_LOSS_WEIGHT = 1.0

DEVICE = torch.device("cpu")

print("Device:", DEVICE)

# ------------------------------------------------
# 1. Load 3-class temporal data
# ------------------------------------------------

X_train = np.load(
    "data/multiclass_3class_X_train.npy"
)

y_train = np.load(
    "data/multiclass_3class_y_train.npy",
    allow_pickle=True
)

X_val = np.load(
    "data/multiclass_3class_X_val.npy"
)

y_val = np.load(
    "data/multiclass_3class_y_val.npy",
    allow_pickle=True
)

X_test = np.load(
    "data/multiclass_3class_X_test.npy"
)

y_test = np.load(
    "data/multiclass_3class_y_test.npy",
    allow_pickle=True
)

print("\nOriginal shapes:")
print("X train:", X_train.shape)
print("y train:", y_train.shape)
print("X val:  ", X_val.shape)
print("y val:  ", y_val.shape)
print("X test: ", X_test.shape)
print("y test: ", y_test.shape)

# ------------------------------------------------
# 2. Convert class labels
# ------------------------------------------------

# The 3-class dataset already stores numeric IDs:
#
# 0 = BENIGN
# 1 = DOS_DDOS
# 2 = RECONNAISSANCE

y_train_id = y_train.astype(np.int64)
y_val_id = y_val.astype(np.int64)
y_test_id = y_test.astype(np.int64)

print("\nLabel values:")
print("Train:", np.unique(y_train_id))
print("Val:  ", np.unique(y_val_id))
print("Test: ", np.unique(y_test_id))
print("\nClass distribution:")

for name, data in [
    ("Train", y_train_id),
    ("Val", y_val_id),
    ("Test", y_test_id)
]:
    print(
        name,
        {
            "BENIGN": int(np.sum(data == 0)),
            "DOS_DDOS": int(np.sum(data == 1)),
            "RECONNAISSANCE": int(np.sum(data == 2))
        }
    )

# ------------------------------------------------
# 3. Scale network-state features
#    Fit ONLY on training data
# ------------------------------------------------

scaler = StandardScaler()

train_flat = X_train.reshape(
    -1,
    X_train.shape[-1]
)

scaler.fit(train_flat)

def scale_sequences(X):
    shape = X.shape

    flat = X.reshape(
        -1,
        shape[-1]
    )

    scaled = scaler.transform(flat)

    return scaled.reshape(shape)


X_train_scaled = scale_sequences(X_train)
X_val_scaled = scale_sequences(X_val)
X_test_scaled = scale_sequences(X_test)

# ------------------------------------------------
# 4. Continuous targets
#
# The next state is already represented by the
# last state of each input sequence's following
# target in our sequence construction.
#
# Because our existing y arrays contain the class
# label, we reconstruct the next network state
# from the original multiclass state sequence.
# ------------------------------------------------

# Load original multiclass states and metadata
states = np.load(
    "data/multiclass_network_states.npy"
)

metadata = __import__("pandas").read_csv(
    "data/multiclass_state_metadata.csv"
)

# Reconstruct the exact same sequence ordering
# used by create_multiclass_sequences.py.

WINDOW_SIZE = 5

target_states = []

for scenario, group in metadata.groupby(
    "source_file",
    sort=False
):

    indices = group.index.to_numpy()

    scenario_states = states[indices]

    n = len(scenario_states)

    for i in range(
        n - WINDOW_SIZE
    ):

        target_states.append(
            scenario_states[
                i + WINDOW_SIZE
            ]
        )

target_states = np.asarray(
    target_states
)

print(
    "\nReconstructed target states:",
    target_states.shape
)

# ------------------------------------------------
# The sequence creator used the same scenario
# ordering, so split targets using the same
# 70/15/15 structure and remove the same rare
# training classes.
# ------------------------------------------------

all_labels = np.load(
    "data/multiclass_state_labels.npy",
    allow_pickle=True
)

sequence_labels = []
sequence_scenarios = []

for scenario, group in metadata.groupby(
    "source_file",
    sort=False
):

    indices = group.index.to_numpy()

    scenario_labels = all_labels[indices]

    n = len(scenario_labels)

    for i in range(
        n - WINDOW_SIZE
    ):

        sequence_labels.append(
            str(
                scenario_labels[
                    i + WINDOW_SIZE
                ]
            )
        )

        sequence_scenarios.append(
            scenario
        )

sequence_labels = np.asarray(
    sequence_labels
)

sequence_scenarios = np.asarray(
    sequence_scenarios
)

# ------------------------------------------------
# Recreate same per-scenario temporal split
# ------------------------------------------------

train_indices = []
val_indices = []
test_indices = []

for scenario in np.unique(
    sequence_scenarios
):

    idx = np.where(
        sequence_scenarios == scenario
    )[0]

    n = len(idx)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_indices.extend(
        idx[:train_end]
    )

    val_indices.extend(
        idx[train_end:val_end]
    )

    test_indices.extend(
        idx[val_end:]
    )

train_indices = np.asarray(
    train_indices
)

val_indices = np.asarray(
    val_indices
)

test_indices = np.asarray(
    test_indices
)

# ------------------------------------------------
# Keep only the same 3 classes
# ------------------------------------------------

allowed = np.array([
    "BENIGN",
    "DOS_DDOS",
    "RECONNAISSANCE"
])

train_keep = np.isin(
    sequence_labels[train_indices],
    allowed
)

val_keep = np.isin(
    sequence_labels[val_indices],
    allowed
)

test_keep = np.isin(
    sequence_labels[test_indices],
    allowed
)

train_indices = train_indices[
    train_keep
]

val_indices = val_indices[
    val_keep
]

test_indices = test_indices[
    test_keep
]

# ------------------------------------------------
# Verify target sizes
# ------------------------------------------------

target_train = target_states[
    train_indices
]

target_val = target_states[
    val_indices
]

target_test = target_states[
    test_indices
]

print("\nTarget shapes:")
print("Train:", target_train.shape)
print("Val:  ", target_val.shape)
print("Test: ", target_test.shape)

# ------------------------------------------------
# Scale continuous targets using same TRAIN scaler
# ------------------------------------------------

target_train_scaled = scaler.transform(
    target_train
)

target_val_scaled = scaler.transform(
    target_val
)

target_test_scaled = scaler.transform(
    target_test
)

# ------------------------------------------------
# Sanity checks
# ------------------------------------------------

assert len(X_train_scaled) == len(
    y_train_id
)

assert len(X_val_scaled) == len(
    y_val_id
)

assert len(X_test_scaled) == len(
    y_test_id
)

assert len(target_train_scaled) == len(
    X_train_scaled
)

assert len(target_val_scaled) == len(
    X_val_scaled
)

assert len(target_test_scaled) == len(
    X_test_scaled
)

print("\nAll dataset sizes match.")

# ------------------------------------------------
# 5. Create PyTorch datasets
# ------------------------------------------------

train_dataset = TensorDataset(
    torch.tensor(
        X_train_scaled,
        dtype=torch.float32
    ),
    torch.tensor(
        target_train_scaled,
        dtype=torch.float32
    ),
    torch.tensor(
        y_train_id,
        dtype=torch.long
    )
)

val_dataset = TensorDataset(
    torch.tensor(
        X_val_scaled,
        dtype=torch.float32
    ),
    torch.tensor(
        target_val_scaled,
        dtype=torch.float32
    ),
    torch.tensor(
        y_val_id,
        dtype=torch.long
    )
)

test_dataset = TensorDataset(
    torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    ),
    torch.tensor(
        target_test_scaled,
        dtype=torch.float32
    ),
    torch.tensor(
        y_test_id,
        dtype=torch.long
    )
)

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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ------------------------------------------------
# 6. Model
# ------------------------------------------------

class MultiTaskWorldModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_classes
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.state_head = nn.Linear(
            hidden_size,
            output_size
        )

        self.class_head = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        hidden = output[:, -1, :]

        predicted_state = self.state_head(
            hidden
        )

        class_logits = self.class_head(
            hidden
        )

        return predicted_state, class_logits


model = MultiTaskWorldModel(
    input_size=15,
    hidden_size=HIDDEN_SIZE,
    output_size=15,
    num_classes=3
).to(DEVICE)

# ------------------------------------------------
# 7. Loss functions
# ------------------------------------------------

state_loss_fn = nn.MSELoss()

# Calculate class weights from TRAIN ONLY
class_counts = np.bincount(
    y_train_id,
    minlength=3
)

class_weights = (
    len(y_train_id)
    /
    (
        3 * class_counts
    )
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print(
    "\nClass weights:",
    class_weights.cpu().numpy()
)

class_loss_fn = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ------------------------------------------------
# 8. Validation function
# ------------------------------------------------

def evaluate(loader):

    model.eval()

    total_state_loss = 0.0
    total_class_loss = 0.0

    all_true = []
    all_pred = []

    total_samples = 0

    with torch.no_grad():

        for xb, y_state, y_class in loader:

            xb = xb.to(DEVICE)
            y_state = y_state.to(DEVICE)
            y_class = y_class.to(DEVICE)

            pred_state, logits = model(xb)

            state_loss = state_loss_fn(
                pred_state,
                y_state
            )

            class_loss = class_loss_fn(
                logits,
                y_class
            )

            batch_size = xb.size(0)

            total_state_loss += (
                state_loss.item()
                * batch_size
            )

            total_class_loss += (
                class_loss.item()
                * batch_size
            )

            pred_class = torch.argmax(
                logits,
                dim=1
            )

            all_true.extend(
                y_class.cpu().numpy()
            )

            all_pred.extend(
                pred_class.cpu().numpy()
            )

            total_samples += batch_size

    avg_state_loss = (
        total_state_loss
        / total_samples
    )

    avg_class_loss = (
        total_class_loss
        / total_samples
    )

    accuracy = accuracy_score(
        all_true,
        all_pred
    )

    balanced_acc = balanced_accuracy_score(
        all_true,
        all_pred
    )

    _, _, macro_f1, _ = (
        precision_recall_fscore_support(
            all_true,
            all_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )
    )

    return (
        avg_state_loss,
        avg_class_loss,
        accuracy,
        balanced_acc,
        macro_f1
    )


# ------------------------------------------------
# 9. Training
# ------------------------------------------------

best_val_macro_f1 = -1

print("\n===============================================")
print("TRAINING")
print("===============================================")

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0
    total_state_loss = 0.0
    total_class_loss = 0.0
    total_samples = 0

    for xb, y_state, y_class in train_loader:

        xb = xb.to(DEVICE)
        y_state = y_state.to(DEVICE)
        y_class = y_class.to(DEVICE)

        optimizer.zero_grad()

        pred_state, logits = model(xb)

        state_loss = state_loss_fn(
            pred_state,
            y_state
        )

        class_loss = class_loss_fn(
            logits,
            y_class
        )

        loss = (
            STATE_LOSS_WEIGHT * state_loss
            +
            CLASS_LOSS_WEIGHT * class_loss
        )

        loss.backward()

        optimizer.step()

        batch_size = xb.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_state_loss += (
            state_loss.item()
            * batch_size
        )

        total_class_loss += (
            class_loss.item()
            * batch_size
        )

        total_samples += batch_size

    train_loss = (
        total_loss
        / total_samples
    )

    train_state_loss = (
        total_state_loss
        / total_samples
    )

    train_class_loss = (
        total_class_loss
        / total_samples
    )

    (
        val_state_loss,
        val_class_loss,
        val_accuracy,
        val_balanced_acc,
        val_macro_f1
    ) = evaluate(val_loader)

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss {train_loss:.4f} | "
        f"State {train_state_loss:.4f} | "
        f"Class {train_class_loss:.4f} | "
        f"Val State {val_state_loss:.4f} | "
        f"Val Acc {val_accuracy:.4f} | "
        f"Val BalAcc {val_balanced_acc:.4f} | "
        f"Val MacroF1 {val_macro_f1:.4f}"
    )

    # Save BEST validation model
    if val_macro_f1 > best_val_macro_f1:

        best_val_macro_f1 = val_macro_f1

        torch.save(
            model.state_dict(),
            "models/multitask_world_model_best.pth"
        )

        print(
            "  -> Saved best model"
        )

# ------------------------------------------------
# 10. Final test evaluation
# ------------------------------------------------

print("\n===============================================")
print("FINAL TEST EVALUATION")
print("===============================================")

model.load_state_dict(
    torch.load(
        "models/multitask_world_model_best.pth",
        map_location=DEVICE
    )
)

(
    test_state_loss,
    test_class_loss,
    test_accuracy,
    test_balanced_acc,
    test_macro_f1
) = evaluate(test_loader)

print(
    f"Test State MSE:       {test_state_loss:.6f}"
)

print(
    f"Test Class Loss:      {test_class_loss:.6f}"
)

print(
    f"Test Accuracy:        {test_accuracy:.4f}"
)

print(
    f"Test Balanced Acc:    {test_balanced_acc:.4f}"
)

print(
    f"Test Macro F1:        {test_macro_f1:.4f}"
)

# ------------------------------------------------
# Detailed classification report
# ------------------------------------------------

model.eval()

all_true = []
all_pred = []

with torch.no_grad():

    for xb, _, y_class in test_loader:

        xb = xb.to(DEVICE)

        _, logits = model(xb)

        pred = torch.argmax(
            logits,
            dim=1
        )

        all_true.extend(
            y_class.numpy()
        )

        all_pred.extend(
            pred.cpu().numpy()
        )

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=[0, 1, 2]
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        all_true,
        all_pred,
        labels=[0, 1, 2],
        zero_division=0
    )
)

names = [
    "BENIGN",
    "DOS_DDOS",
    "RECONNAISSANCE"
]

print("\n===============================================")
print("PER-CLASS RESULTS")
print("===============================================")

for i, name in enumerate(names):

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
    "                BENIGN  DOS_DDOS  RECON"
)

for i, row in enumerate(cm):

    print(
        f"{names[i]:15s} "
        f"{row[0]:7d} "
        f"{row[1]:9d} "
        f"{row[2]:7d}"
    )

# ------------------------------------------------
# Save scaler
# ------------------------------------------------

import joblib

joblib.dump(
    scaler,
    "models/multitask_world_model_scaler.pkl"
)

print(
    "\nSaved:"
)

print(
    "models/multitask_world_model_best.pth"
)

print(
    "models/multitask_world_model_scaler.pkl"
)
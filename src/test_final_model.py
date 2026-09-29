import torch

# Because this script is being run as:
# python src\test_final_model.py
# Python automatically searches the src folder for modules.
from world_model import MultiTaskWorldModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "models/multitask_world_model_final.pth"

INPUT_SIZE = 15
HIDDEN_SIZE = 64
OUTPUT_SIZE = 15
NUM_CLASSES = 3


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("FINAL WORLD MODEL CHECK")
print("=" * 60)

print(
    f"Device: {device}"
)


# ============================================================
# CREATE MODEL ARCHITECTURE
# ============================================================

model = MultiTaskWorldModel(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    num_classes=NUM_CLASSES
)


# ============================================================
# LOAD FINAL CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=False
)


# The training script may save either:
#   {"model_state_dict": ...}
# or:
#   {"state_dict": ...}
# or directly the state dictionary.

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# ============================================================
# LOAD PARAMETERS
# ============================================================

model.load_state_dict(
    state_dict,
    strict=True
)

model.to(device)

model.eval()


print(
    f"Loaded checkpoint: {MODEL_FILE}"
)


# ============================================================
# TEST INPUT
# ============================================================

# Create one dummy input:
#
# batch size     = 1
# history        = 5 states
# features/state = 15

x = torch.randn(
    1,
    5,
    15,
    device=device
)


# ============================================================
# INFERENCE
# ============================================================

with torch.no_grad():

    future_state, class_logits = model(x)


# ============================================================
# PREDICTED CLASS
# ============================================================

predicted_class = torch.argmax(
    class_logits,
    dim=1
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nModel test passed.")

print(
    f"Input shape:          {x.shape}"
)

print(
    f"Future state shape:   {future_state.shape}"
)

print(
    f"Class logits shape:   {class_logits.shape}"
)

print(
    f"Predicted class ID:   {predicted_class.item()}"
)


# ============================================================
# FINAL CONFIRMATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL VERIFICATION SUCCESSFUL")
print("=" * 60)

print(
    "The saved checkpoint is compatible with "
    "src/world_model.py."
)

print(
    "No training was performed."
)
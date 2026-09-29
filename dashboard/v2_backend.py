
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "experimental_v2"

CLASS_NAMES = (
    "BENIGN",
    "DOS_DDOS",
    "RECONNAISSANCE",
)

EXPERIMENTAL_WARNING_THRESHOLD = 0.10


class MultiTaskWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=15,
            hidden_size=64,
            batch_first=True,
        )
        self.state_head = nn.Linear(64, 15)
        self.class_head = nn.Linear(64, 3)

    def forward(self, x):
        output, _ = self.lstm(x)
        hidden = output[:, -1, :]
        return (
            self.state_head(hidden),
            self.class_head(hidden),
        )


class V2PredictionBackend:
    def __init__(self):
        self.model = MultiTaskWorldModel()
        self.model.load_state_dict(
            torch.load(
                MODEL_DIR / "multitask_v2_best.pth",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        self.model.eval()

        self.scaler = joblib.load(
            MODEL_DIR / "multitask_v2_scaler.pkl"
        )

        if self.scaler.n_features_in_ != 15:
            raise ValueError(
                "Expected a scaler with 15 features."
            )

    def predict(
        self,
        states,
        warning_threshold=EXPERIMENTAL_WARNING_THRESHOLD,
    ):
        states = np.asarray(
            states,
            dtype=np.float64,
        )

        if states.shape != (5, 15):
            raise ValueError(
                "Input must contain exactly "
                "5 states with 15 features each."
            )

        if not np.isfinite(states).all():
            raise ValueError(
                "Input contains non-finite values."
            )

        if not 0.0 <= warning_threshold <= 1.0:
            raise ValueError(
                "Warning threshold must be between 0 and 1."
            )

        scaled = self.scaler.transform(states)

        tensor = torch.tensor(
            scaled[np.newaxis, :, :],
            dtype=torch.float32,
        )

        with torch.no_grad():
            next_scaled, logits = self.model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0].numpy()

        next_state = self.scaler.inverse_transform(
            next_scaled.numpy()
        )[0]

        predicted_class = int(
            np.argmax(probabilities)
        )
        ddos_probability = float(
            probabilities[1]
        )

        return {
            "predicted_class": (
                CLASS_NAMES[predicted_class]
            ),
            "class_probabilities": {
                name: float(probabilities[i])
                for i, name in enumerate(CLASS_NAMES)
            },
            "ddos_probability": ddos_probability,
            "experimental_warning": (
                ddos_probability >= warning_threshold
            ),
            "warning_threshold": float(
                warning_threshold
            ),
            "predicted_next_state": (
                next_state.tolist()
            ),
        }

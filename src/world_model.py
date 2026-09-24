import torch
import torch.nn as nn


class MultiTaskWorldModel(nn.Module):
    """
    Multi-task World Model for network attack forecasting.

    Input:
        Sequence of 5 historical network states.
        Each state contains 15 features.

    Outputs:
        1. Future 15-dimensional network state
        2. Future attack-class logits for:
           0 = BENIGN
           1 = DOS_DDOS
           2 = RECONNAISSANCE
    """

    def __init__(
        self,
        input_size=15,
        hidden_size=64,
        output_size=15,
        num_classes=3
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
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: (batch_size, sequence_length, 15)

        Returns
        -------
        future_state : torch.Tensor
            Shape: (batch_size, 15)

        class_logits : torch.Tensor
            Shape: (batch_size, 3)
        """

        lstm_output, _ = self.lstm(x)

        # Use the hidden representation of the latest
        # observed network state.
        last_hidden = lstm_output[:, -1, :]

        # Predict future continuous network state.
        future_state = self.state_head(
            last_hidden
        )

        # Predict future attack class.
        class_logits = self.class_head(
            last_hidden
        )

        return future_state, class_logits
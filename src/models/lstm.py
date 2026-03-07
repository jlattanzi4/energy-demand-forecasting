"""PyTorch LSTM model for energy demand forecasting.

Architecture:
    - 2-layer stacked LSTM with dropout
    - Fully connected output head projecting to forecast horizon
    - Uses the final hidden state from the last timestep

Input:  (batch, seq_len=168, input_size=17)
Output: (batch, forecast_horizon=24)
"""

import torch
import torch.nn as nn


class EnergyLSTM(nn.Module):
    """Multi-step energy demand forecasting with stacked LSTM.

    Args:
        input_size: Number of input features per timestep.
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability between LSTM layers and in the FC head.
        forecast_horizon: Number of future timesteps to predict.
    """

    def __init__(
        self,
        input_size: int = 17,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 24,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_size) input tensor.

        Returns:
            (batch, forecast_horizon) predicted values.
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        output = self.fc(last_hidden)  # (batch, forecast_horizon)
        return output

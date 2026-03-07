"""Temporal Convolutional Network (TCN) for energy demand forecasting.

Architecture:
    - Stack of TemporalBlocks with exponentially increasing dilation
    - Each block: two causal convolutions with weight normalization,
      ReLU activation, dropout, and a residual skip connection
    - Fully connected output head projecting to forecast horizon

Input:  (batch, seq_len=168, input_size=17)
Output: (batch, forecast_horizon=24)

Reference:
    Bai, Kolter, Koltun - "An Empirical Evaluation of Generic Convolutional
    and Recurrent Networks for Sequence Modeling" (2018)
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Module):
    """Causal convolution with left-padding to prevent future information leakage."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.padding,
                dilation=dilation,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove right-side padding to enforce causality
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TemporalBlock(nn.Module):
    """Single temporal block with two causal convolutions and a residual connection.

    Architecture: CausalConv -> ReLU -> Dropout -> CausalConv -> ReLU -> Dropout + Residual
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1x1 convolution for residual if channel dimensions differ
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))

        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilation.

    Dilation pattern: 1, 2, 4, 8, ... (2^i for layer i).
    This gives the network an exponentially large receptive field.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2**i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EnergyTCN(nn.Module):
    """TCN model for multi-step energy demand forecasting.

    Args:
        input_size: Number of input features per timestep.
        num_channels: List of channel sizes for each temporal block.
            Default [64, 64, 64, 64] gives receptive field of
            2 * 4 * (3-1) * (1+2+4+8) = 120 with kernel_size=3.
        kernel_size: Convolution kernel size.
        dropout: Dropout probability.
        forecast_horizon: Number of future timesteps to predict.
    """

    def __init__(
        self,
        input_size: int = 17,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        forecast_horizon: int = 24,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64, 64]

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)

        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_size) input tensor.

        Returns:
            (batch, forecast_horizon) predicted values.
        """
        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        tcn_out = self.tcn(x)  # (batch, num_channels[-1], seq_len)

        # Use the last timestep's representation
        last_out = tcn_out[:, :, -1]  # (batch, num_channels[-1])

        output = self.fc(last_out)  # (batch, forecast_horizon)
        return output

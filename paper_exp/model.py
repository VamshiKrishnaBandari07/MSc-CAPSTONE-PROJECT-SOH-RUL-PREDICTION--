from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """Trim right-side padding so a dilated convolution remains causal."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual TCN block with exponentially dilated causal convolutions."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(self.net(x) + residual)


class TemporalConvNet(nn.Module):
    """Stacked TCN with dilation factors 1, 2, 4, ... as described in the paper."""

    def __init__(self, num_inputs: int, num_channels: Sequence[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        for index, out_channels in enumerate(num_channels):
            dilation = 2 ** index
            in_channels = num_inputs if index == 0 else num_channels[index - 1]
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AdditiveAttention(nn.Module):
    """Additive attention over learned degradation time/voltage states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        energy = self.score(encoder_outputs)
        weights = F.softmax(energy, dim=1)
        context = torch.sum(encoder_outputs * weights, dim=1)
        return context, weights


class PaperCNNTCNLSTMAttention(nn.Module):
    """
    CNN-TCN-LSTM-Attention SOH regressor matching the paper experiment.

    Input shape: [batch, 3, seq_len] for ICA=dQ/dV, DV=dV/dQ, and DC=dI/dV.
    Output shape: [batch, 1] predicted SOH.
    """

    def __init__(
        self,
        input_features: int = 3,
        cnn_out_channels: int = 64,
        tcn_channels: Sequence[int] = (64, 128, 128),
        lstm_hidden: int = 128,
        num_lstm_layers: int = 1,
        head_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(input_features, cnn_out_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.cnn_dropout = nn.Dropout(dropout)

        self.tcn = TemporalConvNet(cnn_out_channels, tcn_channels, kernel_size=3, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=tcn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(lstm_hidden)
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.conv1d.weight)
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)
        for module in self.regressor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for name, parameter in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.cnn_dropout(self.pool(self.relu(self.conv1d(x))))
        x = self.tcn(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        context, attention_weights = self.attention(lstm_out)
        soh = self.regressor(context)
        return soh, attention_weights


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    model = PaperCNNTCNLSTMAttention()
    sample = torch.randn(8, 3, 128)
    soh, attention = model(sample)
    print("--- paper_exp model verification ---")
    print(f"Input: {sample.shape}")
    print(f"SOH output: {soh.shape}")
    print(f"Attention: {attention.shape}")
    print(f"Trainable parameters: {count_parameters(model) / 1e6:.3f}M")


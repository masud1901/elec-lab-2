"""Models for the DeepConvLSTM ablation study.

Reference architecture (matches the workshop's ~4M-parameter model):

    input (B, T=24, C=113)
    -> 4 x Conv2d(64, kernel=(5, 1)) over time, ReLU
    -> (B, T'=8, C=113, F=64) -> flatten to 113*64=7232
    -> dropout 0.5
    -> 2 x LSTM(128)
    -> Linear(128, 18)

The conv layers are 2-D with a kernel of (5, 1), so they convolve over time
and keep the sensor-channel axis as the "width".  This is the variant implied
by the workshop's "~4M parameters" and "0.67 s is the shortest possible
window": 24 - 4*(5-1) = 8 time steps, and 20 - 16 = 4 steps for 0.67 s.

Ablation variants:
    --model cnn       two dense layers of 128 units instead of the recurrent stack
    --rnn gru         GRU cells instead of LSTM cells
    --n-conv 0/1/2    number of convolutional layers (0 = raw 113 channels)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DeepConvLSTM(nn.Module):
    def __init__(
        self,
        n_channels: int = 113,
        n_classes: int = 18,
        window: int = 24,
        n_conv: int = 4,
        conv_filters: int = 64,
        kernel_size: int = 5,
        rnn_type: str = "lstm",
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        model: str = "deepconvlstm",
        dense_hidden: int = 128,
    ):
        super().__init__()
        if model not in ("deepconvlstm", "cnn"):
            raise ValueError(f"Unknown model: {model}")
        if rnn_type not in ("lstm", "gru"):
            raise ValueError(f"Unknown rnn_type: {rnn_type}")
        if n_conv < 0:
            raise ValueError("n_conv must be >= 0")

        self.model = model
        self.n_conv = int(n_conv)
        self.n_channels = int(n_channels)
        self.conv_filters = int(conv_filters)
        self.kernel_size = int(kernel_size)

        self.convs = nn.ModuleList()
        if self.n_conv > 0:
            in_ch = 1
            for _ in range(self.n_conv):
                self.convs.append(
                    nn.Conv2d(in_ch, conv_filters, kernel_size=(kernel_size, 1))
                )
                in_ch = conv_filters
        self.relu = nn.ReLU()

        t_out = window - self.n_conv * (kernel_size - 1)
        if t_out <= 0:
            raise ValueError(
                f"window={window} is too short for {self.n_conv} conv layers "
                f"with kernel {kernel_size} (need > {self.n_conv*(kernel_size-1)})"
            )
        self.t_out = t_out
        in_features = n_channels * conv_filters if self.n_conv > 0 else n_channels
        self.in_features = in_features

        if model == "cnn":
            # Baseline CNN: flatten the whole feature map, then two dense layers.
            self.input_dropout = nn.Dropout(dropout)
            self.dense1 = nn.Linear(t_out * in_features, dense_hidden)
            self.dense2 = nn.Linear(dense_hidden, dense_hidden)
            self.classifier = nn.Linear(dense_hidden, n_classes)
        else:
            self.input_dropout = nn.Dropout(dropout)
            rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=in_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        if self.n_conv > 0:
            x = x.unsqueeze(1)  # (B, 1, T, C)
            for conv in self.convs:
                x = self.relu(conv(x))
            # (B, F, T', C) -> (B, T', C, F) -> (B, T', C*F)
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(x.shape[0], self.t_out, self.in_features)

        if self.model == "cnn":
            x = x.reshape(x.shape[0], -1)
            x = self.input_dropout(x)
            x = self.relu(self.dense1(x))
            x = self.input_dropout(x)
            x = self.relu(self.dense2(x))
            x = self.input_dropout(x)
            return self.classifier(x)

        x = self.input_dropout(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        return self.classifier(x)


def init_weights(module: nn.Module) -> None:
    """Orthogonal initialisation for weight matrices, zero biases."""
    if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LSTM, nn.GRU)):
        for name, param in module.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    model: str = "deepconvlstm",
    rnn: str = "lstm",
    n_conv: int = 4,
    window: int = 24,
    dropout: float = 0.5,
    n_channels: int = 113,
    n_classes: int = 18,
    conv_filters: int = 64,
    kernel_size: int = 5,
    hidden_size: int = 128,
    num_layers: int = 2,
    dense_hidden: int = 128,
) -> DeepConvLSTM:
    return DeepConvLSTM(
        n_channels=n_channels,
        n_classes=n_classes,
        window=window,
        n_conv=n_conv,
        conv_filters=conv_filters,
        kernel_size=kernel_size,
        rnn_type=rnn,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        model=model,
        dense_hidden=dense_hidden,
    )

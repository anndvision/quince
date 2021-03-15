from typing import List
import torch

from torch import nn

from quince.library.modules.spectral_norm import spectral_norm_conv


class Activation(nn.Module):
    def __init__(
        self,
        num_features,
        negative_slope,
        dropout_rate,
        batch_norm,
    ):
        super(Activation, self).__init__()
        self.op = nn.Sequential(
            nn.BatchNorm2d(num_features=num_features) if batch_norm else nn.Identity(),
            nn.LeakyReLU(negative_slope=negative_slope)
            if negative_slope >= 0.0
            else nn.ELU(),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, inputs):
        return self.op(inputs)


class PreactivationConv(nn.Module):
    def __init__(
        self,
        dim_input,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        negative_slope=1.0,
        dropout_rate=0.0,
        batch_norm=False,
        spectral_norm=0.0,
    ):
        super(PreactivationConv, self).__init__()
        self.op = nn.Sequential(
            Activation(
                num_features=dim_input[0],
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
            )
        )
        conv = nn.Conv2d(
            in_channels=dim_input[0],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.op.add_module(
            "conv",
            spectral_norm_conv(conv, spectral_norm, input_dim=dim_input)
            if spectral_norm > 0.0
            else conv,
        )

    def forward(self, inputs):
        return self.op(inputs)


class ResidualConv(nn.Module):
    def __init__(
        self,
        dim_input,
        out_channels,
        bias,
        negative_slope,
        dropout_rate,
        batch_norm,
        spectral_norm,
        stride,
    ):
        super(ResidualConv, self).__init__()
        if dim_input != out_channels:
            self.shortcut = nn.Sequential(nn.Dropout2d(p=dropout_rate))
            conv = nn.Conv2d(
                in_channels=dim_input[0],
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                dilation=1,
                groups=1,
                bias=bias,
            )
            self.shortcut.add_module(
                "linear",
                spectral_norm_conv(conv, spectral_norm, input_dim=dim_input)
                if spectral_norm > 0.0
                else conv,
            )
        else:
            self.shortcut = nn.Identity()

        self.op = nn.Sequential(
            PreactivationConv(
                dim_input=dim_input,
                out_channels=dim_input[0],
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=not batch_norm,
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
            ),
            PreactivationConv(
                dim_input=dim_input,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                dilation=1,
                groups=1,
                bias=not batch_norm,
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
            ),
        )

    def forward(self, inputs):
        return self.op(inputs) + self.shortcut(inputs)


class ResNet(nn.Module):
    def __init__(
        self,
        dim_input: int,
        layers: List[int],
        base_width: int = 64,
        negative_slope: float = 0.0,
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        spectral_norm: float = 0.0,
        stem_kernel_size: int = 7,
        stem_kernel_stride: int = 2,
        stem_kernel_padding: int = 3,
        stem_pool: bool = True,
        output_activation=False,
    ):
        super(ResNet, self).__init__()
        self.dim_input = dim_input
        stem_conv = nn.Conv2d(
            dim_input[0],
            base_width,
            kernel_size=stem_kernel_size,
            stride=stem_kernel_stride,
            padding=stem_kernel_padding,
            bias=not batch_norm,
        )
        self.op = nn.Sequential(
            spectral_norm_conv(stem_conv, spectral_norm, input_dim=dim_input)
            if spectral_norm > 0.0
            else stem_conv,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if stem_pool
            else nn.Identity(),
        )
        dx = dim_input[1] // 2 if stem_pool else dim_input[1]
        dy = dim_input[2] // 2 if stem_pool else dim_input[2]
        for i, block_depth in enumerate(layers):
            for j in range(block_depth - 1):
                self.op.add_module(
                    f"layer_{i + 1:02d}_block_{j + 1:02d}",
                    ResidualConv(
                        dim_input=[base_width * 2 ** i, dx, dy],
                        out_channels=base_width * 2 ** i,
                        bias=not batch_norm,
                        negative_slope=negative_slope,
                        dropout_rate=dropout_rate,
                        batch_norm=batch_norm,
                        spectral_norm=spectral_norm,
                        stride=1,
                    ),
                )
            self.op.add_module(
                f"layer_{i + 1:02d}_block_{block_depth:02d}",
                ResidualConv(
                    dim_input=[base_width * 2 ** i, dx, dy],
                    out_channels=base_width * 2 ** i
                    if i == len(layers) - 1
                    else base_width * 2 ** (i + 1),
                    bias=not batch_norm,
                    negative_slope=negative_slope,
                    dropout_rate=dropout_rate,
                    batch_norm=batch_norm,
                    spectral_norm=spectral_norm,
                    stride=1 if i == len(layers) - 1 else 2,
                ),
            )
            dx = round(1 + (dx + 2 - 3) / 2)
            dy = round(1 + (dy + 2 - 3) / 2)
        self.op.add_module("average_pooling", nn.AdaptiveAvgPool2d((1, 1)))
        self.dim_output = base_width * 2 ** (len(layers) - 1)
        if output_activation:
            self.op.add_module(
                "output_activation",
                Activation(
                    num_features=self.dim_output,
                    negative_slope=negative_slope,
                    dropout_rate=dropout_rate,
                    batch_norm=batch_norm,
                ),
            )

    def forward(self, inputs):
        shape = [-1] + self.dim_input
        return self.op(inputs.reshape(shape)).squeeze(-1).squeeze(-1)

from numpy.core.numeric import identity
import torch

from torch import nn

from quince.library.modules import dense
from quince.library.modules import convolution
from quince.library.modules import variational


class DragonNet(nn.Module):
    def __init__(
        self,
        architecture,
        dim_input,
        dim_hidden,
        dim_output,
        depth,
        negative_slope,
        batch_norm,
        dropout_rate,
        spectral_norm,
    ):
        super(DragonNet, self).__init__()
        self.dim_input = dim_input
        if isinstance(dim_input, list):
            self.encoder = convolution.ResNet(
                dim_input=dim_input,
                layers=[2] * depth,
                base_width=dim_hidden // 8,
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
                stem_kernel_size=5,
                stem_kernel_stride=1,
                stem_kernel_padding=2,
                stem_pool=False,
                activate_output=False,
            )
        elif dim_input == 1:
            self.encoder = nn.Identity()
            self.encoder.dim_output = 1
        else:
            self.encoder = dense.NeuralNetwork(
                architecture=architecture,
                dim_input=dim_input,
                dim_hidden=dim_hidden,
                depth=depth,
                negative_slope=negative_slope,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate,
                spectral_norm=spectral_norm,
                activate_output=False,
            )
        dim_hidden = dim_hidden if self.encoder.dim_output == 1 else dim_hidden // 2
        self.t0_encoder = (
            nn.Sequential(
                dense.ResidualDense(
                    dim_input=self.encoder.dim_output,
                    dim_output=dim_hidden,
                    bias=not batch_norm,
                    negative_slope=1.0
                    if self.encoder.dim_output == 1
                    else negative_slope,
                    dropout_rate=0.0 if self.encoder.dim_output == 1 else dropout_rate,
                    batch_norm=False if self.encoder.dim_output == 1 else batch_norm,
                    spectral_norm=spectral_norm,
                ),
                dense.ResidualDense(
                    dim_input=dim_hidden,
                    dim_output=dim_hidden,
                    bias=not batch_norm,
                    negative_slope=negative_slope,
                    dropout_rate=dropout_rate,
                    batch_norm=batch_norm,
                    spectral_norm=spectral_norm,
                ),
                dense.Activation(
                    dim_input=dim_hidden,
                    negative_slope=negative_slope,
                    dropout_rate=dropout_rate,
                    batch_norm=batch_norm,
                ),
            )
            if self.encoder.dim_output != 1
            else nn.Identity()
        )
        self.t1_encoder = (
            nn.Sequential(
                dense.ResidualDense(
                    dim_input=self.encoder.dim_output,
                    dim_output=dim_hidden,
                    bias=not batch_norm,
                    negative_slope=1.0
                    if self.encoder.dim_output == 1
                    else negative_slope,
                    dropout_rate=0.0 if self.encoder.dim_output == 1 else dropout_rate,
                    batch_norm=False if self.encoder.dim_output == 1 else batch_norm,
                    spectral_norm=spectral_norm,
                ),
                dense.ResidualDense(
                    dim_input=dim_hidden,
                    dim_output=dim_hidden,
                    bias=not batch_norm,
                    negative_slope=negative_slope,
                    dropout_rate=dropout_rate,
                    batch_norm=batch_norm,
                    spectral_norm=spectral_norm,
                ),
                dense.Activation(
                    dim_input=dim_hidden,
                    negative_slope=negative_slope,
                    dropout_rate=dropout_rate,
                    batch_norm=batch_norm,
                ),
            )
            if self.encoder.dim_output != 1
            else nn.Identity()
        )
        self.dim_output = dim_hidden
        self.outcome_density = (
            variational.SplitGMM(
                dim_input=dim_hidden,
                dim_output=dim_output,
            )
            if self.encoder.dim_output != 1
            else nn.Sequential(
                dense.NeuralNetwork(
                    architecture=architecture,
                    dim_input=dim_input + 1,
                    dim_hidden=dim_hidden,
                    depth=depth,
                    negative_slope=negative_slope,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate,
                    spectral_norm=spectral_norm,
                    activate_output=True,
                ),
                variational.GMM(
                    dim_input=dim_hidden,
                    dim_output=dim_output,
                ),
            )
        )
        self.pi_density = nn.Sequential(
            dense.Activation(
                dim_input=self.encoder.dim_output,
                negative_slope=1.0 if self.encoder.dim_output == 1 else negative_slope,
                dropout_rate=0.0 if self.encoder.dim_output == 1 else dropout_rate,
                batch_norm=False if self.encoder.dim_output == 1 else batch_norm,
            ),
            variational.Categorical(
                dim_input=self.encoder.dim_output,
                dim_output=1,
            ),
        )

    def forward(self, inputs):
        phi = self.encoder(inputs[:, :-1])
        pi_density = self.pi_density(phi)
        t = inputs[:, -1:]
        phi = (1 - t) * self.t0_encoder(phi) + t * self.t1_encoder(phi)
        return self.outcome_density(torch.cat([phi, t], dim=-1)), pi_density

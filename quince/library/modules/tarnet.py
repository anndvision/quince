import torch

from torch import nn

from quince.library.modules import dense
from quince.library.modules import convolution


class TARNet(nn.Module):
    def __init__(
        self,
        architecture,
        dim_input,
        dim_hidden,
        dim_treatment,
        depth,
        negative_slope,
        batch_norm,
        dropout_rate,
        spectral_norm,
    ):
        super(TARNet, self).__init__()
        self.dim_input = dim_input
        self.encoder = (
            convolution.ResNet(
                dim_input=dim_input,
                layers=[2, 2, 2, 2],
                base_width=dim_hidden // 8,
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
                stem_kernel_size=5,
                stem_kernel_stride=1,
                stem_kernel_padding=2,
                stem_pool=False,
            )
            if isinstance(dim_input, list)
            else dense.NeuralNetwork(
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
        )
        self.t_encoder = nn.Sequential(
            dense.ResidualDense(
                dim_input=self.encoder.dim_output + dim_treatment,
                dim_output=dim_hidden,
                bias=not batch_norm,
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
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
        self.dim_output = dim_hidden

    def forward(self, inputs):
        phi = self.encoder(inputs[:, :-1])
        t_inputs = torch.cat([phi, inputs[:, -1:]], dim=-1)
        return self.t_encoder(t_inputs)

"""
Spectral Normalization from https://arxiv.org/abs/1802.05957
with additional variable `coeff` or max spectral norm.
From: https://github.com/jhjacobsen/invertible-resnet
Which is based on: https://arxiv.org/abs/1811.00995
Soft Spectral Normalization (not enforced, only <= coeff) for Conv2D layers
Based on: Regularisation of Neural Networks by Enforcing Lipschitz Continuity
    (Gouk et al. 2018)
    https://arxiv.org/abs/1804.04368
"""
import torch
from torch import nn
from torch.nn.functional import normalize, conv2d, conv_transpose2d
from torch.nn.utils.spectral_norm import (
    SpectralNorm,
    SpectralNormLoadStateDictPreHook,
    SpectralNormStateDictHook,
)


class SpectralNormFC(SpectralNorm):
    def compute_weight(self, module, do_power_iteration: bool) -> torch.Tensor:
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(
                        torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
                    )
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor

        # for logging
        sigma_log = getattr(module, self.name + "_sigma")
        sigma_log.copy_(sigma.detach())

        return weight

    @staticmethod
    def apply(
        module: nn.Module,
        coeff: float,
        name: str,
        n_power_iterations: int,
        dim: int,
        eps: float,
    ) -> "SpectralNormFC":
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = SpectralNormFC(name, n_power_iterations, dim, eps)
        fn.coeff = coeff

        weight = module._parameters[name]
        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1))

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


def spectral_norm_fc(
    module,
    coeff: float,
    n_power_iterations: int = 1,
    name: str = "weight",
    eps: float = 1e-12,
    dim: int = None,
):
    """
    Args:
        module (nn.Module): containing module
        coeff (float, optional): coefficient to normalize to
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        name (str, optional): name of weight parameter
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``
    Returns:
        The original module with the spectral norm hook
    Example::
        >>> m = spectral_norm_fc(nn.Linear(20, 40), 2.0)
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])
    """
    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    SpectralNormFC.apply(module, coeff, name, n_power_iterations, dim, eps)
    return module


class SpectralNormConv(SpectralNorm):
    def compute_weight(self, module, do_power_iteration: bool) -> torch.Tensor:
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")

        # get settings from conv-module (for transposed convolution parameters)
        stride = module.stride
        padding = module.padding

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v_s = conv_transpose2d(
                        u.view(self.out_shape),
                        weight,
                        stride=stride,
                        padding=padding,
                        output_padding=(stride[0] - 1, stride[1] - 1),
                    )
                    v = normalize(v_s.view(-1), dim=0, eps=self.eps, out=v)

                    u_s = conv2d(
                        v.view(self.input_dim),
                        weight,
                        stride=stride,
                        padding=padding,
                        bias=None,
                    )
                    u = normalize(u_s.view(-1), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        weight_v = conv2d(
            v.view(self.input_dim), weight, stride=stride, padding=padding, bias=None
        )
        weight_v = weight_v.view(-1)
        sigma = torch.dot(u.view(-1), weight_v)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1, device=weight.device), sigma / self.coeff)
        weight = weight / factor

        # for logging
        sigma_log = getattr(module, self.name + "_sigma")
        sigma_log.copy_(sigma.detach())

        return weight

    @staticmethod
    def apply(module, coeff, input_dim, name, n_power_iterations, eps):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNormConv) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = SpectralNormConv(name, n_power_iterations, eps=eps)
        fn.coeff = coeff
        fn.input_dim = input_dim
        weight = module._parameters[name]

        with torch.no_grad():
            num_input_dim = input_dim[0] * input_dim[1] * input_dim[2] * input_dim[3]
            v = normalize(torch.randn(num_input_dim), dim=0, eps=fn.eps)

            # get settings from conv-module (for transposed convolution)
            stride = module.stride
            padding = module.padding
            # forward call to infer the shape
            u = conv2d(
                v.view(input_dim), weight, stride=stride, padding=padding, bias=None
            )
            fn.out_shape = u.shape
            num_output_dim = (
                fn.out_shape[0] * fn.out_shape[1] * fn.out_shape[2] * fn.out_shape[3]
            )
            # overwrite u with random init
            u = normalize(torch.randn(num_output_dim), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1))

        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


def spectral_norm_conv(
    module,
    coeff,
    input_dim,
    n_power_iterations=1,
    name="weight",
    eps=1e-12,
):
    """
    Applies spectral normalization to Convolutions with flexible max norm
    Args:
        module (nn.Module): containing convolution module
        input_dim (tuple(int, int, int)): dimension of input to convolution
        coeff (float, optional): coefficient to normalize to
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        name (str, optional): name of weight parameter
        eps (float, optional): epsilon for numerical stability in
            calculating norms
    Returns:
        The original module with the spectral norm hook
    Example::
        >>> m = spectral_norm_conv(nn.Conv2D(3, 16, 3), (3, 32, 32), 2.0)
    """

    input_dim_4d = (1, input_dim[0], input_dim[1], input_dim[2])
    SpectralNormConv.apply(module, coeff, input_dim_4d, name, n_power_iterations, eps)

    return module

import torch

from typing import Tuple

from ignite.exceptions import NotComputableError
from ignite.contrib.metrics.regression import _base
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class NegR2Score(_base._BaseRegression):
    def __init__(self, dim_output, output_transform, device):
        self.dim_output = dim_output
        super(NegR2Score, self).__init__(
            output_transform=output_transform, device=device
        )

    @reinit__is_reduced
    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        _base._check_output_types(output)
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.ndimension() == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=-1)

        if y.ndimension() == 2 and y.shape[1] == 1:
            y = y.squeeze(dim=-1)

        self._update((y_pred, y))

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_examples = 0
        self._sum_of_errors = torch.tensor(
            [0.0] * int(self.dim_output), device=self._device
        )
        self._y_sq_sum = torch.tensor([0.0] * int(self.dim_output), device=self._device)
        self._y_sum = torch.tensor([0.0] * int(self.dim_output), device=self._device)

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output
        self._num_examples += y.shape[0]
        self._sum_of_errors += torch.sum(torch.pow(y_pred - y, 2), dim=0).to(
            self._device
        )

        self._y_sum += torch.sum(y, dim=0).to(self._device)
        self._y_sq_sum += torch.sum(torch.pow(y, 2), dim=0).to(self._device)

    @sync_all_reduce("_num_examples", "_sum_of_errors", "_y_sq_sum", "_y_sum")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError(
                "R2Score must have at least one example before it can be computed."
            )
        r2 = 1 - self._sum_of_errors / (
            self._y_sq_sum - (self._y_sum ** 2) / self._num_examples
        )
        return -r2.mean().item()

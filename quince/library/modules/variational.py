import torch

from torch import nn
from torch import distributions


class Normal(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output,
    ):
        super(Normal, self).__init__()
        self.mu = nn.Linear(
            in_features=dim_input,
            out_features=dim_output,
            bias=True,
        )
        sigma = nn.Linear(
            in_features=dim_input,
            out_features=dim_output,
            bias=True,
        )
        self.sigma = nn.Sequential(sigma, nn.Softplus())

    def forward(self, inputs):
        return distributions.Normal(loc=self.mu(inputs), scale=self.sigma(inputs))


class MixtureSameFamily(distributions.MixtureSameFamily):
    def log_prob(self, inputs):
        loss = torch.exp(self.component_distribution.log_prob(inputs))
        loss = torch.sum(loss * self.mixture_distribution.probs, dim=1)
        return torch.log(loss + 1e-7)


class GMM(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output,
    ):
        super(GMM, self).__init__()
        self.mu = nn.Linear(
            in_features=dim_input,
            out_features=dim_output,
            bias=True,
        )
        sigma = nn.Linear(
            in_features=dim_input,
            out_features=dim_output,
            bias=True,
        )
        self.pi = nn.Linear(
            in_features=dim_input,
            out_features=dim_output,
            bias=True,
        )
        self.sigma = nn.Sequential(sigma, nn.Softplus())

    def forward(self, inputs):
        return MixtureSameFamily(
            mixture_distribution=distributions.Categorical(logits=self.pi(inputs)),
            component_distribution=distributions.Normal(
                loc=self.mu(inputs), scale=self.sigma(inputs)
            ),
        )


class Categorical(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output,
    ):
        super(Categorical, self).__init__()
        self.logits = nn.Linear(
            in_features=dim_input,
            out_features=dim_output,
            bias=True,
        )
        self.distribution = (
            distributions.Bernoulli if dim_output == 1 else distributions.Categorical
        )

    def forward(self, inputs):
        return self.distribution(logits=self.logits(inputs))

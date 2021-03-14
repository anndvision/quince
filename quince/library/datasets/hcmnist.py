import torch
import typing
import numpy as np

from torchvision import datasets

from sklearn import model_selection

from quince.library.datasets import utils


class HCMNIST(datasets.MNIST):
    def __init__(
        self,
        root: str,
        lambda_star: float,
        split: str = "train",
        mode: str = "mu",
        p_u: str = "bernoulli",
        gamma: float = 4.0,
        beta: float = 0.75,
        sigma_y: float = 1.0,
        domain: float = 2.0,
        seed: int = 1331,
        transform: typing.Optional[typing.Callable] = None,
        target_transform: typing.Optional[typing.Callable] = None,
        download: bool = False,
    ) -> None:
        train = split == "train" or split == "valid"
        super(HCMNIST, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.data = self.data.view(len(self.targets), -1).numpy()
        self.targets = self.targets.numpy()

        if train:
            (
                data_train,
                data_valid,
                targets_train,
                targets_valid,
            ) = model_selection.train_test_split(
                self.data, self.targets, test_size=0.3, random_state=seed
            )
            self.data = data_train if split == "train" else data_valid
            self.targets = targets_train if split == "train" else targets_valid

        self.mode = mode
        self.dim_input = [1, 28, 28]
        self.dim_treatment = 1
        self.dim_output = 1

        self.phi_model = fit_phi_model(
            root=root,
            edges=torch.arange(-domain, domain + 0.1, (2 * domain) / 10),
        )

        size = (self.__len__(), 1)
        rng = np.random.RandomState(seed=seed)
        if p_u == "bernoulli":
            self.u = rng.binomial(1, 0.5, size=size).astype("float32")
        elif p_u == "uniform":
            self.u = rng.uniform(size=size).astype("float32")
        elif p_u == "beta_bi":
            self.u = rng.beta(0.5, 0.5, size=size).astype("float32")
        elif p_u == "beta_uni":
            self.u = rng.beta(2, 5, size=size).astype("float32")
        else:
            raise NotImplementedError(f"{p_u} is not a supported distribution")

        phi = self.phi
        self.pi = (
            utils.complete_propensity(x=phi, u=self.u, lambda_=lambda_star, beta=beta)
            .astype("float32")
            .ravel()
        )
        self.t = rng.binomial(1, self.pi).astype("float32")
        eps = (sigma_y * rng.normal(size=self.t.shape)).astype("float32")
        self.mu0 = (
            utils.f_mu(x=phi, t=0.0, u=self.u, gamma=gamma).astype("float32").ravel()
        )
        self.mu1 = (
            utils.f_mu(x=phi, t=1.0, u=self.u, gamma=gamma).astype("float32").ravel()
        )
        self.y0 = self.mu0 + eps
        self.y1 = self.mu1 + eps
        self.y = self.t * self.y1 + (1 - self.t) * self.y0
        self.tau = self.mu1 - self.mu0
        self.y_mean = np.array([0.0], dtype="float32")
        self.y_std = np.array([1.0], dtype="float32")

    def __getitem__(self, index):
        x = ((self.data[index].astype("float32") / 255.0) - 0.1307) / 0.3081
        t = self.t[index : index + 1]
        if self.mode == "pi":
            return x, t
        elif self.mode == "mu":
            return np.hstack([x, t]), self.y[index : index + 1]
        else:
            raise NotImplementedError(
                f"{self.mode} not supported. Choose from 'pi'  for propensity models or 'mu' for expected outcome models"
            )

    @property
    def phi(self):
        x = ((self.data.astype("float32") / 255.0) - 0.1307) / 0.3081
        z = np.zeros_like(self.targets.astype("float32"))
        for k, v in self.phi_model.items():
            ind = self.targets == k
            x_ind = x[ind].reshape(ind.sum(), -1)
            means = x_ind.mean(axis=-1)
            z[ind] = utils.linear_normalization(
                np.clip((means - v["mu"]) / v["sigma"], -1.4, 1.4), v["lo"], v["hi"]
            )
        return np.expand_dims(z, -1)

    @property
    def x(self):
        return ((self.data.astype("float32") / 255.0) - 0.1307) / 0.3081


def fit_phi_model(root, edges):
    ds = datasets.MNIST(root=root)
    data = (ds.data.float().div(255) - 0.1307).div(0.3081).view(len(ds), -1)
    model = {}
    digits = torch.unique(ds.targets)
    for i, digit in enumerate(digits):
        lo, hi = edges[i : i + 2]
        ind = ds.targets == digit
        data_ind = data[ind].view(ind.sum(), -1)
        means = data_ind.mean(dim=-1)
        mu = means.mean()
        sigma = means.std()
        model.update(
            {
                digit.item(): {
                    "mu": mu.item(),
                    "sigma": sigma.item(),
                    "lo": lo.item(),
                    "hi": hi.item(),
                }
            }
        )
    return model

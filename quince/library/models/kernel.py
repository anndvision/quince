import torch
import numpy as np

from sklearn.gaussian_process import kernels
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_kernels

from quince.library import utils


class KernelRegressor(object):
    def __init__(
        self,
        dataset,
        initial_length_scale=1.0,
        feature_extractor=None,
        propensity_model=None,
        verbose=False,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.device = propensity_model.device if propensity_model is not None else None
        self.kernel = kernels.RBF(length_scale=initial_length_scale)
        idx = np.argsort(dataset.y.ravel())
        self.x = dataset.x[idx]
        self.t = dataset.t[idx].reshape(-1, 1)
        self.y = dataset.y[idx].reshape(-1, 1)
        self.s = self.y.std()
        self.m = self.y.mean()

        if propensity_model is None:
            propensity_model = LogisticRegression()
            propensity_model = propensity_model.fit(self.x, self.t.ravel())
            self.e = propensity_model.predict_proba(self.x)[:, -1:]
        else:
            with torch.no_grad():
                e = []
                for _ in range(50):
                    e.append(
                        propensity_model.network(
                            torch.tensor(np.hstack([self.x, self.t])).to(
                                propensity_model.device
                            )
                        )[1].probs.to("cpu")
                    )
                self.e = torch.cat(e, dim=-1).mean(1, keepdim=True).numpy()
        self.e = np.clip(self.e, 1e-7, 1 - 1e-7)

        self._gamma = None

        self.alpha_0 = None
        self.alpha_1 = None

        self.beta_0 = None
        self.beta_1 = None

        self.verbose = verbose

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

        self.alpha_0 = utils.alpha_fn(pi=1 - self.e, lambda_=value)
        self.alpha_1 = utils.alpha_fn(pi=self.e, lambda_=value)

        self.beta_0 = utils.beta_fn(pi=1 - self.e, lambda_=value)
        self.beta_1 = utils.beta_fn(pi=self.e, lambda_=value)

    def k(self, x):
        return pairwise_kernels(
            self.embed(x), self.embed(self.x), metric=self.kernel, filter_params=True
        )

    def mu0_w(self, w, k):
        return np.matmul(k, (1 - self.t) * self.y * w) / (
            np.matmul(k, (1 - self.t) * w) + 1e-7
        )

    def mu1_w(self, w, k):
        return np.matmul(k, self.t * self.y * w) / (np.matmul(k, self.t * w) + 1e-7)

    def lambda_top_0(self, u, k):
        t = 1 - self.t
        alpha = np.matmul(k[:, :u], t[:u] * self.alpha_0[:u])
        beta = np.matmul(k[:, u:], t[u:] * self.beta_0[u:])
        alpha_y = np.matmul(k[:, :u], t[:u] * self.alpha_0[:u] * self.y[:u])
        beta_y = np.matmul(k[:, u:], t[u:] * self.beta_0[u:] * self.y[u:])
        return (alpha_y + beta_y) / (alpha + beta)

    def lambda_top_1(self, u, k):
        t = self.t
        alpha = np.matmul(k[:, :u], t[:u] * self.alpha_1[:u])
        beta = np.matmul(k[:, u:], t[u:] * self.beta_1[u:])
        alpha_y = np.matmul(k[:, :u], t[:u] * self.alpha_1[:u] * self.y[:u])
        beta_y = np.matmul(k[:, u:], t[u:] * self.beta_1[u:] * self.y[u:])
        return (alpha_y + beta_y) / (alpha + beta)

    def lambda_bottom_0(self, u, k):
        t = 1 - self.t
        alpha = np.matmul(k[:, u:], t[u:] * self.alpha_0[u:])
        beta = np.matmul(k[:, :u], t[:u] * self.beta_0[:u])
        alpha_y = np.matmul(k[:, u:], t[u:] * self.alpha_0[u:] * self.y[u:])
        beta_y = np.matmul(k[:, :u], t[:u] * self.beta_0[:u] * self.y[:u])
        return (alpha_y + beta_y) / (alpha + beta)

    def lambda_bottom_1(self, u, k):
        t = self.t
        alpha = np.matmul(k[:, u:], t[u:] * self.alpha_1[u:])
        beta = np.matmul(k[:, :u], t[:u] * self.beta_1[:u])
        alpha_y = np.matmul(k[:, u:], t[u:] * self.alpha_1[u:] * self.y[u:])
        beta_y = np.matmul(k[:, :u], t[:u] * self.beta_1[:u] * self.y[:u])
        return (alpha_y + beta_y) / (alpha + beta)

    def mu0(self, k):
        return self.mu0_w(w=(1 - self.e) ** -1, k=k)

    def mu1(self, k):
        return self.mu1_w(w=self.e ** -1, k=k)

    def tau(self, k):
        return self.mu1(k) - self.mu0(k)

    def fit_length_scale(self, dataset, grid):
        best_err = np.inf
        best_h = None
        count = 0
        for h in grid:
            kernel = kernels.RBF(length_scale=h)
            k = pairwise_kernels(
                self.embed(dataset.x),
                self.embed(self.x),
                metric=kernel,
                filter_params=False,
            )
            mu0 = self.mu0(k)
            mu1 = self.mu1(k)
            y = dataset.y.reshape(-1, 1)
            t = dataset.t.reshape(-1, 1)
            err0 = mean_squared_error(y[t == 0], mu0[t == 0])
            err1 = mean_squared_error(y[t == 1], mu1[t == 1])
            err = err0 + err1
            if err < best_err:
                best_err = err
                best_h = h
                count = 0
            elif count < 20:
                count += 1
            else:
                break
            if self.verbose:
                print(f"h-{h:.03f}_err-{err:.03f}")
        self.kernel.length_scale = best_h

    def embed(self, x):
        if self.feature_extractor is None:
            return x
        else:
            with torch.no_grad():
                phi = []
                for i in range(50):
                    phi.append(
                        self.feature_extractor(torch.tensor(x).to(self.device))
                        .to("cpu")
                        .unsqueeze(0)
                    )
                phi = torch.cat(phi).mean(0).numpy()
            return phi

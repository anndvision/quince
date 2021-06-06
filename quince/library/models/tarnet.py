import torch

from ignite import metrics

from torch import optim
from torch.utils import data

from quince.library.models import core
from quince.library.modules import tarnet


class TARNet(core.PyTorchModel):
    def __init__(
        self,
        job_dir,
        architecture,
        dim_input,
        dim_hidden,
        dim_output,
        depth,
        negative_slope,
        batch_norm,
        spectral_norm,
        dropout_rate,
        weight_decay,
        learning_rate,
        batch_size,
        epochs,
        patience,
        num_workers,
        seed,
    ):
        super(TARNet, self).__init__(
            job_dir=job_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            num_workers=num_workers,
        )
        self.network = tarnet.TARNet(
            architecture=architecture,
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            dim_output=dim_output,
            depth=depth,
            negative_slope=negative_slope,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            spectral_norm=spectral_norm,
        )
        self.metrics = {
            "loss": metrics.Average(
                output_transform=lambda x: -x["outputs"].log_prob(x["targets"]).mean(),
                device=self.device,
            )
        }
        self.batch_size = batch_size
        self.best_loss = 1e7
        self.patience = patience
        self.optimizer = optim.Adam(
            params=self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )
        self.network.to(self.device)

    def train_step(self, engine, batch):
        self.network.train()
        inputs, targets = self.preprocess(batch)
        self.optimizer.zero_grad()
        outputs = self.network(inputs)
        loss = -outputs.log_prob(targets).mean()
        loss.backward()
        self.optimizer.step()
        metric_values = {"outputs": outputs, "targets": targets}
        return metric_values

    def tune_step(self, engine, batch):
        self.network.eval()
        inputs, targets = self.preprocess(batch)
        with torch.no_grad():
            outputs = self.network(inputs)
        metric_values = {"outputs": outputs, "targets": targets}
        return metric_values

    def predict_mus(self, ds, batch_size=None):
        dl = data.DataLoader(
            ds,
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        mu_0 = []
        mu_1 = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                batch = self.preprocess(batch)
                covariates = torch.cat([batch[0][:, :-1], batch[0][:, :-1]], 0)
                treatments = torch.cat(
                    [
                        torch.zeros_like(batch[0][:, -1:]),
                        torch.ones_like(batch[0][:, -1:]),
                    ],
                    0,
                )
                inputs = torch.cat([covariates, treatments], -1)
                posterior_predictive = self.network(inputs)
                mu = posterior_predictive.mean
                mus = torch.split(mu, mu.shape[0] // 2, dim=0)
                mu_0.append(mus[0])
                mu_1.append(mus[1])
        return (
            torch.cat(mu_0, 0).to("cpu").numpy(),
            torch.cat(mu_1, 0).to("cpu").numpy(),
        )

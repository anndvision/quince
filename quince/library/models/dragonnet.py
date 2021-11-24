import torch

from torch import optim

from ignite import metrics

from quince.library.models import core
from quince.library.modules import dragonnet
from quince.library.metrics import regression


class DragonNet(core.PyTorchModel):
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
        num_examples,
        learning_rate,
        batch_size,
        epochs,
        patience,
        num_workers,
        seed,
    ):
        super(DragonNet, self).__init__(
            job_dir=job_dir,
            num_examples=num_examples,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            num_workers=num_workers,
        )
        self.network = dragonnet.DragonNet(
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
            "loss": regression.NegR2Score(
                dim_output=dim_output,
                output_transform=lambda x: (x["outputs"][0].mean, x["targets"],),
                device=self.device,
            ),
            "loss_y": metrics.Average(
                output_transform=lambda x: -x["outputs"][0]
                .log_prob(x["targets"])
                .mean(),
                device=self.device,
            ),
            "loss_t": metrics.Average(
                output_transform=lambda x: -x["outputs"][1]
                .log_prob(x["treatments"])
                .mean(),
                device=self.device,
            ),
        }
        self.batch_size = batch_size
        self.best_loss = 1e7
        self.patience = patience
        self.optimizer = optim.Adam(
            params=self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=(0.5 * (1 - dropout_rate)) / num_examples,
        )
        self.network.to(self.device)

    def train_step(self, engine, batch):
        self.network.train()
        inputs, targets = self.preprocess(batch)
        treatments = inputs[:, -1:]
        self.optimizer.zero_grad()
        outputs = self.network(inputs)
        loss = -outputs[0].log_prob(targets).mean()
        loss -= outputs[1].log_prob(treatments).mean()
        loss.backward()
        self.optimizer.step()
        metric_values = {
            "outputs": outputs,
            "targets": targets,
            "treatments": treatments,
        }
        return metric_values

    def tune_step(self, engine, batch):
        self.network.eval()
        inputs, targets = self.preprocess(batch)
        treatments = inputs[:, -1:]
        with torch.no_grad():
            outputs = self.network(inputs)
        metric_values = {
            "outputs": outputs,
            "targets": targets,
            "treatments": treatments,
        }
        return metric_values

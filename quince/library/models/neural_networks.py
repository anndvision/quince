from ray import tune

from torch import nn
from torch import optim

from ignite import metrics

from quince.library.models import core

from quince.library.modules import dense
from quince.library.modules import tarnet
from quince.library.modules import convolution
from quince.library.modules import variational


class _NeuralNetwork(core.PyTorchModel):
    def __init__(
        self,
        job_dir,
        architecture,
        dim_input,
        dim_treatment,
        dim_hidden,
        depth,
        negative_slope,
        batch_norm,
        spectral_norm,
        dropout_rate,
        learning_rate,
        batch_size,
        epochs,
        patience,
        num_workers,
        seed,
    ):
        super(_NeuralNetwork, self).__init__(
            job_dir=job_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            num_workers=num_workers,
        )
        if dim_treatment > 0:
            self.encoder = (
                tarnet.TARNet(
                    architecture=architecture,
                    dim_input=dim_input,
                    dim_hidden=dim_hidden,
                    dim_treatment=dim_treatment,
                    depth=depth,
                    negative_slope=negative_slope,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate,
                    spectral_norm=spectral_norm,
                )
                if isinstance(dim_input, list) or (dim_input > 1)
                else dense.NeuralNetwork(
                    architecture=architecture,
                    dim_input=dim_input + dim_treatment,
                    dim_hidden=dim_hidden,
                    depth=depth,
                    negative_slope=negative_slope,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate,
                    spectral_norm=spectral_norm,
                )
            )
        else:
            self.encoder = (
                convolution.ResNet(
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
                    output_activation=True,
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
                )
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
        self.counter = 0

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
        outputs = self.network(inputs)
        metric_values = {"outputs": outputs, "targets": targets}
        return metric_values

    def on_epoch_completed(self, engine, train_loader, tune_loader):
        train_metrics = self.trainer.state.metrics
        print("Metrics Epoch", engine.state.epoch)
        justify = max(len(k) for k in train_metrics) + 2
        for k, v in train_metrics.items():
            if type(v) == float:
                print("train {:<{justify}} {:<5f}".format(k, v, justify=justify))
                continue
            print("train {:<{justify}} {:<5}".format(k, v, justify=justify))
        self.evaluator.run(tune_loader)
        tune_metrics = self.evaluator.state.metrics
        if tune.is_session_enabled():
            tune.report(mean_loss=tune_metrics["loss"])
        justify = max(len(k) for k in tune_metrics) + 2
        for k, v in tune_metrics.items():
            if type(v) == float:
                print("tune {:<{justify}} {:<5f}".format(k, v, justify=justify))
                continue
            print("tune {:<{justify}} {:<5}".format(k, v, justify=justify))
        is_best = tune_metrics["loss"] < self.best_loss
        self.best_loss = tune_metrics["loss"] if is_best else self.best_loss
        self.counter = 0 if is_best else self.counter + 1
        self.save(is_best=is_best)
        if self.counter == self.patience:
            self.logger.info(
                "Early Stopping: No improvement for {} epochs".format(self.patience)
            )
            engine.terminate()

    def on_training_completed(self, engine, loader):
        self.load(load_best=True)
        self.evaluator.run(loader)
        metric_values = self.evaluator.state.metrics
        print("Metrics Epoch", engine.state.epoch)
        justify = max(len(k) for k in metric_values) + 2
        for k, v in metric_values.items():
            if type(v) == float:
                print("best {:<{justify}} {:<5f}".format(k, v, justify=justify))
                continue
            print("best {:<{justify}} {:<5}".format(k, v, justify=justify))


class CategoricalDensityNetwork(_NeuralNetwork):
    def __init__(
        self,
        job_dir,
        architecture,
        dim_input,
        dim_treatment,
        dim_output,
        dim_hidden,
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
        super(CategoricalDensityNetwork, self).__init__(
            job_dir=job_dir,
            architecture=architecture,
            dim_input=dim_input,
            dim_treatment=dim_treatment,
            dim_hidden=dim_hidden,
            depth=depth,
            negative_slope=negative_slope,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            seed=seed,
        )
        self.network = nn.Sequential(
            self.encoder,
            variational.Categorical(
                dim_input=self.encoder.dim_output,
                dim_output=dim_output,
            ),
        )
        self.optimizer = optim.Adam(
            params=self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )
        self.network.to(self.device)


class GaussainDensityNetwork(_NeuralNetwork):
    def __init__(
        self,
        job_dir,
        architecture,
        dim_input,
        dim_treatment,
        dim_output,
        dim_hidden,
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
        super(GaussainDensityNetwork, self).__init__(
            job_dir=job_dir,
            architecture=architecture,
            dim_input=dim_input,
            dim_treatment=dim_treatment,
            dim_hidden=dim_hidden,
            depth=depth,
            negative_slope=negative_slope,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            seed=seed,
        )
        self.network = nn.Sequential(
            self.encoder,
            variational.Normal(
                dim_input=self.encoder.dim_output,
                dim_output=dim_output,
            ),
        )
        self.optimizer = optim.Adam(
            params=self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )
        self.network.to(self.device)


class GaussianMixtureDensityNetwork(_NeuralNetwork):
    def __init__(
        self,
        job_dir,
        architecture,
        dim_input,
        dim_treatment,
        dim_output,
        dim_hidden,
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
        super(GaussianMixtureDensityNetwork, self).__init__(
            job_dir=job_dir,
            architecture=architecture,
            dim_input=dim_input,
            dim_treatment=dim_treatment,
            dim_hidden=dim_hidden,
            depth=depth,
            negative_slope=negative_slope,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            seed=seed,
        )
        self.network = nn.Sequential(
            self.encoder,
            variational.GMM(
                dim_input=self.encoder.dim_output,
                dim_output=dim_output,
            ),
        )
        self.optimizer = optim.Adam(
            params=self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )
        self.network.to(self.device)

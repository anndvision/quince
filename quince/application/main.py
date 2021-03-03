import ray
import json
import click

from torch import cuda

from pathlib import Path

from quince.library import datasets
from quince.library import models


@click.group(chain=True)
@click.pass_context
def cli(context):
    context.obj = {"n_gpu": cuda.device_count()}


@cli.command("train")
@click.option(
    "--job-dir",
    type=str,
    required=True,
    help="location for writing checkpoints and results",
)
@click.option("--num-trials", default=1, type=int, help="number of trials, default=1")
@click.option(
    "--gpu-per-trial",
    default=0.0,
    type=float,
    help="number of gpus for each trial, default=0",
)
@click.option(
    "--cpu-per-trial",
    default=1.0,
    type=float,
    help="number of cpus for each trial, default=1",
)
@click.option("--verbose", default=False, type=bool, help="verbosity default=False")
@click.option(
    "--seed",
    default=1331,
    type=int,
    help="random number generator seed, default=1331",
)
@click.pass_context
def train(
    context,
    job_dir,
    num_trials,
    gpu_per_trial,
    cpu_per_trial,
    verbose,
    seed,
):
    ray.init(
        num_gpus=context.obj["n_gpu"],
        dashboard_host="127.0.0.1",
        ignore_reinit_error=True,
    )
    gpu_per_trial = 0 if context.obj["n_gpu"] == 0 else gpu_per_trial
    context.obj.update(
        {
            "job_dir": job_dir,
            "num_trials": num_trials,
            "gpu_per_trial": gpu_per_trial,
            "cpu_per_trial": cpu_per_trial,
            "verbose": verbose,
            "seed": seed,
        }
    )


@cli.command("ihdp")
@click.pass_context
@click.option(
    "--root",
    type=str,
    required=True,
    help="location of dataset",
)
@click.option(
    "--hidden-confounding",
    default=True,
    type=bool,
    help="Censor hidden confounder, default=True",
)
def ihdp(
    context,
    root,
    hidden_confounding,
):
    job_dir = Path(context.obj.get("job_dir"))
    dataset_name = "ihdp"
    experiment_dir = job_dir / dataset_name / f"hc-{hidden_confounding}"
    context.obj.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "ds_train": {
                "root": root,
                "split": "train",
                "mode": "mu",
                "hidden_confounding": hidden_confounding,
                "seed": context.obj.get("seed"),
            },
            "ds_valid": {
                "root": root,
                "split": "valid",
                "mode": "mu",
                "hidden_confounding": hidden_confounding,
                "seed": context.obj.get("seed"),
            },
            "ds_test": {
                "root": root,
                "split": "test",
                "mode": "mu",
                "hidden_confounding": hidden_confounding,
                "seed": context.obj.get("seed"),
            },
        }
    )


@cli.command("density-network")
@click.pass_context
@click.option("--dim-hidden", default=400, type=int, help="num neurons")
@click.option("--num-components", default=5, type=int, help="num mixture components")
@click.option("--depth", default=3, type=int, help="depth of feature extractor")
@click.option(
    "--negative-slope",
    default=0.1,
    type=float,
    help="negative slope of leaky relu, default=-1 use elu",
)
@click.option(
    "--dropout-rate", default=0.1, type=float, help="dropout rate, default=0.1"
)
@click.option(
    "--spectral-norm",
    default=0.0,
    type=float,
    help="Spectral normalization coefficient. If 0.0 do not use spectral norm, default=0.0",
)
@click.option(
    "--learning-rate",
    default=1e-3,
    type=float,
    help="learning rate for gradient descent, default=1e-3",
)
@click.option(
    "--batch-size",
    default=32,
    type=int,
    help="number of examples to read during each training step, default=100",
)
@click.option(
    "--epochs", type=int, default=500, help="number of training epochs, default=50"
)
@click.option(
    "--mc-samples",
    type=int,
    default=50,
    help="number of samples from model posterior, default=50",
)
def density_network(
    context,
    dim_hidden,
    num_components,
    depth,
    negative_slope,
    dropout_rate,
    spectral_norm,
    learning_rate,
    batch_size,
    epochs,
    mc_samples,
):
    context.obj.update(
        {
            "dim_hidden": dim_hidden,
            "depth": depth,
            "num_components": num_components,
            "negative_slope": negative_slope,
            "dropout_rate": dropout_rate,
            "spectral_norm": spectral_norm,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "mc_samples": mc_samples,
        }
    )

    @ray.remote(
        num_gpus=context.obj.get("gpu_per_trial"),
        num_cpus=context.obj.get("cpu_per_trial"),
    )
    def trainer(
        config,
        experiment_dir,
        trial,
    ):
        dataset_name = config.get("dataset_name")
        config["ds_train"]["seed"] = trial
        config["ds_valid"]["seed"] = trial + 1 if dataset_name == "synthetic" else trial
        config["ds_test"]["seed"] = trial + 2 if dataset_name == "synthetic" else trial

        experiment_dir = (
            Path(experiment_dir)
            / "density-network"
            / f"dh-{dim_hidden}_nc-{num_components}_dp-{depth}_ns-{negative_slope}_dr-{dropout_rate}_sn-{spectral_norm}_lr-{learning_rate}_bs-{batch_size}_ep-{epochs}"
            / f"{trial:03d}"
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config_path = experiment_dir / "config.json"
        with config_path.open(mode="w") as cp:
            json.dump(config, cp)

        ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
        ds_valid = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))

        out_dir = experiment_dir / "checkpoints" / "mu"
        outcome_model = models.GaussianMixtureDensityNetwork(
            job_dir=out_dir,
            architecture="resnet",
            dim_input=ds_train.dim_input,
            dim_treatment=ds_train.dim_treatment,
            dim_hidden=config.get("dim_hidden"),
            dim_output=config.get("num_components"),
            depth=config.get("depth"),
            negative_slope=config.get("negative_slope"),
            batch_norm=False,
            spectral_norm=config.get("spectral_norm"),
            dropout_rate=config.get("dropout_rate"),
            weight_decay=(0.5 * (1 - config.get("dropout_rate"))) / len(ds_train),
            learning_rate=config.get("learning_rate"),
            batch_size=config.get("batch_size"),
            epochs=config.get("epochs"),
            patience=100,
            num_workers=8,
            seed=config.get("seed"),
        )
        # _ = outcome_model.fit(ds_train, ds_valid)

        ds_train_config = config.get("ds_train")
        ds_train_config["mode"] = "pi"
        ds_train = datasets.DATASETS.get(dataset_name)(**ds_train_config)
        ds_valid_config = config.get("ds_valid")
        ds_valid_config["mode"] = "pi"
        ds_valid = datasets.DATASETS.get(dataset_name)(**ds_valid_config)
        
        out_dir = experiment_dir / "checkpoints" / "pi"
        propensity_model = models.CategoricalDensityNetwork(
            job_dir=out_dir,
            architecture="resnet",
            dim_input=ds_train.dim_input,
            dim_treatment=0,
            dim_hidden=config.get("dim_hidden"),
            dim_output=ds_train.dim_treatment,
            depth=config.get("depth"),
            negative_slope=config.get("negative_slope"),
            batch_norm=False,
            spectral_norm=config.get("spectral_norm"),
            dropout_rate=config.get("dropout_rate"),
            weight_decay=(0.5 * (1 - config.get("dropout_rate"))) / len(ds_train),
            learning_rate=config.get("learning_rate"),
            batch_size=config.get("batch_size"),
            epochs=config.get("epochs"),
            patience=100,
            num_workers=8,
            seed=config.get("seed"),
        )
        _ = propensity_model.fit(ds_train, ds_valid)
        return -1

    results = []
    for trial in range(context.obj.get("num_trials")):
        results.append(
            trainer.remote(
                config=context.obj,
                experiment_dir=context.obj.get("experiment_dir"),
                trial=trial,
            )
        )
    ray.get(results)


if __name__ == "__main__":
    cli()

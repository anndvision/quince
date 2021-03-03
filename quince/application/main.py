import ray
import click

from torch import cuda

from pathlib import Path

from quince.application import workflows


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
    "--ensemble-size",
    type=int,
    default=1,
    help="number of models in ensemble, default=1",
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
    ensemble_size,
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
            "ensemble_size": ensemble_size,
        }
    )

    @ray.remote(
        num_gpus=context.obj.get("gpu_per_trial"),
        num_cpus=context.obj.get("cpu_per_trial"),
    )
    def trainer(**kwargs):
        func = workflows.density_network_trainer(**kwargs)
        return func

    results = []
    for trial in range(context.obj.get("num_trials")):
        for ensemble_id in range(ensemble_size):
            results.append(
                trainer.remote(
                    config=context.obj,
                    experiment_dir=context.obj.get("experiment_dir"),
                    trial=trial,
                    ensemble_id=ensemble_id,
                )
            )
    ray.get(results)


if __name__ == "__main__":
    cli()

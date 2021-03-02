import ray
import click

from torch import cuda

from pathlib import Path

from quince.library import datasets


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


if __name__ == "__main__":
    cli()

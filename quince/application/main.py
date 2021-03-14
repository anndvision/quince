import ray
import math
import click

from torch import cuda

from pathlib import Path

from quince.application import workflows


@click.group(chain=True)
@click.pass_context
def cli(context):
    context.obj = {"n_gpu": cuda.device_count()}


@cli.command("evaluate")
@click.option(
    "--experiment-dir",
    type=str,
    required=True,
    help="location for reading checkpoints",
)
@click.option(
    "--output-dir",
    type=str,
    required=False,
    default=None,
    help="location for writing results",
)
@click.option(
    "--mc-samples",
    type=int,
    required=False,
    default=100,
    help="Number of samples from p(y | x, t), default=100",
)
@click.pass_context
def evaluate(
    context,
    experiment_dir,
    output_dir,
    mc_samples,
):
    output_dir = experiment_dir if output_dir is None else output_dir
    context.obj.update(
        {
            "experiment_dir": experiment_dir,
            "output_dir": output_dir,
            "mc_samples": mc_samples,
        }
    )
    workflows.evaluation.evaluate(
        experiment_dir=Path(experiment_dir),
        output_dir=Path(output_dir),
        mc_samples=mc_samples,
    )


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


@cli.command("hcmnist")
@click.pass_context
@click.option(
    "--root",
    type=str,
    required=True,
    help="location of dataset",
)
@click.option(
    "--lambda-star",
    default=math.exp(1.0),
    type=float,
    help="Ground truth level of hidden confounding, default=2.7",
)
@click.option(
    "--gamma",
    default=4.0,
    type=float,
    help="Coefficient for u effect on y, default=4.0",
)
@click.option(
    "--beta",
    default=0.75,
    type=float,
    help="Coefficient for x effect on t, default=2.0",
)
@click.option(
    "--sigma",
    default=1.0,
    type=float,
    help="standard deviation of random noise in y, default=1.0",
)
@click.option(
    "--domain-limit",
    default=2.0,
    type=float,
    help="Domain of x is [-domain_limit, domain_limit], default=2.5",
)
def hcmnist(
    context,
    root,
    lambda_star,
    gamma,
    beta,
    sigma,
    domain_limit,
):
    job_dir = Path(context.obj.get("job_dir"))
    dataset_name = "hcmnist"
    experiment_dir = (
        job_dir
        / dataset_name
        / f"ls-{lambda_star:.02f}_ga-{gamma:.02f}_be-{beta:.02f}_si-{sigma:.02f}_dl-{domain_limit:.02f}"
    )
    context.obj.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "ds_train": {
                "root": root,
                "lambda_star": lambda_star,
                "split": "train",
                "gamma": gamma,
                "beta": beta,
                "mode": "mu",
                "p_u": "bernoulli",
                "sigma_y": sigma,
                "domain": domain_limit,
                "seed": context.obj.get("seed"),
            },
            "ds_valid": {
                "root": root,
                "lambda_star": lambda_star,
                "split": "valid",
                "gamma": gamma,
                "beta": beta,
                "mode": "mu",
                "p_u": "bernoulli",
                "sigma_y": sigma,
                "domain": domain_limit,
                "seed": context.obj.get("seed") + 1,
            },
            "ds_test": {
                "root": root,
                "lambda_star": lambda_star,
                "split": "test",
                "gamma": gamma,
                "beta": beta,
                "mode": "mu",
                "p_u": "bernoulli",
                "sigma_y": sigma,
                "domain": domain_limit,
                "seed": context.obj.get("seed") + 2,
            },
        }
    )


@cli.command("synthetic")
@click.pass_context
@click.option(
    "--num-examples",
    default=1000,
    type=int,
    help="number of training examples, defaul=1000",
)
@click.option(
    "--lambda-star",
    default=math.exp(1.0),
    type=float,
    help="Ground truth level of hidden confounding, default=2.7",
)
@click.option(
    "--gamma",
    default=4.0,
    type=float,
    help="Coefficient for u effect on y, default=4.0",
)
@click.option(
    "--beta",
    default=0.75,
    type=float,
    help="Coefficient for x effect on t, default=2.0",
)
@click.option(
    "--sigma",
    default=1.0,
    type=float,
    help="standard deviation of random noise in y, default=1.0",
)
@click.option(
    "--domain-limit",
    default=2.0,
    type=float,
    help="Domain of x is [-domain_limit, domain_limit], default=2.5",
)
def synthetic(
    context,
    num_examples,
    lambda_star,
    gamma,
    beta,
    sigma,
    domain_limit,
):
    job_dir = Path(context.obj.get("job_dir"))
    dataset_name = "synthetic"
    experiment_dir = (
        job_dir
        / dataset_name
        / f"ne-{num_examples}_ls-{lambda_star:.02f}_ga-{gamma:.02f}_be-{beta:.02f}_si-{sigma:.02f}_dl-{domain_limit:.02f}"
    )
    context.obj.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "ds_train": {
                "num_examples": num_examples,
                "lambda_star": lambda_star,
                "gamma": gamma,
                "beta": beta,
                "mode": "mu",
                "p_u": "bernoulli",
                "sigma_y": sigma,
                "domain": domain_limit,
                "seed": context.obj.get("seed"),
            },
            "ds_valid": {
                "num_examples": num_examples // 10,
                "lambda_star": lambda_star,
                "gamma": gamma,
                "beta": beta,
                "mode": "mu",
                "p_u": "bernoulli",
                "sigma_y": sigma,
                "domain": domain_limit,
                "seed": context.obj.get("seed") + 1,
            },
            "ds_test": {
                "num_examples": min(num_examples, 2000),
                "lambda_star": lambda_star,
                "gamma": gamma,
                "beta": beta,
                "mode": "mu",
                "p_u": "bernoulli",
                "sigma_y": sigma,
                "domain": domain_limit,
                "seed": context.obj.get("seed") + 2,
            },
        }
    )


@cli.command("density-network")
@click.pass_context
@click.option("--dim-hidden", default=200, type=int, help="num neurons")
@click.option("--num-components", default=5, type=int, help="num mixture components")
@click.option("--depth", default=3, type=int, help="depth of feature extractor")
@click.option(
    "--negative-slope",
    default=-1,
    type=float,
    help="negative slope of leaky relu, default=-1 use elu",
)
@click.option(
    "--dropout-rate", default=0.1, type=float, help="dropout rate, default=0.1"
)
@click.option(
    "--spectral-norm",
    default=3.0,
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
    default=10,
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

from ray import tune
from ray.tune import suggest
from ray.tune import schedulers

from quince.library import models
from quince.library import datasets


def tune_desity_estimator(config):
    dataset_name = config.get("dataset_name")
    ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    ds_valid = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))

    dim_hidden = config.get("dim_hidden")
    dim_output = config.get("num_components")
    depth = config.get("depth")
    negative_slope = config.get("negative_slope")
    dropout_rate = config.get("dropout_rate")
    spectral_norm = config.get("spectral_norm")
    learning_rate = config.get("learning_rate")
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")

    outcome_model = models.GaussianMixtureDensityNetwork(
        job_dir=None,
        architecture="resnet",
        dim_input=ds_train.dim_input,
        dim_treatment=ds_train.dim_treatment,
        dim_hidden=dim_hidden,
        dim_output=dim_output,
        depth=depth,
        negative_slope=negative_slope,
        batch_norm=False,
        spectral_norm=spectral_norm,
        dropout_rate=dropout_rate,
        weight_decay=(0.5 * (1 - dropout_rate)) / len(ds_train),
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=10,
        num_workers=8,
        seed=config.get("seed"),
    )
    _ = outcome_model.fit(ds_train, ds_valid)


def hyper_tune(config):
    space = {
        "dim_hidden": tune.choice([50, 100, 200, 400]),
        "depth": tune.choice([2, 3, 4, 5])
        if config["dataset_name"] != "hcmnist"
        else tune.choice([1, 2, 3, 4]),
        "num_components": tune.choice([1, 2, 3, 4, 5]),
        "negative_slope": tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, -1.0]),
        "dropout_rate": tune.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]),
        "spectral_norm": tune.choice([0.95, 1.0, 1.5, 3.0, 6.0, 12.0, 24.0]),
        "learning_rate": tune.choice([5e-5, 1e-4, 2e-4, 5e-4, 1e-3]),
        "batch_size": tune.choice([16, 32, 64, 100, 200]),
    }
    algorithm = suggest.hyperopt.HyperOptSearch(
        space,
        metric="mean_loss",
        mode="min",
        n_initial_points=20,
    )
    scheduler = schedulers.AsyncHyperBandScheduler()
    analysis = tune.run(
        run_or_experiment=tune_desity_estimator,
        metric="mean_loss",
        mode="min",
        name="hyperopt_density_estimator",
        resources_per_trial={
            "cpu": config.get("cpu_per_trial"),
            "gpu": config.get("gpu_per_trial"),
        },
        num_samples=config.get("max_samples"),
        search_alg=algorithm,
        scheduler=scheduler,
        local_dir=config.get("experiment_dir"),
        config=config,
    )
    print("Best hyperparameters found were: ", analysis.best_config)

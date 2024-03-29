import json
from pathlib import Path

from quince.library import models
from quince.library import datasets


def density_network_trainer(
    config,
    experiment_dir,
    trial,
    ensemble_id,
):
    dataset_name = config.get("dataset_name")
    config["ds_train"]["seed"] = trial
    config["ds_valid"]["seed"] = trial + 1 if dataset_name == "synthetic" else trial
    config["ds_test"]["seed"] = trial + 2 if dataset_name == "synthetic" else trial
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

    experiment_dir = (
        Path(experiment_dir)
        / "density-network"
        / f"dh-{dim_hidden}_nc-{dim_output}_dp-{depth}_ns-{negative_slope}_dr-{dropout_rate}_sn-{spectral_norm}_lr-{learning_rate}_bs-{batch_size}_ep-{epochs}"
        / f"trial-{trial:03d}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_path = experiment_dir / "config.json"
    with config_path.open(mode="w") as cp:
        json.dump(config, cp)

    out_dir = experiment_dir / "checkpoints" / f"model-{ensemble_id}" / "mu"
    if not (out_dir / "best_checkpoint.pt").exists():
        outcome_model = models.GaussianMixtureDensityNetwork(
            job_dir=out_dir,
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
            seed=ensemble_id,
        )
        _ = outcome_model.fit(ds_train, ds_valid)

    ds_train_config = config.get("ds_train")
    ds_train_config["mode"] = "pi"
    ds_train = datasets.DATASETS.get(dataset_name)(**ds_train_config)
    ds_valid_config = config.get("ds_valid")
    ds_valid_config["mode"] = "pi"
    ds_valid = datasets.DATASETS.get(dataset_name)(**ds_valid_config)

    out_dir = experiment_dir / "checkpoints" / f"model-{ensemble_id}" / "pi"
    if not (out_dir / "best_checkpoint.pt").exists():
        propensity_model = models.CategoricalDensityNetwork(
            job_dir=out_dir,
            architecture="resnet",
            dim_input=ds_train.dim_input,
            dim_treatment=0,
            dim_hidden=dim_hidden,
            dim_output=ds_train.dim_treatment,
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
            seed=ensemble_id,
        )
        _ = propensity_model.fit(ds_train, ds_valid)
    return -1

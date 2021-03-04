from quince.library import models


def build(config, experiment_dir, ds):
    outcome_ensemble = []
    propensity_ensemble = []
    for i in range(config.get("ensemble_size")):
        model_dir = experiment_dir / "checkpoints" / f"model-{i}" / "mu"
        outcome_model = models.GaussianMixtureDensityNetwork(
            job_dir=model_dir,
            architecture="resnet",
            dim_input=ds.dim_input,
            dim_treatment=ds.dim_treatment,
            dim_hidden=config.get("dim_hidden"),
            dim_output=config.get("num_components"),
            depth=config.get("depth"),
            negative_slope=config.get("negative_slope"),
            batch_norm=False,
            spectral_norm=config.get("spectral_norm"),
            dropout_rate=config.get("dropout_rate"),
            weight_decay=(0.5 * (1 - config.get("dropout_rate"))) / ds.num_examples,
            learning_rate=config.get("learning_rate"),
            batch_size=config.get("batch_size"),
            epochs=config.get("epochs"),
            patience=100,
            num_workers=8,
            seed=config.get("seed"),
        )
        outcome_model.load(load_best=True)
        outcome_ensemble.append(outcome_model)
        model_dir = experiment_dir / "checkpoints" / f"model-{i}" / "pi"
        propensity_model = models.CategoricalDensityNetwork(
            job_dir=model_dir,
            architecture="resnet",
            dim_input=ds.dim_input,
            dim_treatment=0,
            dim_hidden=config.get("dim_hidden"),
            dim_output=1,
            depth=config.get("depth"),
            negative_slope=config.get("negative_slope"),
            batch_norm=False,
            spectral_norm=config.get("spectral_norm"),
            dropout_rate=config.get("dropout_rate"),
            weight_decay=(0.5 * (1 - config.get("dropout_rate"))) / ds.num_examples,
            learning_rate=config.get("learning_rate"),
            batch_size=config.get("batch_size"),
            epochs=config.get("epochs"),
            patience=100,
            num_workers=8,
            seed=config.get("seed"),
        )
        propensity_model.load(load_best=True)
        propensity_ensemble.append(propensity_model)
    return outcome_ensemble, propensity_ensemble

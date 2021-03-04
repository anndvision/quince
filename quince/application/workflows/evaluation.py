import json

from quince.library import datasets
from quince.library import ensembles


def evaluate(experiment_dir):
    for trial_dir in sorted(experiment_dir.iterdir()):
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)
            dataset_name = config.get("dataset_name")
            ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
            outcome_ensemble, propensity_ensemble = ensembles.build(
                config=config, experiment_dir=trial_dir, ds=ds_train
            )

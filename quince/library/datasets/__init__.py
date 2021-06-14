from quince.library.datasets.ihdp import IHDP
from quince.library.datasets.hcmnist import HCMNIST
from quince.library.datasets.synthetic import Synthetic
from quince.library.datasets.samplers import RandomFixedLengthSampler

DATASETS = {
    "ihdp": IHDP,
    "synthetic": Synthetic,
    "hcmnist": HCMNIST,
}

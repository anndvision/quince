import torch
import pyreadr
import numpy as np

from pathlib import Path

from torch.utils import data

from sklearn import preprocessing
from sklearn import model_selection

_CONTINUOUS_COVARIATES = [
    "bw",
    "b.head",
    "preterm",
    "birth.o",
    "nnhealth",
    "momage",
]

_BINARY_COVARIATES = [
    "sex",
    "twin",
    "mom.lths",
    "mom.hs",
    "mom.scoll",
    "cig",
    "first",
    "booze",
    "drugs",
    "work.dur",
    "prenatal",
    "ark",
    "ein",
    "har",
    "mia",
    "pen",
    "tex",
    "was",
]

_HIDDEN_COVARIATE = [
    "b.marr",
]


class IHDP(data.Dataset):
    def __init__(self, root, split, mode, seed, hidden_confounding):

        root = Path(root)
        df = pyreadr.read_r(str(root / "ihdp.RData"))["ihdp"]
        # Make observational as per Hill 2011
        df = df[~((df["treat"] == 1) & (df["momwhite"] == 0))]
        df = df[
            _CONTINUOUS_COVARIATES + _BINARY_COVARIATES + _HIDDEN_COVARIATE + ["treat"]
        ]
        # Standardize continuous covariates
        df[_CONTINUOUS_COVARIATES] = preprocessing.StandardScaler().fit_transform(
            df[_CONTINUOUS_COVARIATES]
        )
        # Generate response surfaces
        rng = np.random.default_rng(seed)
        x = df[_CONTINUOUS_COVARIATES + _BINARY_COVARIATES]
        u = df[_HIDDEN_COVARIATE]
        t = df["treat"]
        beta_x = rng.choice(
            [0.0, 0.1, 0.2, 0.3, 0.4], size=(24,), p=[0.6, 0.1, 0.1, 0.1, 0.1]
        )
        beta_u = rng.choice(
            [0.1, 0.2, 0.3, 0.4, 0.5], size=(1,), p=[0.2, 0.2, 0.2, 0.2, 0.2]
        )
        mu0 = np.exp((x + 0.5).dot(beta_x) + (u + 0.5).dot(beta_u))
        df["mu0"] = mu0
        mu1 = (x + 0.5).dot(beta_x) + (u + 0.5).dot(beta_u)
        omega = (mu1[t == 1] - mu0[t == 1]).mean(0) - 4
        mu1 -= omega
        df["mu1"] = mu1
        y = t * mu1 + (1 - t) * mu0 + rng.normal(size=t.shape)
        df["y"] = y
        # Train test split
        df_train, df_test = model_selection.train_test_split(
            df, test_size=0.1, random_state=seed
        )
        self.split = split
        # Set x, y, and t values
        self.y_mean = df_train["y"].mean()
        self.y_std = df_train["y"].std()
        covars = _CONTINUOUS_COVARIATES + _BINARY_COVARIATES
        covars = covars + _HIDDEN_COVARIATE if not hidden_confounding else covars
        self.dim_input = len(covars)
        self.dim_output = 1
        if self.split == "test":
            self.x = df_test[covars].to_numpy(dtype="float32")
            self.t = df_test["treat"].to_numpy(dtype="float32")
            self.mu0 = df_test["mu0"].to_numpy(dtype="float32")
            self.mu1 = df_test["mu1"].to_numpy(dtype="float32")
            if mode == "mu":
                self.y = self.mu1 - self.mu0
            elif mode == "pi":
                self.y = self.t
            else:
                raise NotImplementedError("Not a valid mode")
        else:
            df_train, df_valid = model_selection.train_test_split(
                df_train, test_size=0.3, random_state=seed
            )
            if split == "train":
                df = df_train
            elif split == "valid":
                df = df_valid
            else:
                raise NotImplementedError("Not a valid dataset split")
            self.x = df[covars].to_numpy(dtype="float32")
            self.t = df["treat"].to_numpy(dtype="float32")
            self.mu0 = df["mu0"].to_numpy(dtype="float32")
            self.mu1 = df["mu1"].to_numpy(dtype="float32")
            if mode == "mu":
                self.y = df["y"].to_numpy(dtype="float32")
            elif mode == "pi":
                self.y = self.t
            else:
                raise NotImplementedError("Not a valid mode")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        inputs = torch.from_numpy(self.x[idx]).float()
        targets = torch.from_numpy((self.y[idx] - self.y_mean) / self.y_std).float()
        return inputs, targets

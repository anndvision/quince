# quince

Code for [Quantifying Ignorance in Individual-Level Causal-Effect Estimates under Hidden Confounding](https://arxiv.org/abs/2103.04850)

![Image of Gamma Sweep](assets/gamma-sweep.gif)

## Installation

```.bash
$ git clone git@github.com:anndvision/quince.git
$ cd quince
$ conda env create -f environment.yml
$ conda activate quince
```

## Example

### Step 1: Hyperparameter Tuning (optional)

Find the best hyperparameters using the `tune` function, on a dataset like `ihdp` for a `density-network` model.

```.bash
$ quince \
    tune \
        --job-dir ~/experiments/quince/tuning/ \
        --max-samples 200 \
        --gpu-per-trial 0.25 \
    ihdp \
        --root /path/to/quince-repo/assets/ \
    density-network
```

### Step 2: Train ensembles over a number of trials

Here, we use the `train` function to fit an ensemble of `density-network`s on 200 realizations of the `ihdp` with hidden confounding dataset.

```.bash
$ quince \
    train \
        --job-dir ~/experiments/quince/ \
        --num-trials 200 \
        --gpu-per-trial 0.25 \
    ihdp \
        --root assets/ \
    density-network \
        --dim-hidden 400 \
        --num-components 5 \
        --depth 3 \
        --negative-slope -1 \
        --dropout-rate 0.15 \
        --spectral-norm 0.95 \
        --learning-rate 1e-3 \
        --batch-size 32 \
        --epochs 500 \
        --ensemble-size 10

```

### Step 3: Evaluate

```.bash
$ quince \
    evaluate \
        --experiment-dir ~/experiments/quince/ihdp/hc-True/density-network/dh-400_nc-5_dp-3_ns--1.0_dr-0.15_sn-0.95_lr-0.001_bs-32_ep-500/ \
        --mc-samples 100 \
```

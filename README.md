# quince :pear:

Code for [Quantifying Ignorance in Individual-Level Causal-Effect Estimates under Hidden Confounding](https://arxiv.org/abs/2103.04850)

![Image of Gamma Sweep](assets/gamma-sweep.gif)

## :pear: Installation

```.bash
$ git clone git@github.com:anndvision/quince.git
$ cd quince
$ conda env create -f environment.yml
$ conda activate quince
```

## :pear: Example: Replicating IHDP results

### Step 1: Hyperparameter Tuning (optional)

Find the best hyperparameters using the `tune` function, on a dataset like `ihdp` for an `ensemble` model.

```.bash
$ quince \
    tune \
        --job-dir ~/experiments/quince/tuning/ \
        --max-samples 500 \
        --gpu-per-trial 0.2 \
    ihdp \
    ensemble
```

### Step 2: Train ensembles over a number of trials

Here, we use the `train` function to fit an `ensemble` of mixture density networks on 10 realizations of the `ihdp` with hidden confounding dataset. For the full results change `--num-trials 1000`

```.bash
$ quince \
    train \
        --job-dir ~/experiments/quince/ \
        --num-trials 10 \
        --gpu-per-trial 0.2 \
    ihdp \
    ensemble \
        --dim-hidden 200 \
        --num-components 5 \
        --depth 4 \
        --negative-slope 0.3 \
        --dropout-rate 0.5 \
        --spectral-norm 6.0 \
        --learning-rate 5e-4 \
        --batch-size 200 \
        --epochs 500 \
        --ensemble-size 10

```

### Step 3: Evaluate

Plots will be written to the `experiment-dir`
```.bash
$ quince \
    evaluate \
        --experiment-dir ~/experiments/quince/ihdp/hc-True_beta-None/ensemble/dh-200_nc-5_dp-4_ns-0.3_dr-0.5_sn-6.0_lr-0.0005_bs-200_ep-500/ \
    compute-intervals \
        --gpu-per-trial 0.2 \
    compute-intervals-kernel \
        --gpu-per-trial 0.2 \
    plot-deferral \
    plot-errorbars \
        --trial 0
```

## :pear: Replicating Other Results

### Simulated Data

```.bash
$ quince \
    train \
        --job-dir ~/experiments/quince/ \
        --num-trials 50 \
        --gpu-per-trial 0.2 \
    synthetic \
        --lambda-star 1.65
    ensemble \
        --dim-hidden 200 \
        --num-components 5 \
        --depth 4 \
        --negative-slope 0.0 \
        --dropout-rate 0.1 \
        --spectral-norm 6.0 \
        --learning-rate 1e-3 \
        --batch-size 32 \
        --epochs 500 \
        --ensemble-size 10

```

```.bash
$ quince \
    evaluate \
        --experiment-dir ~/experiments/quince/synthetic/ne-1000_ls-1.65_ga-4.00_be-0.75_si-1.00_dl-2.00/ensemble/dh-200_nc-5_dp-4_ns-0.0_dr-0.1_sn-6.0_lr-0.001_bs-32_ep-500/ \
        --mc-samples 100 \
```

Repeat the above for `--lambda-star 2.72` and `--lambda-star 4.48`.

### HCMNIST

```.bash
$ quince \
    train \
        --job-dir ~/experiments/quince/ \
        --num-trials 20 \
        --gpu-per-trial 0.5 \
    hcmnist \
        --root ~/data
        --lambda-star 1.65
    ensemble \
        --dim-hidden 200 \
        --num-components 5 \
        --depth 2 \
        --negative-slope 0.0 \
        --dropout-rate 0.15 \
        --spectral-norm 3.0 \
        --learning-rate 5e-4 \
        --batch-size 200 \
        --epochs 500 \
        --ensemble-size 5

```

```.bash
$ quince \
    evaluate \
        --experiment-dir ~/experiments/quince/hcmnist/ls-1.65_ga-4.00_be-0.75_si-1.00_dl-2.00/ensemble/dh-200_nc-5_dp-2_ns-0.0_dr-0.15_sn-3.0_lr-0.0005_bs-200_ep-500/ \
        --mc-samples 100 \
```

Repeat the above for `--lambda-star 2.72` and `--lambda-star 4.48`.

import math
import json
import torch
import numpy as np

from pathlib import Path

from scipy import stats

from quince.library import utils
from quince.library import models
from quince.library import datasets
from quince.library import plotting


def compute_intervals_ensemble(trial_dir, mc_samples):
    config_path = trial_dir / "config.json"
    with config_path.open(mode="r") as cp:
        config = json.load(cp)
    dataset_name = config.get("dataset_name")
    ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    outcome_ensemble = build_ensemble(
        config=config, experiment_dir=trial_dir, ds=ds_train
    )
    ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))
    _ = get_intervals(
        dataset=ds_test,
        outcome_ensemble=outcome_ensemble,
        mc_samples_y=mc_samples,
        file_path=trial_dir / "intervals.json",
    )


def compute_intervals_kernel(trial_dir):
    config_path = trial_dir / "config.json"
    with config_path.open(mode="r") as cp:
        config = json.load(cp)
    dataset_name = config.get("dataset_name")
    ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    outcome_ensemble = build_ensemble(
        config=config, experiment_dir=trial_dir, ds=ds_train
    )
    ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))
    kr = models.KernelRegressor(
        dataset=ds_train,
        initial_length_scale=1.0,
        feature_extractor=outcome_ensemble[0].encoder.encoder
        if config["dataset_name"] == "hcmnist"
        else None,
        propensity_model=outcome_ensemble[0],
        verbose=False,
    )
    ds_valid = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))
    kr.fit_length_scale(ds_valid, grid=np.arange(0.1, 3.0, 0.002))
    _ = get_intervals_kernel(
        dataset=ds_test, model=kr, file_path=trial_dir / "intervals_kernels.json",
    )


def print_summary(experiment_dir, kernel):
    summary = {"policy_risk": {"risk": {"true": []}, "error": {},}}
    for k in GAMMAS.keys():
        summary["policy_risk"]["risk"].update({k: []})
        summary["policy_risk"]["error"].update({k: []})
    for trial_dir in sorted(experiment_dir.iterdir()):
        if "trial-" not in str(trial_dir):
            continue
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)
        dataset_name = config.get("dataset_name")
        ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))
        tau_true = torch.tensor(ds_test.mu1 - ds_test.mu0).to("cpu")
        pi_true = (
            tau_true >= 0.0 if config["dataset_name"] == "ihdp" else tau_true < 0.0
        )
        intervals = load_intervals(
            file_path=trial_dir / "intervals_kernels.json"
            if kernel
            else trial_dir / "intervals.json",
        )
        update_summaries(
            summary=summary,
            dataset=ds_test,
            intervals=intervals,
            pi_true=pi_true.numpy().astype("float32"),
            epistemic_uncertainty=False,
            lt=False if config["dataset_name"] == "ihdp" else True,
        )
    for k, v in summary["policy_risk"]["risk"].items():
        se = stats.sem(v)
        h = se * stats.t.ppf((1 + 0.95) / 2.0, 20 - 1)
        print(k, np.mean(v), h)
    print("")
    for k, v in summary["policy_risk"]["error"].items():
        se = stats.sem(v)
        h = se * stats.t.ppf((1 + 0.95) / 2.0, 20 - 1)
        print(k, np.mean(v), h)
    print("")


def paired_t_test(experiment_dir):
    summary = {"policy_risk": {"risk": {"true": []}, "error": {},}}
    summary_kernel = {"policy_risk": {"risk": {"true": []}, "error": {},}}
    for k in GAMMAS.keys():
        summary["policy_risk"]["risk"].update({k: []})
        summary["policy_risk"]["error"].update({k: []})
        summary_kernel["policy_risk"]["risk"].update({k: []})
        summary_kernel["policy_risk"]["error"].update({k: []})
    for trial_dir in sorted(experiment_dir.iterdir()):
        if "trial-" not in str(trial_dir):
            continue
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)
        dataset_name = config.get("dataset_name")
        ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))
        tau_true = torch.tensor(ds_test.mu1 - ds_test.mu0).to("cpu")
        pi_true = (
            tau_true >= 0.0 if config["dataset_name"] == "ihdp" else tau_true < 0.0
        )
        intervals = load_intervals(file_path=trial_dir / "intervals.json",)
        intervals_kernel = load_intervals(
            file_path=trial_dir / "intervals_kernels.json"
        )
        update_summaries(
            summary=summary,
            dataset=ds_test,
            intervals=intervals,
            pi_true=pi_true.numpy().astype("float32"),
            epistemic_uncertainty=False,
            lt=False if config["dataset_name"] == "ihdp" else True,
        )
        update_summaries(
            summary=summary_kernel,
            dataset=ds_test,
            intervals=intervals_kernel,
            pi_true=pi_true.numpy().astype("float32"),
            epistemic_uncertainty=False,
            lt=False if config["dataset_name"] == "ihdp" else True,
        )
    print("Paired t-test")
    for k in summary["policy_risk"]["risk"].keys():
        v_density = np.asarray(summary["policy_risk"]["risk"][k])
        v_kernel = np.asarray(summary_kernel["policy_risk"]["risk"][k])
        statistic, pvalue = stats.ttest_rel(v_density, v_kernel)
        print(k, f"{statistic:.06f}", f"{pvalue:.06f}")
    print("")


def plot_deferral(experiment_dir):
    summary = {
        "sweep": {
            "sensitivity": {"pehe": [], "error_rate": [], "defer_rate": [],},
            "ignorance": {"pehe": [], "error_rate": [], "defer_rate": [],},
            "epistemic": {"pehe": [], "error_rate": [], "defer_rate": [],},
        },
    }
    summary_kernel = {
        "sweep": {"sensitivity": {"pehe": [], "error_rate": [], "defer_rate": [],},},
    }
    for trial_dir in sorted(experiment_dir.iterdir()):
        if "trial-" not in str(trial_dir):
            continue
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)
        dataset_name = config.get("dataset_name")
        ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))
        tau_true = torch.tensor(ds_test.mu1 - ds_test.mu0).to("cpu")
        pi_true = (
            tau_true >= 0.0 if config["dataset_name"] == "ihdp" else tau_true < 0.0
        )
        intervals = load_intervals(file_path=trial_dir / "intervals.json",)
        intervals_kernel = load_intervals(
            file_path=trial_dir / "intervals_kernels.json",
        )
        update_ignorance(
            results=summary, intervals=intervals, tau_true=tau_true, pi_true=pi_true,
        )
        update_sensitivity(
            results=summary, intervals=intervals, tau_true=tau_true, pi_true=pi_true,
        )
        update_epistemic(
            results=summary, intervals=intervals, tau_true=tau_true, pi_true=pi_true,
        )
        update_sensitivity(
            results=summary_kernel,
            intervals=intervals_kernel,
            tau_true=tau_true,
            pi_true=pi_true,
        )

    epistemic = summary["sweep"]["epistemic"]
    for k in epistemic.keys():
        epistemic[k] = np.nan_to_num(np.asarray(epistemic[k]).transpose())
    ignorance = summary["sweep"]["ignorance"]
    for k in ignorance.keys():
        ignorance[k] = np.nan_to_num(np.asarray(ignorance[k]).transpose())
    sensitivity = summary["sweep"]["sensitivity"]
    for k in sensitivity.keys():
        sensitivity[k] = np.nan_to_num(np.asarray(sensitivity[k]).transpose())
    sensitivity_kernel = summary_kernel["sweep"]["sensitivity"]
    for k in sensitivity_kernel.keys():
        sensitivity_kernel[k] = np.nan_to_num(
            np.asarray(sensitivity_kernel[k]).transpose()
        )
    plot_sweep(
        ignorance=ignorance,
        epistemic=epistemic,
        sensitivity=sensitivity,
        sensitivity_kernel=sensitivity_kernel,
        mode="error_rate",
        output_dir=experiment_dir,
    )
    plot_sweep(
        ignorance=ignorance,
        epistemic=epistemic,
        sensitivity=sensitivity,
        sensitivity_kernel=sensitivity_kernel,
        mode="pehe",
        output_dir=experiment_dir,
    )


def build_ensemble(config, experiment_dir, ds):
    outcome_ensemble = []
    for i in range(config.get("ensemble_size")):
        model_dir = experiment_dir / "checkpoints" / f"model-{i}" / "mu"
        outcome_model = models.DragonNet(
            job_dir=model_dir,
            architecture="resnet",
            dim_input=ds.dim_input,
            dim_hidden=config.get("dim_hidden"),
            dim_output=config.get("num_components"),
            depth=config.get("depth"),
            negative_slope=config.get("negative_slope"),
            batch_norm=False,
            spectral_norm=config.get("spectral_norm"),
            dropout_rate=config.get("dropout_rate"),
            num_examples=len(ds),
            learning_rate=config.get("learning_rate"),
            batch_size=config.get("batch_size"),
            epochs=config.get("epochs"),
            patience=100,
            num_workers=0,
            seed=config.get("seed"),
        )
        outcome_model.load()
        outcome_ensemble.append(outcome_model)
    return outcome_ensemble


def get_intervals(
    dataset, outcome_ensemble, mc_samples_y, file_path,
):
    if file_path.exists():
        with file_path.open(mode="r") as fp:
            intervals = json.load(fp)
    else:
        x_in = torch.tensor(dataset.x).to(outcome_ensemble[0].device)
        inputs = torch.cat(
            [
                torch.cat([x_in, torch.zeros_like(x_in[:, :1])], dim=-1),
                torch.cat([x_in, torch.ones_like(x_in[:, :1])], dim=-1),
            ]
        )
        intervals = {}
        for k, v in GAMMAS.items():
            tau_mean, tau_bottom, tau_top = predict_tau_ensemble(
                inputs_outcome=inputs,
                outcome_ensemble=outcome_ensemble,
                gamma=v,
                mc_samples_y=mc_samples_y,
            )
            tau_hat = {
                "bottom": (dataset.y_std[0] * tau_bottom).numpy().tolist(),
                "top": (dataset.y_std[0] * tau_top).numpy().tolist(),
                "mean": (dataset.y_std[0] * tau_mean).numpy().tolist(),
            }
            intervals.update({k: tau_hat})
        with file_path.open(mode="w") as fp:
            json.dump(intervals, fp)
    for k, v in intervals.items():
        for k1, v2 in v.items():
            intervals[k][k1] = torch.Tensor(v2)
    return intervals


def predict_tau_ensemble(
    inputs_outcome, outcome_ensemble, gamma, mc_samples_y,
):
    n = inputs_outcome.shape[0] // 2
    tau_tops = []
    tau_bottoms = []
    tau_means = []
    with torch.no_grad():
        for outcome_model in outcome_ensemble:
            outcome_model.network.eval()
            outcome_post, propensity_post = outcome_model.network(inputs_outcome)

            locs = outcome_post.component_distribution.loc
            probs = outcome_post.mixture_distribution.probs
            mu_hat = torch.sum(probs * locs, dim=-1).unsqueeze(0)
            mu_hat_0, mu_hat_1 = torch.split(mu_hat, mu_hat.shape[1] // 2, dim=1)
            e_hat = torch.clip(propensity_post.probs.squeeze(-1), 1e-7, 1 - 1e-7)[:n]

            alpha_1_hat = utils.alpha_fn(e_hat, gamma)
            alpha_0_hat = utils.alpha_fn(1.0 - e_hat, gamma)
            beta_1_hat = utils.beta_fn(e_hat, gamma)
            beta_0_hat = utils.beta_fn(1.0 - e_hat, gamma)

            alpha_0_prime = alpha_0_hat / (beta_0_hat - alpha_0_hat)
            alpha_1_prime = alpha_1_hat / (beta_1_hat - alpha_1_hat)

            y_hat = outcome_post.sample(torch.Size([mc_samples_y]))
            y_hat = torch.sort(y_hat, dim=0)[0]
            y_hat_0, y_hat_1 = torch.split(y_hat, y_hat.shape[1] // 2, dim=1)

            lambda_top_1 = []
            lambda_top_0 = []
            lambda_bottom_1 = []
            lambda_bottom_0 = []
            for i in range(mc_samples_y):
                lambda_top_1.append(
                    utils.lambda_top_func(mu_hat_1, i, y_hat_1, alpha_1_prime)
                )
                lambda_top_0.append(
                    utils.lambda_top_func(mu_hat_0, i, y_hat_0, alpha_0_prime)
                )
                lambda_bottom_1.append(
                    utils.lambda_bottom_func(mu_hat_1, i, y_hat_1, alpha_1_prime)
                )
                lambda_bottom_0.append(
                    utils.lambda_bottom_func(mu_hat_0, i, y_hat_0, alpha_0_prime)
                )
            lambda_top_1 = torch.cat(lambda_top_1)
            lambda_top_0 = torch.cat(lambda_top_0)
            lambda_bottom_1 = torch.cat(lambda_bottom_1)
            lambda_bottom_0 = torch.cat(lambda_bottom_0)

            tau_top = []
            tau_bottom = []
            for i in range(n):
                tau_top.append(
                    lambda_top_1[:, i : i + 1].max(dim=0)[0]
                    - lambda_bottom_0[:, i : i + 1].min(dim=0)[0]
                )
                tau_bottom.append(
                    lambda_bottom_1[:, i : i + 1].min(dim=0)[0]
                    - lambda_top_0[:, i : i + 1].max(dim=0)[0]
                )

            tau_top = torch.cat(tau_top)
            tau_bottom = torch.cat(tau_bottom)

            tau_tops.append(tau_top.unsqueeze(0))
            tau_bottoms.append(tau_bottom.unsqueeze(0))
            tau_means.append(mu_hat_1 - mu_hat_0)

    tau_top = torch.cat(tau_tops).to("cpu")
    tau_bottom = torch.cat(tau_bottoms).to("cpu")
    tau_mean = torch.cat(tau_means).to("cpu")
    return tau_mean, tau_bottom, tau_top


def get_intervals_kernel(dataset, model, file_path):
    if file_path.exists():
        with file_path.open(mode="r") as fp:
            intervals = json.load(fp)
    else:
        intervals = {}
        for k, v in GAMMAS.items():
            tau_mean, tau_bottom, tau_top = predict_tau_kernel(dataset.x, model, v)
            tau_hat = {
                "bottom": tau_bottom.transpose().tolist(),
                "top": tau_top.transpose().tolist(),
                "mean": tau_mean.transpose().tolist(),
            }
            intervals.update({k: tau_hat})
        with file_path.open(mode="w") as fp:
            json.dump(intervals, fp)
    for k, v in intervals.items():
        for k1, v2 in v.items():
            intervals[k][k1] = torch.Tensor(v2)
    return intervals


def load_intervals(file_path,):
    with file_path.open(mode="r") as fp:
        intervals = json.load(fp)
    for k, v in intervals.items():
        for k1, v2 in v.items():
            intervals[k][k1] = torch.Tensor(v2)
    return intervals


def predict_tau_kernel(x, model, gamma):
    model.gamma = gamma
    k = model.k(x)

    lambda_top_1 = []
    lambda_top_0 = []
    lambda_bottom_1 = []
    lambda_bottom_0 = []
    for i in range(k.shape[0]):
        lambda_top_1.append(model.lambda_top_1(i, k).reshape(1, -1))
        lambda_top_0.append(model.lambda_top_0(i, k).reshape(1, -1))
        lambda_bottom_1.append(model.lambda_bottom_1(i, k).reshape(1, -1))
        lambda_bottom_0.append(model.lambda_bottom_0(i, k).reshape(1, -1))
    lambda_top_1 = np.vstack(lambda_top_1)
    lambda_top_0 = np.vstack(lambda_top_0)
    lambda_bottom_1 = np.vstack(lambda_bottom_1)
    lambda_bottom_0 = np.vstack(lambda_bottom_0)

    tau_top = []
    tau_bottom = []
    for i in range(k.shape[0]):
        tau_top.append(
            lambda_top_1[:, i : i + 1].max(axis=0)
            - lambda_bottom_0[:, i : i + 1].min(axis=0)
        )
        tau_bottom.append(
            lambda_bottom_1[:, i : i + 1].min(axis=0)
            - lambda_top_0[:, i : i + 1].max(axis=0)
        )
    tau_top = np.stack(tau_top)
    tau_bottom = np.stack(tau_bottom)
    tau_mean = model.tau(k=k)
    return tau_mean, tau_bottom, tau_top


def plot_errorbars(trial_dir):
    output_dir = trial_dir / "ensemble" / "errorbars"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "config.json"
    with config_path.open(mode="r") as cp:
        config = json.load(cp)
    dataset_name = config.get("dataset_name")
    ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))
    tau_true = torch.tensor(ds_test.mu1 - ds_test.mu0).to("cpu")
    intervals = load_intervals(file_path=trial_dir / "intervals.json",)
    for k in intervals.keys():
        tau_hat = intervals[k]
        tau_mean = tau_hat["mean"].mean(0)
        tau_top = torch.abs(
            tau_hat["top"].mean(0) + 2 * tau_hat["mean"].std(0) - tau_mean
        )
        tau_bottom = torch.abs(
            tau_hat["bottom"].mean(0) - 2 * tau_hat["mean"].std(0) - tau_mean
        )
        plotting.errorbar(
            x=tau_true,
            y=tau_mean,
            y_err=torch.cat([tau_bottom.unsqueeze(0), tau_top.unsqueeze(0)]),
            x_label=r"$\tau(\mathbf{x})$",
            y_label=r"$\widehat{\tau}(\mathbf{x})$",
            marker_label=f"$\log\Gamma = $ {k}",
            x_pad=-20,
            y_pad=-45,
            file_path=output_dir / f"gamma-{k}.png",
        )


def plot_errorbars_kernel(trial_dir):
    output_dir = trial_dir / "kernel" / "errorbars"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "config.json"
    with config_path.open(mode="r") as cp:
        config = json.load(cp)
    dataset_name = config.get("dataset_name")
    ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))
    tau_true = torch.tensor(ds_test.mu1 - ds_test.mu0).to("cpu")
    intervals = load_intervals(file_path=trial_dir / "intervals_kernels.json",)
    for k in intervals.keys():
        tau_hat = intervals[k]
        tau_mean = tau_hat["mean"].mean(0)
        tau_top = torch.abs(tau_hat["top"].mean(0) - tau_mean)
        tau_bottom = torch.abs(tau_hat["bottom"].mean(0) - tau_mean)
        plotting.errorbar(
            x=tau_true,
            y=tau_mean,
            y_err=torch.cat([tau_bottom.unsqueeze(0), tau_top.unsqueeze(0)]),
            x_label=r"$\tau(\mathbf{x})$",
            y_label=r"$\widehat{\tau}(\mathbf{x})$",
            marker_label=f"$\log\Gamma = $ {k}",
            x_pad=-20,
            y_pad=-45,
            file_path=output_dir / f"gamma-{k}.png",
        )


def plot_functions(intervals, ds_train, ds_test, output_dir):
    tau_true = ds_test.mu1 - ds_test.mu0
    domain = ds_test.x.ravel()
    indices = np.argsort(domain)
    for k in intervals.keys():
        tau_hat = intervals[k]
        tau_mean = tau_hat["mean"]
        plotting.functions(
            x=ds_train.x.ravel(),
            t=ds_train.t,
            domain=domain[indices],
            tau_true=tau_true[indices],
            tau_mean=tau_mean[:, indices],
            file_path=output_dir / f"functions_gamma-{k}.png",
        )
        break


def plot_functions_mnist(intervals, ds_train, ds_test, output_dir):
    tau_true = ds_test.mu1 - ds_test.mu0
    domain = ds_test.phi.ravel()
    indices = np.argsort(domain)
    for k in intervals.keys():
        tau_hat = intervals[k]
        tau_mean = tau_hat["mean"]
        plotting.functions(
            x=ds_train.phi.ravel(),
            t=ds_train.t,
            domain=domain[indices],
            tau_true=tau_true[indices],
            tau_mean=tau_mean[:, indices],
            file_path=output_dir / f"functions_gamma-{k}.png",
        )
        break


def plot_ignorance(trial_dir):
    output_dir = trial_dir / "ensemble" / "ignorance"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "config.json"
    with config_path.open(mode="r") as cp:
        config = json.load(cp)
    dataset_name = config.get("dataset_name")
    ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    config["ds_test"]["domain"] = 3.5
    ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))
    tau_true = torch.tensor(ds_test.mu1 - ds_test.mu0).to("cpu")
    outcome_ensemble = build_ensemble(
        config=config, experiment_dir=trial_dir, ds=ds_train
    )
    intervals = get_intervals(
        dataset=ds_test,
        outcome_ensemble=outcome_ensemble,
        mc_samples_y=500,
        file_path=trial_dir / "intervals_ignorance.json",
    )
    tau_true = ds_test.mu1 - ds_test.mu0
    domain = ds_test.x.ravel()
    indices = np.argsort(domain)
    for k in intervals.keys():
        tau_hat = intervals[k]
        tau_mean = tau_hat["mean"].mean(0)
        tau_top = tau_hat["top"].mean(0) + 2 * tau_hat["mean"].std(0)
        tau_bottom = tau_hat["bottom"].mean(0) - 2 * tau_hat["mean"].std(0)
        plotting.pretty_interval(
            x=ds_train.x.ravel(),
            t=ds_train.t,
            domain=domain[indices],
            tau_true=tau_true[indices],
            tau_mean=tau_mean[indices],
            tau_top=tau_top[indices],
            tau_bottom=tau_bottom[indices],
            legend_title=f"$\log \Gamma = {k}$",
            file_path=output_dir / f"gamma-{k}.png",
        )
        if k == "0.0":
            plotting.functions(
                x=ds_train.x.ravel(),
                t=ds_train.t,
                domain=domain[indices],
                tau_true=tau_true[indices],
                tau_mean=tau_hat["mean"][:, indices],
                file_path=output_dir / f"functions.png",
            )
    plotting.rainbow(
        x=ds_train.x.ravel(),
        t=ds_train.t,
        domain=ds_test.x.ravel(),
        tau_true=tau_true.ravel(),
        intervals=intervals,
        file_path=output_dir / f"rainbow.png",
    )


def plot_fillbetween_mnist(intervals, ds_train, ds_test, output_dir):
    tau_true = ds_test.mu1 - ds_test.mu0
    domain = ds_test.phi.ravel()
    indices = np.argsort(domain)
    for k in intervals.keys():
        tau_hat = intervals[k]
        tau_mean = tau_hat["mean"].mean(0)
        tau_top = tau_hat["top"].mean(0) + 2 * tau_hat["mean"].std(0)
        tau_bottom = tau_hat["bottom"].mean(0) - 2 * tau_hat["mean"].std(0)
        plotting.pretty_interval(
            x=ds_train.phi.ravel(),
            t=ds_train.t,
            domain=domain[indices],
            tau_true=tau_true[indices],
            tau_mean=tau_mean[indices],
            tau_top=tau_top[indices],
            tau_bottom=tau_bottom[indices],
            legend_title=f"$\log \Gamma = {k}$",
            file_path=output_dir / f"gamma-{k}.png",
        )


def plot_sweep(
    ignorance, sensitivity, epistemic, mode, output_dir, sensitivity_kernel=None
):
    means_ig, cis_ig = interpolate_values(ignorance["defer_rate"], ignorance[mode])
    means_se, cis_se = interpolate_values(sensitivity["defer_rate"], sensitivity[mode])
    means_ep, cis_ep = interpolate_values(epistemic["defer_rate"], epistemic[mode])
    data = {
        "Ignorance": {
            "mean": np.clip(means_ig, 1e-5, np.inf),
            "ci": cis_ig,
            "color": "C4",
            "line_style": "-",
        },
        "Sensitivity": {
            "mean": np.clip(means_se, 1e-5, np.inf),
            "ci": cis_se,
            "color": "C2",
            "line_style": "-",
        },
        "Uncertainty": {
            "mean": np.clip(means_ep, 1e-5, np.inf),
            "ci": cis_ep,
            "color": "C0",
            "line_style": "-",
        },
    }
    if sensitivity_kernel is not None:
        means_sek, cis_sek = interpolate_values(
            sensitivity_kernel["defer_rate"], sensitivity_kernel[mode]
        )
        data["Sensitivity Kernel"] = {
            "mean": np.clip(means_sek, 1e-5, np.inf),
            "ci": cis_sek,
            "color": "C2",
            "line_style": "--",
        }
    deferral_rates = [i / len(means_ig) for i in range(len(means_ig))]
    plotting.fill_between(
        x=deferral_rates,
        y=data,
        x_label="Deferral Rate",
        y_label="Recommendation Error Rate"
        if mode == "error_rate"
        else r"Standardized $\sqrt{ \epsilon_{PEHE}}$",
        alpha=0.2,
        y_scale="log" if mode == "error_rate" else "linear",
        x_lim=[0, 1.0],
        y_lim=[1e-5, 2e-1] if mode == "error_rate" else None,
        x_pad=-20,
        y_pad=-45,
        legend_loc="upper right",
        file_path=Path(output_dir) / f"{mode}.png",
    )


def update_ignorance(results, intervals, tau_true, pi_true):
    n = len(tau_true)
    pehe = []
    error_rate = []
    defer_rate = []
    tau_hat = intervals["0.0"]
    tau_mean = tau_hat["mean"].mean(0)
    pi_hat = tau_mean > 0.0
    defer_rate.append(0.0)
    pehe.append(torch.sqrt(torch.square(tau_true - tau_mean).mean(0)).item())
    error_rate.append(1 - (pi_hat == pi_true).float().mean().item())
    deferred = []
    for k in intervals.keys():
        tau_hat = intervals[k]
        tau_mean = tau_hat["mean"].mean(0)
        tau_top = tau_hat["top"].mean(0) + 2 * tau_hat["mean"].std(0)
        tau_bottom = tau_hat["bottom"].mean(0) - 2 * tau_hat["mean"].std(0)
        defer = (tau_top >= 0) * (tau_bottom <= 0)
        new_deferrals = list(set(np.where(defer)[0]).difference(set(deferred)))
        rank = torch.min(
            torch.abs(torch.cat([tau_top.unsqueeze(0), tau_bottom.unsqueeze(0)])), dim=0
        )[0]
        if len(new_deferrals) > 0:
            for idx in np.argsort(-rank[new_deferrals]):
                deferred.append(new_deferrals[idx])
                defer_rate.append(len(deferred) / n)
                keep = torch.ones_like(tau_true, dtype=torch.bool)
                keep[deferred] = False
                pehe.append(
                    torch.sqrt(
                        torch.square(tau_true[keep] - tau_mean[keep]).mean(0)
                    ).item()
                )
                error_rate.append((pi_hat[keep] != pi_true[keep]).sum().item() / n)
        pi_hat = tau_bottom > 0.0
    new_deferrals = list(set(set(range(n))).difference(set(deferred)))
    if len(new_deferrals) > 0:
        for idx in np.argsort(rank[new_deferrals]):
            deferred.append(new_deferrals[idx])
            defer_rate.append(len(deferred) / n)
            keep = torch.ones_like(tau_true, dtype=torch.bool)
            keep[deferred] = False
            pehe.append(
                torch.sqrt(torch.square(tau_true[keep] - tau_mean[keep]).mean(0)).item()
            )
            error_rate.append((pi_hat[keep] != pi_true[keep]).sum().item() / n)
    results["sweep"]["ignorance"]["pehe"].append(pehe)
    results["sweep"]["ignorance"]["error_rate"].append(error_rate)
    results["sweep"]["ignorance"]["defer_rate"].append(defer_rate)


def update_epistemic(results, intervals, tau_true, pi_true):
    n = len(tau_true)
    pehe = []
    error_rate = []
    defer_rate = []
    tau_hat = intervals["0.0"]
    tau_var = tau_hat["mean"].var(0)
    tau_var, idx_var = torch.sort(tau_var)
    tau_mean = tau_hat["mean"].mean(0)[idx_var]
    tau_true = tau_true[idx_var]
    pi_true = pi_true[idx_var]
    pi_hat = tau_mean > 0.0
    defer_rate.append(0.0)
    pehe.append(torch.sqrt(torch.square(tau_true - tau_mean).mean(0)).item())
    error_rate.append(1 - (pi_hat == pi_true).float().mean().item())
    for i in range(len(tau_true)):
        defer_rate.append(1 - (len(tau_true[: -(i + 1)]) / len(tau_true)))
        pehe.append(
            torch.sqrt(
                torch.square(tau_true[: -(i + 1)] - tau_mean[: -(i + 1)]).mean(0)
            ).item()
        )
        error_rate.append(
            (pi_hat[: -(i + 1)] != pi_true[: -(i + 1)]).float().sum().item() / n
        )
    results["sweep"]["epistemic"]["pehe"].append(pehe)
    results["sweep"]["epistemic"]["error_rate"].append(error_rate)
    results["sweep"]["epistemic"]["defer_rate"].append(defer_rate)


def update_sensitivity(results, intervals, tau_true, pi_true):
    n = len(tau_true)
    pehe = []
    error_rate = []
    defer_rate = []
    tau_hat = intervals["0.0"]
    tau_mean = tau_hat["mean"].mean(0)
    pi_hat = tau_mean > 0.0
    defer_rate.append(0.0)
    pehe.append(torch.sqrt(torch.square(tau_true - tau_mean).mean(0)).item())
    error_rate.append(1 - (pi_hat == pi_true).float().mean().item())
    deferred = []
    for k in intervals.keys():
        tau_hat = intervals[k]
        tau_mean = tau_hat["mean"].mean(0)
        tau_top = tau_hat["top"].mean(0)
        tau_bottom = tau_hat["bottom"].mean(0)
        defer = (tau_top >= 0) * (tau_bottom <= 0)
        new_deferrals = list(set(np.where(defer)[0]).difference(set(deferred)))
        rank = torch.min(
            torch.abs(torch.cat([tau_top.unsqueeze(0), tau_bottom.unsqueeze(0)])), dim=0
        )[0]
        if len(new_deferrals) > 0:
            for idx in np.argsort(-rank[new_deferrals]):
                deferred.append(new_deferrals[idx])
                defer_rate.append(len(deferred) / n)
                keep = torch.ones_like(tau_true, dtype=torch.bool)
                keep[deferred] = False
                pehe.append(
                    torch.sqrt(
                        torch.square(tau_true[keep] - tau_mean[keep]).mean(0)
                    ).item()
                )
                error_rate.append((pi_hat[keep] != pi_true[keep]).sum().item() / n)
        pi_hat = tau_bottom > 0.0
    new_deferrals = list(set(set(range(n))).difference(set(deferred)))
    if len(new_deferrals) > 0:
        for idx in np.argsort(rank[new_deferrals]):
            deferred.append(new_deferrals[idx])
            defer_rate.append(len(deferred) / n)
            keep = torch.ones_like(tau_true, dtype=torch.bool)
            keep[deferred] = False
            pehe.append(
                torch.sqrt(torch.square(tau_true[keep] - tau_mean[keep]).mean(0)).item()
            )
            error_rate.append((pi_hat[keep] != pi_true[keep]).sum().item() / n)
    results["sweep"]["sensitivity"]["pehe"].append(pehe)
    results["sweep"]["sensitivity"]["error_rate"].append(error_rate)
    results["sweep"]["sensitivity"]["defer_rate"].append(defer_rate)


def interpolate_values(deferral_rate, error_rate):
    means = []
    cis = []
    for i in range(len(deferral_rate)):
        means.append(error_rate[i, :].mean())
        se = stats.sem(error_rate[i, :])
        cis.append(se)
    return np.asarray(means), np.asarray(cis)


def update_summaries(
    summary, dataset, intervals, pi_true, epistemic_uncertainty=False, lt=True
):
    risk = float(
        utils.policy_risk(pi=pi_true, y1=dataset.y1.ravel(), y0=dataset.y0.ravel(),)
    )
    summary["policy_risk"]["risk"]["true"].append(risk)
    for k in GAMMAS.keys():
        tau_hat = intervals[k]
        pi_hat = policy(
            tau_hat=tau_hat, epistemic_uncertainty=epistemic_uncertainty, lt=lt
        )
        risk_hat = utils.policy_risk(
            pi=pi_hat, y1=dataset.y1.ravel(), y0=dataset.y0.ravel()
        )
        summary["policy_risk"]["risk"][k].append(float(risk_hat.numpy()))
        summary["policy_risk"]["error"][k].append(
            float(torch.square(risk_hat - risk).numpy())
        )


def policy(tau_hat, epistemic_uncertainty, lt=True):
    if lt:
        tau_top = (
            tau_hat["top"].mean(0) + 2 * tau_hat["mean"].std(0)
            if epistemic_uncertainty
            else tau_hat["top"].mean(0)
        )
        pi = (tau_top < 0).float()
    else:
        tau_bottom = (
            tau_hat["bottom"].mean(0) - 2 * tau_hat["mean"].std(0)
            if epistemic_uncertainty
            else tau_hat["bottom"].mean(0)
        )
        pi = (tau_bottom >= 0).float()
    return pi


GAMMAS = {
    "0.0": math.exp(0.0),
    "0.1": math.exp(0.1),
    "0.2": math.exp(0.2),
    "0.5": math.exp(0.5),
    "0.7": math.exp(0.7),
    "1.0": math.exp(1.0),
    "1.2": math.exp(1.2),
    "1.5": math.exp(1.5),
    "2.0": math.exp(2.0),
    "2.5": math.exp(2.5),
    "3.0": math.exp(3.0),
    "3.5": math.exp(3.5),
    "4.0": math.exp(4.0),
    "4.5": math.exp(4.5),
    "5.0": math.exp(5.0),
    "5.5": math.exp(5.5),
    "6.0": math.exp(6.0),
    "6.5": math.exp(6.5),
    "7.0": math.exp(7.0),
    "7.5": math.exp(7.5),
    "8.0": math.exp(8.0),
    "8.5": math.exp(8.5),
    "9.0": math.exp(9.0),
    "9.5": math.exp(9.5),
    "10.": math.exp(10.0),
}

DEFERRAL_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

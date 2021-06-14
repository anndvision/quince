import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", palette="colorblind")
params = {
    "figure.constrained_layout.use": True,
    "axes.labelsize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 18,
    "legend.title_fontsize": 18,
    "font.size": 18,
}
plt.rcParams.update(params)

_FUNCTION_COLOR = "#ad8bd6"


def fill_between(
    x,
    y,
    x_label,
    y_label,
    alpha=0.2,
    y_scale="log",
    x_lim=[0, 1.0],
    y_lim=[3e-4, 5],
    x_pad=-20,
    y_pad=-45,
    legend_loc="upper right",
    file_path=None,
):
    _ = plt.figure(figsize=(682 / 72, 512 / 72), dpi=72)
    for k, v in y.items():
        _ = plt.plot(x, v["mean"], c=v["color"], ls=v["line_style"], label=k)
        _ = plt.fill_between(
            x,
            v["mean"] - v["ci"],
            v["mean"] + v["ci"],
            alpha=alpha,
            color=v["color"],
        )
    _ = plt.xlabel(x_label)
    _ = plt.ylabel(y_label)
    _ = plt.yscale(y_scale)
    _ = plt.xlim(x_lim)
    _ = plt.ylim(y_lim)
    _ = plt.tick_params(axis="x", direction="in", pad=x_pad)
    _ = plt.tick_params(axis="y", direction="in", pad=y_pad)
    _ = plt.legend(loc=legend_loc)
    _ = plt.savefig(file_path, dpi=300)
    _ = plt.close()


def errorbar(
    x,
    y,
    y_err,
    x_label,
    y_label,
    marker_label=None,
    x_pad=-20,
    y_pad=-45,
    legend_loc="upper left",
    file_path=None,
):
    _ = plt.figure(figsize=(682 / 72, 512 / 72), dpi=72)
    plt.errorbar(
        x,
        y,
        yerr=y_err,
        linestyle="None",
        marker="o",
        elinewidth=1.0,
        capsize=2.0,
        label=marker_label,
    )
    lim = max(np.abs(x.min()), np.abs(x.max())) * 1.1
    r = np.arange(-lim, lim, 0.1)
    _ = plt.plot(r, r, label="Ground Truth")
    _ = plt.tick_params(axis="x", direction="in", pad=x_pad)
    _ = plt.tick_params(axis="y", direction="in", pad=y_pad)
    _ = plt.xlabel(x_label)
    _ = plt.ylabel(y_label)
    _ = plt.ylim([-lim, lim])
    _ = plt.legend(loc=legend_loc)
    _ = plt.savefig(file_path, dpi=300)
    _ = plt.close()


def pretty_interval(
    x,
    t,
    domain,
    tau_true,
    tau_mean,
    tau_top,
    tau_bottom,
    legend_title=None,
    legend_loc=None,
    file_path=None,
):
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(482 / 72, 512 / 72),
        dpi=300,
        gridspec_kw={"height_ratios": [1, 3]},
    )
    density_axis = ax[0]
    data_axis = ax[1]
    control_color = "C0"
    treatment_color = "C4"

    _ = sns.histplot(
        x=x[t == 0].ravel(),
        bins=64,
        color=control_color,
        fill=True,
        alpha=0.5,
        label=r"$p_{\mathcal{D}}(\mathbf{x} | \mathrm{t}=0)$",
        ax=density_axis,
        stat="density",
    )
    _ = sns.histplot(
        x=x[t == 1].ravel(),
        bins=64,
        color=treatment_color,
        fill=True,
        alpha=0.5,
        label=r"$p_{\mathcal{D}}(\mathbf{x} | \mathrm{t}=1)$",
        ax=density_axis,
        stat="density",
    )
    _ = density_axis.tick_params(
        axis="x", which="both", left=False, right=False, labelbottom=False
    )
    _ = density_axis.set_ylabel("")
    _ = density_axis.set_xlim([-3.5, 3.5])
    _ = density_axis.legend(loc="upper left")

    tau_top = 1.5 * (tau_top / tau_true.max())
    tau_bottom = 1.5 * (tau_bottom / tau_true.max())
    tau_mean = 1.5 * (tau_mean / tau_true.max())
    tau_true = 1.5 * (tau_true / tau_true.max())
    _ = data_axis.plot(
        domain,
        tau_true,
        color="black",
        lw=6,
        ls=":",
        label=r"$\tau(\mathbf{x})$",
    )
    _ = data_axis.fill_between(
        domain,
        tau_top,
        tau_bottom,
        color=_FUNCTION_COLOR,
        alpha=0.5,
        label=r"$\widehat{\tau}(\mathbf{x})$ range",
    )
    _ = data_axis.plot(
        domain,
        tau_mean,
        color=_FUNCTION_COLOR,
        ls="-",
        alpha=1.0,
        label=r"$\widehat{\tau}(\mathbf{x})$",
    )
    _ = data_axis.set_xlabel(r"$\mathbf{x}$")
    _ = data_axis.set_ylim([-1.5, 1.8])
    _ = data_axis.set_xlim([-3.5, 3.5])
    _ = data_axis.legend(loc="upper left")
    _ = plt.savefig(file_path, dpi=300)
    _ = plt.close()


def functions(
    x,
    t,
    domain,
    tau_true,
    tau_mean,
    file_path=None,
):
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(482 / 72, 512 / 72),
        dpi=300,
        gridspec_kw={"height_ratios": [1, 3]},
    )
    density_axis = ax[0]
    data_axis = ax[1]
    control_color = "C0"
    treatment_color = "C4"

    _ = sns.histplot(
        x=x[t == 0].ravel(),
        bins=64,
        color=control_color,
        fill=True,
        alpha=0.5,
        label=r"$p_{\mathcal{D}}(\mathbf{x} | \mathrm{t}=0)$",
        ax=density_axis,
        stat="density",
    )
    _ = sns.histplot(
        x=x[t == 1].ravel(),
        bins=64,
        color=treatment_color,
        fill=True,
        alpha=0.5,
        label=r"$p_{\mathcal{D}}(\mathbf{x} | \mathrm{t}=1)$",
        ax=density_axis,
        stat="density",
    )
    _ = density_axis.tick_params(
        axis="x", which="both", left=False, right=False, labelbottom=False
    )
    _ = density_axis.set_ylabel("")
    _ = density_axis.set_xlim([-3.5, 3.5])
    _ = density_axis.legend(loc="upper left")

    tau_mean = 1.5 * (tau_mean / tau_true.max())
    tau_true = 1.5 * (tau_true / tau_true.max())
    _ = data_axis.plot(
        domain,
        tau_true,
        color="black",
        lw=6,
        ls=":",
        label=r"$\tau(\mathbf{x})$",
    )
    _ = data_axis.plot(
        domain,
        tau_mean.mean(0),
        color=_FUNCTION_COLOR,
        lw=1,
        ls="-",
        alpha=1.0,
        label=r"$\widehat{\tau}_{\mathbf{\omega}}(\mathbf{x})$, $\mathbf{\omega} \sim q(\mathbf{\omega} \mid \mathcal{D})$",
    )
    _ = data_axis.plot(
        domain,
        tau_mean.transpose(1, 0),
        color=_FUNCTION_COLOR,
        lw=1,
        ls="-",
        alpha=0.5,
    )
    _ = data_axis.set_xlabel(r"$\mathbf{x}$")
    _ = data_axis.set_ylim([-1.5, 1.8])
    _ = data_axis.set_xlim([-3.5, 3.5])
    _ = data_axis.legend(loc="upper left")
    _ = plt.savefig(file_path, dpi=300)
    _ = plt.close()


def rainbow(x, t, domain, tau_true, intervals, file_path):
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(482 / 72, 512 / 72),
        dpi=300,
        gridspec_kw={"height_ratios": [1, 3]},
    )
    density_axis = ax[0]
    data_axis = ax[1]
    control_color = "C0"
    treatment_color = "C4"

    indices = np.argsort(domain)
    tau = 1.5 * (tau_true / tau_true.max())

    _ = sns.histplot(
        x=x[t == 0].ravel(),
        bins=64,
        color=control_color,
        fill=True,
        alpha=0.5,
        label=r"$p_{\mathcal{D}}(\mathbf{x} | \mathrm{t}=0)$",
        ax=density_axis,
        stat="density",
    )
    _ = sns.histplot(
        x=x[t == 1].ravel(),
        bins=64,
        color=treatment_color,
        fill=True,
        alpha=0.5,
        label=r"$p_{\mathcal{D}}(\mathbf{x} | \mathrm{t}=1)$",
        ax=density_axis,
        stat="density",
    )
    _ = density_axis.tick_params(
        axis="x", which="both", left=False, right=False, labelbottom=False
    )
    _ = density_axis.set_ylabel("")
    _ = density_axis.set_xlim([-3.5, 3.5])
    _ = density_axis.legend(loc="upper left")

    _ = data_axis.plot(
        domain[indices],
        tau[indices],
        color="black",
        lw=6,
        ls=":",
        label=r"$\tau(\mathbf{x})$",
    )
    _ = data_axis.plot(
        domain[indices],
        1.5 * (intervals["0.0"]["mean"].mean(0)[indices] / tau_true.max()),
        color=_FUNCTION_COLOR,
        ls="-",
        alpha=1.0,
        label=r"$\Gamma=1.0$",
    )
    _ = data_axis.fill_between(
        domain[indices],
        1.5 * (intervals["0.5"]["top"].mean(0)[indices] / tau_true.max()),
        1.5 * (intervals["0.0"]["mean"].mean(0)[indices] / tau_true.max()),
        color=_FUNCTION_COLOR,
        alpha=0.4,
        hatch="\\",
        label=r"$\Gamma=1.7$",
    )
    _ = data_axis.fill_between(
        domain[indices],
        1.5 * (intervals["0.0"]["mean"].mean(0)[indices] / tau_true.max()),
        1.5 * (intervals["0.5"]["bottom"].mean(0)[indices] / tau_true.max()),
        color=_FUNCTION_COLOR,
        hatch="\\",
        alpha=0.4,
    )
    _ = data_axis.fill_between(
        domain[indices],
        1.5 * (intervals["1.0"]["top"].mean(0)[indices] / tau_true.max()),
        1.5 * (intervals["0.5"]["top"].mean(0)[indices] / tau_true.max()),
        color=_FUNCTION_COLOR,
        alpha=0.6,
        hatch="/",
        label=r"$\Gamma=2.7$",
    )
    _ = data_axis.fill_between(
        domain[indices],
        1.5 * (intervals["0.5"]["bottom"].mean(0)[indices] / tau_true.max()),
        1.5 * (intervals["1.0"]["bottom"].mean(0)[indices] / tau_true.max()),
        color=_FUNCTION_COLOR,
        hatch="/",
        alpha=0.6,
    )
    _ = data_axis.fill_between(
        domain[indices],
        1.5 * (intervals["1.5"]["top"].mean(0)[indices] / tau_true.max()),
        1.5 * (intervals["1.0"]["top"].mean(0)[indices] / tau_true.max()),
        color=_FUNCTION_COLOR,
        alpha=0.2,
        hatch="|",
        label=r"$\Gamma=4.5$",
    )
    _ = data_axis.fill_between(
        domain[indices],
        1.5 * (intervals["1.0"]["bottom"].mean(0)[indices] / tau_true.max()),
        1.5 * (intervals["1.5"]["bottom"].mean(0)[indices] / tau_true.max()),
        color=_FUNCTION_COLOR,
        hatch="|",
        alpha=0.2,
    )
    _ = data_axis.set_xlabel(r"$\mathbf{x}$")
    _ = data_axis.set_ylim([-1.5, 1.8])
    _ = data_axis.set_xlim([-3.5, 3.5])
    _ = data_axis.legend(loc="upper left")
    _ = plt.savefig(file_path, dpi=300)
    _ = plt.close()

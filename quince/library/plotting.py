import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", palette="colorblind")
params = {
    "figure.constrained_layout.use": True,
    "axes.labelsize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 22,
    "legend.title_fontsize": 22,
    "font.size": 24,
}
plt.rcParams.update(params)


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
        _ = plt.plot(x, v["mean"], lw=8, c=v["color"], ls=v["line_style"], label=k)
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
    legend_loc = (0.07, 0.55) if legend_loc is None else legend_loc
    tau_top = 1.5 * (tau_top / tau_true.max())
    tau_bottom = 1.5 * (tau_bottom / tau_true.max())
    tau_mean = 1.5 * (tau_mean / tau_true.max())
    tau_true = 1.5 * (tau_true / tau_true.max())
    _ = plt.figure(figsize=(682 / 72, 512 / 72), dpi=72)
    _ = plt.hist(
        [x[t == 0], x[t == 1]],
        bins=50,
        density=True,
        alpha=0.4,
        hatch="X",
        stacked=True,
        label=[
            "$p_{\mathcal{D}}(\mathbf{x}, \mathrm{t}=0)$",
            "$p_{\mathcal{D}}(\mathbf{x}, \mathrm{t}=1)$",
        ],
        color=["C0", "C1"],
    )
    _ = plt.plot(
        domain,
        tau_true,
        color="black",
        lw=6,
        ls=":",
        label=r"$\tau(\mathbf{x})$",
    )
    _ = plt.fill_between(
        domain,
        tau_top,
        tau_bottom,
        color="C6",
        alpha=0.5,
        label=r"$\widehat{\tau}(\mathbf{x})$ range",
    )
    _ = plt.plot(
        domain,
        tau_mean,
        color="C0",
        lw=5,
        ls="-",
        alpha=1.0,
        label=r"$\widehat{\tau}(\mathbf{x})$ biased",
    )
    _ = plt.xlabel(r"$\mathbf{x}$")
    _ = plt.ylim([-1.15, 1.6])
    _ = plt.xlim([-3.8, 3.5])
    _ = plt.tick_params(axis="x", direction="in", pad=-20)
    _ = plt.tick_params(axis="y", direction="in", pad=-45)
    _ = plt.legend(loc=legend_loc, title=legend_title)
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
    tau_mean = 1.5 * (tau_mean / tau_true.max())
    tau_true = 1.5 * (tau_true / tau_true.max())
    _ = plt.figure(figsize=(682 / 72, 512 / 72), dpi=72)
    _ = plt.hist(
        [x[t == 0], x[t == 1]],
        bins=50,
        density=True,
        alpha=0.4,
        hatch="X",
        stacked=True,
        label=[
            "$p_{\mathcal{D}}(\mathbf{x}, \mathrm{t}=0)$",
            "$p_{\mathcal{D}}(\mathbf{x}, \mathrm{t}=1)$",
        ],
        color=["C0", "C1"],
    )
    _ = plt.plot(
        domain,
        tau_true,
        color="black",
        lw=6,
        ls=":",
        label=r"$\tau(\mathbf{x})$",
    )
    _ = plt.plot(
        domain,
        tau_mean.mean(0),
        color="C0",
        lw=1,
        ls="-",
        alpha=1.0,
        label=r"$\widehat{\tau}_{\mathbf{\omega}}(\mathbf{x})$, $\mathbf{\omega} \sim q(\mathbf{\omega} \mid \mathcal{D})$",
    )
    _ = plt.plot(
        domain,
        tau_mean.transpose(1, 0),
        color="C0",
        lw=1,
        ls="-",
        alpha=0.5,
    )
    _ = plt.xlabel(r"$\mathbf{x}$")
    _ = plt.ylim([-1.15, 1.6])
    _ = plt.xlim([-3.8, 3.5])
    _ = plt.tick_params(axis="x", direction="in", pad=-20)
    _ = plt.tick_params(axis="y", direction="in", pad=-45)
    _ = plt.legend(loc=(0.07, 0.68))
    _ = plt.savefig(file_path, dpi=300)
    _ = plt.close()


def rainbow(x, t, domain, tau_true, intervals, file_path):
    indices = np.argsort(domain)
    tau = 1.5 * (tau_true / tau_true.max())
    _ = plt.figure(figsize=(682 / 72, 512 / 72), dpi=72)
    _ = plt.hist(
        [x[t == 0], x[t == 1]],
        bins=50,
        density=True,
        alpha=0.4,
        hatch="X",
        stacked=True,
        label=[
            "$p_{\mathcal{D}}(\mathbf{x}, \mathrm{t}=0)$",
            "$p_{\mathcal{D}}(\mathbf{x}, \mathrm{t}=1)$",
        ],
        color=["C0", "C1"],
    )
    _ = plt.plot(
        domain[indices],
        tau[indices],
        color="black",
        lw=6,
        ls=":",
        label=r"$\tau(\mathbf{x})$",
    )
    _ = plt.plot(
        domain[indices],
        1.5 * (intervals["0.00"]["mean"].mean(0)[indices] / tau_true.max()),
        color="C0",
        lw=6,
        ls="-",
        alpha=1.0,
        label=r"$\Gamma=1.0$",
    )
    _ = plt.fill_between(
        domain[indices],
        1.5 * (intervals["0.50"]["top"].mean(0)[indices] / tau_true.max()),
        1.5 * (intervals["0.00"]["mean"].mean(0)[indices] / tau_true.max()),
        color="C4",
        alpha=0.5,
        hatch="///",
        label=r"$\Gamma=1.7$",
    )
    _ = plt.fill_between(
        domain[indices],
        1.5 * (intervals["0.00"]["mean"].mean(0)[indices] / tau_true.max()),
        1.5 * (intervals["0.50"]["bottom"].mean(0)[indices] / tau_true.max()),
        color="C4",
        hatch="///",
        alpha=0.5,
    )
    _ = plt.fill_between(
        domain[indices],
        1.5 * (intervals["1.00"]["top"].mean(0)[indices] / tau_true.max()),
        1.5 * (intervals["0.50"]["top"].mean(0)[indices] / tau_true.max()),
        color="C9",
        alpha=0.5,
        hatch="//",
        label=r"$\Gamma=2.7$",
    )
    _ = plt.fill_between(
        domain[indices],
        1.5 * (intervals["0.50"]["bottom"].mean(0)[indices] / tau_true.max()),
        1.5 * (intervals["1.00"]["bottom"].mean(0)[indices] / tau_true.max()),
        color="C9",
        hatch="//",
        alpha=0.5,
    )
    _ = plt.fill_between(
        domain[indices],
        1.5 * (intervals["1.50"]["top"].mean(0)[indices] / tau_true.max()),
        1.5 * (intervals["1.00"]["top"].mean(0)[indices] / tau_true.max()),
        color="C6",
        alpha=0.5,
        hatch="/",
        label=r"$\Gamma=4.5$",
    )
    _ = plt.fill_between(
        domain[indices],
        1.5 * (intervals["1.00"]["bottom"].mean(0)[indices] / tau_true.max()),
        1.5 * (intervals["1.50"]["bottom"].mean(0)[indices] / tau_true.max()),
        color="C6",
        hatch="/",
        alpha=0.5,
    )
    _ = plt.xlabel(r"$\mathbf{x}$")
    _ = plt.ylim([-1.15, 1.6])
    _ = plt.xlim([-3.8, 3.5])
    _ = plt.tick_params(axis="x", direction="in", pad=-20)
    _ = plt.tick_params(axis="y", direction="in", pad=-45)
    _ = plt.legend(loc=(0.07, 0.5))
    _ = plt.savefig(file_path, dpi=300)
    _ = plt.close()

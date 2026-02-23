import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def heatmap_proportions(
    df: pd.DataFrame, path: str | None = None, figsize: tuple[float, float] = (8, 0.2),
    dpi: int = 120, cmap: str = "bwr", show: bool = True
):
    f, ax = plt.subplots(1, 1, figsize=(figsize[0], df.shape[0] * figsize[1]), dpi=dpi)
    sns.heatmap(
        df,
        xticklabels=df.columns.tolist(),
        yticklabels=df.index.tolist(),
        annot=True,
        ax=ax, cmap=cmap,
        vmin=0
    )

    if path:
        plt.savefig(path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def bar_proportions(
    melt: pd.DataFrame, path: str | None = None,
    figsize: tuple[float, float] = (0.4, 10), dpi: int = 100, show: bool = True,
    hue_order: list[str] | None = None
):
    melt["est_pct"] = melt["est"] * 100.0

    n_samples = len(melt["sample"].unique())
    
    f, ax = plt.subplots(figsize=(figsize[0], figsize[1] * n_samples + 2), dpi=dpi)
    import seaborn as sns

    sns.barplot(
        data=melt, y="sample", hue="cell_type", x="est_pct", ax=ax, dodge=True,
        hue_order=hue_order
    )

    ax.errorbar(
        x=melt["est_pct"],
        y=np.array([p.get_y() + p.get_height() * 0.5 for p in ax.patches][:melt.shape[0]]),  # type: ignore
        xerr=[melt["min"] * 100.0, melt["max"] * 100.0],
        fmt="none", ecolor="black", elinewidth=figsize[1], capsize=2, capthick=0.5
    )

    sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    )

    plt.xlabel("Cell Proportion [%]")
    plt.ylabel("Sample")

    if path is not None:
        plt.savefig(path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def benchmark_scatter(
    df: pd.DataFrame, x: str = "true", y: str = "est", hue: str | None = "cell_type",
    style: str | None = "sample", figsize=(8, 8), dpi: int = 100, path: str | None = None,
    log: bool = False, legend: bool = True, show: bool = True
):
    rmse = ((df[x] - df[y]) ** 2).mean() ** 0.5
    mad = (df[x] - df[y]).abs().mean()
    r = df[x].corr(df[y])
    
    f, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    if "min" in df.columns:
        ax.errorbar(
            df[x], df[y], yerr=df[["min", "max"]].values.T,
            fmt=",", alpha=.7, zorder=1, c="#c3c3c3",
            label="95% CI", capsize=2
        )

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        style=style,
        edgecolor=(0, 0, 0, 0.8),
        color=(1, 1, 1, 0),
        linewidth=1,
        zorder=2,
        s=80,
    ).set_title(f"RMSE: {rmse:0.2f} MAD: {mad:0.2f} R: {r:0.2f}")

    ax.set_xlabel("True Proportion")
    ax.set_ylabel("Estimated Proportion")
    if log:
        ax.set_yscale("log")
        ax.set_xscale("log")

    ax.plot([0, 1], [0, 1], color="royalblue", label="y=x")

    _legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, ncols=1)
    if legend is False:
        _legend.remove()

    if path:
        plt.savefig(path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def benchmark_corr_per_cell_type(
    df: pd.DataFrame, x: str = "true", y: str = "est", hue: str | None = "cell_type",
    figsize=(8, 8), dpi: int = 100, path: str | None = None,
    log: bool = False, legend: bool = True, show: bool = True
):
    rmse = ((df[x] - df[y]) ** 2).mean() ** 0.5
    r = df[x].corr(df[y])
    
    f, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        edgecolor=(0, 0, 0, 0.8),
        color=(1, 1, 1, 0),
        linewidth=1,
        zorder=2,
        s=80,
        legend=False
    )
    for cell_type, group in df.groupby(hue) if hue is not None else [("", df)]:
        rmse = ((group[x] - group[y]) ** 2).mean() ** 0.5
        r = group[x].corr(group[y])
        sns.regplot(
            data=group, x=x, y=y, ax=ax, label=f"{cell_type} (R: {r:0.2f} RMSE: {rmse:0.2f})",
        )

    ax.set_xlabel("True Proportion")
    ax.set_ylabel("Estimated Proportion")
    if log:
        ax.set_yscale("log")
        ax.set_xscale("log")

    ax.plot([0, 1], [0, 1], color="royalblue", label="y=x")

    _legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, ncols=1)
    if legend is False:
        _legend.remove()

    if path:
        plt.savefig(path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def loss_plot(
    losses: list, title: str | None = None, log: bool = False, path: str | None = None, figsize: tuple[int, int] = (8, 4), dpi: int = 120, show: bool = True
):
    f, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    import seaborn as sns
    sns.lineplot(losses, ax=ax)

    if log:
        ax.set_yscale("log")

    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")

    if title:
        ax.set_title(title)

    if path is not None:
        plt.savefig(path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
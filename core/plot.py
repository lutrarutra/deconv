import tqdm, os

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import scanpy as sc
import pandas as pd

from base import mkdir



def umap_plot(decon, keys, min_dist=0.5, spread=1.0, n_neighbors=15, n_pcs=None, fmt="png", show=False, figsize=(5,5), dpi=100):
    sc.settings.set_figure_params(dpi=dpi, facecolor='white', figsize=figsize)
    sc.pp.neighbors(decon.sadata, random_state=0, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(decon.sadata, min_dist=min_dist, spread=spread, random_state=0)
    sc.pl.umap(decon.sadata,  color=keys, ncols=2, frameon=False, show=show, save=f".{fmt}" if not show else None)


def _plot(df, ax, x, y, logx=False, logy=False, k_poly_fit=None, hue=None, style=None, size=None,):
    if logx:
        _x = np.log(df[x])
    else:
        _x = df[x]
    
    if logy:
        _y = np.log(df[y])
    else:
        _y = df[y]

    sns.scatterplot(
        data=df, x=x, y=y, hue=hue, style=style, size=size,
        edgecolor=(0,0,0,0.8), color=(1,1,1,0), linewidth=1, ax=ax
    )

    if k_poly_fit:
        fit = np.poly1d(np.polyfit(_x, _y, k_poly_fit))

        ax.plot(
            np.exp(_x) if logx else _x, np.exp(fit(_x)) if logy else fit(_x),
            ".", markeredgecolor="royalblue", markeredgewidth=0.5, color=(1,1,1,0),
        )

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")


def plot(decon, x, y, logx=False, logy=False, k_poly_fit=None, hue=None, style=None, size=None, path=None, fmt="png", figsize=(8,8), dpi=100, separate_ct=False, xlim=None, ylim=None):
    plt.style.use("ggplot")

    if separate_ct:
        f, ax = plt.subplots(nrows=decon.n_cell_types, figsize=(figsize[0], figsize[1] * decon.n_cell_types), dpi=dpi)
        
        for i, cell_type in enumerate(decon.cell_types):
            # df = decon.sadata[decon.sadata.obs[decon.params["label_key"]] == cell_type].var
            _plot(decon.sadata.var, ax[i], f"{x}_{cell_type}", f"{y}_{cell_type}", logx=logx, logy=logy, k_poly_fit=k_poly_fit, hue=hue, style=style, size=size)
            ax[i].set_title(f"Cell Type: {cell_type}")
            if xlim != None:
                ax[i].set_xlim(xlim)
            if ylim != None:
                ax[i].set_ylim(ylim)

            ax[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            

    else:
        f, ax = plt.subplots(figsize=figsize, dpi=dpi)
        _plot(decon.sadata.var, ax, x, y, logx=logx, logy=logy, k_poly_fit=k_poly_fit, hue=hue, style=style, size=size)
        if xlim != None:
            ax.set_xlim(xlim)
        if ylim != None:
            ax.set_ylim(ylim)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
    

    if path:
        plt.savefig(os.path.join(path, f"{x}_{y}_plot.{fmt}"), bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        



def dispersion_plot(
    decon, hue="ct_marker_abs_score_max", style="dispersion_outlier",
    size=None, path=None, fmt="png", figsize=(8,8), dpi=100, separate_ct=False):

    plot(
        decon, x="mu", y="cv2", logx=True, logy=True, hue=hue,
        style=style, size=size, path=path, fmt=fmt, figsize=figsize,
        dpi=dpi, k_poly_fit=2, separate_ct=separate_ct
    )


def dropout_plot(
    decon, hue="ct_marker_abs_score_max", style="dropout_outlier",
    size=None, path=None, layer=None, fmt="png", figsize=(8,8),
    dpi=100, separate_ct=False, xlim=None, ylim=(-0.1, 1.1)):

    plot(
        decon, x="nanmu", y="dropout", logx=True, hue=hue,
        style=style, size=size, path=path, fmt=fmt, figsize=figsize,
        dpi=dpi, k_poly_fit=2, separate_ct=separate_ct,
        ylim=ylim, xlim=xlim
    )


def marker_plot(decon, top_n_genes=20, show=False, fmt="png", figsize=(8,8), dpi=100):
    sc.settings.set_figure_params(dpi=dpi, facecolor='white', figsize=figsize)
    sc.tl.dendrogram(decon.sadata, groupby=decon.label_key)
    sc.pl.heatmap(decon.sadata, decon.marker_genes[:top_n_genes], groupby=decon.label_key, cmap="viridis", dendrogram=True, show=show, save=f".{fmt}" if not show else None)


def pseudo_bulk_plot(decon, hue=None, style=None, dir=None, fmt="png", figsize=(8,8), dpi=100):
    plt.style.use("ggplot")

    if dir:
        pbar = tqdm.tqdm(enumerate(decon.bulk_sample_cols))
    else:
        pbar = enumerate(decon.bulk_sample_cols)

    for i, col in pbar:
        f, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        sns.scatterplot(
            data=decon.sadata.var, x="pseudo", y=col, hue=hue, style=style,
            edgecolor=(0,0,0,1), color=(1,1,1,0), ax=ax, linewidth=1
        ).set_title(col)

        ax.plot([0, decon.sadata.var[col].max()], [0, decon.sadata.var[col].max()], label="y=x", c="royalblue")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Pseudo Bulk")
        ax.set_ylabel("Bulk")

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        if dir:
            mkdir(dir)
            plt.savefig(os.path.join(dir, f"{col}.{fmt}"), bbox_inches="tight")
            plt.close()
        else:
            plt.show()

def prediction_plot(mean, Y, path=None, figsize=(8,8), dpi=100):
    plt.style.use("ggplot")
    
    f, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    sns.scatterplot(x=mean, y=Y, edgecolor=(0,0,0,1), color=(1,1,1,0), ax=ax, linewidth=1)
    ax.plot([0, Y.max()], [0, Y.max()], label="y=x", c="royalblue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Deconvoluted Bulk")
    ax.set_ylabel("Bulk")

    if path:
        mkdir(os.path.dirname(path))
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
def proportions_heatmap(df, path=None, figsize=(8, 0.2), dpi=100):
    f, ax = plt.subplots(1, 1, figsize=(figsize[0], df.shape[0] * figsize[1]), dpi=dpi)
    sns.heatmap(df, xticklabels=df.columns, yticklabels=df.index, annot=True, ax=ax, cmap="bwr", vmin=0, vmax=1)

    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def scatter_check(true_df, est_df, figsize=(8,8), dpi=100, path=None):
    samples = true_df.index.values
    nrows = int(np.ceil(len(samples)/2))

    f, ax = plt.subplots(2, nrows, figsize=figsize, dpi=dpi)
    for i, sample in enumerate(samples):
        sns.scatterplot(x=true_df.loc[sample], y=est_df.loc[sample], edgecolor=(0,0,0,1), color=(1,1,1,0), ax=ax[i], linewidth=1)

    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def bar_proportions(proportions_df, path=None, figsize=(0.4,10), dpi=100):
    palette=sns.color_palette("hls", n_colors=len(proportions_df.columns))

    fig, ax = plt.subplots(figsize=(len(proportions_df.index) * figsize[0], figsize[1]), dpi=dpi)

    proportions_df.plot(kind="bar", stacked=True, ax=ax, color=palette, width=0.8, alpha=1.0)

    plt.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=proportions_df.columns)
    ax.grid(False)
    
    # return ax.containers
    for bars in ax.containers:
        labels = []
        for val in bars.datavalues:
            if val > 0.01:
                labels.append(f"{val*100:.0f}%")
            else:
                labels.append("")
        ax.bar_label(bars, labels=labels, label_type="center", fontsize=8, alpha=0.8)


    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def scatter_check(true_df, est_df, hue="cell_type", style=None, figsize=(8,8), dpi=100, path=None):
    df1 = pd.melt(true_df.reset_index(), id_vars=["index"], value_name="true_proportions").set_index(["index", "variable"])
    df2 = pd.melt(est_df.reset_index(), id_vars=["index"], value_name="est_proportions").set_index(["index", "variable"])
    df = pd.concat([df1, df2], axis=1).reset_index()
    df.rename(columns={"index": "sample", "variable":"cell_type"}, inplace=True)

    rmse = ((df["true_proportions"] - df["est_proportions"])**2).mean() ** .5
    mad = (df["true_proportions"] - df["est_proportions"]).abs().mean()
    r = df["true_proportions"].corr(df["est_proportions"])

    f, ax = plt.subplots(figsize=(8,8), dpi=100)
    sns.scatterplot(
        data=df, x="true_proportions", y="est_proportions",
        edgecolor=(0,0,0,0.8), color=(1,1,1,0), linewidth=1,
        hue=hue, style=style
    ).set_title(f"RMSE: {rmse:0.2f} MAD: {mad:0.2f} R: {r:0.2f}")
    
    ax.plot([0, 1], [0, 1], color="royalblue", label="y=x")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
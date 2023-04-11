import os

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import tqdm
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def umap_plot(
    decon,
    keys,
    min_dist=0.5,
    spread=1.0,
    n_neighbors=15,
    n_pcs=None,
    fmt="png",
    show=False,
    figsize=(5, 5),
    dpi=100,
):
    sc.settings.set_figure_params(scanpy=False, dpi=dpi, figsize=figsize)
    sc.pp.neighbors(decon.sadata, random_state=0, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(decon.sadata, min_dist=min_dist, spread=spread, random_state=0)
    sc.pl.umap(
        decon.sadata,
        color=keys,
        ncols=2,
        frameon=False,
        show=show,
        save=f".{fmt}" if not show else None,
    )


def _plot(
    df,
    ax,
    x,
    y,
    logx=False,
    logy=False,
    k_poly_fit=None,
    hue=None,
    style=None,
    size=None,
    palette=None,
):
    if logx:
        _x = np.log(df[x])
    else:
        _x = df[x]

    if logy:
        _y = np.log(df[y])
    else:
        _y = df[y]

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        style=style,
        size=size,
        edgecolor=(0, 0, 0, 0.8),
        color=(1, 1, 1, 0),
        linewidth=1,
        ax=ax,
        palette=palette,
    )

    if k_poly_fit:
        fit = np.poly1d(np.polyfit(_x, _y, k_poly_fit))

        ax.plot(
            np.exp(_x) if logx else _x,
            np.exp(fit(_x)) if logy else fit(_x),
            ".",
            markeredgecolor="royalblue",
            markeredgewidth=0.5,
            color=(1, 1, 1, 0),
        )

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")


def plot(
    decon,
    x,
    y,
    logx=False,
    logy=False,
    k_poly_fit=None,
    hue=None,
    style=None,
    size=None,
    path=None,
    palette=None,
    fmt="png",
    figsize=(8, 8),
    dpi=100,
    separate_ct=False,
    xlim=None,
    ylim=None,
):
    if separate_ct:
        f, ax = plt.subplots(
            nrows=decon.n_cell_types,
            figsize=(figsize[0], figsize[1] * decon.n_cell_types),
            dpi=dpi,
        )

        for i, cell_type in enumerate(decon.cell_types):
            # df = decon.sadata[decon.sadata.obs[decon.params["label_key"]] == cell_type].var
            _plot(
                decon.sadata.var,
                ax[i],
                f"{x}_{cell_type}",
                f"{y}_{cell_type}",
                logx=logx,
                logy=logy,
                k_poly_fit=k_poly_fit,
                hue=hue,
                style=style,
                size=size,
                palette=palette,
            )
            ax[i].set_title(f"Cell Type: {cell_type}")
            if xlim != None:
                ax[i].set_xlim(xlim)
            if ylim != None:
                ax[i].set_ylim(ylim)

            ax[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    else:
        f, ax = plt.subplots(figsize=figsize, dpi=dpi)
        _plot(
            decon.sadata.var,
            ax,
            x,
            y,
            logx=logx,
            logy=logy,
            k_poly_fit=k_poly_fit,
            hue=hue,
            style=style,
            size=size,
            palette=palette,
        )
        if xlim != None:
            ax.set_xlim(xlim)
        if ylim != None:
            ax.set_ylim(ylim)

        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    if path:
        plt.savefig(os.path.join(path, f"{x}_{y}_plot.{fmt}"), bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def dispersion_plot(
    decon,
    hue="ct_marker_abs_score_max",
    style="dispersion_outlier",
    size=None,
    path=None,
    fmt="png",
    figsize=(8, 8),
    dpi=100,
    separate_ct=False,
    palette=None,
):

    plot(
        decon,
        x="mu",
        y="cv2",
        logx=True,
        logy=True,
        hue=hue,
        style=style,
        size=size,
        path=path,
        fmt=fmt,
        figsize=figsize,
        dpi=dpi,
        k_poly_fit=2,
        separate_ct=separate_ct,
        palette=palette,
    )


def gene_weight_hist(
    weights, xlabel, logy=False, path=None, figsize=(8, 8), dpi=80
):
    f, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sns.histplot(weights, bins=20, ax=ax)
    ax.set_xlabel(xlabel)

    if logy:
        ax.set_yscale("log")

    if path:
        plt.savefig(path, bbox_inches="tight")
    
    return f, ax


def marker_plot(decon, top_n_genes=20, show=False, fmt="png", figsize=(8, 8), dpi=100):
    sc.settings.set_figure_params(scanpy=False, dpi=dpi, figsize=figsize)
    sc.tl.dendrogram(decon.sadata, groupby=decon.label_key)
    sc.pl.heatmap(
        decon.sadata,
        decon.marker_genes[:top_n_genes],
        swap_axes=True,
        groupby=decon.label_key,
        cmap="viridis",
        dendrogram=True,
        show=show,
        save=f".{fmt}" if not show else None,
    )


def _dendrogram(model, **kwargs):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[
            i
        ] = current_count  # axes[0][i+1].title.set_text(f"{clusterby.capitalize()} {groups[i]}")

    linkage_matrix = np.column_stack(
        [
            model.children_,
            np.linspace(0, 1, model.distances_.shape[0] + 2)[1:-1],
            counts,
        ]
    ).astype(float)

    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)


def heatmap(
    adata,
    groupby,
    categorical_features,
    var_names=None,
    rank_genes_by=None,
    free_sort_cells=False,
    n_genes=10,
    sort_cells=True,
    sort_genes=True,
    quantiles=(0.0, 1.0),
    cmap="seismic",
    figsize=(20, None),
    dpi=50,
    fig_path=None,
):
    if isinstance(categorical_features, str):
        categorical_features = [categorical_features]

    _grid = rcParams["axes.grid"]
    rcParams["axes.grid"] = False

    palettes = [
        sns.color_palette("tab10"),
        sns.color_palette("Paired"),
        sns.color_palette("Set2"),
    ]

    if var_names is None:
        rank_results = sc.tl.rank_genes_groups(
            adata,
            groupby=rank_genes_by if rank_genes_by else groupby,
            rankby_abs=True,
            method="t-test",
            copy=True,
        ).uns["rank_genes_groups"]
        var_names = np.unique(
            np.array(list(map(list, zip(*rank_results["names"]))))[
                :, :n_genes
            ].flatten()
        )
    else:
        if isinstance(var_names, list):
            var_names = np.array(var_names)

    n_cat = len(categorical_features)
    h_cat = 0.5
    n_vars = len(var_names)
    h_vars = 0.3

    if figsize[1] is None:
        figsize = (figsize[0], n_cat * h_cat + n_vars * h_vars)

    r_cat = int(h_cat * 100.0 / (n_cat * h_cat + n_vars * h_vars))
    r_vars = 100 - r_cat

    print(r_cat)
    print(r_vars)

    f = plt.figure(figsize=figsize, dpi=dpi)

    gs = f.add_gridspec(
        1 + len(categorical_features),
        2,
        hspace=0.2 / figsize[1],
        wspace=0.01,
        height_ratios=[r_cat] * len(categorical_features) + [r_vars],
        width_ratios=[1, 20],
    )

    axes = gs.subplots(sharex="col", sharey="row")

    if sort_genes:
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(
            adata[:, var_names].X.T.toarray()
        )
        gene_dendro = _dendrogram(model, ax=axes[-1, 0], orientation="right")
        gene_order = var_names[gene_dendro["leaves"]]

        icoord = np.array(gene_dendro["icoord"])
        icoord = icoord / (n_genes * 10) * n_genes
        dcoord = np.array(gene_dendro["dcoord"])
        axes[-1, 0].clear()
        for xs, ys in zip(icoord, dcoord):
            axes[-1, 0].plot(ys, xs)
    else:
        gene_order = var_names

    if sort_cells:
        adata.obs["barcode"] = pd.Categorical(adata.obs.index)

        if free_sort_cells:
            if not f"dendrogram_barcode" in adata.uns.keys():
                sc.tl.dendrogram(adata, groupby="barcode", var_names=var_names)
            cell_dendro = adata.uns["dendrogram_barcode"]
            cell_order = cell_dendro["categories_ordered"]
        else:
            cell_order = []
            for cell_type in adata.obs[groupby].cat.categories.tolist():
                # Todo: when reculcustering, the order of the cells is not preserved
                if f"{cell_type}_order" in adata.uns.keys():
                    cell_order.extend(adata.uns[f"{cell_type}_order"])
                else:
                    dendro = sc.tl.dendrogram(
                        adata[adata.obs[groupby] == cell_type],
                        groupby="barcode",
                        inplace=False,
                    )
                    cell_order.extend(dendro["categories_ordered"])
                    adata.uns[f"{cell_type}_order"] = dendro["categories_ordered"]

    else:
        cell_order = adata.obs.sort_values(groupby).index

    data = adata[cell_order, gene_order].layers["logcentered"].toarray().T
    vmin, vmax = np.quantile(data, q=quantiles)

    sns.heatmap(
        data,
        cmap=cmap,
        ax=axes[-1, -1],
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        yticklabels=gene_order,
    )

    for i, categorical_feature in enumerate(categorical_features):
        palette = palettes[i % len(palettes)]
        samples = adata[cell_order, :].obs[categorical_feature].cat.codes
        clr = [palette[s % len(palette)] for s in samples]
        axes[i][1].vlines(np.arange(len(samples)), 0, 1, colors=clr, lw=5, zorder=10)
        axes[i][1].set_yticklabels([])
        axes[i][1].set_ylim([0, 1])
        axes[i][1].patch.set_linewidth(2.0)
        axes[i][1].set_yticks([0.5])
        axes[i][1].set_yticklabels([categorical_feature.capitalize()])

        leg = f.legend(
            title=categorical_feature,
            labels=adata.obs[categorical_feature].cat.categories.tolist(),
            prop={"size": 24},
            bbox_to_anchor=(0.95, 0.9 - 0.3 * i),
            ncol=1,
            frameon=True,
            edgecolor="black",
            loc="upper left",
            facecolor="white",
        )

        plt.gca().add_artist(leg)
        palette = palettes[i % len(palettes)]
        for l, legobj in enumerate(leg.legendHandles):
            legobj.set_color(palette[l % len(palette)])
            legobj.set_linewidth(8.0)

    for ax in axes.flat:
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.set_xticklabels([])

    for ax in axes[:, 0]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    f.colorbar(
        plt.cm.ScalarMappable(
            norm=matplotlib.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0),
            cmap=cmap,
        ),
        ax=axes,
        orientation="vertical",
        fraction=0.05,
        pad=0.01,
        shrink=5.0 / figsize[1],
    )

    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight")

    plt.show()
    rcParams["axes.grid"] = _grid


def clustermap(
    adata,
    clusterby="leiden",
    categorical_features="sample",
    n_genes=10,
    cmap="seismic",
    quantiles=(0.0, 1.0),
    figsize=(25, 25),
    dpi=50,
    fig_path=None,
    min_cells=100,
):

    # Make barcode into a categorical column so that we can dendrogram by cells
    adata.obs["barcode"] = pd.Categorical(adata.obs.index)

    palettes = [
        sns.color_palette("tab10"),
        sns.color_palette("Set2"),
        sns.color_palette("Paired"),
    ]
    _grid = rcParams["axes.grid"]
    rcParams["axes.grid"] = False

    # Filter groups with more than 'min_cells' cells
    cluster_sz = adata.obs.groupby(clusterby).apply(len)
    cluster_sz = cluster_sz[cluster_sz >= min_cells]
    groups = cluster_sz.index.tolist()
    n_groups = len(groups)
    n_cells = min(cluster_sz)
    print(f"Using {n_cells} cells from {n_groups} {clusterby}s")

    # Rank genes and make 2d array
    sc.pp.neighbors(adata, random_state=0)
    rank_results = sc.tl.rank_genes_groups(
        adata, groupby=clusterby, method="t-test", copy=True
    ).uns["rank_genes_groups"]
    gene_groups = np.array(list(map(list, zip(*rank_results["names"]))))[:, :n_genes]

    if type(categorical_features) != list:
        categorical_features = [categorical_features]

    f = plt.figure(figsize=figsize, dpi=dpi)
    gs = f.add_gridspec(
        n_groups + len(categorical_features),
        n_groups + 1,
        hspace=0.05,
        wspace=0.05,
        height_ratios=[1] * len(categorical_features) + [10] * n_groups,
        width_ratios=[2] + [10] * n_groups,
    )
    axes = gs.subplots(sharex="col", sharey="row")

    # Color bar minimum and maximum
    vmin, vmax = np.quantile(adata.layers["logcentered"], q=quantiles)

    # Category label color plot
    for gi, grp in enumerate(categorical_features):
        axes[gi][0].set_yticks([0.5])
        axes[gi][0].set_yticklabels([grp.capitalize()])
        adata.obs[f"{grp}_idx"] = adata.obs[grp].cat.codes

    for i, group_i in enumerate(groups):
        # Index n_cells per cluster and top n_genes
        sample_idx = adata.obs[adata.obs[clusterby] == group_i].sample(n_cells).index
        adata_sample = adata[sample_idx, :].copy()

        dendro_info = sc.tl.dendrogram(
            adata_sample,
            groupby="barcode",
            var_names=gene_groups.flatten(),
            inplace=False,
        )
        dendro_order = dendro_info["categories_ordered"]

        # Gene hierachy
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(
            adata[:, gene_groups[i, :]].X.T.toarray()
        )
        axes[i + len(categorical_features)][0].set_ylim([0, 1])
        gene_dendro = _dendrogram(
            model, ax=axes[i + len(categorical_features), 0], orientation="right"
        )
        gene_groups[i, :] = gene_groups[i, gene_dendro["leaves"]]
        icoord = np.array(gene_dendro["icoord"])
        icoord = icoord / (n_genes * 10) * n_genes
        dcoord = np.array(gene_dendro["dcoord"])

        axes[i + len(categorical_features), 0].clear()
        for xs, ys in zip(icoord, dcoord):
            axes[i + len(categorical_features), 0].plot(ys, xs)

        # Category label color plot
        for gi, grp in enumerate(categorical_features):
            palette = palettes[gi % len(palettes)]
            samples = adata_sample[dendro_order, :].obs[f"{grp}_idx"].values
            clr = [palette[i % len(palette)] for i in samples]
            axes[gi][i + 1].vlines(
                np.arange(len(samples)), 0, 1, colors=clr, lw=5, zorder=10
            )
            axes[gi][i + 1].patch.set_edgecolor("black")
            axes[gi][i + 1].patch.set_linewidth(2.0)

        axes[0][i + 1].set_title(
            f"{clusterby.capitalize()} {groups[i]}"
            if clusterby == "leiden"
            else groups[i],
            fontsize=18,
        )
        axes[i + len(categorical_features)][0].set_ylabel(
            f"{clusterby.capitalize()} {groups[i]}"
            if clusterby == "leiden"
            else groups[i],
            fontsize=18,
        )

        for j, group_j in enumerate(groups):
            data = (
                adata[dendro_order, gene_groups[j, :]].layers["logcentered"].T.toarray()
            )
            sns.heatmap(
                data,
                ax=axes[j + len(categorical_features)][i + 1],
                cmap=cmap,
                center=0,
                vmin=vmin,
                vmax=vmax,
                cbar=False,
                yticklabels=gene_groups[j, :],
            )
            axes[j + len(categorical_features)][i + 1].patch.set_edgecolor("black")
            axes[j + len(categorical_features)][i + 1].patch.set_linewidth(2.0)

    f.colorbar(
        plt.cm.ScalarMappable(
            norm=matplotlib.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0),
            cmap=cmap,
        ),
        ax=axes,
        orientation="vertical",
        fraction=0.05,
        pad=0.05,
        shrink=0.2,
    )

    for gi, grp in enumerate(categorical_features):
        leg = f.legend(
            title=grp,
            labels=adata.obs[grp].cat.categories.tolist(),
            prop={"size": 24},
            bbox_to_anchor=(0.95, 1.0 - 0.1 - gi / 12),
        )
        plt.gca().add_artist(leg)
        palette = palettes[gi % len(palettes)]
        for l, legobj in enumerate(leg.legendHandles):
            legobj.set_color(palette[l % len(palette)])
            legobj.set_linewidth(8.0)

    # Hide frames from dendrogram column
    for i in range(axes.shape[0]):
        axes[i][0].spines["top"].set_visible(False)
        axes[i][0].spines["right"].set_visible(False)
        axes[i][0].spines["bottom"].set_visible(False)
        axes[i][0].spines["left"].set_visible(False)

    for ax in axes.flat:
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.set_xticklabels([])

    f.align_ylabels(axes[:, 0])

    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight")

    plt.show()
    rcParams["axes.grid"] = _grid


# def prediction_plot(mean, Y, path=None, figsize=(8, 8), dpi=100):
#     plt.style.use("ggplot")

#     f, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
#     sns.scatterplot(
#         x=mean, y=Y, edgecolor=(0, 0, 0, 1), color=(1, 1, 1, 0), ax=ax, linewidth=1
#     )
#     ax.plot([0, Y.max()], [0, Y.max()], label="y=x", c="royalblue")
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_xlabel("Deconvoluted Bulk")
#     ax.set_ylabel("Bulk")

#     if path:
#         mkdir(os.path.dirname(path))
#         plt.savefig(path, bbox_inches="tight")
#         plt.close()
#     else:
#         plt.show()


def proportions_heatmap(df, path=None, figsize=(8, 0.2), dpi=100):
    f, ax = plt.subplots(1, 1, figsize=(figsize[0], df.shape[0] * figsize[1]), dpi=dpi)
    sns.heatmap(
        df,
        xticklabels=df.columns,
        yticklabels=df.index,
        annot=True,
        ax=ax,
        cmap="bwr",
        vmin=0,
        vmax=1,
    )

    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



def bar_proportions(proportions_df, path=None, figsize=(0.4, 10), dpi=100):
    palette = sns.color_palette("hls", n_colors=len(proportions_df.columns))

    fig, ax = plt.subplots(
        figsize=(len(proportions_df.index) * figsize[0], figsize[1]), dpi=dpi
    )

    proportions_df.plot(
        kind="bar", stacked=True, ax=ax, color=palette, width=0.8, alpha=1.0
    )

    plt.legend(
        title="Cell Type",
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        labels=proportions_df.columns,
    )
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

def rmse(true_df, est_df):
    df1 = pd.melt(
        true_df.reset_index(), id_vars=["index"], value_name="true_proportions"
    ).set_index(["index", "variable"])

    df2 = pd.melt(
        est_df.reset_index(), id_vars=["index"], value_name="est_proportions"
    ).set_index(["index", "variable"])

    df = pd.concat([df1, df2], axis=1).reset_index()
    df.rename(columns={"index": "sample", "variable": "cell_type"}, inplace=True)

    rmse = ((df["true_proportions"] - df["est_proportions"]) ** 2).mean() ** 0.5
    return rmse


def scatter_check(
    true_df, est_df, hue="cell_type", style=None, figsize=(8, 8), dpi=100, path=None
):
    df1 = pd.melt(
        true_df.reset_index(), id_vars=["index"], value_name="true_proportions"
    ).set_index(["index", "variable"])

    df2 = pd.melt(
        est_df.reset_index(), id_vars=["index"], value_name="est_proportions"
    ).set_index(["index", "variable"])

    df = pd.concat([df1, df2], axis=1).reset_index()
    df.rename(columns={"index": "sample", "variable": "cell_type"}, inplace=True)

    rmse = ((df["true_proportions"] - df["est_proportions"]) ** 2).mean() ** 0.5
    mad = (df["true_proportions"] - df["est_proportions"]).abs().mean()
    r = df["true_proportions"].corr(df["est_proportions"])

    f, ax = plt.subplots(figsize=(8, 8), dpi=100)
    sns.scatterplot(
        data=df,
        x="true_proportions",
        y="est_proportions",
        edgecolor=(0, 0, 0, 0.8),
        color=(1, 1, 1, 0),
        linewidth=1,
        hue=hue,
        style=style,
    ).set_title(f"RMSE: {rmse:0.2f} MAD: {mad:0.2f} R: {r:0.2f}")

    ax.plot([0, 1], [0, 1], color="royalblue", label="y=x")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def xypredictions(df, hue="cell_type", style="sample", figsize=(8, 8), dpi=100, path=None, log=False):
    rmse = ((df["true"] - df["est"]) ** 2).mean() ** 0.5
    mad = (df["true"] - df["est"]).abs().mean()
    r = df["true"].corr(df["est"])
    
    f, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    if "min" in df.columns:
        ax.errorbar(
            df["true"], df["est"], yerr=df[["min", "max"]].values.T,
            fmt=",", alpha=.7, zorder=1, c="#c3c3c3",
            label="+/- sd", capsize=2
        )

    sns.scatterplot(
        data=df,
        x="true",
        y="est",
        hue="cell_type",
        style="sample",
        edgecolor=(0, 0, 0, 0.8),
        color=(1, 1, 1, 0),
        linewidth=1,
        zorder=2,
    ).set_title(f"RMSE: {rmse:0.2f} MAD: {mad:0.2f} R: {r:0.2f}")

    ax.set_xlabel("True Proportion")
    ax.set_ylabel("Estimated Proportion")
    if log:
        ax.set_yscale("log")
        ax.set_xscale("log")


    ax.plot([0, 1], [0, 1], color="royalblue", label="y=x")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return rmse, mad, r

def prediction_plot(decon, i, path=None):
    est = decon.deconvolution_module.pseudo_bulk()[i,:].cpu().numpy()
    est_bulk = np.log1p(est)
    true = decon.adata.varm["bulk"][:, i]
    true_bulk = np.log1p(true)

    rmse = ((true - est) ** 2).mean() ** 0.5
    log_rmse = ((true_bulk - est_bulk) ** 2).mean() ** 0.5

    mu = np.log1p(decon.adata.layers["counts"].mean(0))

    clr = decon.adata.varm["pseudo_factor"][:, i]
    zmin = clr.min()
    zmax = clr.max()
    norm = matplotlib.colors.TwoSlopeNorm(vmin=zmin, vmax=zmax, vcenter=0)

    f, ax = plt.subplots(1, 1, figsize=(8,8), dpi=100)

    sns.scatterplot(
        x=mu,
        y=true_bulk-est_bulk,
        c=clr,
        cmap="seismic",
        norm=norm,
        ax=ax,
        edgecolor=(0, 0, 0, 1),
        linewidth=0.8,
        s=35,
    )
    ax.axhline(0, color="royalblue", linestyle="--", linewidth=1)
    ax.set_ylabel("log1p(bulk) - log1p(est bulk)")
    ax.set_xlabel("log1p(mean gene expression)")
    ax.set_title(f"RMSE: {rmse:.2e} | log RMSE: {log_rmse:0.2f}")
    
    if path is not None:
        plt.savefig(path, bbox_inches="tight")
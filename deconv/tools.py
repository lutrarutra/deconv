import warnings
from typing import Literal

import scanpy as sc
import numpy as np
import pandas as pd

import scipy
import scipy.sparse


def fmt_c(w):
    return " ".join([f"{v:.2f}" for v in w])


def scale_log_center(adata: sc.AnnData, target_sum: float | None = None, norm_factor_key: str | None = None, exclude_highly_expressed: bool = False):
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum, key_added=norm_factor_key, exclude_highly_expressed=exclude_highly_expressed)
    adata.layers["ncounts"] = adata.X.copy()
    sc.pp.log1p(adata)


def __rank_group(adata: sc.AnnData, rank_res: pd.DataFrame, groupby: str, group: str, reference: str, score_type: Literal["welchs", "pval-lfc", "z-score"] = "z-score", scatter_zero_pvals: bool = False) -> pd.DataFrame:
    mapping = {}
    for gene in adata.var_names:
        mapping[gene] = {"z-score": 0.0, "pvals_adj": 0.0, "logFC": 0.0}

    i = rank_res["names"].dtype.names.index(group)

    for genes, scores, pvals, logFC in list(zip(
        rank_res["names"], rank_res["scores"],
        rank_res["pvals_adj"], rank_res["logfoldchanges"]
    )):
        mapping[genes[i]]["z-score"] = scores[i]
        mapping[genes[i]]["pvals_adj"] = pvals[i]
        mapping[genes[i]]["logFC"] = logFC[i]

    df = pd.DataFrame(mapping).T

    _max = -np.log10(np.nanmin(df["pvals_adj"].values[df["pvals_adj"].values != 0]) * 0.1)

    # where pvals_adj is 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_pvals_adj = -np.log10(df["pvals_adj"])

    _shape = log_pvals_adj[log_pvals_adj == np.inf].shape

    if scatter_zero_pvals:
        if _shape[0] > 0:
            print(f"Warning: some p-values ({_shape[0]}) were 0, scattering them around {_max:.1f}")

        log_pvals_adj[log_pvals_adj == np.inf] = _max + np.random.uniform(size=_shape) * _max * 0.2

    df["-log_pvals_adj"] = log_pvals_adj

    group_gex = adata[adata.obs[groupby] == group, :].layers["ncounts"]
    if reference == "rest":
        bg_gex = adata[adata.obs[groupby] != group, :].layers["ncounts"]
    else:
        bg_gex = adata[adata.obs[groupby] == reference, :].layers["ncounts"]

    if scipy.sparse.issparse(group_gex):
        group_gex = group_gex.toarray()
    if scipy.sparse.issparse(bg_gex):
        bg_gex = bg_gex.toarray()

    df["mu_gex"] = np.asarray(group_gex.mean(axis=0)).flatten()
    df["mu_bg"] = np.asarray(bg_gex.mean(axis=0)).flatten()
    df["s_ref"] = np.asarray(group_gex.std(axis=0)).flatten()
    df["s_bg"] = np.asarray(bg_gex.std(axis=0)).flatten()

    if score_type == "welchs":
        df["gene_score"] = (df["mu_gex"] - df["mu_bg"]) / np.sqrt(df["s_ref"]**2 + df["s_bg"]**2)
    elif score_type == "pval-lfc":
        df["gene_score"] = (df["logFC"]) * (1 - df["pvals_adj"])
    elif score_type == "z-score":
        df["gene_score"] = df["z-score"]
    elif score_type == "custom":
        df["gene_score"] = df["pvals_adj"] * (df["mu_gex"] - df["mu_bg"]) / np.sqrt(df["s_ref"]**2 + df["s_bg"]**2)
    else:
        raise ValueError(f"Unknown score type: {score_type}")

    df["abs_score"] = np.abs(df["gene_score"])
    df["significant"] = df["pvals_adj"] < 0.05

    df.index.name = f"{group}_vs_{reference}"

    return df.sort_values("abs_score", ascending=False)


def rank_marker_genes(
    adata: sc.AnnData, groupby: str, reference: str | None, corr_method: Literal["bonferroni", "benjamini-hochberg"] = "benjamini-hochberg",
    method: Literal["t-test", "wilcoxon", "t-test_overestim_var"] = "t-test", layer: str | None = None, score_type: Literal["welchs", "pval-lfc", "z-score"] = "z-score",
) -> dict[str, pd.DataFrame]:
    if adata.obs[groupby].dtype.name != "category":
        adata.obs[groupby] = adata.obs[groupby].astype("category")

    if reference is None:
        reference = "rest"

    if (rank_res := sc.tl.rank_genes_groups(
        adata, groupby=groupby, method=method, corr_method=corr_method, copy=True,
        reference=reference, layer=layer
    )) is None:
        raise ValueError()
    
    rank_res = rank_res.uns["rank_genes_groups"]
    categories = rank_res["names"].dtype.names

    res = {}

    for i, group in enumerate(categories):
        res[f"{str(group)} vs. {reference}"] = __rank_group(adata=adata, rank_res=rank_res, groupby=groupby, group=group, reference=reference, score_type=score_type)

    return res


def sub_cluster(adata: sc.AnnData, groupby: str, key_added: str = "sub_type", leiden_res: float = 0.1):
    adata.obs[key_added] = ""
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    for i, group in enumerate(adata.obs[groupby].cat.categories):
        view = adata[adata.obs[groupby] == group, :]
        res = sc.tl.leiden(
            view,
            resolution=leiden_res,
            copy=True,
            key_added="sub_type",
            random_state=0,
        ).obs["sub_type"]

        adata.obs.loc[res.index, key_added] = res.values
        adata.obs.loc[res.index, key_added] = adata.obs.loc[res.index, "sub_type"].apply(lambda x: f"{group}_{x}")

    adata.obs[key_added] = adata.obs[key_added].astype("category")
import threading, warnings
from typing import Literal

import scanpy as sc
import numpy as np
import pandas as pd

import scipy


def fmt_c(w):
    return " ".join([f"{v:.2f}" for v in w])

def scale_log_center(adata, target_sum=None, norm_factor_key=None, exclude_highly_expressed=False):
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum, key_added=norm_factor_key, exclude_highly_expressed=exclude_highly_expressed)
    adata.layers["ncounts"] = adata.X.copy()
    sc.pp.log1p(adata)
    adata.layers["centered"] = np.asarray(
        adata.layers["counts"] - adata.layers["counts"].mean(axis=0)
    )
    adata.layers["logcentered"] = np.asarray(adata.X - adata.X.mean(axis=0))


def _rank_group(adata, rank_res, groupby, idx, ref_name, logeps):
    mapping = {}
    for gene in adata.var_names:
        mapping[gene] = {"z-score": 0.0, "pvals_adj": 0.0, "logFC": 0.0}

    for genes, scores, pvals, logFC in list(zip(
        rank_res["names"], rank_res["scores"],
        rank_res["pvals_adj"], rank_res["logfoldchanges"]
    )):
        mapping[genes[idx]]["z-score"] = scores[idx]
        mapping[genes[idx]]["pvals_adj"] = pvals[idx]
        mapping[genes[idx]]["logFC"] = logFC[idx]

    df = pd.DataFrame(mapping).T

    _max = -np.log10(np.nanmin(df["pvals_adj"].values[df["pvals_adj"].values != 0]) * 0.1)

    # where pvals_adj is 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_pvals_adj = -np.log10(df["pvals_adj"])
    
    _shape = log_pvals_adj[log_pvals_adj == np.inf].shape
    if _shape[0] > 0:
        print(f"Warning: some p-values ({_shape[0]}) were 0, scattering them around {_max:.1f}")
    
    log_pvals_adj[log_pvals_adj == np.inf] = _max + np.random.uniform(size=_shape) * _max * 0.2

    df["-log_pvals_adj"] = log_pvals_adj

    df["significant"] = df["pvals_adj"] < 0.05

    group_idx = adata.obs[groupby].astype("str").astype("category").cat.categories.tolist().index(ref_name)

    min_logfc, max_logfc = np.quantile(df["logFC"], [0.05, 0.95])

    df["mu_expression"] = adata.varm[f"mu_expression_{groupby}"][:, group_idx]
    df["log_mu_expression"] = adata.varm[f"log_mu_expression_{groupby}"][:, group_idx]
    assert df["log_mu_expression"].isna().any() == False
    df["cv"] = adata.varm[f"cv_{groupby}"][:, group_idx]

    df["gene_score"] = (
        # np.clip(df["logFC"], min_logfc, max_logfc) *
        # df["logFC"] * (1-df["pvals_adj"]) * (1.0/df["cv"])
        np.sign(df["logFC"]) *
        (1-df["pvals_adj"]) * np.abs(df["z-score"])
        # df["z-score"]
        # * (1/np.log1p(df["cv"]))
        # df["significant"]
        # np.sign(df["logFC"]) * (1-df["pvals_adj"]) #* df["log_mu_expression"]# * (1.0 - df["dropout"])
    )
    # p = (adata[adata.obs[groupby].astype("category").cat.codes == group_idx, :].layers["counts"].sum(0) / adata.layers["counts"].sum(0)).copy()
    # print(p.shape)
    # # df["weight"] = 
    # df["score"] = np.log1p(df["weight"]) * df["-log_pvals_adj"] * df["logFC"]

    df["abs_score"] = np.abs(df["gene_score"])

    df.index.name = f"{ref_name}_vs_rest"

    return df


def sub_cluster(adata, groupby, key_added="sub_type", leiden_res=0.1):
    adata.obs[key_added] = ""
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


def combine(adata, bulk_df):
    # Checks if bulk genes are given as columns or as rows
    common_genes = np.intersect1d(adata.var_names, bulk_df.index.astype(str))
    common_genes_t = np.intersect1d(adata.var_names, bulk_df.columns.astype(str))

    if len(common_genes) < len(common_genes_t):
        common_genes = common_genes_t
        bulk_df = bulk_df.T

    assert (len(common_genes) > 0), "No common genes found between bulk and single cell data"

    adata = adata[:, common_genes].copy()
    adata.varm["bulk"] = bulk_df.loc[common_genes].values.astype(np.float32)
    
    if "counts" not in adata.layers.keys():
        adata.varm["pseudo"] = adata.X.sum(0).reshape(-1, 1)
    else:
        adata.varm["pseudo"] = adata.layers["counts"].sum(0)

    adata.varm["bulk"] *= (adata.varm["pseudo"].sum() / adata.varm["bulk"].sum(0))

    adata.varm["pseudo_factor"] = (np.log1p(adata.varm["bulk"]) - np.log1p(adata.varm["pseudo"]))

    adata.uns["bulk_samples"] = bulk_df.columns.tolist()

    print(f"scRNA-seq data - cells: {adata.shape[0]}, genes: {adata.shape[1]}")
    print(f"bulk RNA-seq data - samples: {adata.varm['bulk'].shape[1]}, genes: {adata.varm['bulk'].shape[0]}")

    return adata

def group_stats(adata, groupby, eps=1e-8):
    n_groups = len(adata.obs[groupby].cat.categories)
    n_genes = adata.n_vars
    
    adata.varm[f"mu_expression_{groupby}"] = np.empty((n_genes, n_groups), np.float32)
    adata.varm[f"var_expression_{groupby}"] = np.empty((n_genes, n_groups), np.float32)
    adata.varm[f"cv_{groupby}"] = np.empty((n_genes, n_groups), np.float32)
    adata.varm[f"log_mu_expression_{groupby}"] = np.empty((n_genes, n_groups), np.float32)
    # adata.varm[f"log_var_expression_{groupby}"] = np.empty((n_genes, n_groups), np.float32)
    adata.varm[f"dropout_{groupby}"] = np.empty((n_genes, n_groups), np.float32)
    # adata.varm[f"nan_mu_expression_{groupby}"] = np.empty((n_genes, n_groups), np.float32)
    # adata.varm[f"nan_log_mu_expression_{groupby}"] = np.empty((n_genes, n_groups), np.float32)
    # adata.varm[f"dropout_weight_{groupby}"] = np.empty((n_genes, n_groups), np.float32)


    for i, group in enumerate(adata.obs[groupby].cat.categories):
        a = adata[adata.obs[groupby] == group, :].layers["counts"]
        if isinstance(a, scipy.sparse.csr_matrix):
            a = a.toarray()

        adata.varm[f"mu_expression_{groupby}"][:, i] = np.asarray(a.mean(axis=0)).flatten()

        assert np.isnan(adata.varm[f"mu_expression_{groupby}"][:, i]).any() == False

        adata.varm[f"var_expression_{groupby}"][:, i] = np.asarray(a.var(axis=0)).flatten()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = a.std(axis=0) / adata.varm[f"mu_expression_{groupby}"][:, i]
        # nans where std and mu is 0, and posinf where std > 0 and mu is 0
        adata.varm[f"cv_{groupby}"][:, i] = np.nan_to_num(cv, nan=0.0, posinf=0.0)

        adata.varm[f"log_mu_expression_{groupby}"][:, i] = np.asarray(np.log1p(a).mean(0)).flatten()

        # assert np.isnan(adata.varm[f"log_mu_expression_{groupby}"][:, i]).any() == False

        # adata.varm[f"log_var_expression_{groupby}"][:, i] = np.asarray(
        #     np.log1p(a)
        # ).var(0).flatten()

        adata.varm[f"dropout_{groupby}"][:, i] = np.asarray(
            (a == 0).mean(0)
        ).flatten()


def rank_marker_genes(
    adata, groupby, reference="rest", corr_method="benjamini-hochberg", logeps=-500, copy=False,
    method: Literal["t-test", "logreg", "wilcoxon", "t-test_overestim_var"] = "t-test"
):
    if adata.obs[groupby].dtype.name != "category":
        adata.obs[groupby] = adata.obs[groupby].astype("category")

    if f"mu_expression_{groupby}" not in adata.varm.keys():
        group_stats(adata, groupby)

    rank_res = sc.tl.rank_genes_groups(
        adata, groupby=groupby, method=method, corr_method=corr_method, copy=True, reference=reference
    ).uns["rank_genes_groups"]

    res = {}

    for i, ref in enumerate(rank_res["names"].dtype.names):
        res[f"{str(ref)} vs. {reference}"] = _rank_group(adata, rank_res, groupby, i, ref, logeps)

    if not copy:
        if "de" not in adata.uns:
            adata.uns["de"] = {}

        adata.uns["de"][groupby] = res

        print(f"DE: added results to 'adata.uns['de']['{groupby}']'")
    else:
        return res


def read_data(path):
    if path.endswith(".csv"):
        delim = ","
    elif path.endswith(".tsv"):
        delim = "\t"
    else:
        assert False, "Unknown file type, only .csv and .tsv are supported"

    adata = sc.read_csv(path, delimiter=delim)
    adata.var_names_make_unique()

    return adata

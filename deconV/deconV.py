import argparse
import glob
import os

# Custom tools
import sys
import time
import warnings
from typing import Literal

import deconV.plot as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as S
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import yaml
from matplotlib import rcParams

from . import base


def fmt_c(w):
    return " ".join([f"{v:.2f}" for v in w])


class DeconV:
    def __init__(self, sc_adata, bulk_adata, cell_types, params, use_sub_types=False):
        self.params = params
        self.sadata = sc_adata
        self.badata = bulk_adata
        self.cell_types = cell_types
        self.n_all_genes = self.sadata.n_vars
        self.n_used_genes = self.n_all_genes
        self.n_cell_types = len(cell_types)
        self.eps = 1e-4
        self.use_sub_types = use_sub_types
        self.label_key = "sub_type" if use_sub_types else self.params["cell_type_key"]
        self.sub_types = []
        self.n_labels = -1 if use_sub_types else self.n_cell_types
        self.gene_weights = None
        self.normalising_constant = torch.tensor([1.0])

        self.sadata.var["pseudo"] = self.sadata.layers[self.params["layer"]].sum(0)

        self.init_stats()
        self.sadata.var["outlier"] = False

    def init_stats(self):
        # Neighbors for clustering
        sc.pp.pca(self.sadata)
        sc.pp.neighbors(self.sadata, random_state=0)

        # Pseudobulk vs. Bulk
        self.sadata.var["log_bulk_residual"] = 0.0
        for i in range(self.badata.shape[0]):
            self.sadata.var[f"bulk_{i}"] = self.badata.layers[self.params["layer"]][
                i, :
            ]
            self.sadata.var[f"bulk_residual_{i}"] = self.sadata.var[
                f"bulk_{i}"
            ] - np.log1p(self.sadata.var["pseudo"])
            self.sadata.var[f"log_bulk_residual_{i}"] = self.badata.X[i, :] - np.log1p(
                self.sadata.var["pseudo"]
            )
            self.sadata.var["log_bulk_residual"] += self.sadata.var[
                f"log_bulk_residual_{i}"
            ]

        self.sadata.var["log_bulk_residual"] /= self.badata.shape[0]

        sc.tl.pca(self.sadata)

        sc.tl.rank_genes_groups(
            self.sadata, self.params["cell_type_key"], method="t-test"
        )

        self.marker_genes = list(
            sum(self.sadata.uns["rank_genes_groups"]["names"].tolist(), ())
        )

        self.sadata.var = self.sadata.var.drop(
            columns=[k for k in self.sadata.var.columns if "ct_marker_" in k]
        )

        """
        ---------------------------------------
                Cell Type Specific Stats
        ---------------------------------------
        """
        dispersion = torch.empty(self.n_all_genes, self.n_cell_types)

        for i, cell_type in enumerate(self.cell_types):
            layer = self.sadata[
                self.sadata.obs[self.params["cell_type_key"]] == cell_type, :
            ].layers[self.params["layer"]]
            self.sadata.var[f"mu_{cell_type}"] = np.maximum(
                np.nan_to_num(layer.mean(0), nan=0.0), self.eps
            )
            self.sadata.var[f"std_{cell_type}"] = np.maximum(
                np.nan_to_num(layer.std(0), nan=0.0), self.eps
            )

            ncounts = self.sadata[
                self.sadata.obs[self.params["cell_type_key"]] == cell_type, :
            ].layers["ncounts"]
            nancounts = ncounts.copy()
            nancounts[nancounts == 0] = np.nan

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.sadata.var[f"nanmu_{cell_type}"] = np.nan_to_num(
                    np.nanmean(nancounts, 0), nan=self.eps
                )

            self.sadata.var[f"dropout_{cell_type}"] = (layer == 0).mean(0)
            self.sadata.var[f"cv2_{cell_type}"] = (
                self.sadata.var[f"std_{cell_type}"] / self.sadata.var[f"mu_{cell_type}"]
            ) ** 2

            dispersion[:, i] = torch.tensor(
                self.sadata.var[f"cv2_{cell_type}"] / self.sadata.var[f"mu_{cell_type}"]
            )

        self.sadata.var["dispersion_min"] = torch.min(dispersion, dim=1).values
        self.sadata.var["dispersion_max"] = torch.max(dispersion, dim=1).values
        self.sadata.var["dispersion_mu"] = torch.mean(dispersion, dim=1)
        self.sadata.var["log_dispersion_min"] = np.log(
            self.sadata.var["dispersion_min"]
        )
        self.sadata.var["log_dispersion_max"] = np.log(
            self.sadata.var["dispersion_max"]
        )
        self.sadata.var["log_dispersion_mu"] = np.log(self.sadata.var["dispersion_mu"])

        """ 
        ---------------------------------------
                Marker gene strength
        ---------------------------------------
        """
        mapping = {}
        for cell_type in self.cell_types:
            mapping[cell_type] = {}
            mapping[cell_type]["pvals_adj"] = {}
            mapping[cell_type]["score"] = {}
            mapping[cell_type]["logFC"] = {}

        for genes, scores, pvals, logFC in zip(
            self.sadata.uns["rank_genes_groups"]["names"],
            self.sadata.uns["rank_genes_groups"]["scores"],
            self.sadata.uns["rank_genes_groups"]["pvals_adj"],
            self.sadata.uns["rank_genes_groups"]["logfoldchanges"],
        ):
            for i, (cell_type, _) in enumerate(genes.dtype.descr):
                mapping[cell_type]["score"][genes[i]] = scores[i]
                mapping[cell_type]["pvals_adj"][genes[i]] = pvals[i]
                mapping[cell_type]["logFC"][genes[i]] = logFC[i]

        dfs = {}
        for cell_type in self.cell_types:
            dfs[cell_type] = pd.DataFrame(mapping[cell_type])
            dfs[cell_type]["-log_pvals_adj"] = -np.log10(dfs[cell_type]["pvals_adj"])
            self.sadata.var.loc[
                list(mapping[cell_type]["score"].keys()), f"ct_marker_score_{cell_type}"
            ] = list(mapping[cell_type]["score"].values())
            self.sadata.var.loc[
                list(mapping[cell_type]["pvals_adj"].keys()),
                f"ct_marker_pval_{cell_type}",
            ] = list(mapping[cell_type]["pvals_adj"].values())

        self.sadata.uns["ranked_genes"] = dfs

        marker_score_mu = self.sadata.var.loc[
            :, self.sadata.var.columns.str.contains("ct_marker_score_")
        ].mean(axis=1)
        marker_score_max = self.sadata.var.loc[
            :, self.sadata.var.columns.str.contains("ct_marker_score_")
        ].max(axis=1)
        marker_score_min = self.sadata.var.loc[
            :, self.sadata.var.columns.str.contains("ct_marker_score_")
        ].min(axis=1)
        marker_abs_score_mu = (
            self.sadata.var.loc[
                :, self.sadata.var.columns.str.contains("ct_marker_score_")
            ]
            .abs()
            .mean(axis=1)
        )
        marker_abs_score_max = (
            self.sadata.var.loc[
                :, self.sadata.var.columns.str.contains("ct_marker_score_")
            ]
            .abs()
            .max(axis=1)
        )
        marker_abs_score_min = (
            self.sadata.var.loc[
                :, self.sadata.var.columns.str.contains("ct_marker_score_")
            ]
            .abs()
            .min(axis=1)
        )

        self.sadata.var["ct_marker_score_mu"] = marker_score_mu
        self.sadata.var["ct_marker_score_max"] = marker_score_max
        self.sadata.var["ct_marker_score_min"] = marker_score_min
        self.sadata.var["ct_marker_abs_score_mu"] = marker_abs_score_mu
        self.sadata.var["ct_marker_abs_score_max"] = marker_abs_score_max
        self.sadata.var["ct_marker_abs_score_min"] = marker_abs_score_min
        self.sadata.var["log_marker_score_mu"] = np.log1p(marker_abs_score_mu)
        self.sadata.var["log_marker_score_max"] = np.log1p(marker_abs_score_max)
        self.sadata.var["log_marker_score_min"] = np.log1p(marker_abs_score_min)

        marker_pval_mu = self.sadata.var.loc[
            :, self.sadata.var.columns.str.contains("ct_marker_pval_")
        ].mean(axis=1)
        marker_pval_max = self.sadata.var.loc[
            :, self.sadata.var.columns.str.contains("ct_marker_pval_")
        ].max(axis=1)
        marker_pval_min = self.sadata.var.loc[
            :, self.sadata.var.columns.str.contains("ct_marker_pval_")
        ].min(axis=1)

        self.sadata.var["ct_marker_pval_mu"] = marker_pval_mu
        self.sadata.var["ct_marker_pval_max"] = marker_pval_max
        self.sadata.var["ct_marker_pval_min"] = marker_pval_min

        """ 
        ---------------------------------------
                        Dropout
        ---------------------------------------
        """
        # ncounts = sc.pp.normalize_total(self.sadata, layer=self.params["layer"], inplace=False)["X"]
        nancounts = self.sadata.layers["ncounts"].copy()
        nancounts[nancounts == 0] = np.nan

        self.sadata.var["nanmu"] = np.nanmean(nancounts, 0)
        self.sadata.var["dropout"] = (self.sadata.layers["ncounts"] == 0).mean(0)

        logx = np.log(self.sadata.var["nanmu"])
        y = self.sadata.var["dropout"]

        fit = np.poly1d(np.polyfit(logx, y, 2))

        self.sadata.var["dropout_residual"] = y - fit(logx)

        """ 
        ---------------------------------------
                        Dispersion
        ---------------------------------------
        """
        ncounts = self.sadata.layers["ncounts"]
        self.sadata.var["cv2"] = (ncounts.std(0) / ncounts.mean(0)) ** 2
        self.sadata.var["mu"] = ncounts.mean(0)
        self.sadata.var["cv2/mu"] = self.sadata.var["cv2"] / self.sadata.var["nanmu"]
        self.sadata.var["log_cv2/mu"] = (
            np.log(self.sadata.var["cv2"]) / self.sadata.var["nanmu"]
        )
        self.sadata.var["dropout/mu"] = (
            self.sadata.var["dropout"] / self.sadata.var["nanmu"]
        )

        logx = np.log(self.sadata.var["mu"])
        logy = np.log(self.sadata.var["cv2"])

        fit = np.poly1d(np.polyfit(logx, logy, 2))

        self.sadata.var["dispersion_residual"] = logy - fit(logx)
        self.sadata.var["dispersion_residual_dx"] = np.abs(
            self.sadata.var["dispersion_residual"]
        )

        # self.sadata.var["gene_scale"] = self.badata

    def filter_outliers(
        self,
        dispersion_lims=(-np.inf, np.inf),
        dropout_mu_lim=2,
        dropout_lim=1.0,
        marker_zscore_lim=1.0,
        pseudobulk_lims=(-10, 10),
    ):
        marker_zscore_lim = (
            self.sadata.var["ct_marker_abs_score_max"].max() * marker_zscore_lim
        )

        self.sadata.var["dispersion_outlier"] = (
            self.sadata.var["dispersion_residual"] < dispersion_lims[0]
        ) | (self.sadata.var["dispersion_residual"] > dispersion_lims[1])
        self.sadata.var["dropout_outlier"] = (
            self.sadata.var["dropout/mu"] > dropout_mu_lim
        ) | (self.sadata.var["dropout"] > dropout_lim)
        self.sadata.var["pseudobulk_outlier"] = (
            self.sadata.var["log_bulk_residual"] < pseudobulk_lims[0]
        ) | (self.sadata.var["log_bulk_residual"] > pseudobulk_lims[1])

        self.sadata.var["outlier"] = (
            self.sadata.var["dispersion_outlier"] | self.sadata.var["dropout_outlier"]
        )
        self.sadata.var.loc[
            self.sadata.var["ct_marker_abs_score_max"] > marker_zscore_lim, "outlier"
        ] = False

    def sub_cluster(self, leiden_res=0.1):
        self.sadata.obs["sub_type"] = ""

        for cell_type in self.cell_types:
            adata = self.sadata[
                self.sadata.obs[self.params["cell_type_key"]] == cell_type
            ]
            res = sc.tl.leiden(
                adata,
                resolution=leiden_res,
                copy=True,
                key_added="sub_type",
                random_state=0,
            ).obs["sub_type"]
            self.sadata.obs.loc[res.index, "sub_type"] = res.values
            self.sadata.obs.loc[res.index, "sub_type"] = self.sadata.obs.loc[
                res.index, "sub_type"
            ].apply(lambda x: f"{cell_type}_{x}")

        self.sub_types = list(self.sadata.obs["sub_type"].unique())
        if self.use_sub_types:
            self.n_labels = len(self.sub_types)

    def init_dataset(
        self,
        adata=None,
        use_outliers=True,
        gene_weight_key=None,
        plot_gene_weight_hist=True,
    ):
        if adata is None:
            adata = self.sadata

        self.X = []

        for i, cell_type in enumerate(
            self.sub_types if self.use_sub_types else self.cell_types
        ):
            if use_outliers:
                _x = adata[adata.obs[self.label_key] == cell_type, :].layers[
                    self.params["layer"]
                ]
            else:
                _x = adata[
                    adata.obs[self.label_key] == cell_type, ~adata.var["outlier"]
                ].layers[self.params["layer"]]

            self.X.append(torch.tensor(_x))

        if use_outliers:
            self.Y = torch.tensor(self.badata.layers[self.params["layer"]])
        else:
            self.Y = torch.tensor(
                self.badata[:, ~adata.var["outlier"]].layers[self.params["layer"]]
            )

        assert (
            self.X[0].shape[1] == self.Y.shape[1]
        ), f"{self.X[0].shape} != {self.Y.shape}"
        self.n_bulk_samples = self.Y.shape[0]
        self.n_used_genes = self.Y.shape[1]

        # Gene weights
        if gene_weight_key is not None:
            assert (
                gene_weight_key in adata.var.columns
            ), f"Key: {gene_weight_key} not found in sadata.var"
            self.gene_weights = torch.tensor(
                adata[:, ~self.sadata.var["outlier"] | use_outliers]
                .var[gene_weight_key]
                .values
            )
            if "pval" in gene_weight_key:
                self.gene_weights = 1 - self.gene_weights
            self.gene_weights = (self.gene_weights - self.gene_weights.min()) / (
                self.gene_weights.max() - self.gene_weights.min()
            )
            if plot_gene_weight_hist:
                pl.gene_weight_hist(self.gene_weights, gene_weight_key)
        else:
            self.gene_weights = None

        # Normalising constant for loss function (makes loss comparable between datasets)
        self.normalising_constant = 1.0 / torch.tensor(
            np.nanmean(
                np.nan_to_num(
                    self.sadata.layers["counts"].sum(0) / self.badata.layers["counts"],
                    posinf=np.nan,
                ),
                axis=1,
            )
        )

    def init_signature(self):
        print(
            f"Creating signature for cell types: {self.sub_types if self.use_sub_types else self.cell_types}..."
        )

        self.loc = torch.empty(self.n_used_genes, self.n_labels)
        self.scale = torch.empty(self.n_used_genes, self.n_labels)

        for i, _ in enumerate(self.cell_types):
            self.loc[:, i] = self.X[i].mean(0)
            self.scale[:, i] = self.X[i].std(0)

        assert torch.isnan(self.loc).sum() == 0, "NaN found in loc"
        assert torch.isnan(self.scale).sum() == 0, "NaN found in scale"

    def _fit(self, model, y):
        if self.params["tqdm"]:
            pbar = tqdm.tqdm(range(self.params["epochs"]))
        else:
            pbar = range(self.params["epochs"])
        # pbar = range(self.params["epochs"])

        optim = torch.optim.Adam(model.parameters(), lr=self.params["lr"])

        for i in pbar:
            optim.zero_grad()
            loss = model(y)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.00001)
            optim.step()
            if i % 50 == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.1f}",
                        "p": fmt_c(model.get_proportions()),
                        "lib_size": f"{model.get_lib_size().item():.1f}",
                    }
                )

    def deconvolute(self):
        results = np.empty((self.n_bulk_samples, self.n_cell_types))

        for sample_i, y in enumerate(self.Y):
            print(f"Sample: {sample_i}/{self.n_bulk_samples}")

            if self.params["model_type"] == "normal":
                model = base.NSM(
                    self.loc,
                    self.scale,
                    gene_weights=self.gene_weights,
                    norm=self.normalising_constant[sample_i],
                )
            elif self.params["model_type"] == "mse":
                model = base.MSEM(self.loc)
            elif self.params["model_type"] == "poisson":
                model = base.PSM(
                    self.loc,
                    gene_weights=self.gene_weights,
                    norm=self.normalising_constant[sample_i],
                )

            self._fit(model, y)
            proportions = model.get_proportions().detach().numpy()

            if self.use_sub_types:
                for i, cell_type in enumerate(self.sub_types):
                    print(f"{cell_type}: {proportions[i]:.3f} | ", end="")
                print()

            if self.use_sub_types:
                results[sample_i, :] = self.sum_sub_proportions(proportions)
                for i, cell_type in enumerate(self.cell_types):
                    print(f"{cell_type}: {results[sample_i, i]:.3f} | ", end="")
                print()
            else:
                results[sample_i, :] = proportions

            if not self.params["jupyter"]:
                pl.prediction_plot(
                    model.get_mean().detach(),
                    y,
                    path=os.path.join(
                        self.params["outdir"],
                        self.params["model_type"],
                        f"sample_{self.badata.obs.index[sample_i]}.{self.params['fig_fmt']}",
                    ),
                )

        df = pd.DataFrame(results, columns=self.cell_types, index=self.badata.obs.index)
        df = df.reindex(sorted(df.columns), axis=1)

        print("Plotting results...")
        if not self.params["jupyter"]:
            if self.n_bulk_samples > 1:
                pl.proportions_heatmap(
                    df,
                    path=os.path.join(
                        self.params["outdir"],
                        self.params["model_type"],
                        f"proportions.{self.params['fig_fmt']}",
                    ),
                )

            pl.bar_proportions(
                df,
                path=os.path.join(
                    self.params["outdir"],
                    self.params["model_type"],
                    f"bar.{self.params['fig_fmt']}",
                ),
            )
            df.to_csv(
                os.path.join(
                    self.params["outdir"], self.params["model_type"], "proportions.csv"
                ),
                sep=",",
            )
        else:
            if self.n_bulk_samples > 1:
                pl.proportions_heatmap(df)

            pl.bar_proportions(df)
            return df

    def sum_sub_proportions(self, sub_proportions):
        d = {}
        for i, sub_type in enumerate(self.sub_types):
            cell_type = "_".join(sub_type.split("_")[:-1])
            if cell_type not in d.keys():
                d[cell_type] = sub_proportions[i]
            else:
                d[cell_type] += sub_proportions[i]

        return np.array([d[cell_type] for cell_type in self.cell_types])


def preprocess(sc_adata, bulk_adata, params):

    if params["selected_ct"] is not None:
        sc_adata.obs.loc[
            ~sc_adata.obs[params["cell_type_key"]].isin(params["selected_ct"]),
            params["cell_type_key"],
        ] = "other"

        if params["ignore_others"]:
            sc_adata = sc_adata[sc_adata.obs[params["cell_type_key"]] != "other"].copy()

    assert (
        sc_adata.n_obs > 0
    ), "No cells found in scRNA-seq data, check 'selected_ct' and 'cell_type_key' parameters..."

    sc.pp.filter_cells(sc_adata, min_genes=200)
    sc.pp.filter_genes(sc_adata, min_cells=3)

    common_genes = list(
        set(bulk_adata.var.index.values).intersection(set(sc_adata.var.index.values))
    )
    assert (
        len(common_genes) > 0
    ), "No common genes found between bulk and single cell data"
    bulk_adata = bulk_adata[:, common_genes].copy()
    sc_adata = sc_adata[:, common_genes].copy()

    sc_adata.layers["counts"] = sc_adata.X.copy()
    bulk_adata.layers["counts"] = bulk_adata.X.copy()

    sc.pp.normalize_total(sc_adata, target_sum=None)
    sc_adata.layers["ncounts"] = sc_adata.X.copy()
    sc.pp.log1p(sc_adata)

    sc.pp.normalize_total(bulk_adata, target_sum=None)
    bulk_adata.layers["ncounts"] = bulk_adata.X.copy()
    sc.pp.log1p(bulk_adata)

    sc_adata.layers["centered"] = sc_adata.layers["counts"] - sc_adata.layers[
        "counts"
    ].mean(axis=0)
    sc_adata.layers["logcentered"] = sc_adata.X - sc_adata.X.mean(axis=0)

    if params["n_top_genes"] > 0:
        sc.pp.highly_variable_genes(
            sc_adata,
            min_mean=0.01,
            max_mean=3,
            min_disp=0.5,
            subset=True,
            n_top_genes=params["n_top_genes"],
        )
        bulk_adata = bulk_adata[:, sc_adata.var.index.values].copy()

    return sc_adata, bulk_adata


def read_data(path, is_bulk=False, transpose_bulk=False):
    if path.endswith(".csv"):
        delim = ","
    elif path.endswith(".tsv"):
        delim = "\t"
    else:
        assert False, "Unknown file type, only .csv and .tsv are supported"

    adata = sc.read_csv(path, delimiter=delim)
    adata.var_names_make_unique()
    if is_bulk:
        if transpose_bulk:
            adata = adata.T
        elif adata.shape[1] == 1:
            print("Only one gene found in bulk file, transposing...")
            adata = adata.T

    return adata

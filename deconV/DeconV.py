import os, warnings
from typing import Literal

from . import plot as pl
from . import tools as tl
from . import base

import numpy as np
import pandas as pd
import scanpy as sc

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import tqdm

import scout


class DeconV:
    def __init__(self, adata, label_key, layer="counts"):
        self.layer = layer
        self.adata = adata
        self.label_key = label_key
        self.cell_types = self.adata.obs[self.label_key].astype("category").cat.categories.tolist()
        self.n_cell_types = len(self.cell_types)
        self.eps = 1e-4

        self.init_stats()
        self.adata.var["outlier"] = False

    def init_stats(self):
        # Neighbors for clustering
        sc.pp.pca(self.adata)
        sc.pp.neighbors(self.adata, random_state=0)
        scout.tl.rank_marker_genes(self.adata, groupby=self.label_key)

    def filter_outliers(
        self,
        dispersion_lims=(-np.inf, np.inf),
        dropout_mu_lim=2,
        dropout_lim=1.0,
        marker_zscore_lim=1.0,
        pseudobulk_lims=(-10, 10),
    ):
        marker_zscore_lim = (
            self.adata.var["ct_marker_abs_score_max"].max() * marker_zscore_lim
        )

        self.adata.var["dispersion_outlier"] = (
            self.adata.var["dispersion_residual"] < dispersion_lims[0]
        ) | (self.adata.var["dispersion_residual"] > dispersion_lims[1])
        self.adata.var["dropout_outlier"] = (
            self.adata.var["dropoutxmu"] > dropout_mu_lim
        ) | (self.adata.var["dropout"] > dropout_lim)
        self.adata.var["pseudobulk_outlier"] = (
            self.adata.var["log_bulk_residual"] < pseudobulk_lims[0]
        ) | (self.adata.var["log_bulk_residual"] > pseudobulk_lims[1])

        self.adata.var["outlier"] = (
            self.adata.var["dispersion_outlier"] | self.adata.var["dropout_outlier"]
        )
        self.adata.var.loc[
            self.adata.var["ct_marker_abs_score_max"] > marker_zscore_lim, "outlier"
        ] = False

    def init_dataset(
        self, weight_type=None, use_outliers=True, plot_gene_weight_hist=True,
        weight_agg: Literal["min", "max", "mean", "geom_mean"] = "mean"
    ):
        self.X = []

        for i, cell_type in enumerate(self.cell_types):
            if use_outliers:
                _x = self.adata[self.adata.obs[self.label_key] == cell_type, :].layers[self.layer]
            else:
                _x = self.adata[self.adata.obs[self.label_key] == cell_type, ~self.adata.var["outlier"]].layers[self.layer]

            self.X.append(torch.tensor(_x))

        if use_outliers:
            self.Y = torch.tensor(self.adata.varm["bulk"].T)
        else:
            self.Y = torch.tensor(self.adata[:, ~self.adata.var["outlier"]].varm["bulk"].T)

        assert (self.X[0].shape[1] == self.Y.shape[1]), f"{self.X[0].shape} != {self.Y.shape}"

        # ---------------------------
        #       Gene weights
        # ---------------------------

        if weight_type is not None:
            scores = np.empty((self.adata.shape[1], self.n_cell_types))
        
            for i, ct in enumerate(self.cell_types):
                scores[:, i] = self.adata.uns["de"]["cellType"][f"{ct} vs. rest"][weight_type].values
                if "pvals_adj" == weight_type:
                    scores[:, i] = 1.0-scores[:, i]

            scores = pd.DataFrame(scores, index=self.adata.var_names, columns=self.cell_types)
            if weight_agg == "mean":
                self.adata.var["weight"] = scores.mean(axis=1)
            elif weight_agg == "geom_mean":
                self.adata.var["weight"] = np.exp(np.log(scores + self.eps).mean(axis=1))
            elif weight_agg == "min":
                self.adata.var["weight"] = scores.min(axis=1)
            elif weight_agg == "max":
                self.adata.var["weight"] = scores.max(axis=1)

            self.gene_weights = torch.tensor(
                self.adata[:, ~self.adata.var["outlier"] | use_outliers].var["weight"].values
            )

            self.gene_weights = (self.gene_weights - self.gene_weights.min()) / (
                self.gene_weights.max() - self.gene_weights.min()
            )

            if plot_gene_weight_hist:
                pl.gene_weight_hist(self.gene_weights, f"Gene Weight ({weight_type} | {weight_agg})")
        else:
            self.gene_weights = None

        # Normalising constant for loss function (makes loss comparable between datasets)
        self.normalising_constant = np.nanmean(
            np.nan_to_num(self.adata.layers["counts"].sum(0) / self.adata.varm["bulk"].T, posinf=np.nan),
            axis=1
        )


    def get_signature(self, use_outlier_genes=False):
        if not use_outlier_genes:
            n_used_genes = (~self.adata.var["outlier"]).sum()
        else:
            n_used_genes = self.adata.n_vars

        loc = torch.empty(n_used_genes, self.n_cell_types)
        scale = torch.empty(n_used_genes, self.n_cell_types)

        for i, _ in enumerate(self.cell_types):
            loc[:, i] = self.X[i].mean(0)
            scale[:, i] = self.X[i].std(0)

        assert not torch.isnan(loc).any(), "NaN found in loc"
        assert not torch.isnan(scale).any(), "NaN found in scale"

        return loc, scale

    def _fit(self, model, y, num_epochs, lr):
        pbar = tqdm.tqdm(range(num_epochs))

        optim = torch.optim.Adam(model.parameters(), lr=lr)

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
                        "p": tl.fmt_c(model.get_proportions()),
                        "lib_size": f"{model.get_lib_size().item():.1f}",
                    }
                )

    def deconvolute(self, model_type: Literal["poisson", "normal", "mse"] = "poisson", num_epochs=5000, lr=0.01):
        n_bulk_samples = self.adata.varm["bulk"].shape[1]
        results = np.empty((n_bulk_samples, self.n_cell_types))
        loc, scale = self.get_signature()

        for sample_i, y in enumerate(self.Y):
            print(f"Sample: {sample_i}/{n_bulk_samples}")

            if model_type == "normal":
                model = base.NSM(
                    loc, scale,
                    gene_weights=self.gene_weights,
                    norm=self.normalising_constant[sample_i],
                )
            elif model_type == "mse":
                model = base.MSEM(self.loc)
            elif model_type == "poisson":
                model = base.PSM(
                    loc,
                    gene_weights=self.gene_weights,
                    norm=self.normalising_constant[sample_i],
                )
            else:
                assert False

            self._fit(model, y, num_epochs=num_epochs, lr=lr)
            proportions = model.get_proportions().detach().numpy()

            results[sample_i, :] = proportions


        df = pd.DataFrame(results, columns=self.cell_types, index=self.adata.uns["bulk_samples"])
        df = df.reindex(sorted(df.columns), axis=1)


        if n_bulk_samples > 1:
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

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
    def __init__(self, adata, cell_type_key, layer="ncounts", sub_type_key=None):
        self.use_sub_types = sub_type_key is not None
        self.layer = layer
        self.adata = adata

        self.cell_type_key = cell_type_key
        self.cell_types = self.adata.obs[cell_type_key].astype(str).astype("category").cat.categories.tolist()
        self.n_cell_types = len(self.cell_types)

        self.sub_type_key = sub_type_key
        self.sub_types = self.adata.obs[sub_type_key].astype("category").cat.categories.tolist() if self.use_sub_types else None
        self.n_sub_types = len(self.sub_types) if self.use_sub_types else None

        self.label_key = self.cell_type_key if not self.use_sub_types else self.sub_type_key
        self.labels  = self.cell_types if not self.use_sub_types else self.sub_types
        self.n_labels = len(self.labels)

        self.n_bulk_samples = self.adata.varm["bulk"].shape[1]

        self.eps = 1e-4

        self.init_stats()
        self.adata.var["outlier"] = False

    def init_stats(self):
        # Neighbors for clustering
        # if "X_pca" not in self.adata.obsm.keys():
        #     print("Computing PCA...")
        #     sc.pp.pca(self.adata)
        # if "distances" not in self.adata.obsp.keys():
        #     print("Computing neighbors...")
        #     sc.pp.neighbors(self.adata, random_state=0)
        # if "de" not in self.adata.uns.keys() or self.label_key not in self.adata.uns["de"].keys():
        # scout.tl.rank_marker_genes(self.adata, groupby=self.label_key)
        pass


    def filter_outliers(
        self,
        dropout_factor_quantile=0.99,
        pseudobulk_lims=(-10, 10),
        aggregate: Literal["mean", "median", "geom_mean", "min", "max"] = "mean"
    ):
        # self.adata.var["dispersion_outlier"] = (
        #     self.adata.var["dispersion_residual"] < dispersion_lims[0]
        # ) | (self.adata.var["dispersion_residual"] > dispersion_lims[1])
        if aggregate == "mean":
            dropout_factor = self.adata.varm[f"dropout_weight_{self.label_key}"].mean(1)
        elif aggregate == "median":
            dropout_factor = np.median(self.adata.varm[f"dropout_weight_{self.label_key}"], 1)
        elif aggregate == "geom_mean":
            dropout_factor = np.exp(np.mean(np.log(self.adata.varm[f"dropout_weight_{self.label_key}"]+1e-4), 1))
        elif aggregate == "min":
            dropout_factor = self.adata.varm[f"dropout_weight_{self.label_key}"].min(1)
        elif aggregate == "max":
            dropout_factor = self.adata.varm[f"dropout_weight_{self.label_key}"].max(1)

        _lim = np.quantile(self.adata.varm[f"dropout_weight_{self.label_key}"], dropout_factor_quantile, axis=0).mean()

        self.adata.varm["dropout_outlier"] = dropout_factor > _lim

        # self.adata.var["dropout_outlier"] = (
        #     dropout_factor > dropout_factor_lim
        # )# | (self.adata.var["dropout"] > dropout_lim)

        self.adata.varm["pseudobulk_outlier"] = (
            self.adata.varm["pseudo_factor"] < pseudobulk_lims[0]
        ) | (self.adata.varm["pseudo_factor"] > pseudobulk_lims[1])

        # self.adata.var["outlier"] = (
        #     self.adata.var["dispersion_outlier"] | self.adata.var["dropout_outlier"]
        # )

        self.adata.varm["outlier"] = (
            (self.adata.varm["dropout_outlier"] | self.adata.varm["pseudobulk_outlier"].T).T
        )
        # self.adata.var.loc[
        #     self.adata.var["ct_marker_abs_score_max"] > marker_zscore_lim, "outlier"
        # ] = False

    def init_dataset(
        self, weight_type=None, quantiles=(0.05, 0.95),
        weight_agg: Literal["min", "max", "mean", "geom_mean", "median"] = "mean", inverse_weight=False, log_weight=False,
    ):
        self.X = []

        for i, cell_type in enumerate(self.labels):
            _x = self.adata[self.adata.obs[self.label_key].cat.codes == i, :].layers[self.layer]
            self.X.append(_x)

        self.Y = torch.tensor(self.adata.varm["bulk"].T)

        assert (self.X[0].shape[1] == self.Y.shape[1]), f"{self.X[0].shape} != {self.Y.shape}"

        # ---------------------------
        #       Gene weights
        # ---------------------------

        if weight_type is not None:
            if f"{weight_type}_{self.label_key}" not in self.adata.varm_keys():
                scores = np.empty((self.adata.shape[1], self.n_labels))
                for i, ct in enumerate(self.labels):
                        scores[:, i] = self.adata.uns["de"][self.label_key][f"{ct} vs. rest"][weight_type].values
                        if "pvals_adj" == weight_type:
                            scores[:, i] = 1.0-scores[:, i]

            else:
                scores = self.adata.varm[f"{weight_type}_{self.label_key}"]

            # scores = pd.DataFrame(scores, index=self.adata.var_names, columns=self.cell_types)
            if weight_agg == "mean":
                gene_weights = scores.mean(axis=1)
            elif weight_agg == "geom_mean":
                gene_weights = np.exp(np.log(scores + self.eps).mean(axis=1))
            elif weight_agg == "min":
                gene_weights = scores.min(axis=1)
            elif weight_agg == "max":
                gene_weights = scores.max(axis=1)

            _min, _max = np.quantile(gene_weights, quantiles)
            gene_weights = gene_weights.clip(_min, _max)

            if log_weight:
                gene_weights = np.log1p(gene_weights)


            gene_weights = (gene_weights - gene_weights.min(0)) / (
                gene_weights.max(0) - gene_weights.min(0)
            )

            if inverse_weight:
                gene_weights = 1.0 - gene_weights

            self.adata.varm["gene_weights"] = gene_weights

        else:
            self.adata.varm["gene_weights"] = np.ones(self.adata.n_vars)

        # Normalising constant for loss function (makes loss comparable between datasets)
        # self.normalising_constant = np.nanmean(
        #     np.nan_to_num(self.adata.layers["counts"].sum(0) / self.adata.varm["bulk"].T, posinf=np.nan),
        #     axis=1
        # )
        self.normalising_constant = np.ones(1)


    def get_signature(self, quantiles=(0.0, 0.3), signature_type: Literal["mean", "median", "nanmean"] = "mean"):
        loc = torch.empty(self.adata.n_vars, self.n_labels)
        scale = torch.empty(self.adata.n_vars, self.n_labels)

        for i, _ in enumerate(self.labels):
            _x = self.X[i].copy()
            _min, _max = np.quantile(_x, quantiles, axis=0)
            mask = (_min <= _x) & (_x <= _max)
            _x[~mask] = np.nan
            _x[:, (np.isnan(_x).sum(0) == _x.shape[0])] = 0

            if signature_type == "mean":
                loc[:, i] = torch.tensor(self.X[i].mean(0))
            elif signature_type == "median":
                loc[:, i] = torch.tensor(np.quantile(self.X[i], 0.5, axis=0))
            elif signature_type == "nanmean":
                loc[:, i] = torch.tensor(np.nanmean(_x, 0))
                
            scale[:, i] = torch.tensor(self.X[i].std(0))

        assert not torch.isnan(loc).any(), "NaN found in loc"
        assert not torch.isnan(scale).any(), "NaN found in scale"

        return loc, scale

    def sum_sub_proportions(self, sub_proportions):
        d = {}
        for i, sub_type in enumerate(self.sub_types):
            cell_type = "_".join(sub_type.split("_")[:-1])
            if cell_type not in d.keys():
                d[cell_type] = sub_proportions[i]
            else:
                d[cell_type] += sub_proportions[i]

        return np.array([d[cell_type] for cell_type in self.cell_types])

    def _fit(self, model, y, num_epochs, lr, progress_bar=True):
        if progress_bar:
            pbar = tqdm.tqdm(range(num_epochs))
        else:
            pbar = range(num_epochs)

        optim = torch.optim.Adam(model.parameters(), lr=lr)

        for i in pbar:
            optim.zero_grad()
            loss = model(y)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.00001)
            optim.step()
            if progress_bar and (i % 50 == 0):
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.1f}",
                        "p": tl.fmt_c(model.get_proportions()),
                        "lib_size": f"{model.get_lib_size().item():.1f}",
                    }
                )

    def deconvolute(
        self, model_type: Literal["poisson", "normal", "lrm"] = "poisson", log_loss=False,
        num_epochs=5000, lr=0.1, use_outlier_genes=False, signature_quantiles=(0, 0.95),
        plot=False, progress_bar=True
    ):
        mu, scale = self.get_signature()

        res = np.empty((self.n_bulk_samples, self.n_labels))

        if model_type == "poisson":
            Model = base.PSM
        elif model_type == "normal":
            Model = base.NSM
        elif model_type == "lrm":
            Model = base.LRM

        for i in range(self.n_bulk_samples):
            print(f"Sample: {i+1}/{self.n_bulk_samples}", end="\r")
            bulk = self.adata.varm["bulk"][:, i]

            mask = (
                (bulk != 0) & 
                (~(mu.sum(1) == 0).numpy())
            )


            lrm = Model(self.n_labels, log=log_loss)

            lrm.fit(mu[mask], scale[mask], bulk[mask], num_iter=num_epochs, lr=lr, progress=progress_bar)

            res[i, :] = lrm.get_proportions().detach().numpy()

        return res

        # n_bulk_samples = self.adata.varm["bulk"].shape[1]
        # results = np.empty((n_bulk_samples, self.n_cell_types))
        # loc, scale = self.get_signature(quantiles=signature_quantiles)

        # for sample_i, y in enumerate(self.Y):
        #     if not use_outlier_genes:
        #         mask = ~self.adata.varm["outlier"][:, sample_i]
        #         gene_weights = torch.tensor(self.adata.varm["gene_weights"])
        #     else:
        #         mask = np.ones(self.adata.n_vars, dtype=bool)
        #         gene_weights = torch.tensor(self.adata.varm["gene_weights"])

        #     mask = mask & (~(loc.sum(1) == 0)).numpy()

        #     print(f"Sample: {sample_i+1}/{n_bulk_samples}", end=" | ")
        #     # if progress_bar:
        #     print(f"Using {mask.sum()} genes ({mask.sum()*100 / self.adata.n_vars:.1f}%)", end="\r")

        #     if model_type == "normal":
        #         model = base.NSM(
        #             loc[mask], scale[mask],
        #             gene_weights=gene_weights[mask],
        #             # norm=self.normalising_constant[sample_i],
        #         )
        #     elif model_type == "mse":
        #         model = base.MSEM(loc[mask], gene_weights=gene_weights[mask])
        #     elif model_type == "poisson":
        #         model = base.PSM(
        #             loc[mask],
        #             gene_weights=gene_weights[mask],
        #             #norm=self.normalising_constant[sample_i],
        #         )
        #     else:
        #         assert False

        #     self._fit(model, y[mask], num_epochs=num_epochs, lr=lr, progress_bar=progress_bar)
        #     proportions = model.get_proportions().detach().numpy()

        #     # if self.use_sub_types:
        #     #     for i, cell_type in enumerate(self.sub_types):
        #     #         print(f"{cell_type}: {proportions[i]:.3f} | ", end="")
        #     #     print()

        #     if self.use_sub_types:
        #         results[sample_i, :] = self.sum_sub_proportions(proportions)
        #         for i, cell_type in enumerate(self.cell_types):
        #             print(f"{cell_type}: {results[sample_i, i]:.3f} | ", end="")
        #         print()
        #     else:
        #         results[sample_i, :] = proportions

        # df = pd.DataFrame(results, columns=self.cell_types, index=self.adata.uns["bulk_samples"])
        # df = df.reindex(sorted(df.columns), axis=1)

        # if plot:
        #     if n_bulk_samples > 1:
        #         pl.proportions_heatmap(df)

        #     pl.bar_proportions(df)
        # return df

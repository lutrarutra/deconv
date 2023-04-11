import os, warnings
from typing import Literal

from . import plot as pl
from . import tools as tl

import numpy as np
import pandas as pd
import scanpy as sc

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist

from matplotlib import pyplot as plt

import tqdm

from . import models

import scout
class DeconV:
    def __init__(
            self, adata, cell_type_key,
            model_type: Literal["static", "beta", "gamma", "lognormal", "nb"] = "static",
            dropout_type: Literal["separate", "shared", None] = "separate",
            sub_type_key=None, device="cpu"
        ):
        self.model_type = model_type
        
        self.use_sub_types = sub_type_key is not None
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

        if self.model_type == "static":
            self.deconvolution_module = models.Static(self.adata, self.label_key, dropout_type=dropout_type, device=device)
        elif self.model_type == "beta":
            self.deconvolution_module = models.Beta(self.adata, self.label_key, dropout_type=dropout_type, device=device)
        elif self.model_type == "gamma":
            self.deconvolution_module = models.Gamma(self.adata, self.label_key, dropout_type=dropout_type, device=device)
        elif self.model_type == "lognormal":
            self.deconvolution_module = models.LogNormal(self.adata, self.label_key, dropout_type=dropout_type, device=device)
        elif self.model_type == "nb":
            self.deconvolution_module = models.NB(self.adata, self.label_key, dropout_type=dropout_type, device=device)
        else:
            raise ValueError("Unknown model type: {}".format(self.model_type))

    def fit_reference(self, lr=0.1, lrd=0.995, num_epochs=500, batch_size=None, seed=None, pyro_validation=True, layer="counts", fp_hack=False):
        self.deconvolution_module.fit_reference(
            lr=lr, lrd=lrd,
            num_epochs=num_epochs,
            batch_size=batch_size,
            seed=seed,
            pyro_validation=pyro_validation,
            layer=layer,
            fp_hack=fp_hack
        )

    def deconvolute(self, model_dropout=True, bulk=None, lr=0.1, lrd=0.995, num_epochs=1000, progress=True):
        self.deconvolution_module.deconvolute(
            model_dropout=model_dropout,
            bulk=bulk,
            lr=lr, lrd=lrd,
            num_epochs=num_epochs,
            progress=progress
        )

        proportions = self.deconvolution_module.get_proportions()
        if self.use_sub_types:
            proportions = self.sum_sub_proportions(proportions)

        return proportions
    
    def get_results_df(self, quantiles=(0.025, 0.975)):
        if self.deconvolution_module.concentrations is None:
            raise ValueError("Please run deconvolute() first.")
        
        limits = np.empty((self.n_bulk_samples, self.n_labels, 2))

        proportions = self.deconvolution_module.get_proportions().cpu()
        if self.use_sub_types:
            proportions = self.sum_sub_proportions(proportions).cpu()
        else:
            for i in range(self.n_bulk_samples):
                p_dist = dist.Dirichlet(self.deconvolution_module.concentrations[i])
                ps = p_dist.sample((10000,)).cpu()
                q = np.quantile(ps, quantiles, 0)
                limits[i, :, :] = q.T

            _min = pd.DataFrame(
                limits[:, :, 0],
                columns=self.adata.obs[self.cell_type_key].cat.categories.to_list(),
                index=self.adata.uns["bulk_samples"]
            ).reset_index().melt(id_vars="index")

            _max = pd.DataFrame(
                limits[:, :, 1],
                columns=self.adata.obs[self.cell_type_key].cat.categories.to_list(),
                index=self.adata.uns["bulk_samples"]
            ).reset_index().melt(id_vars="index")

        res_df = pd.DataFrame(
            proportions,
            columns=self.adata.obs[self.cell_type_key].cat.categories.to_list(),
            index=self.adata.uns["bulk_samples"]
        ).reset_index().melt(id_vars="index")

        res_df.rename(columns={"index": "sample", "value": "est", "variable":"cell_type"}, inplace=True)

        if not self.use_sub_types:
            # TODO: temp fix floating point precision...
            res_df["min"] = (res_df["est"] - _min["value"]).clip(lower=0.0)
            res_df["max"] = (_max["value"] - res_df["est"]).clip(lower=0.0)

        # assert res_df["min"].min() > 0, res_df["min"].min()
        # assert res_df["max"].min() > 0, res_df["max"].min()
        return res_df


    def sum_sub_proportions(self, sub_proportions):
        d = {}
        for i, sub_type in enumerate(self.sub_types):
            cell_type = "_".join(sub_type.split("_")[:-1])
            if cell_type not in d.keys():
                d[cell_type] = sub_proportions[:, i]
            else:
                d[cell_type] += sub_proportions[:, i]

        return np.array([d[cell_type] for cell_type in self.cell_types]).T
    
    def check_fit(self, path=None):
        f, ax = plt.subplots(self.n_labels, self.n_labels, figsize=(20, 20), dpi=100)
        res = scout.tl.rank_marker_genes(self.adata, self.label_key, copy=True)
        for i in range(self.n_labels):
            for j in range(self.n_labels):
                gene = res[f"{self.labels[i]} vs. rest"].sort_values("gene_score", ascending=False).index[0]
                ax[i,0].set_ylabel(gene)
                ax[self.n_labels-1,j].set_xlabel(self.labels[j])
                ax[i,j].set_yticks([])
                gene_i = self.adata.var_names.tolist().index(gene)
                self.deconvolution_module.plot_pdf(gene_i, j, ax=ax[i, j])

        if path is not None:
            plt.savefig(path, bbox_inches="tight")

        plt.show()
import os
from math import nan
from typing import Literal

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class PSM(nn.Module):
    def __init__(
        self,
        loc,
        gene_weights=None,
        gene_scale=None,
        lib_size=torch.tensor(1000),
        norm=torch.tensor(1.0),
    ):
        super().__init__()
        self.loc = loc
        self.gene_weights = gene_weights
        self.gene_scale = gene_scale
        self.norm = norm
        self.log_cell_counts = nn.Parameter(
            torch.ones(loc.shape[1]) * torch.log(lib_size / loc.shape[1])
        )
        self.eps = torch.tensor(1e-5)

    def get_proportions(self):
        with torch.no_grad():
            return F.softmax(self.log_cell_counts, 0)

    def get_lib_size(self):
        with torch.no_grad():
            return torch.sum(self.log_cell_counts.exp())

    def get_distribution(self):
        loc = torch.sum(self.loc * self.log_cell_counts.exp(), 1)

        assert torch.isnan(loc).sum() == 0

        return D.Poisson(loc)

    def forward(self, x):
        d = self.get_distribution()

        x = x.round()

        if self.gene_weights != None:
            return (-d.log_prob(x) * self.gene_weights).mean() * self.norm

        return -d.log_prob(x).mean() * self.norm


class NSM(nn.Module):
    def __init__(
        self,
        loc,
        scale,
        gene_weights=None,
        gene_scale=None,
        lib_size=torch.tensor(2000000),
        norm=torch.tensor(1.0),
    ):
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.gene_weights = gene_weights
        self.gene_scale = gene_scale
        self.log_cell_counts = nn.Parameter(
            torch.ones(loc.shape[1]) * torch.log(lib_size / loc.shape[1])
        )
        self.norm = norm
        self.eps = torch.tensor(1e-5)

    def get_proportions(self):
        with torch.no_grad():
            return F.softmax(self.log_cell_counts, 0)

    def get_lib_size(self):
        with torch.no_grad():
            return torch.sum(self.log_cell_counts.exp())

    def get_distribution(self):
        loc = torch.sum(self.loc * self.log_cell_counts.exp(), 1)
        scale = torch.max(
            self.eps,
            torch.sqrt(torch.sum(self.scale**2 * self.log_cell_counts.exp(), 1)),
        )

        assert torch.isnan(loc).sum() == 0

        return D.Normal(loc, scale)

    def forward(self, x):
        d = self.get_distribution()

        if self.gene_weights != None:
            return (-d.log_prob(x) * self.gene_weights).mean() * self.norm

        return -d.log_prob(x).mean() * self.norm


class MSEM(nn.Module):
    def __init__(self, loc, gene_weights=None):
        super().__init__()
        self.loc = loc
        self.gene_weights = gene_weights
        self.proportions = nn.Parameter(torch.ones(loc.shape[1]))
        self.lib_size = nn.Parameter(torch.tensor(8.0))

    def get_proportions(self, proportions=None):
        if weights is None:
            weights = self.weights

        proportions = F.relu(proportions)
        # return weights / weights.sum()
        return F.softmax(proportions, dim=0)

    def get_mean(self, lib_size=None, proportions=None):
        if lib_size is None:
            lib_size = self.lib_size
        if proportions is None:
            proportions = self.weights

        proportions = self.get_proportions()

        loc = lib_size.exp() * torch.sum(self.loc * proportions, 1)
        return loc

    def forward(self, x):
        loc = self.get_mean()

        return F.mse_loss(loc, x)


def create_dataset(
    adata,
    bulk_adata,
    genes=None,
    layer="counts",
    label_key="leiden",
    res_limits=(0.001, 1000),
):
    if genes is None:
        _adata = adata.copy()
        _bulk_adata = bulk_adata.copy()
    else:
        _adata = adata[:, adata.var.index.isin(genes)].copy()
        _bulk_adata = bulk_adata[:, bulk_adata.var.index.isin(genes)].copy()

    X = []

    pseudo = _adata.layers[layer].sum(0)

    for i, cell_type in enumerate(_adata.obs[label_key].cat.categories):
        _x = _adata[_adata.obs[label_key] == cell_type].layers[layer]
        X.append(torch.tensor(_x))

    Y = torch.tensor(_bulk_adata.layers[layer]).squeeze()

    c = torch.concat(X, dim=0)

    if res_limits != None:
        res = (pseudo / pseudo.max()) / (Y / Y.max())
        mask = (c == 0).sum(0) / c.shape[0] < 0.2
        X = [x[:, mask] for x in X]
        Y = Y[mask]
        pseudo = pseudo[mask]

    return X, Y, torch.tensor(pseudo)

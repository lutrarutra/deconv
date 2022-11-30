from math import nan

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from typing import Literal

import os

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class NSM(nn.Module):
    def __init__(self, loc, scale, gene_weights=None, gene_scale=None, norm=torch.tensor(1.0)):
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.gene_weights = gene_weights
        self.gene_scale = gene_scale
        self.norm = norm
        self.proportions = nn.Parameter(torch.ones(loc.shape[1]))
        self.lib_size = nn.Parameter(torch.tensor(8.0))
        self.eps = torch.tensor(1e-5)

    def get_proportions(self, proportions=None):
        if proportions is None:
            proportions = self.proportions

        proportions = F.relu(proportions)
        # return weights / weights.sum()
        return F.softmax(proportions, dim=0)

    def get_mean(self, lib_size=None, proportions=None):
        return self.get_distribution(lib_size, proportions).mean.detach()

    def get_distribution(self, lib_size=None, proportions=None):
        if lib_size is None:
            lib_size = self.lib_size
        if proportions is None:
            proportions = self.get_proportions()
       
        ls = lib_size.exp()

        loc = ls * torch.sum(self.loc * proportions, 1)
        scale = torch.max(self.eps, torch.sqrt(ls * torch.sum(self.scale**2 * proportions, 1)))

        assert torch.isnan(loc).sum() == 0, f"{ls}, {proportions}"

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

def create_dataset(adata, bulk_adata, genes=None, layer="counts", label_key="leiden", res_limits=(0.001, 1000)):
    if genes is None:
        _adata = adata.copy()
        _bulk_adata = bulk_adata.copy()
    else:
        _adata = adata[:,adata.var.index.isin(genes)].copy()
        _bulk_adata = bulk_adata[:,bulk_adata.var.index.isin(genes)].copy()

    X = []

    pseudo = _adata.layers[layer].sum(0)

    for i, cell_type in enumerate(_adata.obs[label_key].cat.categories):
        _x = _adata[_adata.obs[label_key] == cell_type].layers[layer]
        X.append(torch.tensor(_x))

    Y = torch.tensor(_bulk_adata.layers[layer]).squeeze()

    c = torch.concat(X, dim=0)

    if res_limits != None:
        res = (pseudo/pseudo.max()) / (Y / Y.max())
        mask = (c == 0).sum(0) / c.shape[0] < 0.2
        X = [x[:,mask] for x in X]
        Y = Y[mask]
        pseudo = pseudo[mask]

    return X, Y, torch.tensor(pseudo)
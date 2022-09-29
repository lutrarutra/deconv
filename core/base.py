from lib2to3.pytree import NegatedPattern
from math import nan
import scvi
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from typing import Literal

import os

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ProportionDecoder(nn.Module):
    def __init__(self, n_genes, n_cell_types, n_hidden=128):
        super().__init__()

        self.n_genes = n_genes
        self.n_cell_types = n_cell_types

        self.decoder = nn.Sequential(
            nn.Linear(self.n_genes, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, self.n_cell_types),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.decoder(x)


class Encoder(nn.Module):
    def __init__(self, n_input, n_hidden=128, n_latent=10):
        super().__init__()
        self.var_eps = 1e-4

        self.body = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(n_hidden, n_latent)
        self.var_head = nn.Linear(n_hidden, n_latent)

    def forward(self, x):
        _x = self.body(x)

        z_mu = self.mu_head(_x)
        z_var = torch.exp(self.var_head(_x)) + self.var_eps
        
        return z_mu, z_var

class DistributionDecoder(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=128, distribution: Literal["normal", "nb", "zinb", "poisson"] = "normal"):
        super().__init__()

        self.distribution = distribution

        self.body = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
        )

        if distribution == "normal":
            self.head = nn.ModuleDict({
                "mu": nn.Linear(n_hidden, n_output),
                "var": nn.Linear(n_hidden, n_output),
            })
            

        elif distribution == "nb":
            self.head = nn.ModuleDict({
                "mu": nn.Sequential(
                        nn.Linear(n_hidden, n_output),
                        nn.ReLU(),
                ),
                "scale": nn.Sequential(
                        nn.Linear(n_hidden, n_output),
                        nn.Softmax(dim=-1),
                ),
                "theta": nn.Sequential(
                        nn.Linear(n_hidden, n_output),
                        nn.ReLU(),
                ),
            })

        elif distribution == "zinb":
            self.head = nn.ModuleDict({
                "mu": nn.Sequential(
                        nn.Linear(n_hidden, n_output),
                        nn.ReLU(),
                ),
                "theta": nn.Sequential(
                        nn.Linear(n_hidden, n_output),
                        nn.ReLU(),
                ),
                "log_dropout": nn.Sequential(
                        nn.Linear(n_hidden, n_output),
                        nn.ReLU(),
                ),
                "scale": nn.Sequential(
                        nn.Linear(n_hidden, n_output),
                        nn.Softmax(dim=-1),
                )
            })

        
        elif distribution == "poisson":
            self.head = nn.ModuleDict({
                "rate": nn.Sequential(
                        nn.Linear(n_hidden, n_output),
                        nn.ReLU(),
                ),
                "scale": nn.Sequential(
                        nn.Linear(n_hidden, n_output),
                        nn.Softmax(dim=-1),
                ),
            })
            
        
        else:
            assert False, "Unknown distribution"

  
    def forward(self, x):
        _x = self.body(x)

        # Distribution parameters
        params = {}
        for param in self.head.keys():
            params[param] = self.head[param](_x)

        return params
        

class ParameterDecoder(nn.Module):
    def __init__(self, n_output, distribution: Literal["normal", "nb", "zinb", "poisson"] = "normal"):
        super().__init__()

        self.distribution = distribution

        if distribution == "normal":
            self.params = nn.ParameterDict({
                "mu": nn.Parameter(torch.ones(n_output, dtype=torch.float32)),
                "var": nn.Parameter(torch.ones(n_output, dtype=torch.float32)),
            })
            

        elif distribution == "nb":
            self.params = nn.ParameterDict({
                "mu": nn.Parameter(torch.ones(n_output, dtype=torch.float32)),
                "scale": nn.Parameter(torch.ones(n_output, dtype=torch.float32)),
                "theta": nn.Parameter(torch.ones(n_output, dtype=torch.float32))
            })

        elif distribution == "zinb":
            self.params = nn.ParameterDict({
                "mu": nn.Parameter(torch.ones(n_output, dtype=torch.float32)),
                "theta": nn.Parameter(torch.ones(n_output, dtype=torch.float32)),
                "log_dropout": nn.Parameter(torch.ones(n_output, dtype=torch.float32)),
                "scale": nn.Parameter(torch.ones(n_output, dtype=torch.float32))
            })
        
        elif distribution == "poisson":
            self.params = nn.ParameterDict({
                "rate": nn.Parameter(torch.ones(n_output, dtype=torch.float32)),
                "scale": nn.Parameter(torch.ones(n_output, dtype=torch.float32)),
            })
            
        
        else:
            assert False, "Unknown distribution"

        
    def forward(self, x):
        for p in self.parameters():
            p.data.clamp_(min=0.0)
        return self.params

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
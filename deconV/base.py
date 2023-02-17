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
        # print(self.log_cell_counts)
        self.eps = torch.tensor(1e-5)

    def get_proportions(self):
        with torch.no_grad():
            return F.softmax(self.log_cell_counts, 0)

    def get_lib_size(self):
        with torch.no_grad():
            return torch.sum(self.log_cell_counts.exp())

    def get_distribution(self):
        assert torch.isnan(self.loc).any() == False
        assert torch.isnan(self.log_cell_counts).any() == False, self.log_cell_counts

        loc = torch.sum(self.loc * self.log_cell_counts.exp(), 1)

        assert torch.isnan(loc).sum() == 0

        return D.Poisson(loc)

    def forward(self, x):
        d = self.get_distribution()

        x = x.round()

        if self.gene_weights is not None:
            return (-d.log_prob(x) * self.gene_weights).mean() * self.norm
        
        return -d.log_prob(x).mean() * self.norm


class NSM(nn.Module):
    def __init__(
        self,
        loc,
        scale,
        gene_weights,
        gene_scale=None,
        lib_size=torch.tensor(1000),
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

        if self.gene_weights is not None:
            return (-d.log_prob(x) * self.gene_weights).mean() * self.norm

        return -d.log_prob(x).mean() * self.norm


class MSEM(nn.Module):
    def __init__(self, loc, gene_weights=None, lib_size=torch.tensor(1000)):
        super().__init__()
        self.loc = loc
        self.gene_weights = gene_weights
        self.log_cell_counts = nn.Parameter(
            torch.ones(loc.shape[1]) * torch.log(lib_size / loc.shape[1])
        )
        self.lib_size = nn.Parameter(torch.tensor(8.0))

    def get_proportions(self):
        with torch.no_grad():
            return F.softmax(self.log_cell_counts, 0)

    def get_lib_size(self):
        with torch.no_grad():
            return torch.sum(self.log_cell_counts.exp())

    def get_mean(self, lib_size=None, proportions=None):
        loc = torch.sum(self.loc * self.log_cell_counts.exp(), 1)
        return loc

    def forward(self, x):
        loc = self.get_mean()

        loss = F.mse_loss(loc, x, reduction="none")
        if self.gene_weights is not None:
            loss = loss * self.gene_weights
        
        return loss.mean()


import tqdm
from typing import Literal

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from .base import Base, logits2probs

from matplotlib import pyplot as plt

class Static(Base):
    def __init__(self, adata, labels_key, dropout_type: Literal["separate", "shared", None] = "separate", device="cpu"):
        super().__init__(adata, labels_key, dropout_type, device)

        self.log_rate0 = 0.0 * torch.ones(self.n_genes, self.n_labels, device=self.device)


    def ref_model(self, sc_counts, labels):
        if self.ref_dropout_type == "separate":
            dropout_logits = pyro.param(
                "dropout_logits",
                torch.zeros(self.n_genes, self.n_labels, device=self.device),
                constraint=dist.constraints.real
            )
            dropout_logits = dropout_logits[:, labels].T

        elif self.ref_dropout_type == "shared":
            dropout_logits = pyro.param(
                "dropout_logits",
                torch.zeros(self.n_genes, device=self.device),
                constraint=dist.constraints.real
            )
            
        elif self.ref_dropout_type is not None:
            raise ValueError("Unknown dropout type")

        log_rate = pyro.param(
            "log_rate", self.log_rate0,
            dist.constraints.real
        )
            
        with pyro.plate("obs", len(labels), device=self.device), poutine.scale(scale=1.0/len(labels)):
            rate = log_rate.exp()[:, labels].T
            
            if self.ref_dropout_type is not None:
                obs_dist = dist.ZeroInflatedPoisson(
                    rate=rate, gate_logits=dropout_logits,
                ).to_event(1)
            else:
                obs_dist = dist.Poisson(
                    rate=rate
                ).to_event(1)

            pyro.sample("sc_obs", obs_dist, obs=sc_counts)
    
        
    def ref_guide(self, sc_counts, labels):
        pass


    def dec_model(self, bulk):
        n_samples = len(bulk)

        concentrations = pyro.param(
            "concentrations",
            torch.ones((n_samples, self.n_labels), device=self.device, dtype=torch.float64),
            constraint=dist.constraints.positive
        )

        cell_counts = pyro.param(
            "cell_counts",
            1e7 * torch.ones(n_samples, device=self.device),
            constraint=dist.constraints.positive
        )

        rate = self.params["log_rate"].exp()

        with pyro.plate("samples", n_samples, device=self.device):
            proportions = pyro.sample("proportions", dist.Dirichlet(concentrations))

            rate = torch.sum(proportions.unsqueeze(0) * rate.unsqueeze(1), dim=-1)
            rate = cell_counts * rate

            if self.dec_model_dropout:
                dropout = logits2probs(self.params["dropout_logits"])
                if self.ref_dropout_type == "separate":
                    dropout = torch.sum(proportions.unsqueeze(0) * dropout.unsqueeze(1), dim=-1)

                if dropout.dim() == 2:
                    dropout = dropout.T
                    
                bulk_dist = dist.ZeroInflatedPoisson(rate=rate.T, gate_logits=dropout).to_event(1)
            else:
                bulk_dist = dist.Poisson(rate=rate.T).to_event(1)

            pyro.sample("bulk", bulk_dist, obs=bulk)


    def dec_guide(self, bulk):
        n_samples = len(bulk)
        
        concentrations = pyro.param(
            "concentrations",
            torch.ones((n_samples, self.n_labels), device=self.device, dtype=torch.float64),
            constraint=dist.constraints.positive
        )

        with pyro.plate("samples", n_samples, device=self.device):
            pyro.sample("proportions", dist.Dirichlet(concentrations))

    def pseudo_bulk(self):
        if self.concentrations is None:
            raise ValueError("Run deconvolute() first")
        
        theta = self.params["log_rate"].exp()

        proportions = self.get_proportions()
        rate = torch.sum(proportions.unsqueeze(0) * theta.unsqueeze(1), dim=-1)
        rate = self.get_cell_counts() * rate

        if self.dec_model_dropout:
            dropout = logits2probs(self.params["dropout_logits"])
            if self.ref_dropout_type == "separate":
                dropout = torch.sum(proportions.unsqueeze(0) * dropout.unsqueeze(1), dim=-1)

            if dropout.dim() == 2:
                dropout = dropout.T
                
            bulk_dist = dist.ZeroInflatedPoisson(rate=rate.T, gate_logits=dropout)
        else:
            bulk_dist = dist.Poisson(rate=rate.T)

        return bulk_dist.mean


    def plot_pdf(self, gene_i, ct_i, n_samples=5000, ax=None):
        gex = self.adata[self.adata.obs[self.labels_key].cat.codes == ct_i, gene_i].layers["counts"].toarray()
        rate = self.params["log_rate"][gene_i, ct_i].exp()

        if self.ref_dropout_type is not None:
            if self.ref_dropout_type == "separate":
                dropout = logits2probs(self.params["dropout_logits"][gene_i, ct_i])
            elif self.ref_dropout_type == "shared":
                dropout = logits2probs(self.params["dropout_logits"][gene_i])
            else:
                raise ValueError("Unknown dropout type")
            
            x = dist.ZeroInflatedPoisson(rate=rate, gate=dropout).sample((n_samples,)).cpu()
        else:
            x = dist.Poisson(rate=rate).sample((n_samples,)).cpu()


        if ax is None:
            f, ax = plt.subplots(figsize=(4,4), dpi=120)

        counts, bins = np.histogram(gex, bins=20, density=True)

        ax.hist(bins[:-1], bins, weights=counts, alpha=0.8, color='red')
        ax.hist(x, bins=bins, density=True, alpha=0.6, color='orange')
        ax.axvline(gex.mean(), color='royalblue')
        return ax
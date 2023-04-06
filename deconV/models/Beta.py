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

class Beta(Base):
    def __init__(self, adata, labels_key, dropout_type: Literal["separate", "shared", None] = "separate", device="cpu"):
        super().__init__(adata, labels_key, dropout_type, device)

        self.alpha0 = 1.0 * torch.ones((self.n_genes, self.n_labels), device=self.device, dtype=torch.float64)
        self.beta0 = 50.0 * torch.ones((self.n_genes, self.n_labels), device=self.device, dtype=torch.float64)

        self.mu_lib_size0 = 10 * torch.ones(self.n_labels, device=self.device, dtype=torch.float64)
        self.std_lib_size0 = 0.1 * torch.ones(self.n_labels, device=self.device, dtype=torch.float64)


    def ref_model(self, sc_counts, labels):
        if self.ref_dropout_type == "separate":
            dropout_logits = pyro.param(
                "dropout_logits",
                torch.zeros(self.n_genes, self.n_labels, device=self.device, dtype=torch.float64),
                constraint=dist.constraints.real
            )
            dropout_logits = dropout_logits[:, labels].T

        elif self.ref_dropout_type == "shared":
            dropout_logits = pyro.param(
                "dropout_logits",
                torch.zeros(self.n_genes, device=self.device, dtype=torch.float64),
                constraint=dist.constraints.real
            )
        elif self.ref_dropout_type is not None:
            raise ValueError("Unknown dropout type")
        
        alpha = pyro.param(
            "alpha", self.alpha0,
            dist.constraints.positive
        )

        beta = pyro.param(
            "beta", self.beta0,
            dist.constraints.positive
        )

        mu_lib_size = pyro.param(
            "mu_lib_size", self.mu_lib_size0, 
            constraint=dist.constraints.positive
        )

        std_lib_size = pyro.param(
            "std_lib_size", self.std_lib_size0,
            constraint=dist.constraints.positive
        )

        with pyro.plate("labels", self.n_labels, device=self.device):
            lib_size = pyro.sample("lib_size", dist.Gamma(mu_lib_size, std_lib_size))
            with pyro.plate("genes", self.n_genes, device=self.device):
                theta = pyro.sample("theta", dist.Beta(alpha, beta))
            
        with pyro.plate("obs", len(labels), device=self.device), poutine.scale(scale=1.0/len(labels)):
            rate = theta[:, labels].T * lib_size[labels].unsqueeze(-1)
            
            if self.ref_dropout_type is not None:
                obs_dist = dist.ZeroInflatedPoisson(
                    rate=rate, gate_logits=dropout_logits,
                ).to_event(1)
            else:
                obs_dist = dist.Poisson(
                    rate=rate
                ).to_event(1)
            
            pyro.sample("sc", obs_dist, obs=sc_counts)
        
    
        
    def ref_guide(self, sc_counts, labels):
        alpha = pyro.param(
            "alpha", self.alpha0,
            dist.constraints.positive
        )

        beta = pyro.param(
            "beta", self.beta0,
            dist.constraints.positive
        )

        mu_lib_size = pyro.param(
            "mu_lib_size", self.mu_lib_size0, 
            constraint=dist.constraints.positive
        )

        std_lib_size = pyro.param(
            "std_lib_size", self.std_lib_size0,
            constraint=dist.constraints.positive
        )

        with pyro.plate("labels", self.n_labels, device=self.device):
            pyro.sample("lib_size", dist.Gamma(mu_lib_size, std_lib_size))
            with pyro.plate("genes", self.n_genes, device=self.device):
                assert torch.isnan(alpha).sum() == 0, torch.isnan(alpha).nonzero(as_tuple=True)[0]
                pyro.sample("theta", dist.Beta(alpha, beta))
            

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

        alpha = self.params["alpha"]
        beta = self.params["beta"]

        mu_lib_size = self.params["mu_lib_size"]
        std_lib_size = self.params["std_lib_size"]
        
        with pyro.plate("labels", self.n_labels, device=self.device):
            lib_size = pyro.sample("lib_size", dist.Gamma(mu_lib_size, std_lib_size))
            with pyro.plate("genes", self.n_genes, device=self.device):
                theta = pyro.sample("theta", dist.Beta(alpha, beta))

        with pyro.plate("samples", n_samples, device=self.device):
            proportions = pyro.sample("proportions", dist.Dirichlet(concentrations))

            ct_rate = lib_size * theta
            rate = torch.sum(proportions.unsqueeze(0) * ct_rate.unsqueeze(1), dim=-1)
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

        alpha = self.params["alpha"]
        beta = self.params["beta"]

        mu_lib_size = self.params["mu_lib_size"]
        std_lib_size = self.params["std_lib_size"]

        with pyro.plate("labels", self.n_labels, device=self.device):
            pyro.sample("lib_size", dist.Gamma(mu_lib_size, std_lib_size))
            with pyro.plate("genes", self.n_genes, device=self.device):
                pyro.sample("theta", dist.Beta(alpha, beta))

    def pseudo_bulk(self):
        if self.concentrations is None:
            raise ValueError("Run deconvolute() first")
        
        alpha = self.params["alpha"]
        beta = self.params["beta"]

        mu_lib_size = self.params["mu_lib_size"]
        std_lib_size = self.params["std_lib_size"]

        lib_size = dist.Gamma(mu_lib_size, std_lib_size).mean
        theta = dist.Beta(alpha, beta).mean

        proportions = self.get_proportions()
        ct_rate = lib_size * theta
        rate = torch.sum(proportions.unsqueeze(0) * ct_rate.unsqueeze(1), dim=-1)
        rate = self.get_cell_counts() * rate

        if self.dec_model_dropout:
            dropout = logits2probs(self.params["dropout_logits"])
            if self.ref_dropout_type == "separate":
                dropout = torch.sum(proportions.unsqueeze(0) * dropout.unsqueeze(1), dim=-1)
                
            bulk_dist = dist.ZeroInflatedPoisson(rate=rate.T, gate_logits=dropout.T)
        else:
            bulk_dist = dist.Poisson(rate=rate.T)

        return bulk_dist.mean


    def plot_pdf(self, gene_i, ct_i, n_samples=5000, ax=None):
        gex = self.adata[self.adata.obs[self.labels_key].cat.codes == ct_i, gene_i].layers["counts"].toarray()

        a = self.params["alpha"][gene_i, ct_i]
        b = self.params["beta"][gene_i, ct_i]
        theta = dist.Beta(a, b).sample((n_samples,))

        mu_lib_size = self.params["mu_lib_size"][ct_i]
        std_lib_size = self.params["std_lib_size"][ct_i]
        lib_size = dist.Gamma(mu_lib_size, std_lib_size).sample((n_samples,))
        rate = theta * lib_size

        if self.ref_dropout_type is not None:
            if self.ref_dropout_type == "separate":
                dropout_logits = self.params["dropout_logits"][gene_i, ct_i]
            elif self.ref_dropout_type == "shared":
                dropout_logits = self.params["dropout_logits"][gene_i]
            else:
                raise ValueError("Unknown dropout type")
            x = dist.ZeroInflatedPoisson(rate=rate, gate_logits=dropout_logits).sample().cpu()
        else:
            x = dist.Poisson(rate=rate).sample().cpu()


        if ax is None:
            f, ax = plt.subplots(figsize=(4,4), dpi=120)

        counts, bins = np.histogram(gex, bins=20, density=True)

        ax.hist(bins[:-1], bins, weights=counts, alpha=0.8, color='red')
        ax.hist(x, bins=bins, density=True, alpha=0.6, color='orange')
        ax.axvline(gex.mean(), color='royalblue')
        return ax
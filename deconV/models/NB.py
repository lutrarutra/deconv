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

class NB(Base):
    def __init__(self, adata, labels_key, dropout_type: Literal["separate", "shared", None] = "separate", device="cpu"):
        super().__init__(adata, labels_key, dropout_type, device)

        self.mu_total_count0 = 5 * torch.ones((self.n_labels, self.n_genes), device=self.device)
        self.std_total_count0 = 1 * torch.ones((self.n_labels, self.n_genes), device=self.device)

        self.alpha0 = 1.0 * torch.ones(self.n_genes, device=self.device)
        self.beta0 = 50.0 * torch.ones(self.n_genes, device=self.device)


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

        alpha = pyro.param(
            "alpha", self.alpha0,
            dist.constraints.positive
        )

        beta = pyro.param(
            "beta", self.beta0,
            dist.constraints.positive
        )

        mu_total_count = pyro.param(
            "mu_total_count",
            self.mu_total_count0,
            constraint=dist.constraints.real
        )

        std_total_count = pyro.param(
            "std_total_count",
            self.std_total_count0,
            constraint=dist.constraints.positive
        )

        with pyro.plate("genes", self.n_genes, device=self.device):
            probs = pyro.sample("probs", dist.Beta(alpha, beta))
            with pyro.plate("labels", self.n_labels, device=self.device):
                total_count = pyro.sample("total_count", dist.LogNormal(mu_total_count, std_total_count))

            
            with pyro.plate("cells", len(labels), device=self.device), poutine.scale(scale=1.0/len(labels)):
                total_count = total_count[labels,:]

                if self.ref_dropout_type is not None:
                    obs_dist = dist.ZeroInflatedNegativeBinomial(
                        total_count=total_count,
                        probs=probs,
                        gate_logits=dropout_logits,
                    )
                else:
                    obs_dist = dist.NegativeBinomial(
                        total_count=total_count,
                        probs=probs,
                    )

                pyro.sample("sc_obs", obs_dist, obs=sc_counts)
    
        
    def ref_guide(self, sc_counts, labels):
        alpha = pyro.param(
            "alpha", self.alpha0,
            dist.constraints.positive
        )

        beta = pyro.param(
            "beta", self.beta0,
            dist.constraints.positive
        )

        mu_total_count = pyro.param(
            "mu_total_count",
            self.mu_total_count0,
            constraint=dist.constraints.real
        )

        std_total_count = pyro.param(
            "std_total_count",
            self.std_total_count0,
            constraint=dist.constraints.positive
        )

        with pyro.plate("genes", self.n_genes, device=self.device):
            pyro.sample("probs", dist.Beta(alpha, beta))
            with pyro.plate("labels", self.n_labels, device=self.device):
                pyro.sample("total_count", dist.LogNormal(mu_total_count, std_total_count))

        # with pyro.plate("genes", self.n_genes, device=self.device):
        #     pyro.sample("probs", dist.Beta(alpha, beta))

    def dec_model(self, bulk):
        n_samples = len(bulk)

        concentrations = pyro.param(
            "concentrations",
            torch.ones((n_samples, self.n_labels), device=self.device, dtype=torch.float64),
            constraint=dist.constraints.positive
        )

        cell_counts = pyro.param(
            "cell_counts",
            1e7 * torch.ones(n_samples, device=self.device, dtype=torch.float64),
            constraint=dist.constraints.positive
        )

        alpha = self.params["alpha"]
        beta = self.params["beta"]

        mu_total_count = self.params["mu_total_count"]
        std_total_count = self.params["std_total_count"]

        with pyro.plate("genes", self.n_genes, device=self.device):
            probs = pyro.sample("probs", dist.Beta(alpha, beta))
            with pyro.plate("labels", self.n_labels, device=self.device):
                ct_total_count = pyro.sample("total_count", dist.LogNormal(mu_total_count, std_total_count))

        assert torch.isnan(probs).sum() == 0, torch.isnan(probs).nonzero(as_tuple=True)[0]
        assert torch.isnan(ct_total_count).sum() == 0, torch.isnan(ct_total_count).nonzero(as_tuple=True)[0]

        with pyro.plate("samples", n_samples, device=self.device):
            proportions = pyro.sample("proportions", dist.Dirichlet(concentrations))
            total_count = torch.sum(proportions.unsqueeze(-1) * ct_total_count.unsqueeze(0), dim=1)
            total_count = cell_counts * total_count.T

            if self.dec_model_dropout:
                dropout = logits2probs(self.params["dropout_logits"])
                if self.ref_dropout_type == "separate":
                    dropout = torch.sum(proportions.unsqueeze(0) * dropout.unsqueeze(1), dim=-1)
                
                bulk_dist = dist.ZeroInflatedNegativeBinomial(
                    total_count=total_count.T,
                    probs=probs,
                    gate_logits=dropout.T
                ).to_event(1)
            else:
                bulk_dist = dist.NegativeBinomial(
                    total_count=total_count.T,
                    probs=probs,
                ).to_event(1)

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

        mu_total_count = self.params["mu_total_count"]
        std_total_count = self.params["std_total_count"]

        with pyro.plate("genes", self.n_genes, device=self.device):
            pyro.sample("probs", dist.Beta(alpha, beta))
            with pyro.plate("labels", self.n_labels, device=self.device):
                pyro.sample("total_count", dist.LogNormal(mu_total_count, std_total_count))
        
        # with pyro.plate("genes", self.n_genes, device=self.device):
        #     pyro.sample("probs", dist.Beta(alpha, beta))

    def pseudo_bulk(self):
        if self.concentrations is None:
            raise ValueError("Run deconvolute() first")
        
        alpha = self.params["alpha"]
        beta = self.params["beta"]

        probs = dist.Beta(alpha, beta).mean

        mu_total_count = self.params["mu_total_count"]
        std_total_count = self.params["std_total_count"]
        ct_total_count = dist.LogNormal(mu_total_count, std_total_count).mean

        proportions = self.get_proportions()
        total_count = torch.sum(proportions.unsqueeze(-1) * ct_total_count.unsqueeze(0), dim=1)
        total_count = self.get_cell_counts() * total_count.T

        if self.dec_model_dropout:
            dropout = logits2probs(self.params["dropout_logits"])
            if self.ref_dropout_type == "separate":
                dropout = torch.sum(proportions.unsqueeze(0) * dropout.unsqueeze(1), dim=-1)

            if dropout.dim() == 2:
                dropout = dropout.T
                    

            bulk_dist = dist.ZeroInflatedNegativeBinomial(
                total_count=total_count.T,
                probs=probs,
                gate=dropout
            )
        else:
            bulk_dist = dist.NegativeBinomial(
                total_count=total_count.T,
                probs=probs,
            )

        return bulk_dist.mean


    def plot_pdf(self, gene_i, ct_i, n_samples=5000, ax=None):
        gex = self.adata[self.adata.obs[self.labels_key].cat.codes == ct_i, gene_i].layers["counts"].toarray()

        a = self.params["alpha"][gene_i]
        b = self.params["beta"][gene_i]
        mu_total_count = self.params["mu_total_count"][ct_i, gene_i]
        std_total_count = self.params["std_total_count"][ct_i, gene_i]
        probs = dist.Beta(a, b).sample((n_samples,))

        total_count = dist.LogNormal(mu_total_count, std_total_count).sample((n_samples,))

        if self.ref_dropout_type is not None:
            if self.ref_dropout_type == "separate":
                dropout_logits = self.params["dropout_logits"][gene_i, ct_i]
            elif self.ref_dropout_type == "shared":
                dropout_logits = self.params["dropout_logits"][gene_i]
            else:
                raise ValueError("Unknown dropout type")
            x = dist.ZeroInflatedNegativeBinomial(
                total_count=total_count,
                probs=probs,
                gate_logits=dropout_logits.T
            ).sample().cpu()
        else:
            x = dist.NegativeBinomial(
                total_count=total_count,
                probs=probs,
            ).sample().cpu()

        if ax is None:
            f, ax = plt.subplots(figsize=(4,4), dpi=120)

        counts, bins = np.histogram(gex, bins=20, density=True)

        ax.hist(bins[:-1], bins, weights=counts, alpha=0.8, color='red')
        ax.hist(x, bins=bins, density=True, alpha=0.6, color='orange')
        ax.axvline(gex.mean(), color='royalblue')
        return ax
    
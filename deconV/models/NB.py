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

        self.log_total_count0 = 5 * torch.ones(self.n_genes, self.n_labels, device=self.device)

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
            dropout_logits = dropout_logits.T
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

        log_total_count = pyro.param(
            "log_total_count",
            self.log_total_count0,
            constraint=dist.constraints.real
        )

        with pyro.plate("genes", self.n_genes, device=self.device):
            probs = pyro.sample("probs", dist.Beta(alpha, beta))
            
        with pyro.plate("obs", len(labels), device=self.device), poutine.scale(scale=1.0/len(labels)):
            total_count = log_total_count.exp()[:, labels].T
            if self.ref_dropout_type is not None:
                obs_dist = dist.ZeroInflatedNegativeBinomial(
                    total_count=total_count,
                    probs=probs,
                    gate_logits=dropout_logits,
                ).to_event(1)
            else:
                obs_dist = dist.NegativeBinomial(
                    total_count=total_count,
                    probs=probs,
                ).to_event(1)

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

        with pyro.plate("genes", self.n_genes, device=self.device):
            pyro.sample("probs", dist.Beta(alpha, beta))

    def dec_model(self, bulk):
        n_samples = len(bulk)

        log_concentration = torch.zeros((n_samples, self.n_labels), device=self.device)

        log_cell_counts = pyro.param(
            "log_cell_counts",
            7.0 * torch.ones(n_samples, device=self.device),
            constraint=dist.constraints.positive
        )

        alpha = self.params["alpha"]
        beta = self.params["beta"]

        ct_total_count = self.params["log_total_count"].exp()
        
        with pyro.plate("genes", self.n_genes, device=self.device):
            probs = pyro.sample("probs", dist.Beta(alpha, beta))

        with pyro.plate("samples", n_samples, device=self.device):
            proportions = pyro.sample("proportions", dist.Dirichlet(log_concentration.exp()))

            total_count = torch.sum(proportions.unsqueeze(0) * ct_total_count.unsqueeze(1), dim=-1)
            total_count = log_cell_counts.exp() * total_count

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
        
        log_concentration = pyro.param(
            "log_concentration",
            torch.zeros((n_samples, self.n_labels), device=self.device),
            constraint=dist.constraints.real
        )

        with pyro.plate("samples", n_samples, device=self.device):
            pyro.sample("proportions", dist.Dirichlet(log_concentration.exp()))

        alpha = self.params["alpha"]
        beta = self.params["beta"]
        
        with pyro.plate("genes", self.n_genes, device=self.device):
            pyro.sample("probs", dist.Beta(alpha, beta))

    def pseudo_bulk(self, n_samples=1000):
        if self.log_concentrations is None:
            raise ValueError("Run deconvolute() first")
        
        alpha = self.params["alpha"]
        beta = self.params["beta"]

        ct_total_count = self.params["log_total_count"].exp()

        probs = dist.Beta(alpha, beta).mean

        proportions = self.get_proportions()
        total_count = torch.sum(proportions.unsqueeze(0) * ct_total_count.unsqueeze(1), dim=-1)
        total_count = self.get_cell_counts() * total_count

        if self.dec_model_dropout:
            dropout = logits2probs(self.params["dropout_logits"])
            if self.ref_dropout_type == "separate":
                dropout = torch.sum(proportions.unsqueeze(0) * dropout.unsqueeze(1), dim=-1)

            bulk_dist = dist.ZeroInflatedNegativeBinomial(
                total_count=total_count.T,
                probs=probs,
                gate=dropout.T
            )
        else:
            bulk_dist = dist.NegativeBinomial(
                total_count=total_count.T,
                probs=probs,
            )

        return bulk_dist.sample((n_samples,)).mean(0)


    def plot_pdf(self, gene_i, ct_i, n_samples=5000, ax=None):
        gex = self.adata[self.adata.obs[self.labels_key].cat.codes == ct_i, gene_i].layers["counts"].toarray()

        a = self.params["alpha"][gene_i]
        b = self.params["beta"][gene_i]
        probs = dist.Beta(a, b).sample((n_samples,))

        total_count = self.params["log_total_count"][gene_i, ct_i].exp()

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
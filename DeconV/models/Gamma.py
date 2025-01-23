from typing import Literal

import numpy as np
import scanpy as sc
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from .base import ReferenceModel, DeconvolutionModel, logits2probs

from matplotlib import pyplot as plt


class GammaRefModel(ReferenceModel):
    def __init__(self, adata: sc.AnnData, labels_key: str, dropout_type: Literal["separate", "shared", None], device: Literal["cpu", "cuda"], layer: str | None, fp_hack: bool):
        super().__init__(adata=adata, labels_key=labels_key, dropout_type=dropout_type, device=device, layer=layer, fp_hack=fp_hack)
        self.alpha0 = 1.0 * torch.ones(self.n_genes, self.n_labels, device=self.device)
        self.beta0 = 0.1 * torch.ones(self.n_genes, self.n_labels, device=self.device)
        self.dec_model_cls = GammaDecModel

    def get_params(self, genes: list[str] | np.ndarray | None) -> dict[str, torch.Tensor]:
        if self._params is None:
            raise ValueError("Run fit_reference() first")
        
        alpha = self._params["alpha"]
        beta = self._params["beta"]

        if self.dropout_enabled:
            dropout_logits = self._params["dropout_logits"]

        if genes is not None:
            gene_idx = [self.var_names.get_loc(g) for g in genes]

            alpha = alpha[gene_idx]
            beta = beta[gene_idx]

            if self.dropout_enabled:
                dropout_logits = dropout_logits[gene_idx]

        p = dict(alpha=alpha, beta=beta)
        if self.dropout_enabled:
            p["dropout_logits"] = dropout_logits

        return p

    def model(self, sc_counts: torch.Tensor, labels: torch.Tensor):
        if self.dropout_type == "separate":
            dropout_logits = pyro.param(
                "dropout_logits",
                torch.zeros(self.n_genes, self.n_labels, device=self.device),
                constraint=dist.constraints.real
            )
            dropout_logits = dropout_logits[:, labels].T

        elif self.dropout_type == "shared":
            dropout_logits = pyro.param(
                "dropout_logits",
                torch.zeros(self.n_genes, device=self.device),
                constraint=dist.constraints.real
            )

        elif self.dropout_type is not None:
            raise ValueError("Unknown dropout type")

        alpha = pyro.param(
            "alpha", self.alpha0,
            dist.constraints.positive
        )

        beta = pyro.param(
            "beta", self.beta0,
            dist.constraints.positive
        )
        
        with pyro.plate("labels", self.n_labels, device=self.device):
            with pyro.plate("genes", self.n_genes, device=self.device):
                theta = pyro.sample("theta", dist.Gamma(alpha, beta))
            
        with pyro.plate("obs", len(labels), device=self.device), poutine.scale(scale=1.0 / len(labels)):
            rate = theta[:, labels].T
            
            if self.dropout_type is not None:
                obs_dist = dist.ZeroInflatedPoisson(
                    rate=rate, gate_logits=dropout_logits,
                ).to_event(1)
            else:
                obs_dist = dist.Poisson(
                    rate=rate
                ).to_event(1)

            pyro.sample("sc_obs", obs_dist, obs=sc_counts)
        
    def guide(self, sc_counts: torch.Tensor, labels: torch.Tensor):
        alpha = pyro.param(
            "alpha", self.alpha0,
            dist.constraints.positive
        )

        beta = pyro.param(
            "beta", self.beta0,
            dist.constraints.positive
        )

        with pyro.plate("labels", self.n_labels, device=self.device):
            with pyro.plate("genes", self.n_genes, device=self.device):
                pyro.sample("theta", dist.Gamma(alpha, beta))

    def pseudo_bulk(self, ref_params: dict[str, torch.Tensor], model_dropout: bool):
        pass
    #     if self.concentrations is None:
    #         raise ValueError("Run deconvolute() first")
    #     alpha = ref_params["alpha"]
    #     beta = ref_params["beta"]
        
    #     theta = dist.Gamma(alpha, beta).mean

    #     proportions = self.get_proportions()
    #     rate = torch.sum(proportions.unsqueeze(0) * theta.unsqueeze(1), dim=-1)
    #     rate = self.get_cell_counts() * rate

    #     if model_dropout:
    #         dropout = logits2probs(ref_params["dropout_logits"])
    #         if self.dropout_type == "separate":
    #             dropout = torch.sum(proportions.unsqueeze(0) * dropout.unsqueeze(1), dim=-1)
                
    #         if dropout.dim() == 2:
    #             dropout = dropout.T

    #         bulk_dist = dist.ZeroInflatedPoisson(rate=rate.T, gate_logits=dropout)
    #     else:
    #         bulk_dist = dist.Poisson(rate=rate.T)

    #     return bulk_dist.mean

    def plot_pdf(self, adata: sc.AnnData, cell_type: str, gene: str, ax: plt.Axes, n_samples: int = 5000):
        if self._params is None:
            raise ValueError("Run fit_reference() first")
        
        gene_idx = self.var_names.get_loc(gene)
        cell_type_idx = adata.obs[self.labels_key].cat.categories.get_loc(cell_type)
        
        gex = adata[adata.obs[self.labels_key] == cell_type, gene].layers["counts"].toarray()  # type: ignore

        a = self._params["alpha"][gene_idx, cell_type_idx]
        b = self._params["beta"][gene_idx, cell_type_idx]
        theta = dist.Gamma(a, b).sample(torch.Size((n_samples,)))

        if self.dropout_type is not None:
            if self.dropout_type == "separate":
                dropout_logits = self._params["dropout_logits"][gene_idx, cell_type_idx]
            elif self.dropout_type == "shared":
                dropout_logits = self._params["dropout_logits"][gene_idx]
            else:
                raise ValueError("Unknown dropout type")
            x = dist.ZeroInflatedPoisson(rate=theta, gate_logits=dropout_logits).sample().cpu()
        else:
            x = dist.Poisson(rate=theta).sample().cpu()

        counts, bins = np.histogram(gex, bins=20, density=True)

        ax.hist(bins[:-1], bins, weights=counts, alpha=0.8, color='red')
        ax.hist(x, bins=bins, density=True, alpha=0.6, color='orange')
        ax.axvline(gex.mean(), color='royalblue')
        return ax


class GammaDecModel(DeconvolutionModel):
    def model(self):
        concentrations = pyro.param(
            "concentrations",
            torch.ones((self.n_samples, self.n_labels), device=self.device, dtype=torch.float64),
            constraint=dist.constraints.positive
        )

        cell_counts = pyro.param(
            "cell_counts",
            1e7 * torch.ones(self.n_samples, device=self.device),
            constraint=dist.constraints.positive
        )

        alpha = self.ref_params["alpha"]
        beta = self.ref_params["beta"]
        
        with pyro.plate("labels", self.n_labels, device=self.device):
            with pyro.plate("genes", self.n_genes, device=self.device):
                theta = pyro.sample("theta", dist.Gamma(alpha, beta))

        with pyro.plate("samples", self.n_samples, device=self.device):
            proportions = pyro.sample("proportions", dist.Dirichlet(concentrations))

            rate = torch.sum(proportions.unsqueeze(0) * theta.unsqueeze(1), dim=-1)
            rate = cell_counts * rate

            if self.model_dropout:
                dropout = logits2probs(self.ref_params["dropout_logits"])
                if self.dropout_type == "separate":
                    dropout = torch.sum(proportions.unsqueeze(0) * dropout.unsqueeze(1), dim=-1)
                    
                if dropout.dim() == 2:
                    dropout = dropout.T

                bulk_dist = dist.ZeroInflatedPoisson(rate=rate.T, gate_logits=dropout).to_event(1)
            else:
                bulk_dist = dist.Poisson(rate=rate.T).to_event(1)

            pyro.sample("bulk", bulk_dist, obs=self.bulk_gex)

    def guide(self):
        concentrations = pyro.param(
            "concentrations",
            torch.ones((self.n_samples, self.n_labels), device=self.device, dtype=torch.float64),
            constraint=dist.constraints.positive
        )

        with pyro.plate("samples", self.n_samples, device=self.device):
            pyro.sample("proportions", dist.Dirichlet(concentrations))

        alpha = self.ref_params["alpha"]
        beta = self.ref_params["beta"]

        with pyro.plate("labels", self.n_labels, device=self.device):
            with pyro.plate("genes", self.n_genes, device=self.device):
                pyro.sample("theta", dist.Gamma(alpha, beta))
from typing import Literal

import numpy as np
import scanpy as sc
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from .base import ReferenceModel, DeconvolutionModel, logits2probs

from matplotlib import pyplot as plt


class NBRefModel(ReferenceModel):
    def __init__(self, adata: sc.AnnData, labels_key: str, dropout_type: Literal["separate", "shared", None], device: Literal["cpu", "cuda"], layer: str | None, fp_hack: bool):
        super().__init__(adata=adata, labels_key=labels_key, dropout_type=dropout_type, device=device, layer=layer, fp_hack=fp_hack)

        self.mu_total_count0 = 5 * torch.ones((self.n_labels, self.n_genes), device=self.device)
        self.std_total_count0 = 1 * torch.ones((self.n_labels, self.n_genes), device=self.device)

        self.alpha0 = 1.0 * torch.ones(self.n_genes, device=self.device)
        self.beta0 = 50.0 * torch.ones(self.n_genes, device=self.device)

        self.dec_model_cls = NBDecModel

    def get_params(self, genes: list[str] | np.ndarray | None) -> dict[str, torch.Tensor]:
        if self._params is None:
            raise ValueError("Run fit_reference() first")
        
        alpha = self._params["alpha"]
        beta = self._params["beta"]
        mu_total_count = self._params["mu_total_count"]
        std_total_count = self._params["std_total_count"]

        if self.dropout_enabled:
            dropout_logits = self._params["dropout_logits"]

        if genes is not None:
            gene_idx = [self.var_names.get_loc(g) for g in genes]

            alpha = alpha[gene_idx]
            beta = beta[gene_idx]
            mu_total_count = mu_total_count[:, gene_idx]
            std_total_count = std_total_count[:, gene_idx]

            if self.dropout_enabled:
                dropout_logits = dropout_logits[gene_idx]

        p = dict(alpha=alpha, beta=beta, mu_total_count=mu_total_count, std_total_count=std_total_count)
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

            with pyro.plate("cells", len(labels), device=self.device), poutine.scale(scale=1.0 / len(labels)):
                total_count = total_count[labels, :]

                if self.dropout_type is not None:
                    obs_dist = dist.ZeroInflatedNegativeBinomial(
                        total_count=total_count,
                        probs=probs,
                        gate_logits=dropout_logits,
                    )
                else:
                    obs_dist = dist.NegativeBinomial(  # type: ignore
                        total_count=total_count,
                        probs=probs,
                    )

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

    def pseudo_bulk(self, ref_params: dict[str, torch.Tensor], model_dropout: bool):
        pass
        # if self.concentrations is None:
        #     raise ValueError("Run deconvolute() first")
        
        # alpha = ref_params["alpha"]
        # beta = ref_params["beta"]

        # probs = dist.Beta(alpha, beta).mean

        # mu_total_count = ref_params["mu_total_count"]
        # std_total_count = ref_params["std_total_count"]
        # ct_total_count = dist.LogNormal(mu_total_count, std_total_count).mean

        # proportions = self.get_proportions()
        # total_count = torch.sum(proportions.unsqueeze(-1) * ct_total_count.unsqueeze(0), dim=1)
        # total_count = self.get_cell_counts() * total_count.T

        # if model_dropout:
        #     dropout = logits2probs(ref_params["dropout_logits"])
        #     if self.dropout_type == "separate":
        #         dropout = torch.sum(proportions.unsqueeze(0) * dropout.unsqueeze(1), dim=-1)

        #     if dropout.dim() == 2:
        #         dropout = dropout.T

        #     bulk_dist = dist.ZeroInflatedNegativeBinomial(
        #         total_count=total_count.T,
        #         probs=probs,
        #         gate=dropout
        #     )
        # else:
        #     bulk_dist = dist.NegativeBinomial(
        #         total_count=total_count.T,
        #         probs=probs,
        #     )

        # return bulk_dist.mean

    def plot_pdf(self, adata: sc.AnnData, cell_type: str, gene: str, ax: plt.Axes, n_samples: int = 5000):
        if self._params is None:
            raise ValueError("Run fit_reference() first")
        
        gene_idx = self.var_names.get_loc(gene)
        cell_type_idx = adata.obs[self.labels_key].cat.categories.get_loc(cell_type)
        
        gex = adata[adata.obs[self.labels_key] == cell_type, gene].layers["counts"].toarray()  # type: ignore

        a = self._params["alpha"][gene_idx]
        b = self._params["beta"][gene_idx]
        mu_total_count = self._params["mu_total_count"][cell_type_idx, gene_idx]
        std_total_count = self._params["std_total_count"][cell_type_idx, gene_idx]
        probs = dist.Beta(a, b).sample(torch.Size((n_samples,)))

        total_count = dist.LogNormal(mu_total_count, std_total_count).sample(torch.Size((n_samples,)))

        if self.dropout_type is not None:
            if self.dropout_type == "separate":
                dropout_logits = self._params["dropout_logits"][gene_idx, cell_type_idx]
            elif self.dropout_type == "shared":
                dropout_logits = self._params["dropout_logits"][gene_idx]
            else:
                raise ValueError("Unknown dropout type")
                    
            x = dist.ZeroInflatedNegativeBinomial(
                total_count=total_count,
                probs=probs,
                gate_logits=dropout_logits
            ).sample().cpu()
        else:
            x = dist.NegativeBinomial(  # type: ignore
                total_count=total_count,
                probs=probs,
            ).sample().cpu()

        counts, bins = np.histogram(gex, bins=20, density=True)

        ax.hist(bins[:-1], bins, weights=counts, alpha=0.8, color='red')
        ax.hist(x, bins=bins, density=True, alpha=0.6, color='orange')
        ax.axvline(gex.mean(), color='royalblue')
        return ax


class NBDecModel(DeconvolutionModel):
    def model(self):
        concentrations = pyro.param(
            "concentrations",
            torch.ones((self.n_samples, self.n_labels), device=self.device, dtype=torch.float64),
            constraint=dist.constraints.positive
        )

        cell_counts = pyro.param(
            "cell_counts",
            1e7 * torch.ones(self.n_samples, device=self.device, dtype=torch.float64),
            constraint=dist.constraints.positive
        )

        alpha = self.ref_params["alpha"]
        beta = self.ref_params["beta"]
        mu_total_count = self.ref_params["mu_total_count"]
        std_total_count = self.ref_params["std_total_count"]

        with pyro.plate("genes", self.n_genes, device=self.device):
            probs = pyro.sample("probs", dist.Beta(alpha, beta))
            with pyro.plate("labels", self.n_labels, device=self.device):
                ct_total_count = pyro.sample("total_count", dist.LogNormal(mu_total_count, std_total_count))

        with pyro.plate("samples", self.n_samples, device=self.device):
            proportions = pyro.sample("proportions", dist.Dirichlet(concentrations))
            total_count = torch.sum(proportions.unsqueeze(-1) * ct_total_count.unsqueeze(0), dim=1)
            total_count = cell_counts * total_count.T

            if self.model_dropout:
                dropout = logits2probs(self.ref_params["dropout_logits"])
                if self.dropout_type == "separate":
                    dropout = torch.sum(proportions.unsqueeze(0) * dropout.unsqueeze(1), dim=-1)
                
                bulk_dist = dist.ZeroInflatedNegativeBinomial(
                    total_count=total_count.T,
                    probs=probs,
                    gate_logits=dropout.T
                ).to_event(1)
            else:
                bulk_dist = dist.NegativeBinomial(  # type: ignore
                    total_count=total_count.T,
                    probs=probs,
                ).to_event(1)

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

        mu_total_count = self.ref_params["mu_total_count"]
        std_total_count = self.ref_params["std_total_count"]

        with pyro.plate("genes", self.n_genes, device=self.device):
            pyro.sample("probs", dist.Beta(alpha, beta))
            with pyro.plate("labels", self.n_labels, device=self.device):
                pyro.sample("total_count", dist.LogNormal(mu_total_count, std_total_count))
import tqdm
from typing import Literal

import numpy as np
import scanpy as sc
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
import pyro.optim
import pyro.util
from pyro.infer import SVI, config_enumerate, Trace_ELBO

from matplotlib import pyplot as plt

from . import models
from . import tools as tl


class DeconV:
    def __init__(
        self, adata: sc.AnnData, cell_type_key: str,
        bulk: sc.AnnData | pd.DataFrame,
        model_type: Literal["static", "beta", "gamma", "lognormal", "nb"] = "gamma",
        dropout_type: Literal["separate", "shared", None] = "separate",
        sub_type_key: str | None = None, device: Literal["cpu", "cuda"] = "cpu",
        layer: str | None = "counts", fp_hack: bool = False,
        ignore_genes: list[str] | np.ndarray | None = None,
        top_n_variable_genes: int | None = None,
        model_bulk_dropout: bool = True
    ):
        self.device = device
        self.adata = sc.AnnData(
            X=adata.layers[layer].copy() if layer is not None else adata.X.copy(),  # type: ignore
            obs=adata.obs[[cell_type_key]], var=adata.var[[]]
        )

        if sub_type_key is not None:
            self.adata.obs[sub_type_key] = adata.obs[sub_type_key]

        tl.scale_log_center(self.adata)

        if isinstance(bulk, pd.DataFrame):
            self.sample_names = bulk.index.values.tolist()
            
            self.genes = list(set(self.adata.var_names.tolist()) & set(bulk.columns.tolist()))
            if ignore_genes is not None:
                self.genes = [gene for gene in self.genes if gene not in ignore_genes]
            if len(self.genes) == 0:
                raise ValueError("No common genes between the reference and the bulk data. Please check that columns in the bulk data match the gene names in 'adata.var_names'.")
            
        elif isinstance(bulk, sc.AnnData):
            self.sample_names = bulk.obs.index.values.tolist()
            
            self.genes = list(set(self.adata.var_names.tolist()) & set(bulk.var_names.tolist()))
            if ignore_genes is not None:
                self.genes = [gene for gene in self.genes if gene not in ignore_genes]
            if len(self.genes) == 0:
                raise ValueError("No common genes between the reference and the bulk data. Please check that columns in the bulk data match the gene names in 'adata.var_names'.")
        else:
            raise ValueError("bulk must be a pandas DataFrame or an AnnData object")
        
            print(f"Using {len(self.genes)} common genes between the reference and the bulk data.")

        self.adata = self.adata[:, self.genes].copy()  # type: ignore

        if top_n_variable_genes is not None:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=top_n_variable_genes, subset=True)
            self.genes = self.adata.var_names.tolist()

        print(f"Using {len(self.genes)} common genes between the reference and the bulk data.")

        if isinstance(bulk, pd.DataFrame):
            bulk_gex = torch.tensor(bulk[self.genes].values.copy(), dtype=torch.float32, device=self.device).round()
        elif isinstance(bulk, sc.AnnData):
            bulk_gex = torch.tensor(bulk[:, self.genes].X.copy(), dtype=torch.float32, device=self.device).round()  # type: ignore

        self.model_type = model_type
        self.use_sub_types = sub_type_key is not None

        self.cell_type_key = cell_type_key
        self.cell_types = adata.obs[cell_type_key].astype("category").cat.categories.tolist()
        self.n_cell_types = len(self.cell_types)

        self.sub_type_key = sub_type_key
        self.sub_types = adata.obs[sub_type_key].astype("category").cat.categories.tolist() if self.use_sub_types else None
        self.n_sub_types = len(self.sub_types) if self.sub_types is not None else None

        self.label_key = self.cell_type_key if self.sub_type_key is None else self.sub_type_key
        self.labels = self.cell_types if self.sub_types is None else self.sub_types
        self.n_labels = len(self.labels)

        if self.model_type == "static":
            self._ref_model = models.StaticRefModel(adata=self.adata, labels_key=self.label_key, dropout_type=dropout_type, device=device, layer="counts", fp_hack=fp_hack)
        elif self.model_type == "beta":
            self._ref_model = models.BetaRefModel(adata=self.adata, labels_key=self.label_key, dropout_type=dropout_type, device=device, layer="counts", fp_hack=fp_hack)
        elif self.model_type == "gamma":
            self._ref_model = models.GammaRefModel(adata=self.adata, labels_key=self.label_key, dropout_type=dropout_type, device=device, layer="counts", fp_hack=fp_hack)
        elif self.model_type == "lognormal":
            self._ref_model = models.LogNormalRefModel(adata=self.adata, labels_key=self.label_key, dropout_type=dropout_type, device=device, layer="counts", fp_hack=fp_hack)
        elif self.model_type == "nb":
            self._ref_model = models.NBRefModel(adata=self.adata, labels_key=self.label_key, dropout_type=dropout_type, device=device, layer="counts", fp_hack=fp_hack)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self._dec_model = self._ref_model.dec_model_cls(bulk_gex=bulk_gex, common_genes=self.genes, ref_model=self._ref_model, model_dropout=model_bulk_dropout)
        self.__concentrations: None | torch.Tensor = None

    def fit_reference(
        self, lr: float = 0.1, lrd: float = 0.999, num_epochs: int = 2000, batch_size: int | None = None,
        seed: int | None = None, pyro_validation: bool = True
    ):
        self._ref_model.fit(
            lr=lr, lrd=lrd,
            num_epochs=num_epochs,
            batch_size=batch_size,
            seed=seed,
            pyro_validation=pyro_validation,
        )
        self._dec_model.set_ref_params(self._ref_model.get_params(self.genes))

    def deconvolute(
        self, model_dropout: bool = True, lr: float = 0.1, lrd: float = 0.999, num_epochs: int = 1000, progress: bool = True,
    ) -> pd.DataFrame:

        if self._ref_model._params is None:
            raise ValueError("You must fit the reference first: 'fit_reference()'")
        if model_dropout and self._ref_model.dropout_type is None:
            raise ValueError("You must fit the reference with dropout to deconvolute with dropout")
        
        pyro.clear_param_store()
        optim = pyro.optim.ClippedAdam(dict(lr=lr, lrd=lrd))
        
        guide = config_enumerate(self._dec_model.guide, "parallel", expand=True)

        svi = SVI(self._dec_model.model, guide, optim=optim, loss=Trace_ELBO())

        if progress:
            pbar = tqdm.tqdm(range(num_epochs), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}", dynamic_ncols=True)
        else:
            pbar = range(num_epochs)

        for epoch in pbar:
            self.deconvolution_loss = svi.step()
            if isinstance(pbar, tqdm.tqdm):
                pbar.set_postfix(
                    loss=f"{self.deconvolution_loss:.2e}",
                    lr=f"{list(optim.get_state().values())[0]['param_groups'][0]['lr']:.2e}",
                )

        self.__concentrations = pyro.param("concentrations").clone().detach()
        # self.cell_counts = pyro.param("cell_counts").clone().detach()

        # if ignore_genes is not None:
        #     self.n_genes = _n_genes
        #     for key in self.params.keys():
        #         self.params[key] = _params[key].clone()

        # return self.get_proportions()
        return self.proportions
    
    @property
    def concentrations(self) -> torch.Tensor:
        if self.__concentrations is None:
            raise ValueError("Please run deconvolute() first.")
        return self.__concentrations
    
    @property
    def proportions(self) -> pd.DataFrame:
        proportions = dist.Dirichlet(self.concentrations).mean.cpu().numpy()
        
        if self.use_sub_types:
            proportions = self.__sum_sub_proportions(proportions)

        proportions = pd.DataFrame(
            proportions,
            columns=self.cell_types,
            index=self.sample_names
        )
        return proportions
    
    def get_results_df(self, quantiles: tuple[float, float] = (0.025, 0.975)) -> pd.DataFrame:
        limits = np.empty((self._dec_model.n_samples, self.n_labels, 2))

        proportions = self.proportions
        if not self.use_sub_types:
            for i in range(self._dec_model.n_samples):
                p_dist = dist.Dirichlet(self.concentrations[i])
                ps = p_dist.sample(torch.Size((10000,))).cpu()
                q = np.quantile(ps, quantiles, 0)
                limits[i, :, :] = q.T

            _min = pd.DataFrame(
                limits[:, :, 0],
                columns=self.cell_types,
                index=self.sample_names
            ).reset_index().melt(id_vars="index")

            _max = pd.DataFrame(
                limits[:, :, 1],
                columns=self.cell_types,
                index=self.sample_names
            ).reset_index().melt(id_vars="index")

        res_df = pd.DataFrame(
            proportions,
            columns=self.cell_types,
            index=self.sample_names
        ).reset_index().melt(id_vars="index")

        res_df.rename(columns={"index": "sample", "value": "est", "variable": "cell_type"}, inplace=True)

        if not self.use_sub_types:
            # TODO: temp fix floating point precision...
            res_df["min"] = (res_df["est"] - _min["value"]).clip(lower=0.0)
            res_df["max"] = (_max["value"] - res_df["est"]).clip(lower=0.0)

        # assert res_df["min"].min() > 0, res_df["min"].min()
        # assert res_df["max"].min() > 0, res_df["max"].min()
        return res_df

    def __sum_sub_proportions(self, sub_proportions: np.ndarray) -> np.ndarray:
        if self.sub_types is None:
            raise ValueError("Subtypes are not used..")
        d = {}
        for i, sub_type in enumerate(self.sub_types):
            cell_type = "_".join(sub_type.split("_")[:-1])
            if cell_type not in d.keys():
                d[cell_type] = sub_proportions[:, i]
            else:
                d[cell_type] += sub_proportions[:, i]

        return np.array([d[cell_type] for cell_type in self.cell_types]).T
    
    def check_fit(self, path: str | None = None):
        f, ax = plt.subplots(self.n_labels, self.n_labels, figsize=(20, 20), dpi=100)
        res = tl.rank_marker_genes(self.adata, groupby=self.label_key, reference=None)
        for i in range(self.n_labels):
            for j in range(self.n_labels):
                gene = res[f"{self.labels[i]} vs. rest"].sort_values("gene_score", ascending=False).index[0]
                ax[i, 0].set_ylabel(gene)
                ax[self.n_labels - 1, j].set_xlabel(self.labels[j])
                ax[i, j].set_yticks([])
                self._ref_model.plot_pdf(adata=self.adata, cell_type=self.labels[j], gene=gene, ax=ax[i, j])

        if path is not None:
            plt.savefig(path, bbox_inches="tight")

        plt.show()
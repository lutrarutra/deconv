from abc import abstractmethod, ABC
import tqdm, os
from typing import Literal

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, config_enumerate, Trace_ELBO

import torch

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import scipy

import pickle

def logits2probs(logits):
    return 1.0 / (1.0 + (-logits).exp())
    
class RefDataSet(torch.utils.data.Dataset):
    def __init__(self, adata, label_key, layer="counts", device="cpu", fp_hack=False) -> None:
        super().__init__()

        if layer == "X" or layer == None:
            _layer = adata.X
        else:
            _layer = adata.layers[layer]

        if scipy.sparse.issparse(_layer):
            x = _layer.toarray()
        else:
            x = _layer

        self.sc_counts = torch.tensor(x, dtype=torch.float32, device=device)
        
        if not np.equal(np.mod(x, 1), torch.tensor(0.0)).all():
            print("Warning: single-cell counts are not integers, make sure you provided the correct layer with raw counts.")
            print("Rounding counts to integers.")
            self.sc_counts = (self.sc_counts + torch.tensor(1e-8)).round()

        self.labels = torch.tensor(adata.obs[label_key].cat.codes.values, dtype=torch.long, device=device)

        # Fixes floating point error; When all of the genes are zero gradients tend to become NaN
        if fp_hack:
            for label in np.unique(self.labels.cpu()):
                mask = (self.sc_counts[self.labels == label,:].sum(0) == 0)
                idx = (self.labels == label).nonzero(as_tuple=True)[0][0]
                self.sc_counts[idx, mask] = 1

    def __len__(self):
        return len(self.sc_counts)

    def __getitem__(self, idx):
        return self.sc_counts[idx], self.labels[idx], idx


class Base(ABC):
    def __init__(self, adata, labels_key, dropout_type: Literal["separate", "shared", None] = "separate", device="cpu"):
        self.adata = adata
        self.labels_key = labels_key
        self.n_genes = adata.n_vars
        self.n_labels = len(adata.obs[labels_key].cat.categories)
        
        self.params = None
        self.concentrations = None

        self.ref_dropout_type = dropout_type
        self.device = device


    def fit_reference(self, lr=0.1, lrd=0.999, num_epochs=500, batch_size=None, seed=None, pyro_validation=True, layer="counts", fp_hack=False):
        pyro.clear_param_store()
        pyro.set_rng_seed(0)

        if seed is not None:        
            pyro.util.set_rng_seed(seed)

        pyro.enable_validation(pyro_validation)

        optim = pyro.optim.ClippedAdam(dict(lr=lr, lrd=lrd))
        guide = config_enumerate(self.ref_guide, "parallel", expand=True)
        svi = SVI(self.ref_model, guide, optim=optim, loss=Trace_ELBO())
        
        dataset = RefDataSet(self.adata, self.labels_key, device=self.device, layer=layer, fp_hack=fp_hack)

        if batch_size is None:
            batch_size = self.adata.n_obs
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        pbar = tqdm.tqdm(range(num_epochs), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}", dynamic_ncols=True)
        for epoch in pbar:
            losses = []
            for x, labels, _ in loader:
                self.reference_loss = svi.step(x, labels)
                losses.append(self.reference_loss)

            pbar.set_postfix(
                loss=f"{np.mean(self.reference_loss):.2e}",
                lr=f"{list(optim.get_state().values())[0]['param_groups'][0]['lr']:.2e}",
            )

        self.params = dict()
        for param in pyro.get_param_store():
            self.params[str(param)] = pyro.param(param).clone().detach()

    def deconvolute(self, model_dropout, ignore_genes=None, bulk=None, lr=0.1, lrd=0.995, num_epochs=1000, progress=True):
        assert self.params is not None, "You must fit the reference first"
        if model_dropout and self.ref_dropout_type is None:
            raise ValueError("You must fit the reference with dropout to deconvolute with dropout")
        
        self.dec_model_dropout = model_dropout

        if bulk is None:
            bulk = (torch.tensor(self.adata.varm["bulk"].T, dtype=torch.float32, device=self.device)).round()
        if isinstance(bulk, np.ndarray):
            bulk = torch.tensor(bulk, dtype=torch.float32, device=self.device).round()
        
        _params = dict()
        if ignore_genes is not None:
            mask = torch.tensor(~self.adata.var.index.isin(ignore_genes), dtype=torch.bool, device=self.device)
            _n_genes = self.n_genes
            self.n_genes = mask.sum()

            for key in self.params.keys():
                _params[key] = self.params[key].clone()

                if mask.shape[0] == self.params[key].shape[0]:
                    self.params[key] = self.params[key][mask]
                elif self.params[key].dim() > 1 and mask.shape[0] == self.params[key].shape[1]:
                    self.params[key] = self.params[key][:,mask]

            bulk = bulk[:, mask]

        assert bulk.shape[1] == self.n_genes, "The bulk data must have the same number of genes as the reference data"
        
        def get_optim_params(param_name):
            if param_name == "cell_counts":
                return dict(lr=lr, lrd=lrd)
            else:
                return dict(lr=lr, lrd=lrd)

        pyro.clear_param_store()
        optim = pyro.optim.ClippedAdam(get_optim_params)
        
        guide = config_enumerate(self.dec_guide, "parallel", expand=True)

        svi = SVI(self.dec_model, guide, optim=optim, loss=Trace_ELBO())

        if progress:
            pbar = tqdm.tqdm(range(num_epochs), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}", dynamic_ncols=True)
        else:
            pbar = range(num_epochs)

        for epoch in pbar:
            self.deconvolution_loss = svi.step(bulk)
            if progress:
                pbar.set_postfix(
                    loss=f"{self.deconvolution_loss:.2e}",
                    lr=f"{list(optim.get_state().values())[0]['param_groups'][0]['lr']:.2e}",
                )
            
        self.concentrations = pyro.param("concentrations").clone().detach()
        self.cell_counts = pyro.param("cell_counts").clone().detach()

        if ignore_genes is not None:
            self.n_genes = _n_genes
            for key in self.params.keys():
                self.params[key] = _params[key].clone()

    def get_proportions(self):
        if self.concentrations is None:
            raise ValueError("You must deconvolute first")
        return dist.Dirichlet(self.concentrations).mean 

    def get_cell_counts(self):
        if self.cell_counts is None:
            raise ValueError("You must deconvolute first")
        return self.cell_counts

    @abstractmethod
    def ref_model(self, sc_counts, labels):
        pass

    @abstractmethod
    def ref_guide(self, sc_counts, labels):
        pass

    @abstractmethod
    def dec_model(self, bulk):
        pass

    @abstractmethod
    def dec_guide(self, bulk):
        pass

    @abstractmethod
    def plot_pdf(self, gene_i, ct_i, n_samples=5000, ax=None):
        pass

    @abstractmethod
    def pseudo_bulk(self):
        pass

    def save_model(self, dir):
        if self.concentrations is None:
            raise ValueError("You must deconvolute first")

        if not os.path.exists(dir):
            os.makedirs(dir)
        
        with open(os.path.join(dir, "concentrations.npy"), "wb") as f:
            np.save(f, self.concentrations.cpu().numpy())

        with open(os.path.join(dir, "ref.pkl"), "wb") as f:
            params = self.params
            params["model_type"] = self.__class__.__name__
            params["ref_dropout_type"] = self.ref_dropout_type
            params["dec_model_dropout"] = self.dec_model_dropout
            params["genes"] = self.adata.var_names
            params["cell_types"] = list(self.adata.obs[self.labels_key].cat.categories)
            pickle.dump(params, f)

    # @classmethod
    # def load_model(cls, dir, adata, labels_key):
    #     with open(os.path.join(dir, "concentrations.npy"), "rb") as f:
    #         concentrations = torch.tensor(np.load(f))


    #     with open(os.path.join(dir, "ref.pkl"), "rb") as f:
    #         params = pickle.load(f)

    #     model_type = params.pop("model_type")
    #     ref_dropout_type = params.pop("ref_dropout_type")
    #     dec_model_dropout = params.pop("dec_model_dropout")
    #     # TODO: check that the genes and cell types match with adata
    #     genes = params.pop("genes")
    #     cell_types = params.pop("cell_types")

    #     if model_type == "NB":
    #         model = NB(adata, labels_key, ref_dropout_type)

    #     elif model_type == "lognormal":
    #         model = LogNormal(adata, labels_key, ref_dropout_type)

    #     elif model_type == "beta":
    #         model = Beta(adata, labels_key, ref_dropout_type)

    #     elif model_type == "gamma":
    #         model = Gamma(adata, labels_key, ref_dropout_type)

    #     elif model_type == "static":
    #         model = Static(adata, labels_key, ref_dropout_type)

        
    #     model.concentrations = concentrations
    #     model.params = params
    #     model.dec_model_dropout = dec_model_dropout

    #     return model


    # def res_df(self):
    #     assert self.concentrations is not None, "You must deconvolute first"
    #     n_bulk_samples = self.concentrations.shape[0]
    #     quantiles = np.empty((n_bulk_samples, self.n_labels, 2))

    #     res = dist.Dirichlet(self.concentrations).mean

    #     for i in range(n_bulk_samples):
    #         p_dist = dist.Dirichlet(self.concentrations[i])
    #         ps = p_dist.sample((10000,))
    #         q = np.quantile(ps, (0.025, 0.975), 0)
    #         quantiles[i, :, :] = q.T

    #     _min = pd.DataFrame(
    #         quantiles[:, :, 0],
    #         columns=self.adata.obs["labels"].cat.categories.to_list(),
    #         index=self.adata.uns["bulk_samples"]
    #     ).reset_index().melt(id_vars="index")

    #     _max = pd.DataFrame(
    #         quantiles[:, :, 1],
    #         columns=self.adata.obs["labels"].cat.categories.to_list(),
    #         index=self.adata.uns["bulk_samples"]
    #     ).reset_index().melt(id_vars="index")

    #     res_df = pd.DataFrame(
    #         res,
    #         columns=self.adata.obs["labels"].cat.categories.to_list(),
    #         index=self.adata.uns["bulk_samples"]
    #     ).reset_index().melt(id_vars="index")

    #     res_df.rename(columns={"index": "sample", "value": "est", "variable":"cell_type"}, inplace=True)
    #     res_df["min"] = res_df["est"] - _min["value"]
    #     res_df["max"] = _max["value"] - res_df["est"]
    #     return res_df
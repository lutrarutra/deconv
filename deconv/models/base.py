from abc import abstractmethod, ABC
from typing import Literal

import tqdm

import pyro
import pyro.distributions as dist
import pyro.optim
import pyro.util
from pyro.infer import SVI, config_enumerate, Trace_ELBO

import torch
import torch.utils.data
import scanpy as sc
import numpy as np

from matplotlib import pyplot as plt

import scipy


def logits2probs(logits):
    return 1.0 / (1.0 + (-logits).exp())
    
    
class RefDataSet(torch.utils.data.Dataset):
    def __init__(self, adata, label_key: str, layer: str | None, device: Literal["cpu", "cuda"], fp_hack: bool):
        super().__init__()

        if layer == "X" or layer is None:
            x = adata.X
        else:
            x = adata.layers[layer]

        if scipy.sparse.issparse(x):
            x = x.toarray()

        self.sc_counts = torch.tensor(x, dtype=torch.float32, device=device)
        
        if not np.equal(np.mod(x, 1), torch.tensor(0.0)).all():
            print("Warning: single-cell counts are not integers, make sure you provided the correct layer with raw counts.")
            print("Rounding counts to integers.")
            self.sc_counts = (self.sc_counts).round()

        self.labels = torch.tensor(adata.obs[label_key].cat.codes.values, dtype=torch.long, device=device)

        # Fixes floating point error; When all of the genes are zero gradients tend to become NaN
        if fp_hack:
            for label in np.unique(self.labels.cpu()):
                mask = (self.sc_counts[self.labels == label, :].sum(0) == 0)
                idx = (self.labels == label).nonzero(as_tuple=True)[0][0]
                self.sc_counts[idx, mask] = 1

    def __len__(self):
        return len(self.sc_counts)

    def __getitem__(self, idx):
        return self.sc_counts[idx], self.labels[idx], idx


class ReferenceModel(ABC):
    def __init__(
        self, adata: sc.AnnData, labels_key: str, dropout_type: Literal["separate", "shared", None], device: Literal["cpu", "cuda"],
        layer: str | None, fp_hack: bool
    ):
        self.labels_key = labels_key
        self.n_genes = adata.n_vars
        self.n_labels = len(adata.obs[labels_key].cat.categories)
        self._params = None

        self.dropout_type = dropout_type
        self.device = device
        self.dataset = RefDataSet(adata, self.labels_key, device=self.device, layer=layer, fp_hack=fp_hack)
        self.var_names = adata.var_names
        self.training_history = None

    def fit(
        self, lr: float, lrd: float, num_epochs: int, batch_size: int | None,
        seed: int | None, pyro_validation: bool
    ):
        pyro.clear_param_store()
        pyro.set_rng_seed(0)

        if seed is not None:
            pyro.util.set_rng_seed(seed)

        pyro.enable_validation(pyro_validation)

        optim = pyro.optim.ClippedAdam(dict(lr=lr, lrd=lrd))
        guide = config_enumerate(self.guide, "parallel", expand=True)
        svi = SVI(self.model, guide, optim=optim, loss=Trace_ELBO())

        if batch_size is None:
            batch_size = len(self.dataset)
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size)

        pbar = tqdm.tqdm(range(num_epochs), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}", dynamic_ncols=True)
        training_history = []
        for epoch in pbar:
            losses = []
            for x, labels, _ in loader:
                self.reference_loss: torch.Tensor = svi.step(x, labels)  # type: ignore
                self.reference_loss /= self.n_genes
                losses.append(self.reference_loss)

            epoch_loss = np.mean(self.reference_loss)
            pbar.set_postfix(
                loss=f"{np.mean(self.reference_loss):.2e}",
                lr=f"{list(optim.get_state().values())[0]['param_groups'][0]['lr']:.2e}",
            )
            training_history.append(epoch_loss)

        self._params = dict()
        for param in pyro.get_param_store():
            self._params[str(param)] = pyro.param(param).clone().detach()
        
        self.training_history: list[float] | None = training_history

    @abstractmethod
    def get_params(self, genes: list[str] | np.ndarray | None) -> dict[str, torch.Tensor]:
        pass
    
    @property
    def dropout_enabled(self):
        return self.dropout_type is not None
    
    @abstractmethod
    def model(self, sc_counts: torch.Tensor, labels: torch.Tensor):
        pass

    @abstractmethod
    def guide(self, sc_counts: torch.Tensor, labels: torch.Tensor):
        pass
    
    @abstractmethod
    def plot_pdf(self, adata: sc.AnnData, cell_type: str, gene: str, ax: plt.Axes, n_samples: int = 5000):
        pass

    @abstractmethod
    def pseudo_bulk(self):
        pass


class DeconvolutionModel(ABC):
    def __init__(self, bulk_gex: torch.Tensor, common_genes: list[str] | np.ndarray, ref_model: ReferenceModel, model_dropout: bool):
        self.device = ref_model.device
        self.bulk_gex = bulk_gex
        self.model_dropout = model_dropout
        self.n_labels = ref_model.n_labels
        self.n_samples = len(bulk_gex)
        self.n_genes = len(common_genes)
        self.dropout_type = ref_model.dropout_type
        self.__ref_params: dict[str, torch.Tensor] | None = None

    def set_ref_params(self, ref_params: dict[str, torch.Tensor]):
        self.__ref_params = ref_params

    @property
    def ref_params(self):
        if self.__ref_params is None:
            raise ValueError("Run 'fit_reference()' first")
        return self.__ref_params

    @abstractmethod
    def model(self, bulk: torch.Tensor, ref_params: dict[str, torch.Tensor], model_dropout: bool):
        pass

    @abstractmethod
    def guide(self, bulk: torch.Tensor, ref_params: dict[str, torch.Tensor], model_dropout: bool):
        pass
    
    def get_concentrations(self) -> torch.Tensor:
        return pyro.param("concentrations").clone().detach()
    
    def get_cell_counts(self) -> torch.Tensor:
        return pyro.param("cell_counts").clone().detach()
    
    def get_proportions(self) -> torch.Tensor:
        return dist.Dirichlet(self.get_concentrations()).mean
    

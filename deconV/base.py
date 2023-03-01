import os, tqdm
from typing import Literal

from abc import ABC, abstractmethod

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .tools import fmt_c

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class SignatureDataset(torch.utils.data.Dataset):
    def __init__(self, mu, scale, bulk) -> None:
        super().__init__()
        self.mu = mu
        self.scale = scale
        self.y = torch.tensor(bulk, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.mu[idx, :], self.scale[idx, :], self.y[idx]
    

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.inf

    def early_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class BaseModel(ABC, nn.Module):
    def __init__(self, n_cell_types, locked_proportions, gene_weights=None, log_loss=False):
        super(BaseModel, self).__init__()
        self.log_loss = log_loss
        self.n_cell_types = n_cell_types

        self.log_cell_counts = nn.Parameter(torch.ones(self.n_cell_types) / self.n_cell_types)
        self.gene_weights = gene_weights
        if type(self.gene_weights) is np.ndarray:
            self.gene_weights = torch.from_numpy(self.gene_weights).float()

    def get_proportions(self):
        return F.softmax(self.log_cell_counts, dim=0)

    def get_lib_size(self):
        return self.log_cell_counts.exp().sum()

    def get_counts(self):
        return self.log_cell_counts.exp()
    
    @abstractmethod
    def process_batch(self, x, y) -> float:
        pass

    @abstractmethod
    def forward(self, signature):
        pass

    def fit(self, mu, scale, bulk_sample, num_iter=5000, batch_size=-1, lr=0.1, weight_decay=0.0, progress=True):
        if batch_size > 0:
            loader = torch.utils.data.DataLoader(SignatureDataset(mu, scale, bulk_sample), batch_size=batch_size, shuffle=True)
        else:
            loader = [(mu, scale, torch.from_numpy(bulk_sample).float())]

        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        if progress:
            pbar = tqdm.tqdm(range(num_iter))
        else:
            pbar = range(num_iter)
        
        early_stopper = EarlyStopper(patience=100, min_delta=100)
        for i in pbar:
            L = 0.0
            for mu_batch, scale_batch, y_batch in loader:
                L += self.process_batch(mu_batch, scale_batch, y_batch)
            
            if progress:
                pbar.set_postfix(dict(proportions=fmt_c(self.get_proportions()), lib_size=f"{self.get_lib_size():.1f}", loss=f"{L:.1f}"))

            if early_stopper.early_stop(L):
                break

class LRM(BaseModel):
    def __init__(self, n_cell_types, gene_weights=None, log=False):
        super(LRM, self).__init__(n_cell_types, gene_weights, log_loss=log)

    def process_batch(self, mu_batch, scale_batch, y):
        self.optim.zero_grad()
        _x = self(mu_batch)
        
        if self.log_loss:
            loss = F.mse_loss(torch.log(_x), torch.log(y), reduction="none", reduce=None)
        else:
            loss = F.mse_loss(_x, y, reduction="none", reduce=None)

        if self.gene_weights is not None:
            loss = loss * self.gene_weights

        loss = loss.sum()
        loss.backward()
        self.optim.step()
        return loss.item()

    def forward(self, mu):
        _x = torch.sum(mu * self.log_cell_counts.exp(), dim=1)
        return _x
    
class PSM(BaseModel):
    def __init__(self, n_cell_types, gene_weights=None, log=False):
        super(PSM, self).__init__(n_cell_types, gene_weights, log_loss=False)
        if log:
            print("Log loss is not supported for Poisson regression. Setting log_loss=False")

    def process_batch(self, mu_batch, scale_batch, bulk_batch):
        self.optim.zero_grad()
        l = self(mu_batch)

        loss = F.poisson_nll_loss(l, bulk_batch.round(), log_input=False, full=True, reduction="none", reduce=None)

        if self.gene_weights is not None:
            loss = loss * self.gene_weights
        loss = loss.sum()

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.00001)
        self.optim.step()
        return loss.item()

    def forward(self, mu):
        l = torch.sum(mu * self.log_cell_counts.exp(), dim=1)
        assert torch.isnan(l).any() == False
        return l
    

class NSM(BaseModel):
    def __init__(self, n_cell_types, gene_weights=None, log=False):
        super(NSM, self).__init__(n_cell_types, gene_weights, log_loss=log)
        self.eps = torch.tensor(1e-5)

    def process_batch(self, mu_batch, scale_batch, bulk_batch):
        self.optim.zero_grad()
        mu, var = self(mu_batch, scale_batch)

        if self.log_loss:
            loss = F.gaussian_nll_loss(torch.log(mu), torch.log(bulk_batch), var=var, reduction="none", full=True)
        else:
            loss = F.gaussian_nll_loss(mu, bulk_batch, var=var, reduction="none", full=True)
        if self.gene_weights is not None:
            loss = loss * self.gene_weights
        loss = loss.sum()
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.00001)
        self.optim.step()
        return loss.item()

    def forward(self, mu, scale):
        mu = torch.sum(mu * self.log_cell_counts.exp(), dim=1)
        var = torch.max(self.eps, torch.sum(scale**2 * self.log_cell_counts.exp(), 1))
        assert torch.isnan(mu).any() == False
        return mu, var


# class PSM(nn.Module):
#     def __init__(
#         self,
#         loc,
#         gene_weights=None,
#         gene_scale=None,
#         lib_size=torch.tensor(1000),
#         norm=torch.tensor(1.0),
#     ):
#         super().__init__()
#         self.loc = loc
#         self.gene_weights = gene_weights
#         self.gene_scale = gene_scale
#         self.norm = norm
#         self.log_cell_counts = nn.Parameter(
#             torch.ones(loc.shape[1]) * torch.log(lib_size / loc.shape[1])
#         )
#         # print(self.log_cell_counts)
#         self.eps = torch.tensor(1e-5)

#     def get_proportions(self):
#         with torch.no_grad():
#             return F.softmax(self.log_cell_counts, 0)

#     def get_lib_size(self):
#         with torch.no_grad():
#             return torch.sum(self.log_cell_counts.exp())

#     def get_distribution(self):
#         assert torch.isnan(self.loc).any() == False
#         assert torch.isnan(self.log_cell_counts).any() == False, self.log_cell_counts

#         loc = torch.sum(self.loc * self.log_cell_counts.exp(), 1)

#         assert torch.isnan(loc).sum() == 0

#         return D.Poisson(loc)

#     def forward(self, x):
#         d = self.get_distribution()

#         x = x.round()

#         if self.gene_weights is not None:
#             return (-d.log_prob(x) * self.gene_weights).mean() * self.norm
        
#         return -d.log_prob(x).mean() * self.norm


# class NSM(nn.Module):
#     def __init__(
#         self,
#         loc,
#         scale,
#         gene_weights,
#         gene_scale=None,
#         lib_size=torch.tensor(1000),
#         norm=torch.tensor(1.0),
#     ):
#         super().__init__()
#         self.loc = loc
#         self.scale = scale
#         self.gene_weights = gene_weights
#         self.gene_scale = gene_scale
#         self.log_cell_counts = nn.Parameter(
#             torch.ones(loc.shape[1]) * torch.log(lib_size / loc.shape[1])
#         )
#         self.norm = norm
#         self.eps = torch.tensor(1e-5)

#     def get_proportions(self):
#         with torch.no_grad():
#             return F.softmax(self.log_cell_counts, 0)

#     def get_lib_size(self):
#         with torch.no_grad():
#             return torch.sum(self.log_cell_counts.exp())

#     def get_distribution(self):
#         loc = torch.sum(self.loc * self.log_cell_counts.exp(), 1)
#         scale = torch.max(
#             self.eps,
#             torch.sqrt(torch.sum(self.scale**2 * self.log_cell_counts.exp(), 1)),
#         )

#         assert torch.isnan(loc).sum() == 0

#         return D.Normal(loc, scale)
    
#     def fit(self, mu, bulk, num_iter=100, batch_size=32, lr=0.001, weigh_decay=0.0, progress=True):
#         loader = torch.utils.data.DataLoader(SignatureDataset(mu, bulk), batch_size=batch_size, shuffle=True)

#         self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weigh_decay)

#         if progress:
#             pbar = tqdm.tqdm(range(num_iter))
#         else:
#             pbar = range(num_iter)

#         for i in pbar:
#             L = 0.0
#             for x_batch, y_batch in loader:
#                 self.optim.zero_grad()
#                 loss = self(x_batch)
#                 L += loss.item()
#                 loss.backward()
#                 self.optim.step()

#             pbar.set_postfix(dict(proportions=fmt_c(self.get_proportions()), lib_size=f"{self.get_lib_size():.1f}", loss=L))

#     def forward(self, x):
#         d = self.get_distribution()

#         if self.gene_weights is not None:
#             return (-d.log_prob(x) * self.gene_weights).mean() * self.norm

#         return -d.log_prob(x).mean() * self.norm





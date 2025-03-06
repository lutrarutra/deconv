import scipy.sparse
import tqdm

import numpy as np
import scipy
import scanpy as sc
import pandas as pd


def __generate_pseudo_bulk_sample(adata: sc.AnnData, groupby: str, proportions: list | np.ndarray, n_cells: int, layer: str | None) -> np.ndarray:
    pseudo = np.zeros(adata.n_vars, dtype=np.float32)

    idxs = np.random.choice(len(proportions), n_cells, p=proportions).tolist()

    counts = [(idx, idxs.count(idx)) for idx in set(idxs)]

    lens = adata.obs.groupby(groupby).size()
    
    for ct_i, c in counts:
        subset = adata[adata.obs[groupby].cat.codes == ct_i, :]
        gex = subset.X if layer is None or layer == "X" else subset.layers[layer]
        assert gex is not None
        if isinstance(gex, scipy.sparse.spmatrix):
            gex = gex.toarray()
        
        ridx = np.random.randint(0, lens[ct_i], (c))

        for ri in ridx:
            pseudo += gex[ri, :]

    return pseudo


def generate_pseudo_bulk_samples(adata: sc.AnnData, groupby: str, n_samples: int, layer: str | None = "counts") -> sc.AnnData:
    n_groups = len(adata.obs[groupby].cat.categories)
    props = np.random.dirichlet(np.ones(n_groups), n_samples)
    
    bulks = []

    columns = adata.obs[groupby].cat.categories.tolist()
    proportions = pd.DataFrame([], columns=columns)

    cell_counts = []

    for i in tqdm.tqdm(range(n_samples)):
        n_cells = np.random.randint(500, 10000)
        cell_counts.append(n_cells)
        bulks.append(__generate_pseudo_bulk_sample(adata, groupby, props[i], n_cells, layer=layer))
        proportions = pd.concat([proportions, pd.DataFrame([props[i].tolist()], columns=columns, index=[i])])

    proportions.index = "pseudo_bulk_" + proportions.index.astype(str)

    bulk_counts = pd.DataFrame(bulks, columns=adata.var_names).T
    pseudo = sc.AnnData(X=bulk_counts.values.T, uns={"proportions": proportions}, obs={"n_cells": cell_counts})
    pseudo.var.index = adata.var_names
    pseudo.obs.index = "pseudo_bulk_" + pseudo.obs.index.astype(str)
    return pseudo
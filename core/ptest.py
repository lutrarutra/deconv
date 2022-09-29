import itertools
import multiprocessing
import deconV as dV
from base import NSM, MSEM
import plot as pl

import glob, tqdm, time, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc
import scvi
import seaborn as sns
import tqdm


dV.params = {
    "jupyter": True,
    "tqdm": False,
    "cell_type_key": "cellType",
    "layer": "ncounts",
    "index_col": 0,
    "selected_ct": ["0", "1", "2"],
    "model_type": "prob",
    "ignore_others": True,
    "n_top_genes": -1,
    "plot_pseudo_bulk": False,
    "lr": 0.01,
    "epochs": 5000,
    "fig_fmt": "png",
    "indir": "../../data/GSE136148/",
    "outdir": "out",
    "figsize": (8,8),
    "dpi": 80,
}

sadata = dV.read_data(os.path.join(dV.params["indir"], "sc.tsv"))
print(f"scRNA-seq data - cells: {sadata.shape[0]}, genes: {sadata.shape[1]}")

print("Reading bulk data...")
badata = dV.read_data(os.path.join(dV.params["indir"], "bulk.tsv"), is_bulk=True)
print(f"bulk RNA-seq data - samples: {badata.shape[0]}, genes: {badata.shape[1]}")

print("Reading pheno data...")
pheno_df = pd.read_csv(os.path.join(dV.params["indir"], "pdata.tsv"), sep="\t", index_col=dV.params["index_col"])
pheno_df.index.name = None

sadata.obs = pd.concat([sadata.obs, pheno_df], axis=1)
assert dV.params["cell_type_key"] in sadata.obs.columns, f"{dV.params['cell_type_key']} not in obs columns"
sadata.obs[dV.params["cell_type_key"]] = sadata.obs[dV.params["cell_type_key"]].astype(str)
sadata.obs

print("Preprocessing data...")
sadata, badata = dV.preprocess(sadata, badata)
print("After preprocessing:")
print(f"scRNA-seq data - cells: {sadata.shape[0]}, genes: {sadata.shape[1]}")
print(f"bulk RNA-seq data - samples: {badata.shape[0]}, genes: {badata.shape[1]}")

assert sadata.shape[1] == badata.shape[1], "scRNA-seq and bulk RNA-seq data have different number of genes"

cell_types = list(sadata.obs[dV.params['cell_type_key']].unique())
# print(cell_types)

param_range = {
    "layer": ["ncounts", "counts", None],
    # "ignore_others": [False, True],
    # "n_top_genes": [-1, 1000, 2000, 5000, 10000],
    "use_sub_types": [False],
    "gene_weights_key": [None, "cv2"],
    "dropout_res_lim": [0.1, 0.2, 0.3, 0.4, 0.5],
    "dropout_lim": [0.95, 0.975, 0.99, 0.995, 0.999, 1.0],
    "marker_zscore_lim": [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0],
    "dispersion_lims":[(-3,1), (-2,2), (-1,1), (-1,2), (-3,3), (-2, 0), (0, 2), (-1, 3)]
}


combs = list(itertools.product(*param_range.values()))


results = []

pool = multiprocessing.Pool(6)

def prc(comb):
    p = dict(zip(param_range.keys(), comb))
    decon = dV.DeconV(sadata, badata, cell_types, dV.params, use_sub_types=p["use_sub_types"], gene_weights_key=p["gene_weights_key"])

    decon.filter_outliers(dropout_res_lim=p["dropout_res_lim"], dropout_lim=p["dropout_lim"], marker_zscore_lim=p["marker_zscore_lim"], dispersion_lims=p["dispersion_lims"])

    decon.init_dataset(use_outliers=False)
    decon.init_signature()

    est_df = decon.deconvolute()
    return est_df.values

with multiprocessing.Pool(6) as p:
    res = p.map(prc, combs)
    print(res)
    results.append(res)
    print(f"Progress: {len(results)}/{len(combs)}")



pd.DataFrame(results).to_csv("results.csv")



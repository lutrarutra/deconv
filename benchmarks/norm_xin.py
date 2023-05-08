import deconV as dv
import scout

import glob, tqdm, time, os, argparse, json
import torch
import pyro
import pyro.distributions as dist
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

import pandas as pd
import numpy as np
import scanpy as sc
import scvi
import seaborn as sns
import tqdm
import scout

import itertools

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

PARAMS = {
    "dropout_type": ["separate"],
    "model_type": ["gamma", "beta", "nb", "static", "lognormal"],
    "bulk_dropout": [True],
    "target_sum": ["lognorm", False, None, 1e6],
    "exclude_highly_expressed": [True, False],
}

def read_inputs(indir):
    reference_file = os.path.join(indir, "sc.tsv")
    reference_mdata_file = os.path.join(indir, "pdata.tsv")
    bulk_file = os.path.join(indir, "bulk.tsv")
    cell_types = ["alpha", "delta", "gamma", "beta", "lognormal"]
    true_df = pd.read_csv(os.path.join(indir, "true.tsv"), sep="\t", index_col=0)
    true_df = true_df.reindex(sorted(true_df.columns), axis=1)

    sadata = dv.tl.read_data(reference_file)

    pheno_df = pd.read_csv(reference_mdata_file, sep="\t", index_col=0)
    pheno_df.index.name = None
    common_cells = list(set(pheno_df.index.tolist()) & set(sadata.obs_names.tolist()))

    sadata = sadata[common_cells, :].copy()
    pheno_df = pheno_df.loc[common_cells, :].copy()
    sadata.obs["labels"] = pheno_df["cellType"].tolist()
    sadata.obs["labels"] = sadata.obs["labels"].astype("category")

    sadata = sadata[sadata.obs["labels"].astype("str").isin(cell_types), :].copy()

    print(sadata.obs.groupby("labels").size())

    bulk_df = pd.read_csv(bulk_file, sep="\t", index_col=None)
    if bulk_df.iloc[:,0].dtype == "O":
        bulk_df.set_index(bulk_df.columns[0], inplace=True)
    print(f"bulk RNA-seq data - samples: {bulk_df.shape[0]}, genes: {bulk_df.shape[1]}")

    sc.pp.filter_cells(sadata, min_genes=200)
    sc.pp.filter_genes(sadata, min_counts=100)
    adata = dv.tl.combine(sadata, bulk_df)
    del sadata
    
    return adata, true_df


def run_benchmark(outdir, adata, true_df, device):
    ps = list(itertools.product(*PARAMS.values()))

    for i, values in enumerate(ps):
        params = dict(zip(PARAMS.keys(), values))
        model_type = params["model_type"]
        dropout_type = params["dropout_type"]
        bulk_dropout = params["bulk_dropout"]
        target_sum = params["target_sum"]
        exclude_highly_expressed = params["exclude_highly_expressed"]

        if target_sum == False:
            layer = "counts"
        elif target_sum == "lognorm":
            layer = "X"
            target_sum = None
        else:
            layer = "ncounts"
        
        if "counts" in adata.layers.keys():
            adata.X = adata.layers["counts"].copy()
            del adata.layers["counts"]
            del adata.layers["ncounts"]

        scout.tl.scale_log_center(adata, target_sum=target_sum, exclude_highly_expressed=exclude_highly_expressed)

        print(f"Model {(i+1)}/{len(ps)}: {' '.join([f'{k}: {v},' for k,v in params.items()])}")

        if os.path.exists(os.path.join(outdir, "done.json")):
            with open(os.path.join(outdir, "done.json"), "r") as f:
                done = json.load(f)
        else:
            done = []

        if bulk_dropout and (dropout_type == None):
            continue

        if target_sum == False and exclude_highly_expressed:
            continue

        if params in done:
            print("Already calculated!")
            continue
        else:
            done.append(params)

        out_dir = os.path.join(outdir, model_type)
        mkdir(out_dir)

        decon = dv.DeconV(
            adata, cell_type_key="labels",
            dropout_type=dropout_type,
            model_type=model_type, sub_type_key=None,
            device=device
        )

        decon.fit_reference(num_epochs=2000, lr=0.1, lrd=0.999, layer=layer, fp_hack=True)

        decon.deconvolute(model_dropout=bulk_dropout, lrd=0.999, lr=0.1, num_epochs=1000, progress=False)

        res_df = decon.get_results_df()
        res_df["true"] = true_df.melt()["value"]
        rmse, mad, r = dv.pl.xypredictions(res_df)
        plt.close()

        if target_sum == 1e6:
            normalisation = "cpm"
        elif target_sum == False:
            normalisation = "raw"
        elif target_sum == None:
            if layer == "ncounts":
                normalisation = "median"
            elif layer == "X":
                normalisation = "lognorm"
            else:
                assert False, "Not implemented"
        else:
            normalisation = f"{target_sum:e}"

        with open(os.path.join(outdir, "losses.txt"), "a") as f:
            f.write(model_type + "\t")
            f.write(normalisation + "\t")
            f.write(str(exclude_highly_expressed) + "\t")
            f.write(str(decon.deconvolution_module.reference_loss) + "\t")
            f.write(str(decon.deconvolution_module.deconvolution_loss) + "\t")
            f.write(str(rmse) + "\t")
            f.write(str(mad) + "\t")
            f.write(str(r) + "\n")

        with open(os.path.join(outdir, "done.json"), "w") as f:
            json.dump(done, f)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", type=str, help="Input directory", required=True)
    parser.add_argument("-o", "--outdir", type=str, help="Output directory", required=True)

    args = parser.parse_args()

    adata, true_df = read_inputs(args.indir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mkdir(args.outdir)

    if not os.path.exists(os.path.join(args.outdir, "losses.txt")):    
        with open(os.path.join(args.outdir, "losses.txt"), "a") as f:
            f.write("distribution\tnormalisation\texclude_highly_expressed\tref_nll_loss\tdeconv_nll_loss\trmse\tmad\tr\n")

    run_benchmark(args.outdir, adata, true_df, device)



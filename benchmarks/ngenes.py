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
    "model_type": ["gamma", "beta", "nb"],
    "bulk_dropout": [True],
    "n_genes": [12500, 10000, 7500, 5000, 2500, 1000, 500, 100]
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
    scout.tl.scale_log_center(adata, target_sum=None, exclude_highly_expressed=True)
    return adata, true_df


def run_benchmark(outdir, adata, true_df, device):
    ps = list(itertools.product(*PARAMS.values()))

    for i, values in enumerate(ps):
        params = dict(zip(PARAMS.keys(), values))
        model_type = params["model_type"]
        dropout_type = params["dropout_type"]
        bulk_dropout = params["bulk_dropout"]
        n_genes = params["n_genes"]
        print(f"Model {i+1}/{len(ps)}: {' '.join([f'{k}: {v},' for k,v in params.items()])}")

        if os.path.exists(os.path.join(outdir, "done.json")):
            with open(os.path.join(outdir, "done.json"), "r") as f:
                done = json.load(f)
        else:
            done = []

        if bulk_dropout and (dropout_type == None):
            continue

        if params in done:
            print("Already calculated!")
            continue
        else:
            done.append(params)

        out_dir = os.path.join(outdir, model_type)
        mkdir(out_dir)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_genes, subset=False)
        
        decon = dv.DeconV(
            adata[:, adata.var["highly_variable"]], cell_type_key="labels",
            dropout_type=dropout_type,
            model_type=model_type, sub_type_key=None,
            device=device
        )

        decon.fit_reference(num_epochs=2000, lr=0.1, lrd=0.999, layer="counts", fp_hack=True)

        suffix = f"n{n_genes}"

        decon.check_fit(path=os.path.join(out_dir, f"ref_fit_{suffix}.pdf"))
        plt.close()

        proportions = decon.deconvolute(model_dropout=bulk_dropout, lrd=0.999, lr=0.1, num_epochs=1000).cpu()
        pd.DataFrame(proportions, index=adata.uns["bulk_samples"], columns=decon.cell_types).to_csv(
            os.path.join(out_dir, f"proportions_{suffix}.tsv"), sep="\t"
        )

        res_df = decon.get_results_df()
        res_df["true"] = true_df.melt()["value"]
        rmse, mad, r = dv.pl.xypredictions(res_df, path=os.path.join(out_dir, f"xy_{suffix}.pdf"))
        plt.close()
        mkdir(os.path.join(out_dir, "pseudo"))

        for i in range(decon.n_bulk_samples):
            dv.pl.prediction_plot(decon, i, os.path.join(out_dir, "pseudo", f"sample_{i}_{suffix}.pdf"))
            plt.close()

        # decon.deconvolution_module.save_model(os.path.join(out_dir, f"model_{suffix}"))
        with open(os.path.join(outdir, "losses.txt"), "a") as f:
            f.write(model_type + "_" + suffix)
            f.write("\t")
            f.write(str(decon.deconvolution_module.reference_loss))
            f.write("\t")
            f.write(str(decon.deconvolution_module.deconvolution_loss))
            f.write("\t")
            f.write(str(rmse))
            f.write("\t")
            f.write(str(mad))
            f.write("\t")
            f.write(str(r))
            f.write("\n")

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

    with open(os.path.join(args.outdir, "losses.txt"), "a") as f:
        f.write("distribution\tref_nll_loss\tdeconv_nll_loss\trmse\tmad\tr\n")

    run_benchmark(args.outdir, adata, true_df, device)



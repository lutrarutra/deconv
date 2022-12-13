import glob, tqdm, time, os, argparse, yaml, warnings
from typing import Literal

# Custom tools
import sys

from base import NSM, MSEM
import plot as pl

import deconV as dV

import pandas as pd
import numpy as np
import scanpy as sc
import tqdm

def main(args):
    print("Reading scRNA-seq data...")
    sadata = dV.read_data(os.path.join(params["indir"], args.sc))
    print(f"scRNA-seq data - cells: {sadata.shape[0]}, genes: {sadata.shape[1]}")

    if args.bulk:
        print("Reading bulk data...")
        badata = dV.read_data(os.path.join(params["indir"], args.bulk), is_bulk=True, transpose_bulk=args.transpose_bulk)
        print(f"bulk RNA-seq data - samples: {badata.shape[0]}, genes: {badata.shape[1]}")
    else:
        print("No bulk RNA-seq data provided, creating pseudo bulk data")
        badata = sc.AnnData(sadata.X.sum(0).reshape(1,-1), var=sadata.var)

    print("Reading pheno data...")
    pheno_df = pd.read_csv(os.path.join(params["indir"], args.pheno), sep="\t" if args.pheno.endswith(".tsv") else ",", index_col=params["index_col"])
    pheno_df.index.name = None

    sadata.obs = pd.concat([sadata.obs, pheno_df], axis=1)
    assert params["cell_type_key"] in sadata.obs.columns, f"{params['cell_type_key']} not in obs columns"
    sadata.obs[params["cell_type_key"]] = sadata.obs[params["cell_type_key"]].astype(str)

    print("Preprocessing data...")
    sadata, badata = dV.preprocess(sadata, badata)
    print("After preprocessing:")
    print(f"scRNA-seq data - cells: {sadata.shape[0]}, genes: {sadata.shape[1]}")
    print(f"bulk RNA-seq data - samples: {badata.shape[0]}, genes: {badata.shape[1]}")

    assert sadata.shape[1] == badata.shape[1], "scRNA-seq and bulk RNA-seq data have different number of genes"

    pl.umap_plot(sadata, params["label_key"], fmt=params["fig_fmt"])
    pl.dispersion_plot(sadata, path=params["outdir"], layer="counts", fmt=params["fig_fmt"])

    cell_types = list(sadata.obs[params['label_key']].unique())
    
    decon = dV.DeconV(sadata, badata, cell_types, params)

    if params["plot_pseudo_bulk"]:
        print("Plottting bulk vs. pseudo bulk...")
        pl.pseudo_bulk_plot(
            decon.sadata, decon.bulk_sample_cols,
            dir=os.path.join(params["outdir"], "pseudo_bulk") if not params["jupyter"] else None,
            fmt=params["fig_fmt"], figsize=params["figsize"], dpi=params["dpi"]
        )

    print("Deconvolving...")
    decon.deconvolute()
    

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sc", type=str, help="Path to scRNA-seq data file (.csv/.tsv)", required=True)
    parser.add_argument("-b", "--bulk", type=str, help="Path to bulk RNA-seq data file (.csv/.tsv)", required=False)
    parser.add_argument("-p", "--pheno", type=str, help="Path to pheno data table file (.csv/.tsv)", required=True)
    parser.add_argument("-c", "--config", type=str, help="Path to parameters/config file (.yaml/.yml)", required=True)
    parser.add_argument("-o", "--outdir", type=str, help="Output directory", required=False, default="out")
    parser.add_argument("-i", "--indir", type=str, help="Input directory", required=False, default="")
    parser.add_argument("-t", "--ptrue", type=str, help="Path to true cell proportions file (.csv/.tsv)", required=False)
    parser.add_argument("--transpose_bulk", action="store_true", help="Transpose bulk data")
    args = parser.parse_args()

    args = parser.parse_args()

    # Read params.yaml file
    with open(args.config, "r") as f:
        params = yaml.safe_load(f)

    params["outdir"] = os.path.join(args.outdir, time.strftime('%Y%m%d-%H%M%S'))
    sc.settings.figdir = params["outdir"]
    params["indir"] = args.indir
    print("Output directory: ", params["outdir"])

    # convert selected_ct to string (QoL)
    if params["selected_ct"]:
        params["selected_ct"] = list(map(str, params["selected_ct"]))

    main(args)
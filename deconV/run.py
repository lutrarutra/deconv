import deconV as dv
import scout

import time, argparse, os, yaml
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

import pandas as pd
import numpy as np
import scanpy as sc

import scout


def main(params):
    sadata = dv.tl.read_data(params["ref_file"])
    print(f"scRNA-seq data - cells: {sadata.shape[0]}, genes: {sadata.shape[1]}")

    print("Reading pheno data...")
    pheno_df = pd.read_csv(params["ref_annot_file"], sep="\t", index_col=0)
    pheno_df.index.name = None

    common_cells = list(set(pheno_df.index.tolist()) & set(sadata.obs_names.tolist()))
    sadata = sadata[common_cells, :].copy()
    pheno_df = pheno_df.loc[common_cells, :].copy()
    sadata.obs[params["label_key"]] = pheno_df[params["label_key"]].tolist()
    sadata.obs.groupby(params["label_key"]).size()

    print("Reading bulk data...")
    bulk_df = pd.read_csv(params["bulk_file"], sep="\t", index_col=None)
    if bulk_df.iloc[:, 0].dtype == "O":
        bulk_df.set_index(bulk_df.columns[0], inplace=True)
    print(f"bulk RNA-seq data - samples: {bulk_df.shape[0]}, genes: {bulk_df.shape[1]}")

    if params["selected_ct"] is not None and len(params["selected_ct"]) > 0:
        sadata = sadata[sadata.obs[params["label_key"]].astype("str").isin(params["selected_ct"]), :].copy()

    sadata.obs[params["label_key"]] = sadata.obs[params["label_key"]].astype("category")
    sadata.obs.groupby(params["label_key"]).size()

    print("Preprocessing data...")
    sc.pp.filter_cells(sadata, min_genes=params["min_genes"])
    sc.pp.filter_genes(sadata, min_cells=params["min_cells"])

    adata = dv.tl.combine(sadata, bulk_df)
    scout.tl.scale_log_center(
        adata, target_sum=None if params["target_sum"] == -1 else params["target_sum"],
        exclude_highly_expressed=params["exclude_highly_expressed"]
    )

    if params["use_sub_type"]:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        scout.tl.sub_cluster(adata, params["label_key"])
    

    decon = dv.DeconV(
        adata, cell_type_key=params["label_key"],
        sub_type_key="sub_type" if params["use_sub_type"] else None,
        layer=params["layer"]
    )

    decon.filter_outliers(
        dropout_factor_quantile=params["dropout_factor_quantile"],
        pseudobulk_lims=params["pseudobulk_lims"],
        aggregate="max"
    )

    decon.init_dataset(
        weight_type=params["weight_type"], weight_agg=params["weight_agg"],
        inverse_weight=False, log_weight=False, quantiles=params["weight_quantiles"]
    )

    os.mkdir(params["outdir"])

    dv.pl.gene_weight_hist(
        decon.adata.varm["gene_weights"],
        f"Gene Weight ({params['weight_type']} | {params['weight_agg']})",
        logy=False,
        path=os.path.join(params["outdir"], f"gene_weight_hist.{params['fig_fmt']}"),
    )

    est_df = decon.deconvolute(
        model_type="poisson",
        num_epochs=5000,
        lr=0.01,
        use_outlier_genes=False, plot=False
    )

    est_df.to_csv(os.path.join(params["outdir"], "estimate.tsv"), sep="\t")

    if params["true_proportions"] is not None:
        true_df = pd.read_csv(params["true_proportions"], sep="\t", index_col=0)
        print("Evaluating deconvolution...")
        dv.pl.scatter_check(true_df, est_df, style="sample", path=os.path.join(params["outdir"], f"scatter_check.{params['fig_fmt']}"))


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to parameters/config file (.yaml/.yml)", required=True)
    
    parser.add_argument("-s", "--sc", type=str, help="Path to scRNA-seq data file (.csv/.tsv)", default=None, required=False)
    parser.add_argument("-b", "--bulk", type=str, help="Path to bulk RNA-seq data file (.csv/.tsv)", default=None, required=False)
    parser.add_argument("-p", "--pheno", type=str, help="Path to annotation data table file (.csv/.tsv)", default=None, required=False)
    parser.add_argument("-o", "--outdir", type=str, help="Output directory", required=False, default="out")
    parser.add_argument("-t", "--true", type=str, help="True proportions", required=False, default=None)

    args = parser.parse_args()

    # Read params.yaml file
    with open(args.config, "r") as f:
        params = yaml.safe_load(f)

    if args.sc is not None:
        params["ref_file"] = args.sc
    
    if args.bulk is not None:
        params["bulk_file"] = args.bulk

    if args.pheno is not None:
        params["ref_annot_file"] = args.pheno

    params["outdir"] = os.path.join(args.outdir, time.strftime('%Y%m%d-%H%M%S'))

    params["true_proportions"] = args.true

    # convert selected_ct to string (QoL)
    if params["selected_ct"]:
        params["selected_ct"] = list(map(str, params["selected_ct"]))

    main(params)
import time, yaml, argparse, os, itertools, tqdm, random

import deconV as dv
import scout

import pandas as pd
import numpy as np
import scanpy as sc

PARAMS = {
    # "target_sum": [None, 1, 1e4, 1e6],
    "exclude_highly_expressed": [False],
    "layer": ["ncounts", "counts"],
    "use_sub_type": [False],
    "weight_type": ["abs_score"],
    "weight_agg": ["min"],
    "weight_quantiles": [(0, 1)],
    "dropout_factor_quantile": [0.8, 0.85, 0.9, 0.95],
    "pseudobulk_lims": [(-10, 10)],
    "signature_quantiles": [(0, 1), (0, 0.95), (0, 0.9), (0, 0.85), (0, 0.8)],
}

def deconvolute(adata, cell_type_key, params):
    decon = dv.DeconV(
        adata, cell_type_key=cell_type_key,
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

    est_df = decon.deconvolute(
        model_type="poisson",
        num_epochs=5000,
        lr=0.01,
        use_outlier_genes=False,
        signature_quantiles=params["signature_quantiles"],
        progress_bar=False
    )

    return est_df


def main(args):
    sadata = dv.tl.read_data(args["ref_file"])
    print(f"scRNA-seq data - cells: {sadata.shape[0]}, genes: {sadata.shape[1]}")

    print("Reading pheno data...")
    pheno_df = pd.read_csv(args["ref_annot_file"], sep="\t", index_col=0)
    pheno_df.index.name = None

    common_cells = list(set(pheno_df.index.tolist()) & set(sadata.obs_names.tolist()))
    sadata = sadata[common_cells, :].copy()
    pheno_df = pheno_df.loc[common_cells, :].copy()
    sadata.obs[args["cell_type_key"]] = pheno_df[args["cell_type_key"]].tolist()
    sadata.obs.groupby(args["cell_type_key"]).size()

    print("Reading bulk data...")
    bulk_df = pd.read_csv(args["bulk_file"], sep="\t", index_col=None)
    if bulk_df.iloc[:, 0].dtype == "O":
        bulk_df.set_index(bulk_df.columns[0], inplace=True)
    print(f"bulk RNA-seq data - samples: {bulk_df.shape[0]}, genes: {bulk_df.shape[1]}")

    if args["selected_ct"] is not None and len(args["selected_ct"]) > 0:
        sadata = sadata[sadata.obs[args["cell_type_key"]].astype("str").isin(args["selected_ct"]), :].copy()

    sadata.obs[args["cell_type_key"]] = sadata.obs[args["cell_type_key"]].astype("category")
    sadata.obs.groupby(args["cell_type_key"]).size()

    print("Preprocessing data...")
    sc.pp.filter_cells(sadata, min_genes=200)
    sc.pp.filter_genes(sadata, min_cells=3)

    adata = dv.tl.combine(sadata, bulk_df)
    scout.tl.scale_log_center(
        adata, target_sum=None,
        exclude_highly_expressed=True
    )

    sc.pp.pca(adata, random_state=0)
    sc.pp.neighbors(adata, random_state=0)

    scout.tl.sub_cluster(adata, args["cell_type_key"])

    sadata.layers["counts"] = sadata.X.copy()

    scout.tl.rank_marker_genes(adata, groupby=args["cell_type_key"])
    # scout.tl.rank_marker_genes(sadata, groupby="sub_type")
    best_rmse = np.inf
    best_params = None

    ps = list(itertools.product(*PARAMS.values()))
    random.shuffle(ps)
    pbar = tqdm.tqdm(ps)

    with open("results.txt", "w") as f:
        f.write("rmse\tproportions\t" + "\t".join(PARAMS.keys()) + "\n")

    for values in pbar:
        params = dict(zip(PARAMS.keys(), values))
        est_df = deconvolute(adata, args["cell_type_key"], params)
        r = np.corrcoef(est_df.values.flatten(), np.array([0.6, 0.3, 0.1]))[0, 1]
        rmse = np.sqrt(((est_df.values.flatten() - np.array([0.6, 0.3, 0.1]))**2).sum())

        with open("results.txt", "a") as f:
            f.write(f"{rmse:.5f}\t{' '.join([f'{v:.3f}' for v in est_df.values.flatten()])}\t")
            f.write('\t'.join([str(v) for v in params.values()]))
            f.write("\n")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            best_est = est_df.values.flatten()
            pbar.set_postfix({
                "best_rmse": best_rmse,
                "best_est": best_est,
            })


if __name__ == "__main__":
    main({
        "cell_type_key": "cellType",
        "selected_ct": ["0", "1", "2"],
        "bulk_file": "./data/GSE136148/bulk.tsv",
        "ref_annot_file": "./data/GSE136148/pdata.tsv",
        "ref_file": "./data/GSE136148/sc.tsv",
    })

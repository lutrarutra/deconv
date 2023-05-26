import DeconV as dv
import scout

import glob, tqdm, time, os, argparse, json
import torch
import matplotlib.pyplot as plt

import pandas as pd
import scanpy as sc
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
    reference_file = os.path.join(indir, "sc.txt")
    reference_mdata_file = os.path.join(indir, "pdata.txt")
    bulk_file = os.path.join(indir, "bulk.txt")
    true_df = pd.read_table(os.path.join(indir, "proportions.txt"), index_col=0)
    true_df.sort_index(axis="columns", inplace=True)
    true_df.index.name = "sample"
    cell_types = ["alpha", "delta", "gamma", "beta"]

    adata = sc.read_csv(reference_file, first_column_names=True, delimiter="\t")

    pheno_df = pd.read_table(reference_mdata_file, index_col=0)
    pheno_df.index.name = None
    common_cells = list(set(pheno_df.index.tolist()) & set(adata.obs_names.tolist()))

    adata = adata[common_cells, :].copy()
    pheno_df = pheno_df.loc[common_cells, :].copy()
    adata.obs["labels"] = pheno_df["cellType"].tolist()
    adata.obs["labels"] = adata.obs["labels"].astype("category")

    adata = adata[adata.obs["labels"].astype("str").isin(cell_types), :].copy()

    print(adata.obs.groupby("labels").size())

    bulk_df = pd.read_table(bulk_file, index_col=None)
    if bulk_df.iloc[:,0].dtype == "O":
        bulk_df.set_index(bulk_df.columns[0], inplace=True)
    print(f"bulk RNA-seq data - samples: {bulk_df.shape[0]}, genes: {bulk_df.shape[1]}")

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_counts=100)
    adata = dv.tl.combine(adata, bulk_df)
    scout.tl.scale_log_center(adata)

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

        res_melt = decon.get_results_df()
        true_melt = true_df.reset_index().melt(id_vars="sample").rename(columns={"value":"true", "variable":"cell_type"})
        assert (true_melt["sample"] == res_melt["sample"]).all()
        assert (true_melt["cell_type"] == res_melt["cell_type"]).all()
        res_melt["true"] = true_melt["true"].values

        rmse, mad, r = dv.pl.xypredictions(res_melt, figsize=(5,5), dpi=150, legend=False)
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



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
    "dropout_type": ["separate", "shared", None],
    "model_type": ["gamma", "beta", "nb", "static", "lognormal"],
    "bulk_dropout": [True, False],
}

def read_inputs(indir):
    reference_file = os.path.join(indir, "reference.h5ad")
    bulk_file = os.path.join(indir, "bulk.txt")
    cell_types = [
        'CD4 T', 'Monocytes',
        'B cells', 'CD8 T',
        'NK', 'Monocytes',
        'DCs']
    true_df = pd.read_csv(os.path.join(indir, "true.csv"), index_col=0)
    true_df = true_df.reindex(sorted(true_df.columns), axis=1)

    adata = sc.read_h5ad(reference_file)
    adata.X = adata.X.astype("float32").toarray()

    adata = adata[adata.obs["labels"].astype("str").isin(cell_types), :].copy()

    print(adata.obs.groupby("labels").size())

    bulk_df = pd.read_table(bulk_file, index_col=0)

    print(f"bulk RNA-seq data - samples: {bulk_df.shape[0]}, genes: {bulk_df.shape[1]}")

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = dv.tl.combine(adata, bulk_df)
    scout.tl.scale_log_center(adata, target_sum=None, exclude_highly_expressed=True)
    return adata, true_df


def run_benchmark(outdir, adata, true_df, device):
    ps = list(itertools.product(*PARAMS.values()))

    for i, values in enumerate(ps):
        params = dict(zip(PARAMS.keys(), values))
        model_type = params["model_type"]
        dropout_type = params["dropout_type"]
        bulk_dropout = params["bulk_dropout"]

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

        decon = dv.DeconV(
            adata, cell_type_key="labels",
            dropout_type=dropout_type,
            model_type=model_type, sub_type_key=None,
            device=device
        )

        decon.fit_reference(num_epochs=2000, lr=0.1, lrd=0.999, layer="counts", fp_hack=False)

        suffix = f"{dropout_type}{'_bd' if bulk_dropout else ''}"

        decon.check_fit(path=os.path.join(out_dir, f"ref_fit_{suffix}.pdf"))
        plt.close()
        
        proportions = decon.deconvolute(model_dropout=bulk_dropout, lrd=0.999, lr=0.1, num_epochs=1000).cpu()
        pd.DataFrame(proportions, index=adata.uns["bulk_samples"], columns=decon.cell_types).to_csv(
            os.path.join(out_dir, f"proportions_{suffix}.tsv"), sep="\t"
        )

        res_melt = decon.get_results_df()
        true_melt = true_df.reset_index().melt(id_vars="sample").rename(columns={"value":"true", "variable":"cell_type"})
        assert (true_melt["sample"] == res_melt["sample"]).all()
        assert (true_melt["cell_type"] == res_melt["cell_type"]).all()
        res_melt["true"] = true_melt["true"].values

        rmse, mad, r = dv.pl.xypredictions(res_melt, figsize=(5,5), dpi=150, path=os.path.join(out_dir, f"xy_{suffix}.pdf"), legend=False)

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



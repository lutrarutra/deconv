import argparse
import os
import yaml

import torch
import pandas as pd
import scanpy as sc

import deconv as dv


def mkdir(path: str) -> str:
    _dir = os.path.dirname(path)
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return path


def run(adata: sc.AnnData, bulk: sc.AnnData | pd.DataFrame, args: argparse.Namespace) -> int:
    decon = dv.DeconV(
        adata=adata,
        bulk=bulk,
        cell_type_key=args.cell_type_key,
        model_type=args.model,
        top_n_variable_genes=args.num_genes,
        layer=args.layer if args.layer != "X" else None,
        fp_hack=args.fp_hack,
        device=args.device,
        dropout_type=args.ref_dropout if args.ref_dropout != "none" else None,
        model_bulk_dropout=args.dec_dropout,
    )

    print("Fitting reference model...")
    decon.fit_reference(num_epochs=args.ref_epochs, lr=args.ref_lr, lrd=args.ref_lrd)

    if not args.no_plots:
        decon.check_fit(path=mkdir(os.path.join(args.outdir, "figures", f"gex_dist_fit.{args.figure_fmt}")), show=False)
        decon.plot_reference_losses(path=mkdir(os.path.join(args.outdir, "figures", f"ref_fit_history.{args.figure_fmt}")), show=False)

    print("Deconvoluting cell types...")
    proportions = decon.deconvolute(num_epochs=args.dec_epochs, lr=args.dec_lr, lrd=args.dec_lrd)

    if not args.no_plots:
        decon.plot_bar_proportions(
            show=False, path=mkdir(os.path.join(args.outdir, "figures", f"bar_cell_proportions.{args.figure_fmt}")),
        )
        decon.plot_heatmap_proportions(
            show=False, path=mkdir(os.path.join(args.outdir, "figures", f"heatmap_cell_proportions.{args.figure_fmt}")),
        )
        decon.plot_deconvolution_losses(
            show=False, path=mkdir(os.path.join(args.outdir, "figures", f"dec_fit_history.{args.figure_fmt}")),
        )

    proportions.to_csv(mkdir(os.path.join(args.outdir, "predicted_proportions.csv")), index=False)
    decon.get_results_df().to_csv(mkdir(os.path.join(args.outdir, "results.csv")), index=False)
    
    with open(mkdir(os.path.join(args.outdir, "config.yaml")), "w") as f:
        d = vars(args)
        d["version"] = dv.__version__
        yaml.dump(d, f)

    print(f"Done!\nResults saved to '{args.outdir}'")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="Deconv", description="Probabilistic Cell Type Deconvolution of Bulk RNA-seq.")

    parser.add_argument("--ref", type=str, required=True, help="Path to Single-cell reference (.h5ad/.csv/.tsv/.txt). If you provide as matrix (csv/.tsv/.txt), make sure the genes are specified as column names (first row) and cell-types are specified as row names (first column).")
    parser.add_argument("--bulk", type=str, required=True, help="Path to bulk RNA-seq data (.csv/.tsv/.txt/.h5ad). Make sure the genes are specified as column names (first row) and samples are specified as row names (first column).")
    parser.add_argument("--outdir", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--cell-type-key", type=str, required=False, help="Key in 'adata.obs' that contains cell type labels.")

    parser.add_argument("--transpose-bulk", action="store_true", default=False, help="Transpose bulk data.")
    parser.add_argument("--no-plots", action="store_true", default=False, help="Do not save plots.")
    parser.add_argument("--figure-fmt", type=str, default="pdf", choices=["pdf", "png"], help="Format for saving figures.")
    
    parser.add_argument("--model", type=str, default="nb", choices=["nb", "gamma", "beta", "lognormal", "static"], help="Model to use for gene expression modelling.")
    parser.add_argument("--num-genes", type=int, default=None, required=False, help="Number of genes to use for deconvolution (top-n variable genes). Default: all genes.")
    parser.add_argument("--layer", type=str, default="X", help="Layer with raw counts in AnnData object. Default: 'X'.")
    parser.add_argument("--fp-hack", action="store_true", default=False, help="Use the floating point underflow for numerical stability.")
    
    parser.add_argument("--ref-epochs", type=int, default=2000, help="Number of epochs for reference model training.")
    parser.add_argument("--ref-lr", type=float, default=0.1, help="Learning rate for reference model training.")
    parser.add_argument("--ref-lrd", type=float, default=0.999, help="Learning rate decay for reference model training.")
    parser.add_argument("--ref-dropout", default="separate", type=str, choices=["separate", "shared", "none"], help="Reference model dropout type. 'separate' - Unique for each cell type; 'shared' - Shared across all cell types; 'none' - No dropout.")

    parser.add_argument("--dec-dropout", type=str, choices=["auto", "true", "false"], help="Use dropout in the bulk gene expression distribution. Default: 'auto' - Use dropout if 'ref-dropout' is not 'none'.")
    parser.add_argument("--dec-epochs", type=int, default=1000, help="Number of epochs for deconvolution model training.")
    parser.add_argument("--dec-lr", type=float, default=0.1, help="Learning rate for deconvolution model.")
    parser.add_argument("--dec-lrd", type=float, default=0.999, help="Learning rate decay for deconvolution model.")
    
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use for computations. Default: 'cuda' if available.")
    parser.add_argument("--ncores", type=int, default=-1, help="Number of cores to use, default: no limit (-1).")

    parser.add_argument("--version", action="version", version=f"v{dv.__version__}")

    args = parser.parse_args()

    print(f"DeconV v{dv.__version__}")

    if not os.path.exists(args.ref):
        print(f"File '{args.ref}' not found.")
        return 1
    
    if not os.path.exists(args.bulk):
        print(f"File '{args.bulk}' not found.")
        return 1
    
    if args.ref.endswith(".h5ad") and args.cell_type_key is None:
        print("'--cell-type-key' is required for AnnData reference.")
        return 1

    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA is not available.")
            return 1
    elif args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {args.device}")

    if args.ref.endswith(".h5ad"):
        adata = sc.read_h5ad(args.ref)
    else:
        gex = pd.read_csv(
            args.ref,
            sep="," if args.ref.endswith(".csv") else "\t",
            index_col=0
        )
        adata = sc.AnnData(X=gex.values, var=pd.DataFrame(index=gex.columns))
        adata.obs["cell_type"] = gex.index.tolist()
        args.cell_type_key = "cell_type"
        if adata.n_obs == adata.obs["cell_type"].unique().shape[0]:
            print("First column of the reference matrix must be cell type labels.")
            return 1

    if args.dec_dropout == "auto":
        args.dec_dropout = args.ref_dropout != "none"
    elif args.dec_dropout == "true":
        if args.ref_dropout == "none":
            print("Cannot use dropout in the deconvolution model if the reference model does not use dropout.")
            return 1
        args.dec_dropout = True
    else:
        args.dec_dropout = False
    
    if args.cell_type_key not in adata.obs.columns:
        print(f"Key '{args.cell_type_key}' not found in 'adata.obs.columns'.")
        return 1
    
    if args.layer != "X":
        if args.layer not in adata.layers.keys():
            print(f"Layer '{args.layer}' not found in 'adata.layers'.")
            return 1
        
    print(f"Cell-types: {adata.obs.groupby(args.cell_type_key, observed=False).size()}")
        
    if args.ncores != -1:
        torch.set_num_threads(args.ncores)

    if args.bulk.endswith(".h5ad"):
        bulk = sc.read_h5ad(args.bulk)
    else:
        bulk = pd.read_csv(
            args.bulk,
            sep="," if args.bulk.endswith(".csv") else "\t",
            index_col=0
        )
        if args.transpose_bulk:
            bulk = bulk.T

    try:
        return run(adata=adata, bulk=bulk, args=args)
    except ValueError as e:
        print(e)
        return 1


if __name__ == "__main__":
    exit(main())

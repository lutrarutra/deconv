# DeconV: Probabilistic Cell Type Deconvolution of Bulk RNA-seq

- [BioRxiv Pre-print](https://www.biorxiv.org/content/10.1101/2023.12.07.570524v1)

- Single-cell RNA-seq reference
- Probabilistic Model
- Proportions inferred as distributions with confidence intervals
- GPU acceleration with Pyro & PyTorch
- Integrates with ScanPy and AnnData-object
- MIT License

![](https://github.com/lutrarutra/deconv/blob/main/deconv/figures/banner.png?raw=true)

## Installation
### Conda
- CPU only:
    - `conda install lutrarutra::deconv`
- or GPU (CUDA):
    - `conda install lutrarutra::deconv pytorch-gpu`
- New Environment:
    - `conda create -n deconv lutrarutra::deconv # CPU only`
    - `conda create -n deconv lutrarutra::deconv pytorch-gpu  # GPU`
    
### PIP
0. Download
    - `git clone https://github.com/lutrarutra/deconv` or `wget https://github.com/lutrarutra/deconv/releases/download/v0.9.3/deconv.v0.9.3.zip`
    - `unzip deconv.v0.9.3.zip` 
    - `cd deconv.v0.9.3`
1. Install:
    - Normal mode:
        - `pip install .`
    - or editable mode (dev):
        - `pip install -e .`

## How to run?
1. Run as a notebook or Python script. See `examples/` for examples:
    ```python
    import deconv
    model = deconv.DeconV(
        adata,
        bulk=bulk_df,                   # bulk_df is a pandas DataFrame with genes as columns and samples as rows
        cell_type_key="labels",         # cell_type_key is the column key in adata.obs that holds the cell type annotations 
        dropout_type="separate",        # 'separate', 'shared', or None
        model_type="gamma",             # 'nb', 'gamma', 'beta', 'lognormal', or 'static'    
        device=device,                  # 'cpu' or 'cuda'
        layer=None,                     # You can specify layer where raw counts are stored. None denotes adata.X.
        top_n_variable_genes=10000,     # Number of top variable genes to use, None to use all
    )
    model.fit_reference(num_epochs=2000, lr=0.1, lrd=0.999)

    proportions = model.deconvolute(lrd=0.999, lr=0.1, num_epochs=1000)
    ```
2. Or use the command line interface (CLI) with `deconv`:
    ```bash
    deconv --ref data/xin/sc.tsv --bulk data/xin/bulk.txt --outdir out --fp-hack --model gamma
    ```
    - Reference (`deconv --help`):
        ```bash
        usage: Deconv [-h] --ref REF --bulk BULK --outdir OUTDIR [--cell-type-key CELL_TYPE_KEY] [--transpose-bulk] [--no-plots] [--figure-fmt {pdf,png}] [--model {nb,gamma,beta,lognormal,static}]
                    [--num-genes NUM_GENES] [--layer LAYER] [--fp-hack] [--ref-epochs REF_EPOCHS] [--ref-lr REF_LR] [--ref-lrd REF_LRD] [--ref-dropout {separate,shared,none}]
                    [--dec-dropout {auto,true,false}] [--dec-epochs DEC_EPOCHS] [--dec-lr DEC_LR] [--dec-lrd DEC_LRD] [--device {auto,cpu,cuda}] [--ncores NCORES] [--version]

        Probabilistic Cell Type Deconvolution of Bulk RNA-seq.

        options:
        -h, --help            show this help message and exit
        --ref REF             Path to Single-cell reference (.h5ad/.csv/.tsv/.txt). If you provide as matrix (csv/.tsv/.txt), make sure the genes are in the columns and cell-types are specified in the first
                                column.
        --bulk BULK           Path to bulk RNA-seq data (.csv/.tsv/.txt/.h5ad). Make sure genes are in the columns and samples are in the rows
        --outdir OUTDIR       Path to output directory.
        --cell-type-key CELL_TYPE_KEY
                                Key in 'adata.obs' that contains cell type labels.
        --transpose-bulk      Transpose bulk data.
        --no-plots            Do not save plots.
        --figure-fmt {pdf,png}
                                Format for saving figures.
        --model {nb,gamma,beta,lognormal,static}
                                Model to use for gene expression modelling.
        --num-genes NUM_GENES
                                Number of genes to use for deconvolution (top-n variable genes). Default: all genes.
        --layer LAYER         Layer with raw counts in AnnData object. Default: 'X'.
        --fp-hack             Use the floating point underflow for numerical stability.
        --ref-epochs REF_EPOCHS
                                Number of epochs for reference model training.
        --ref-lr REF_LR       Learning rate for reference model training.
        --ref-lrd REF_LRD     Learning rate decay for reference model training.
        --ref-dropout {separate,shared,none}
                                Reference model dropout type. 'separate' - Unique for each cell type; 'shared' - Shared across all cell types; 'none' - No dropout.
        --dec-dropout {auto,true,false}
                                Use dropout in the bulk gene expression distribution. Default: 'auto' - Use dropout if 'ref-dropout' is not 'none'.
        --dec-epochs DEC_EPOCHS
                                Number of epochs for deconvolution model training.
        --dec-lr DEC_LR       Learning rate for deconvolution model.
        --dec-lrd DEC_LRD     Learning rate decay for deconvolution model.
        --device {auto,cpu,cuda}
                                Device to use for computations. Default: 'cuda' if available.
        --ncores NCORES       Number of cores to use, default: no limit (-1).
        --version             show program's version number and exit
        ```

## Problems
- Numerical stability / floating point underflow:
    - `Expected parameter loc ... of distribution ..., scale: .. to satisfy the constraint Real(), but found invalid values:`
    
    - When there are many zeros in the data, the model can underflow during training.

    - Use `fp_hack=True` (python) or `--fp-hack` (bash/cli) to enable the floating point underflow hack. This will help with numerical stability. This turns 0s in the gene expression matrix into 1s.

## Figures
![](https://github.com/lutrarutra/deconv/blob/main/examples/figures/pbmc_benchmark_scatter.png?raw=true)


![](https://github.com/lutrarutra/deconv/blob/main/examples/figures/pbmc_bar_proportions.png?raw=true)

![](https://github.com/lutrarutra/deconv/blob/main/examples/figures/pbmc_heatmap_proportions.png?raw=true)

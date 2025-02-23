# DeconV: Probabilistic Cell Type Deconvolution from Bulk RNA-seq

- single-cell RNA-seq reference
- Probabilistic Model
- Proportions inferred as distributions
- GPU acceleration with Pyro & PyTorch
- Integrates with ScanPy and AnnData-object

![](https://github.com/lutrarutra/deconv/blob/main/DeconV/figures/banner.png?raw=true)

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
    - `wget https://github.com/lutrarutra/deconv/releases/download/v0.9.2/deconv.v0.9.2.zip` or `git clone https://github.com/lutrarutra/deconv`
    - `unzip deconv.v0.9.2.zip` 
    - `cd deconv.v0.9.2`
1. Install dependencies
    - `pip install -r requirements.txt`
2. Install DeconV
    - `pip install -e .`
## How to run?
3. Run as a notebook or Python script. See `examples/` for examples:
    ```python
        decon = dv.DeconV(
            adata,
            bulk=bulk_df,                   # bulk_df is a pandas DataFrame with genes as columns and samples as rows
            cell_type_key="labels",         # cell_type_key is the column key in adata.obs that holds the cell type annotations 
            dropout_type="separate",        # 'separate', 'shared', or None
            model_type="gamma",             # 'nb', 'gamma', 'beta', 'lognormal', or 'static'    
            device=device,                  # 'cpu' or 'cuda'
            layer=None,                     # You can specify layer where raw counts are stored. None denotes adata.X.
            top_n_variable_genes=10000,     # Number of top variable genes to use, None to use all
        )
        decon.fit_reference(num_epochs=2000, lr=0.1, lrd=0.999)

        proportions = decon.deconvolute(model_dropout=True, lrd=0.999, lr=0.1, num_epochs=1000)
    ```


![](https://github.com/lutrarutra/deconv/blob/main/DeconV/figures/xin_xy.png?raw=true)

![](https://github.com/lutrarutra/deconv/blob/main/DeconV/figures/xin_boxes.png?raw=true)

![](https://github.com/lutrarutra/deconv/blob/main/DeconV/figures/xin_bars.png?raw=true)

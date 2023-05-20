# DeconV: Probabilistic Cell Type Deconvolution from Bulk RNA-seq

![](https://github.com/lutrarutra/deconv/blob/main/deconV/figures/banner.png?raw=true)

## How to run? (Tested with Python 3.10.9)
0. `cd deconv`
1. Install dependencies
    - `pip install -r requirements.txt`
2. Install DeconV
    - `pip install -e .`
3. Run as a notebook or Python script. See `examples/` for examples:
    ```python
        decon = dv.DeconV(
            adata, cell_type_key="labels",  # cell_type_key is the column key in adata.obs that holds the cell type annotations 
            dropout_type="separate",        # separate, shared, or None
            model_type="gamma",             # Gamma, Beta, nb, lognormal, or static    
            device=device                   # GPU acceleration
        )
        decon.fit_reference(num_epochs=2000, lr=0.1, lrd=0.999, layer="counts")

        proportions = decon.deconvolute(model_dropout=True, lrd=0.999, lr=0.1, num_epochs=1000)
    ```


![](https://github.com/lutrarutra/deconv/blob/main/deconV/figures/xin_xy.png?raw=true)

![](https://github.com/lutrarutra/deconv/blob/main/deconV/figures/xin_boxes.png?raw=true)

![](https://github.com/lutrarutra/deconv/blob/main/deconV/figures/xin_bars.png?raw=true)
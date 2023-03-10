import scanpy as sc
import numpy as np

def fmt_c(w):
    return " ".join([f"{v:.2f}" for v in w])


def combine(adata, bulk_df):
    # Checks if bulk genes are given as columns or as rows
    common_genes = np.intersect1d(adata.var_names, bulk_df.index.astype(str))
    common_genes_t = np.intersect1d(adata.var_names, bulk_df.columns.astype(str))

    if len(common_genes) < len(common_genes_t):
        common_genes = common_genes_t
        bulk_df = bulk_df.T

    assert (len(common_genes) > 0), "No common genes found between bulk and single cell data"

    adata = adata[:, common_genes].copy()
    adata.varm["bulk"] = bulk_df.loc[common_genes].values.astype(np.float32)
    
    if "counts" not in adata.layers.keys():
        adata.varm["pseudo"] = adata.X.sum(0).reshape(-1, 1)
    else:
        adata.varm["pseudo"] = adata.layers["counts"].sum(0)

    adata.varm["bulk"] *= (adata.varm["pseudo"].sum() / adata.varm["bulk"].sum(0))

    adata.varm["pseudo_factor"] = (np.log1p(adata.varm["bulk"]) - np.log1p(adata.varm["pseudo"]))

    adata.uns["bulk_samples"] = bulk_df.columns.tolist()

    print(f"scRNA-seq data - cells: {adata.shape[0]}, genes: {adata.shape[1]}")
    print(f"bulk RNA-seq data - samples: {adata.varm['bulk'].shape[1]}, genes: {adata.varm['bulk'].shape[0]}")

    return adata


def read_data(path):
    if path.endswith(".csv"):
        delim = ","
    elif path.endswith(".tsv"):
        delim = "\t"
    else:
        assert False, "Unknown file type, only .csv and .tsv are supported"

    adata = sc.read_csv(path, delimiter=delim)
    adata.var_names_make_unique()

    return adata

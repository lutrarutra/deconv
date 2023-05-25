library(MuSiC)
# library(reticulate)
# library(dplyr)
# library(DESeq2)
# library(Biobase)
library(SingleCellExperiment)

source("https://raw.githubusercontent.com/DimitriMeistermann/veneR/main/loadFun.R")

sc_counts <- t(read.csv("data/pbmc_counts.csv", header=T, row.names=1, sep=","))
sc_pdata <- read.csv("data/pbmc_celltypes.csv")

# sc.eset <- ExpressionSet(assayData=sc_counts, phenoData = sc_pdata)
sc.sce = SingleCellExperiment(list(counts=sc_counts), colData=sc_pdata)

bulk_counts <- as.matrix(read.table("data/pbmc_bulk_data.csv", header=T, row.names=1, sep=","))
# bulk.eset <- ExpressionSet(assayData=bulk_counts)
# bulk.mtx = exprs(bulk.eset)

# dim(bulk_counts)

# dim(bulk.eset)
# head(exprs(bulk.eset))

# head(bulk.mtx)

names(colData(sc.sce))

est = music_prop(
    bulk.mtx=bulk_counts, sc.sce=sc.sce, clusters='labels',
    select.ct=NULL, samples='batch'
)

fastWrite(est$"Est.prop.weighted", "music_proportions.csv", sep=",")

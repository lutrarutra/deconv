library(MuSiC)
library(SingleCellExperiment)

sc_counts <- t(read.table("data/xin/sc.txt", header=T, row.names=1, sep="\t"))
sc_pdata <- read.table("data/xin/pdata.txt", header=T, row.names=1, sep="\t")

sc.sce = SingleCellExperiment(list(counts=sc_counts), colData=sc_pdata)

bulk_counts <- as.matrix(t(read.table("data/xin/bulk.txt", header=T, row.names=1, sep="\t")))

est = music_prop(
    bulk.mtx=bulk_counts, sc.sce=sc.sce, clusters='cellType',
    select.ct=c('alpha', 'beta', 'delta', 'gamma'), samples='sampleID'
)

fastWrite(est$"Est.prop.weighted", "music_proportions.csv", sep=",")

library(MuSiC)
library(reticulate)
library(dplyr)
library(DESeq2)
library(Biobase)
library(SingleCellExperiment)

setwd("~/Documents/dev/fimm/data/xin")
source("https://raw.githubusercontent.com/DimitriMeistermann/veneR/main/loadFun.R")


XinT2D.eset = readRDS('XinT2Deset.rds')
XinT2D.eset
XinT2D.construct.full = bulk_construct(XinT2D.eset, clusters = 'cellType', samples = 'SubjectName')
XinT2D.construct.full$prop.real = relative.ab(XinT2D.construct.full$num.real, by.col = FALSE)
bulk.eset = XinT2D.construct.full$Bulk.counts

# sc.eset <- ExpressionSet(assayData=sc_counts, phenoData = sc_pdata)
sc.sce = SingleCellExperiment(list(counts=sc_counts), colData=sc_pdata)

bulk_counts <- as.matrix(read.table("data/xin_bulk_data.csv", header=T, row.names=1, sep=","))

names(colData(sc.sce))

est = music_prop(
    bulk.mtx=bulk_counts, sc.sce=sc.sce, clusters='labels',
    select.ct=NULL, samples='sampleID'
)

fastWrite(est$"Est.prop.weighted", "music_proportions.csv", sep=",")

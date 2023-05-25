library(MuSiC)
library(reticulate)
library(dplyr)
library(DESeq2)
library(Biobase)
library(SingleCellExperiment)

setwd("~/Documents/dev/deconv_benchmark/deconv/benchmarks/3rd-party/music/xin")
source("https://raw.githubusercontent.com/DimitriMeistermann/veneR/main/loadFun.R")


XinT2D.sce = readRDS('XinT2Dsce.rds')

XinT2D.construct.full = bulk_construct(XinT2D.eset, clusters = 'cellType', samples = 'SubjectName')
XinT2D.construct.full$prop.real = relative.ab(XinT2D.construct.full$num.real, by.col = FALSE)
bulk.eset = XinT2D.construct.full$Bulk.counts

est = music_prop(
    bulk.mtx=bulk_counts, sc.sce=sc.sce, clusters='labels',
    select.ct=NULL, samples='sampleID'
)

fastWrite(est$"Est.prop.weighted", "music_proportions.csv", sep=",")

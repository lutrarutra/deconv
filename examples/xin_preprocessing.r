library(MuSiC)
source("https://raw.githubusercontent.com/DimitriMeistermann/veneR/main/loadFun.R")

getwd()
setwd("~/Documents/dev/fimm/data/xin/")

# Single-cell for artificial bulk
# SingleCellExperiment from: https://xuranw.github.io/MuSiC/articles/pages/data.html
# Single cell RNA-seq data of pancreatic islet from healthy and diseased individuals(Xin et al.)
# XinT2D.eset = readRDS('raw/XinT2Deset.rds')
XinT2D.sce = readRDS('raw/XinT2Dsce.rds')

# Single-cell Reference
# ExpressionSet from:  https://xuranw.github.io/MuSiC/articles/pages/data.html
# Single cell RNA-seq data of pancreatic islets from healthy individuals (Segerstolpe et al.)
sc.eset = readRDS('raw/EMTABesethealthy.rds')

XinT2D.construct.full = bulk_construct(XinT2D.sce, clusters='cellType', samples='SubjectName')
XinT2D.construct.full$prop.real = relative.ab(XinT2D.construct.full$num.real, by.col=FALSE)
bulk.eset = XinT2D.construct.full$bulk.counts

fastWrite(as.matrix(XinT2D.construct.full$prop.real), "proportions.txt")

fastWrite(t(as.matrix((bulk.eset))), "bulk.txt")

fastWrite(t(as.matrix(exprs((sc.eset)))), "sc.txt")

fastWrite(as.matrix(pData(sc.eset)), "pdata.txt")

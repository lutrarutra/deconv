
library(MuSiC)
source("https://raw.githubusercontent.com/DimitriMeistermann/veneR/main/loadFun.R")

setwd("~/Documents/dev/fimm/data/xin")

XinT2D.eset = readRDS('XinT2Deset.rds')
XinT2D.eset
XinT2D.construct.full = bulk_construct(XinT2D.eset, clusters = 'cellType', samples = 'SubjectName')
XinT2D.construct.full$prop.real = relative.ab(XinT2D.construct.full$num.real, by.col = FALSE)
bulk.eset = XinT2D.construct.full$Bulk.counts

sc.eset = readRDS("EMTABesethealthy.rds")

sc.eset$"SubjectName"

prop = music_prop(
    bulk.eset=bulk.eset, sc.eset=sc.eset,
    clusters="cellType", samples="sampleID",
    select.ct = c('alpha', 'beta', 'delta', 'gamma')
)
prop$Est.prop.weighted

fastWrite(t(as.matrix(exprs((bulk.eset)))), "bulk.tsv")

# fastWrite(t(), "./bulk.tsv"))
fastWrite(t(as.matrix(exprs((sc.eset)))), "sc.tsv")

fastWrite(as.matrix(pData(sc.eset)), "pdata.tsv")

varLabels(sc.eset)

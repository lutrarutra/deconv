#! /bin/bash
cd data
scaden simulate --pattern "*_counts.txt"
scaden process data.h5ad pbmc_bulk_data.txt
scaden train processed.h5ad --steps 5000 --model_dir model
scaden predict --model_dir model pbmc_bulk_data.txt
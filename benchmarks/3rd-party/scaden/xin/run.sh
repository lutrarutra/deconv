#! /bin/bash
cd data
scaden simulate --n_samples 32000 --pattern "*_counts.txt"
scaden process data.h5ad xin_bulk_data.txt
scaden train processed.h5ad --steps 5000 --model_dir model
scaden predict --model_dir model xin_bulk_data.txt
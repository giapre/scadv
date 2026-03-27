#!/bin/bash

while read -r s; do
    sbatch --export=SUB_ID=$s 6_extract_synthetic_data_features.slurm
done < subject_list.txt
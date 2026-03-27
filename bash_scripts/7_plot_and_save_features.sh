#!/bin/bash

while read -r s; do
    sbatch --export=SUB_ID=$s 7_plot_and_save_features.slurm
done < subject_list.txt
#!/bin/bash

while read -r s; do
    sbatch --export=SUB_ID=$s 4_plot_patients_setup_for_sims.slurm
done < subject_list.txt
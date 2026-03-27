#!/bin/bash

while read -r s; do
    sbatch --export=SUB_ID=$s 3_create_params_for_simulations.slurm
done < subject_list.txt
#!/bin/bash

while read -r s; do
    sbatch --export=SUB_ID=$s 5_run_simulation.slurm
done < subject_list.txt
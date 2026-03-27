#!/bin/bash

while read -r s; do
    sbatch --export=SUB_ID=$s 5bis_stack_simulations.slurm
done < subject_list.txt
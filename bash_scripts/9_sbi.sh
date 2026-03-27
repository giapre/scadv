#!/bin/bash

while read -r sub ses; do
    sbatch --export=SUB_ID=$sub,SES_ID=$ses 9_sbi.slurm
done < subject_and_ses_list_t0.txt
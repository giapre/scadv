#!/bin/bash

while read -r sub ses; do
    sbatch --export=SUB_ID=$sub,SES_ID=$ses 8_extract_emp_data_features.slurm
done < subject_and_ses_list_t0.txt
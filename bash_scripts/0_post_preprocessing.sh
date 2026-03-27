#!/bin/bash

while read -r sub ses; do
    sbatch --export=SUB_ID=$sub,SES_ID=$ses 0_post_preprocessing.slurm
done < subject_and_ses_list.txt
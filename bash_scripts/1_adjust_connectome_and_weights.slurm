#!/bin/bash
#
#SBATCH --job-name adjust_connectome
#SBATCH --partition=jobs-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6GB
#SBATCH --output /data/core-psy-archive/projects/VBT_SCZ/scadv/logs/adjust_%A_%j.out
#SBATCH --error  /data/core-psy-archive/projects/VBT_SCZ/scadv/logs/adjust_%A_%j.err
#SBATCH --account=core-psy

# SUBJECT must be passed when submitting
# Example:
# sbatch --export=SUB_ID=sub-1864 run_adjust_connectome.slurm

singularity exec \
-B /data/core-psy-archive:/data/core-psy-archive \
--pwd /data/core-psy-archive/projects/VBT_SCZ/scadv/synth_pat/scripts \
/data/core-psy-archive/projects/VBT_SCZ/vbt-preproc-26-02-10.sif \
python3 1_adjust_connectome_and_weights.py ${SUB_ID}
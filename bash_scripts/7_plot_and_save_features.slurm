#!/bin/bash
#SBATCH --job-name=run_simulation
#SBATCH --partition=jobs-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16GB
#SBATCH --output=/data/core-psy-archive/projects/VBT_SCZ/scadv/logs/7_plot_and_save_features%A_%a.out
#SBATCH --error=/data/core-psy-archive/projects/VBT_SCZ/scadv/logs/7_plot_and_save_features%A_%a.err
#SBATCH --account=core-psy

# SUBJECT must be passed as an environment variable SUB_ID
if [ -z "$SUB_ID" ]; then
    echo "Error: SUB_ID not set"
    exit 1
fi

singularity exec \
    --nv \
    -B /data/core-psy-archive/data/PRONIA/test_vbt_pipe:/data/core-psy-archive/data/PRONIA/test_vbt_pipe \
    --pwd /data/core-psy-archive/projects/VBT_SCZ/scadv/synth_pat/scripts \
    /data/core-psy-archive/projects/VBT_SCZ/vbt-preproc-26-02-10.sif \
    python3 -u 7_plot_and_save_features.py ${SUB_ID}
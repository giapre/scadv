#!/bin/bash
#SBATCH --job-name=postproc_fmri
#SBATCH --partition=jobs-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12GB
#SBATCH --output=/data/core-psy-archive/projects/VBT_SCZ/scadv/logs/9_sbi_%A_%a.out
#SBATCH --error=/data/core-psy-archive/projects/VBT_SCZ/scadv/logs/9_sbi_%A_%a.err
#SBATCH --account=core-psy

# Check required variables
if [ -z "$SUB_ID" ]; then
    echo "Error: SUB_ID not set"
    exit 1
fi

if [ -z "$SES_ID" ]; then
    echo "Error: SES_ID not set"
    exit 1
fi

echo "Running subject: $SUB_ID session: $SES_ID"

singularity exec \
    -B /data/core-psy-archive/data/PRONIA:/data/core-psy-archive/data/PRONIA \
    --pwd /data/core-psy-archive/projects/VBT_SCZ/scadv/synth_pat/scripts \
    /data/core-psy-archive/projects/VBT_SCZ/vbt-preproc-26-02-10.sif \
    python3 -u 9_sbi.py ${SUB_ID} ${SES_ID}
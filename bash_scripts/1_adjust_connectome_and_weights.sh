for s in $(ls /data/core-psy-archive/data/PRONIA/test_vbt_pipe/vbt_derivatives/freesurfer | grep sub-); do
  sbatch --export=SUB_ID=$s 1_adjust_connectome_and_weights.slurm
done
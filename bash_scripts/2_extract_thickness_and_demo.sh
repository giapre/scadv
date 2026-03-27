for s in $(ls /data/core-psy-archive/data/PRONIA/test_vbt_pipe/vbt_derivatives/freesurfer | grep sub-); do
  sbatch --export=SUB_ID=$s 2_extract_thickness_and_demo.slurm
done
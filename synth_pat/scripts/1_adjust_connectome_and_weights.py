import os
import sys
import numpy as np
from utils import adjust_dopamine_connectome, adjust_serotonine_connectome, adjust_serotonin_lengths

subj = sys.argv[1]

base_dir = '/data/core-psy-archive/data/PRONIA/test_vbt_pipe/vbt_derivatives/freesurfer'
atlas = 'dk'

if not subj.startswith("sub-"):
    print("Invalid subject format")
    sys.exit(1)

if subj.endswith(".html"):
    print("Invalid subject format")
    sys.exit(1)

subj_path = os.path.join(base_dir, f'{subj}')

if not os.path.isdir(subj_path):
    print(f"Skipping {subj}: no folder at {subj_path}")
    sys.exit(0)

lengths_file = os.path.join(subj_path, "dwi/dk_lengths.txt")
weights_file = os.path.join(subj_path, "dwi/dk_weights.txt")

if not (os.path.exists(lengths_file) and os.path.exists(weights_file)):
    print(f"Skipping {subj}: missing files")
    sys.exit(0)

print(f"Processing {subj}")

save_dir = os.path.join(base_dir, subj, 'pipe')
os.makedirs(save_dir, exist_ok=True)
# Expected output files
dopa_file = os.path.join(save_dir, f'{atlas}_weights_with_dopa.csv')
sero_file = os.path.join(save_dir, f'{atlas}_weights_with_sero_and_dopa.csv')
lengths_out_file = os.path.join(save_dir, f'{atlas}_lengths_with_sero_and_dopa.csv')

# Skip if everything already exists
if all(os.path.exists(f) for f in [dopa_file, sero_file, lengths_out_file]):
    print(f"Skipping {subj}: outputs already exist")
    sys.exit(0)
#os.makedirs(save_dir, exist_ok=True)

dopa_weights_df = adjust_dopamine_connectome(subj, weights_file, atlas)
dopa_weights_df.to_csv(dopa_file)

sero_weights_df = adjust_serotonine_connectome(subj, dopa_file, atlas)
sero_weights_df.to_csv(sero_file)

lengths_df = adjust_serotonin_lengths(subj, lengths_file, atlas)
lengths_df.to_csv(lengths_out_file)

assert lengths_df.shape == sero_weights_df.shape

print(f'{subj} done!')
import os
import glob
import sys
import numpy as np
import pandas as pd

from paths import Paths

freesurfer_dir = '/data/core-psy-archive/data/PRONIA/test_vbt_pipe/vbt_derivatives/freesurfer/'
subject_id   = sys.argv[1] 
ses = sys.argv[2]

DERIV_DIR = os.path.join(freesurfer_dir, f'{subject_id}_{ses}', 'pipe')
if not os.path.exists(DERIV_DIR):
    ses = 'ses-t0'
    DERIV_DIR = os.path.join(freesurfer_dir, f'{subject_id}_{ses}', 'pipe')

ppc_dir = f'{DERIV_DIR}/post_pred_check'
type_of_confounds = Paths.TYPE_OF_CONFOUNDS
type_of_sweep = Paths.TYPE_OF_SWEEP

save_file_name = f'{ppc_dir}/{subject_id}_{ses}_{type_of_confounds}_{type_of_sweep}_PPC.npz'

if os.path.exists(save_file_name):
    print(f"Skipping {save_file_name}: file already exists")
    sys.exit()

print(f'Running {subject_id}, {ses}, {type_of_confounds}, {type_of_sweep}')

bolds = []
params = []
files_to_delete = []

param_names = ['ws', 'njdopa_ctx', 'njdopa_str']

for match in glob.glob(f'{ppc_dir}/{subject_id}_{ses}_{type_of_confounds}_*.npz'):
    if match.endswith('PPC.npz'):
        continue

    data = np.load(match)
    
    bold = data['bold']
    
    # fix singleton dimension 
    if bold.ndim == 3 and bold.shape[2] == 1:
        bold = np.squeeze(bold, axis=2)

    bolds.append(bold)
    params.append(data['params'])
    files_to_delete.append(match)

# Convert to arrays
bold_array = np.stack(bolds, axis=-1)   # (time, regions, simulations)
params_array = np.stack(params, axis=0)

# Save stacked file
np.savez(
    save_file_name,
    bold=bold_array,
    params=params_array,
    param_names=param_names
)

print(f"Saved stacked simulations to {save_file_name}")

# Verify file before deleting
try:
    test = np.load(save_file_name)
    assert 'bold' in test and 'params' in test
    print("Verification successful, deleting individual files...")

    for f in files_to_delete:
        os.remove(f)

    print(f"Deleted {len(files_to_delete)} individual simulation files.")

except Exception as e:
    print("Verification failed, NOT deleting files.")
    print(e)
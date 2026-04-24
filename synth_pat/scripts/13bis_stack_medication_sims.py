import os
import sys
import numpy as np
import pandas as pd
import jax.numpy as jp
import scipy.sparse
import gast_model as gm

from paths import Paths
from simulation_utils import run_bold_sweep

subject_id = sys.argv[1]
ses = sys.argv[2]
type_of_sweep = Paths.TYPE_OF_SWEEP
type_of_confounds = Paths.TYPE_OF_CONFOUNDS
PID_DERIV_DIR = os.path.join(Paths.DERIVATIVES, "freesurfer", f'{subject_id}_{ses}/pipe')
if not os.path.exists(PID_DERIV_DIR):
    PID_DERIV_DIR = os.path.join(Paths.DERIVATIVES, "freesurfer", f'{subject_id}_ses-t0/pipe')

MED_DIR = f'{PID_DERIV_DIR}/medication'
SWEEP_RES_DIR = f'{MED_DIR}/simulations'
out_file = f'{MED_DIR}/{subject_id}_{ses}_{type_of_confounds}_{type_of_sweep}_medication_effect.npz'
if os.path.exists(out_file):
    print('Stacked simulation already exists, exiting')
    sys.exit()

med_zi = []
bolds = []
med_params = []
medication_name = []
est_params = []
files_to_delete = []
for sim_name in os.listdir(SWEEP_RES_DIR):
    if not sim_name.endswith('.npz'):
            continue
    sim_file = f'{SWEEP_RES_DIR}/{sim_name}'
    sim_array = np.load(sim_file)
    med_zi.append(int(sim_name.split('_')[2])) # to be modified and loaded from the sim file like all other things, med_zi.append(sim_file['med_zi'])
    bolds.append(sim_array['bold'])
    med_params.append(sim_array['med_params'])
    medication_name.append(sim_array['medication_name']) 
    est_params.append(sim_array['est_params'])
    files_to_delete.append(sim_file)

    
est_param_names = sim_array['est_param_names']
med_param_names = sim_array['med_param_names']


# Convert to arrays
bold_array = np.stack(bolds, axis=-1)   
bold_array = np.squeeze(bold_array, axis=2) # (time, regions, simulations)
med_params = np.stack(med_params, axis=0)
est_params = np.stack(est_params, axis=0)
medication_name = np.array(medication_name)
med_zi =np.array(med_zi)

# Save stacked file
np.savez(
    out_file,
    bold=bold_array,
    medication=medication_name,
    med_zi=med_zi,
    med_params=med_params,
    med_param_names=med_param_names,
    est_params=est_params,
    est_param_names=est_param_names
)

print(f"Saved stacked simulations to {out_file}")

# Verify file before deleting
try:
    test = np.load(out_file)
    assert 'bold' in test and 'med_params' in test
    print("Verification successful, deleting individual files...")

    for f in files_to_delete:
        os.remove(f)

    print(f"Deleted {len(files_to_delete)} individual simulation files.")

except Exception as e:
    print("Verification failed, NOT deleting files.")
    print(e)
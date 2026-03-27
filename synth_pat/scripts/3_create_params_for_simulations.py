import os
import sys
import numpy as np
import pandas as pd
import jax.numpy as jp

from paths import Paths
from simulation_utils import (
stack_connectomes,
setup_delays,
setup_ja,
setup_receptors,
adjust_ja_for_midbrain
)

RESULTS_DIR = Paths.RESULTS
DATA_DIR = Paths.DATA
DERIVATIVES_DIR = Paths.DERIVATIVES

subj = sys.argv[1]
print(f'Going with {subj}')
freesurfer_dir = '/data/core-psy-archive/data/PRONIA/test_vbt_pipe/vbt_derivatives/freesurfer/'
DER_PID_DIR =os.path.join(freesurfer_dir, subj, 'pipe')

output_csv = os.path.join(DER_PID_DIR, "Receptors.npy")
if os.path.exists(output_csv):
    print(f"Skipping {subj}: file already exists")
    sys.exit(0)

# ------------------------
# Load data
# ------------------------

W = pd.read_csv(os.path.join(DER_PID_DIR, "dk_weights_with_sero_and_dopa.csv"), index_col=0)
L = pd.read_csv(os.path.join(DER_PID_DIR, "dk_lengths_with_sero_and_dopa.csv"), index_col=0)
zscores = pd.read_csv(os.path.join(DER_PID_DIR, "cortical_thick_zscores.csv"), index_col='SubjectID')

regions_names = L.columns.to_list()

# ------------------------
# Model setup
# ------------------------

setup = {
    "v_c": 3.9,
    "dt": 0.1,
}

Ceids = stack_connectomes(W)
np.save(f'{DER_PID_DIR}/Ceids.npy', Ceids)

idelays = setup_delays(L, Ceids, setup["v_c"], setup["dt"])
np.save(f'{DER_PID_DIR}/idelays.npy', idelays)

mean_Ja = 12
std_Ja = 1.2
Ja = setup_ja(zscores, W, subj, mean_Ja, std_Ja)
Ja = adjust_ja_for_midbrain(Ja, regions_names)
Ja = np.save(f'{DER_PID_DIR}/Ja.npy', Ja)

Rd1, Rd2, Rsero = setup_receptors()
np.save(f'{DER_PID_DIR}/Receptors.npy', [Rd1, Rd2, Rsero])

print(f'{subj} done!')
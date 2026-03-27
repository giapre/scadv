import os
import sys
import numpy as np
import pandas as pd
import jax.numpy as jp
import scipy.sparse
import gast_model as gm
import time

from paths import Paths
from simulation_utils import run_bold_sweep

# ------------------------
# Parse args
# ------------------------

freesurfer_dir = '/data/core-psy-archive/data/PRONIA/test_vbt_pipe/vbt_derivatives/freesurfer/'
subject_id   = sys.argv[1]          # sub-XXXX
ws_values = jp.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
njdopa_ctx_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
njdopa_str_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

DERIV_DIR = os.path.join(freesurfer_dir, subject_id, 'pipe')
output_dir = f'{DERIV_DIR}/simulations'
os.makedirs(output_dir, exist_ok=True)
# ------------------------
# Load subject-specific data
# ------------------------

L = pd.read_csv(os.path.join(DERIV_DIR, "dk_lengths_with_sero_and_dopa.csv"), index_col=0)
regions_names = L.columns.to_list()

Ceids = np.load(os.path.join(DERIV_DIR, "Ceids.npy"))
idelays = np.load(os.path.join(DERIV_DIR, "idelays.npy"))
Ja = np.load(os.path.join(DERIV_DIR, "Ja.npy"))
Rd1, Rd2, Rsero = np.load(os.path.join(DERIV_DIR, "Receptors.npy"))

# ------------------------
# Model setup
# ------------------------

setup = {
    "Seids": scipy.sparse.csr_matrix(Ceids),
    "idelays": idelays,
    "params": gm.sigm_d1d2sero_default_theta,
    "v_c": 3.9,
    "horizon": 650,
    "num_item": ws_values.shape[0],
    "dt": 0.1,
    "num_skip": 10,
    "num_time": 300000,
    "init_state": jp.array([.01, -55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(10,1),
    "noise": 0.15,
}

print(f'starting patient {subject_id}')

for njdopa_ctx in njdopa_ctx_values:
    for njdopa_str in njdopa_str_values:
        output_file = f'{output_dir}/ctx={njdopa_ctx}_str={njdopa_str}'

        if os.path.exists(f'{output_file}.npz'):
            print(f"Skipping {output_file}: file already exists")
            continue

        # ------------------------
        # Dopamine scaling
        # ------------------------

        JJdopa = np.ones(len(regions_names)) * njdopa_ctx
        for region in ['L.PU', 'R.PU', 'L.CA', 'R.CA', 'L.PA', 'R.PA', 'L.AC', 'R.AC']:
            idx = regions_names.index(region)
            JJdopa[idx] = njdopa_str
        JJdopa = JJdopa[:, None]

        # ------------------------
        # Parameter update
        # ------------------------

        theta = gm.sigm_d1d2sero_default_theta._replace(
            I=46.5,
            Ja=Ja,
            Jsa=Ja,
            Jsg=13,
            Jg=0,
            Jdopa=100000 * JJdopa,
            Rd1=Rd1,
            Rd2=Rd2,
            Rs=Rsero,
            Sd1=-10.0,
            Sd2=-10.0,
            Ss=-40.0,
            Zd1=0.5,
            Zd2=1.0,
            Zs=0.25,
            we=0.3,
            wi=1.,
            wd=1,
            ws=ws_values,
            sigma_V=setup["noise"],
            sigma_u=0.1 * setup["noise"],
        )

        setup["params"] = theta

        # ------------------------
        # Run simulation
        # ------------------------

        toc = time.time()
        bold = run_bold_sweep((theta, setup))
        tic = time.time()
        bold = np.asarray(bold)

        np.savez(output_file, bold=bold)
        print(f'it took {tic - toc}')
        print(f'simulation saved at {output_file}')

print(f'{subject_id} done')
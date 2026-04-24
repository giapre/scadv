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

med_dic = pd.read_csv(f'{Paths.RESOURCES}/mediction_sweep_params.csv')
best_params_df = pd.read_csv(f'{PID_DERIV_DIR}/post_pred_check/{subject_id}_{ses}_{type_of_confounds}_{type_of_sweep}_best_100pca_PPC.csv')

SWEEP_RES_DIR = f'{PID_DERIV_DIR}/medication/simulations'
os.makedirs(SWEEP_RES_DIR, exist_ok=True)
# ------------------------
# Load subject-specific data
# ------------------------

L = pd.read_csv(os.path.join(PID_DERIV_DIR, "dk_lengths_with_sero_and_dopa.csv"), index_col=0)
regions_names = L.columns.to_list()

Ceids = np.load(os.path.join(PID_DERIV_DIR, "Ceids.npy"))
idelays = np.load(os.path.join(PID_DERIV_DIR, "idelays.npy"))
Ja = np.load(os.path.join(PID_DERIV_DIR, "Ja.npy"))
Rd1, Rd2, Rsero = np.load(os.path.join(PID_DERIV_DIR, "Receptors.npy"))

# ------------------------
# Model setup
# ------------------------

setup = {
    "Seids": scipy.sparse.csr_matrix(Ceids),
    "idelays": idelays,
    "params": gm.sigm_d1d2sero_default_theta,
    "v_c": 3.9,
    "horizon": 650,
    "num_item": 1,
    "dt": 0.1,
    "num_skip": 10,
    "num_time": 300000,
    "init_state": jp.array([.01, -55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(10,1),
    "noise": 0.15,
}

# Sweep over medication and estimated parameters 

for med_idx in med_dic.index:

    Z_D1 = float(med_dic.loc[med_idx, 'Z_D1'])
    Z_D2 = float(med_dic.loc[med_idx, 'Z_D2'])
    Z_S  = float(med_dic.loc[med_idx, 'Z_S'])
    medication_name = med_dic.loc[med_idx, 'medication']
    med_zi = med_dic.loc[med_idx, 'z_i']

    med_params = [Z_D1, Z_D2, Z_S]

    print(f"Running subject: {subject_id}, {medication_name}, {med_idx}")

    for params_idx in best_params_df.index[:50]:

        output_file = f'{SWEEP_RES_DIR}/{medication_name}_zi_{med_zi}_params_{params_idx}.npz'
        if os.path.exists(output_file):
            print('Simulation already exists, going to next one')
            continue

        print(f"Running subject: {subject_id}, {medication_name}_{med_idx}, params_idx: {params_idx}")
        njdopa_ctx_est = float(best_params_df.loc[params_idx, 'njdopa_ctx'])
        njdopa_str_est = float(best_params_df.loc[params_idx, 'njdopa_str'])
        ws_est = float(best_params_df.loc[params_idx, 'ws'])
        est_params = [ws_est, njdopa_ctx_est, njdopa_str_est]

        # ------------------------
        # Dopamine scaling
        # ------------------------

        JJdopa = np.ones(len(regions_names)) * njdopa_ctx_est

        for region in ['L.PU', 'R.PU', 'L.CA', 'R.CA', 'L.PA', 'R.PA', 'L.AC', 'R.AC']:
            idx = regions_names.index(region)
            JJdopa[idx] = njdopa_str_est

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
            Zd1=Z_D1,
            Zd2=Z_D2,
            Zs=Z_S,
            we=0.3,
            wi=1.,
            wd=1,
            ws=ws_est,
            sigma_V=setup["noise"],
            sigma_u=0.1 * setup["noise"],
        )

        setup["params"] = theta

        bold = run_bold_sweep((theta, setup))
        bold = np.asarray(bold)

        np.savez(output_file, bold=bold, med_params=med_params, med_param_names=['Z_D1', 'Z_D2', 'Z_S'], medication_name = medication_name, med_zi=med_zi, est_params=est_params, est_param_names=['ws', 'njdopa_ctx', 'njdopa_str'])

print('Done!')
import os
import sys
import numpy as np
import pandas as pd
import jax.numpy as jp
import scipy.sparse
import synth_pat.scripts.gast_model as gm

from synth_pat.paths import Paths
from synth_pat.scripts.simulation_utils import run_bold_sweep

subject_id = sys.argv[1]
scaler = int(sys.argv[2])
medication = sys.argv[3]

med_dic = pd.read_csv(f'{Paths.RESOURCES}/mediction_sweep_params.csv', index_col='k_i')
med_dic = med_dic[med_dic['medication']==medication]
Z_D1 = 0.5 - med_dic.loc[scaler, 'Z_D1']
Z_D2 = 1.0 - med_dic.loc[scaler, 'Z_D2']
Z_S = 0.5 - med_dic.loc[scaler, 'Z_S']
med_params = [Z_D1, Z_D2, Z_S]

type_of_sim = Paths.TYPE_OF_SWEEP

print(f"Running subject: {subject_id}")

DERIV_DIR = os.path.join(Paths.DATA, "derivatives", subject_id)
PID_RESULTS = os.path.join(Paths.RESULTS, subject_id)

os.makedirs(f'{PID_RESULTS}/{type_of_sim}', exist_ok=True)
output_file = f'{PID_RESULTS}/{type_of_sim}/medication={medication}_ki={scaler}'

try:
    est_params = np.load(f'{PID_RESULTS}/bold_sweep_traditional_estimated_mode_params.npz')
except:
    est_params = np.load(f'{PID_RESULTS}/bold_sweep_daniela_estimated_mode_params.npz')

njdopa_ctx_est = 10**est_params['njdopa_ctx']
njdopa_str_est = 10**est_params['njdopa_str']
ws_est = 10**est_params['ws']

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
    "num_item": 1,
    "dt": 0.1,
    "num_skip": 10,
    "num_time": 300000,
    "init_state": jp.array([.01, -55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(10,1),
    "noise": 0.16,
}

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

os.makedirs(PID_RESULTS, exist_ok=True)

np.savez(output_file, bold=bold, params=med_params, est_params=est_params)

print(f"{subject_id} Done")
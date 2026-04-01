import os
import sys
import numpy as np
import pandas as pd
import jax.numpy as jp
import scipy.sparse
import gast_model as gm

from paths import Paths
from simulation_utils import run_bold_sweep

# ------------------------
# Parse args
# ------------------------

freesurfer_dir = '/data/core-psy-archive/data/PRONIA/test_vbt_pipe/vbt_derivatives/freesurfer/'
subject_id   = sys.argv[1] 
ses = sys.argv[2]
DERIV_DIR = os.path.join(freesurfer_dir, f'{subject_id}_{ses}', 'pipe')
output_dir = f'{DERIV_DIR}/post_pred_check'
os.makedirs(output_dir, exist_ok=True)
type_of_confounds = Paths.TYPE_OF_CONFOUNDS

post_distr_arr = np.load(f'{DERIV_DIR}/sbi/{ses}_sweep_simulations_{type_of_confounds}_params_post_distr.npz')

for n_iter in range(500):
    output_file = f'{output_dir}/{subject_id}_{ses}_{type_of_confounds}_{n_iter}'
    if os.path.exists(f'{output_file}.npz'):
            print(f"Skipping {output_file}: file already exists")
            continue

    print(f'Running {subject_id}, {ses},{n_iter}, {type_of_confounds}')

    ws_est = 10**np.random.choice(post_distr_arr['ws']) 
    njdopa_ctx_est = 10**np.random.choice(post_distr_arr['njdopa_ctx'])
    njdopa_str_est = 10**np.random.choice(post_distr_arr['njdopa_str'])      # sub-XXXX
    params = [ws_est, njdopa_ctx_est, njdopa_str_est]
    param_names = ['ws', 'njdopa_ctx', 'njdopa_str']

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
        "noise": 0.15,
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
        Zd1=0.5,
        Zd2=1.0,
        Zs=0.25,
        we=0.3,
        wi=1.,
        wd=1,
        ws=ws_est,
        sigma_V=setup["noise"],
        sigma_u=0.1 * setup["noise"],
    )

    setup["params"] = theta

    # ------------------------
    # Run simulation
    # ------------------------

    bold = run_bold_sweep((theta, setup))

    bold = np.asarray(bold)

    np.savez(output_file, bold=bold, params=params, param_names=param_names)
    print(f'simulation saved at {output_file}')

print(f'{subject_id} done')
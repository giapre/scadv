import os
import sys
import numpy as np
import pandas as pd
import jax.numpy as jp
import scipy.sparse
import synth_pat.scripts.gast_model as gm

from synth_pat.paths import Paths
from synth_pat.scripts.simulation_utils import (
    stack_connectomes,
    setup_delays,
    setup_ja,
    setup_receptors,
    adjust_ja_for_midbrain
)

RESULTS_DIR = Paths.RESULTS
DATA_DIR = Paths.DATA
DERIVATIVES_DIR = Paths.DERIVATIVES

for PID in os.listdir(DERIVATIVES_DIR):
    print(f'doing {PID}')
    pid = PID.split('-')[1]
    DER_PID_DIR = f'{DERIVATIVES_DIR}/sub-{pid}'
    # ------------------------
    # Load data
    # ------------------------

    W = pd.read_csv(os.path.join(DER_PID_DIR, "dk_weights_with_sero_and_dopa.csv"), index_col=0)
    L = pd.read_csv(os.path.join(DER_PID_DIR, "dk_lengths_with_sero_and_dopa.csv"), index_col=0)
    zscores = pd.read_csv(os.path.join(DATA_DIR, "zscore_full_chinese_all_with_fu_cortical_thick.csv"), index_col='SubjectID')

    regions_names = L.columns.to_list()

    # ------------------------
    # Model setup
    # ------------------------

    setup = {
        "Seids": [],
        "idelays": [],
        "params": gm.sigm_d1d2sero_default_theta,
        "v_c": 3.9,
        "horizon": 650,
        "num_item": 1,
        "dt": 0.1,
        "num_skip": 10,
        "num_time": 300000,
        "init_state": jp.array([.01, -55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(10,1),
        "noise": 0.0631,
    }

    Ceids = stack_connectomes(W)
    np.save(f'{DER_PID_DIR}/Ceids.npy', Ceids)

    idelays = setup_delays(L, Ceids, setup["v_c"], setup["dt"])
    np.save(f'{DER_PID_DIR}/idelays.npy', idelays)

    mean_Ja = 12
    std_Ja = 1.2
    Ja = setup_ja(zscores, W, pid, mean_Ja, std_Ja)
    Ja = adjust_ja_for_midbrain(Ja, regions_names)
    Ja = np.save(f'{DER_PID_DIR}/Ja.npy', Ja)

    Rd1, Rd2, Rsero = setup_receptors()
    np.save(f'{DER_PID_DIR}/Receptors.npy', [Rd1, Rd2, Rsero])

    print(f'{PID} done!')
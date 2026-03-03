import os
import sys
import numpy as np
import pandas as pd
import jax.numpy as jp
import scipy.sparse
import synth_pat.scripts.gast_model as gm

from synth_pat.paths import Paths
from synth_pat.scripts.simulation_utils import run_bold_sweep

# ------------------------
# Parse args
# ------------------------

subject_id   = sys.argv[1]          # sub-XXXX
ws           = float(sys.argv[2])
njdopa_ctx   = float(sys.argv[3])
njdopa_str   = float(sys.argv[4])
output_file  = sys.argv[5]

DERIV_DIR = os.path.join(Paths.DATA, "derivatives", subject_id)

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
    we=0.4,
    wi=1.,
    wd=1,
    ws=ws,
    sigma_V=setup["noise"],
    sigma_u=0.1 * setup["noise"],
)

setup["params"] = theta

# ------------------------
# Run simulation
# ------------------------

bold = run_bold_sweep((theta, setup))

bold = np.asarray(bold)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
np.savez(output_file, bold=bold)
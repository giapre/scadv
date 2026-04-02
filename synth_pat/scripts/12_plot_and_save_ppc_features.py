from paths import Paths
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
import os
import sys
from plot_utils import save_feat_and_color_by_param_for_ppc, basic_3d_sweep_plot_with_planes_for_ppc

subject_id = sys.argv[1]
ses = sys.argv[2]
RESULTS_DIR = f"{Paths.DERIVATIVES}/freesurfer/{subject_id}_{ses}/pipe"
if not os.path.exists(RESULTS_DIR):
    RESULTS_DIR = f"{Paths.DERIVATIVES}/freesurfer/{subject_id}_ses-t0/pipe"
PPC_DIR = f"{RESULTS_DIR}/post_pred_check"
type_of_sweep =  Paths.TYPE_OF_SWEEP
type_of_confunds = Paths.TYPE_OF_CONFOUNDS
print(f'doing {subject_id}, {type_of_sweep}, {type_of_confunds}')

sweep_file = f"{RESULTS_DIR}/simulations/{subject_id}_{ses}_{type_of_sweep}_extracted_features.csv"
ppc_file = f"{PPC_DIR}/{subject_id}_{ses}_{type_of_confunds}_{type_of_sweep}_best_100_PPC.csv"
emp_file = f"{RESULTS_DIR}/{subject_id}_{ses}_{type_of_confunds}_extracted_emp_features.csv"
outpath = f'{PPC_DIR}/figures/{ses}_{type_of_confunds}_{type_of_sweep}/'
os.makedirs(outpath, exist_ok=True)

sweep_df = pd.read_csv(sweep_file, index_col=0)
ppc_df = pd.read_csv(ppc_file, index_col=0)
emp_df = pd.read_csv(emp_file, index_col=0)

params =['ws', 'njdopa_ctx', 'njdopa_str']
p1_name, p2_name, p3_name = params
vars_x_dic = {'ws': 'serotonin', 'njdopa_ctx': 'cortical dopamine', 'njdopa_str': 'striatal dopamine'}
sweep_df[params] = np.log10(sweep_df[params])
ppc_df[params] = np.log10(ppc_df[params])

metrics = ['VAR_FCD',
'GBC',
'L.CA_FC',
'L.CA-L.CER',
'L.CA_ALFF',
'R.CA_ALFF',
'L.PTR_ALFF',
'L.HI_ALFF',
'L.CACG_ALFF']
cmap='viridis'
for var_to_plot in metrics:
    save_name = f'{outpath}/3d_{var_to_plot}_best_100_ppc'
    basic_3d_sweep_plot_with_planes_for_ppc(sweep_df, ppc_df, p1_name, p2_name, p3_name, var_to_plot, vars_x_dic, cmap, save_name)

# === PCA ===

from analysis_utils import do_pca
cols = [col for col in sweep_df.columns if col not in params]
sweep_r, ppc_r, emp_r = do_pca(sweep_df[cols], ppc_df, emp_df)
out_name = f'{outpath}/{ses}_{type_of_confunds}_{type_of_sweep}_best_100_ppc_pca'
save_feat_and_color_by_param_for_ppc(params, sweep_r, ppc_r, emp_r, sweep_df, out_name)

print('Done!')


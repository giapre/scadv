# To do: finish plot_pca_with_base_med_and_emp 
import pandas as pd
import numpy as np
import os
import sys
from paths import Paths
from plot_utils import plot_med_results #plot_pca_with_base_med_and_emp

subject_id = sys.argv[1]
ses = sys.argv[2]
RESULTS_DIR = f"{Paths.DERIVATIVES}/freesurfer/{subject_id}_{ses}/pipe"
if not os.path.exists(RESULTS_DIR):
    RESULTS_DIR = f"{Paths.DERIVATIVES}/freesurfer/{subject_id}_ses-t0/pipe"
MED_DIR = f"{RESULTS_DIR}/medication"
PPC_DIR = f"{RESULTS_DIR}/post_pred_check"
type_of_sweep =  Paths.TYPE_OF_SWEEP
type_of_confunds = Paths.TYPE_OF_CONFOUNDS
demo = pd.read_csv(Paths.DEMO, index_col='PSN')
subject_id_idx = int(subject_id.split('-')[1])
remission = demo.loc[subject_id_idx, 'Remission']
print(f'doing {subject_id}, {type_of_sweep}, {type_of_confunds}')

#med_file = f"{MED_DIR}/{subject_id}_{ses}_{type_of_confunds}_{type_of_sweep}_medication_effect_extracted_features.csv"
sweep_file = f"{RESULTS_DIR}/simulations/{subject_id}_{ses}_{type_of_sweep}_extracted_features.csv" # original df with general parameters sweep
med_file = f"{MED_DIR}/{subject_id}_{ses}_{type_of_confunds}_{type_of_sweep}_medication_effect_extracted_features.csv" # df with the simulated effects of medications
ppc_file = f"{PPC_DIR}/{subject_id}_{ses}_{type_of_confunds}_{type_of_sweep}_best_100pca_PPC.csv" # posterior predictive check
base_emp_file = f"{RESULTS_DIR}/{subject_id}_ses-t0_{type_of_confunds}_extracted_emp_features.csv" # before drug was administered
fup_emp_file = f"{RESULTS_DIR}/{subject_id}_ses-t1_{type_of_confunds}_extracted_emp_features.csv" # follow-up after drug was administered
fig_outpath = f'{MED_DIR}/figures/{type_of_confunds}_{type_of_sweep}/'
os.makedirs(fig_outpath, exist_ok=True)

sweep_df = pd.read_csv(sweep_file, index_col=0)
med_df = pd.read_csv(med_file, index_col=0)
ppc_df = pd.read_csv(ppc_file, index_col=0)
base_emp_df = pd.read_csv(base_emp_file, index_col=0)
fup_emp_df = pd.read_csv(fup_emp_file, index_col=0)
base_emp_pid_data = base_emp_df.iloc[0]
fup_emp_pid_data = fup_emp_df.iloc[0]

print(f'{subject_id} going for general PCA')

from analysis_utils import drop_high_corr_features
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

feat = ['L.PU-L.CACG',
'L.PU-L.RACG',
'R.PU-R.CACG',
'R.PU-R.RACG',
'L.PU-L.IN',
'R.PU-R.IN',
'L.CA-L.HI',
'R.CA-R.HI',
'VAR_FCD']

params = ['ws', 'njdopa_ctx', 'njdopa_str']
sweep_df[params] = np.log10(sweep_df[params])
sweep_reduced = drop_high_corr_features(sweep_df)
sweep_reduced.drop(columns=params, inplace=True)
scaler = StandardScaler()
sweep_scaled = scaler.fit_transform(sweep_reduced.values)
ppc_reduced = ppc_df[sweep_reduced.columns]
base_emp_reduced = base_emp_pid_data[sweep_reduced.columns]
fup_emp_reduced = fup_emp_pid_data[sweep_reduced.columns]
ppc_scaled = scaler.transform(ppc_reduced.values)
base_emp_scaled = scaler.transform(base_emp_reduced.values.reshape(1,-1))
fup_emp_scaled = scaler.transform(fup_emp_reduced.values.reshape(1,-1))

# PCA projection
pca = PCA(n_components=2)
sweep_r = pca.fit_transform(sweep_scaled)
ppc_r = pca.transform(ppc_scaled)
base_emp_r = pca.transform(base_emp_scaled)
fup_emp_r = pca.transform(fup_emp_scaled)

for medication in ['amisulpride', 'aripiprazole', 'clozapine', 'olanzapine']:#np.unique(med_df['medication']):
    med_df_specific_med = med_df[med_df['medication']==medication]
    med_reduced = med_df_specific_med[sweep_reduced.columns]
    med_scaled = scaler.transform(med_reduced.values)
    med_r = pca.transform(med_scaled)

    outpath = f'{fig_outpath}/{subject_id}_{ses}_{type_of_sweep}_{type_of_confunds}_{medication}_PCA.png'
    #plot_pca_with_base_med_and_emp(params, medication, sweep_r, med_r, ppc_r, base_emp_r, fup_emp_r, sweep_df, remission, outpath)

    errors = np.linalg.norm(med_r - fup_emp_r, axis=1)
    med_df.loc[med_df_specific_med.index, 'error'] = errors
med_df['error'] = med_df['error']/med_df['error'].max()


print('Doing the error')

thing_to_plot = 'error'
save_path = f'{fig_outpath}/{subject_id}_{ses}_{type_of_sweep}_{type_of_confunds}_{thing_to_plot}.png'
plot_med_results(thing_to_plot, med_df, subject_id, remission, save_path)

print('Done')

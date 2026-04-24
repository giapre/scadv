import pandas as pd
import numpy as np
import os
import sys
from paths import Paths
from plot_utils import plot_med_results, save_feat_and_color_by_param_for_med

subject_id = sys.argv[1]
ses = sys.argv[2]
RESULTS_DIR = f"{Paths.DERIVATIVES}/freesurfer/{subject_id}_{ses}/pipe"
if not os.path.exists(RESULTS_DIR):
    RESULTS_DIR = f"{Paths.DERIVATIVES}/freesurfer/{subject_id}_ses-t0/pipe"
MED_DIR = f"{RESULTS_DIR}/medication"
type_of_sweep =  Paths.TYPE_OF_SWEEP
type_of_confunds = Paths.TYPE_OF_CONFOUNDS
demo = pd.read_csv(Paths.DEMO, index_col='PSN')
subject_id_idx = int(subject_id.split('-')[1])
remission = demo.loc[subject_id_idx, 'Remission']
print(f'doing {subject_id}, {type_of_sweep}, {type_of_confunds}')

#med_file = f"{MED_DIR}/{subject_id}_{ses}_{type_of_confunds}_{type_of_sweep}_medication_effect_extracted_features.csv"
sweep_file = f"{RESULTS_DIR}/simulations/{subject_id}_{ses}_{type_of_sweep}_extracted_features.csv"
med_file = f"{MED_DIR}/{subject_id}_{ses}_{type_of_confunds}_{type_of_sweep}_medication_effect_extracted_features.csv" # 
base_emp_file = f"{RESULTS_DIR}/{subject_id}_ses-t0_{type_of_confunds}_extracted_emp_features.csv" # before drug was administered
fup_emp_file = f"{RESULTS_DIR}/{subject_id}_ses-t1_{type_of_confunds}_extracted_emp_features.csv" # follow-up after drug was administered
fig_outpath = f'{MED_DIR}/figures/{type_of_confunds}_{type_of_sweep}/'
os.makedirs(fig_outpath, exist_ok=True)

sweep_df = pd.read_csv(sweep_file, index_col=0)
med_df = pd.read_csv(med_file, index_col=0)
base_emp_df = pd.read_csv(base_emp_file, index_col=0)
fup_emp_df = pd.read_csv(fup_emp_file, index_col=0)
base_emp_pid_data = base_emp_df.iloc[0]
fup_emp_pid_data = fup_emp_df.iloc[0]

score_results_name = f'{MED_DIR}/{subject_id}_{ses}_{type_of_confunds}_{type_of_sweep}_medication_score.csv'

if not os.path.isfile(score_results_name):
    print(f'{subject_id} does not have the scores yet!')
    sys.exit()

score_results_df = pd.read_csv(score_results_name)
print(f'processing {subject_id}')

for thing_to_plot in ['sim_med-sim_base_score', 'sim_med-sim_base_diff', 'sim_med-emp_base_score', 'sim_med-emp_base_diff', 'sim_med-emp_med_diff']:
    save_path = f'{fig_outpath}/{subject_id}_{ses}_{type_of_sweep}_{type_of_confunds}_{thing_to_plot}.png'
    plot_med_results(thing_to_plot, score_results_df, subject_id, remission, save_path)

print(f'{subject_id} going for PCA')

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
#sweep_reduced = drop_high_corr_features(sweep_df)
#sweep_reduced.drop(columns=params, inplace=True)
sweep_reduced = sweep_df[feat]
scaler = StandardScaler()
sweep_scaled = scaler.fit_transform(sweep_reduced.values)
base_emp_reduced = base_emp_pid_data[sweep_reduced.columns]
fup_emp_reduced = fup_emp_pid_data[sweep_reduced.columns]
base_emp_scaled = scaler.transform(base_emp_reduced.values.reshape(1,-1))
fup_emp_scaled = scaler.transform(fup_emp_reduced.values.reshape(1,-1))
# PCA projection
pca = PCA(n_components=5)
sweep_r = pca.fit_transform(sweep_scaled)
base_emp_r = pca.transform(base_emp_scaled)
fup_emp_r = pca.transform(fup_emp_scaled)

for medication in np.unique(med_df['medication']):
    med_df_specific_med = med_df[med_df['medication']==medication]
    med_reduced = med_df_specific_med[sweep_reduced.columns]
    med_scaled = scaler.transform(med_reduced.values)
    med_r = pca.transform(med_scaled)

    outpath = f'{fig_outpath}/{subject_id}_{ses}_{type_of_sweep}_{type_of_confunds}_{medication}_PCA.png'
    save_feat_and_color_by_param_for_med(params, medication, sweep_r, med_r, base_emp_r, fup_emp_r, sweep_df, remission, outpath)

print('Done')

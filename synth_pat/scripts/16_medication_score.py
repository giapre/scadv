import pandas as pd
import numpy as np
import os
import sys
from paths import Paths
from plot_utils import plot_med_results

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

med_file = f"{MED_DIR}/{subject_id}_{ses}_{type_of_confunds}_{type_of_sweep}_medication_effect_extracted_features.csv"
base_emp_file = f"{RESULTS_DIR}/{subject_id}_ses-t0_{type_of_confunds}_extracted_emp_features.csv" # before drug was administered
fup_emp_file = f"{RESULTS_DIR}/{subject_id}_ses-t1_{type_of_confunds}_extracted_emp_features.csv" # follow-up after drug was administered
fig_outpath = f'{MED_DIR}/figures/{type_of_confunds}_{type_of_sweep}/'
os.makedirs(fig_outpath, exist_ok=True)

med_df = pd.read_csv(med_file, index_col=0)
base_emp_df = pd.read_csv(base_emp_file, index_col=0)
fup_emp_df = pd.read_csv(fup_emp_file, index_col=0)
emp_pid_data = base_emp_df.iloc[0]
emp_pid_fup_data = fup_emp_df.iloc[0]

feat = ['L.PU-L.CACG',
'L.PU-L.RACG',
'R.PU-R.CACG',
'R.PU-R.RACG',
'L.PU-L.IN',
'R.PU-R.IN',
'L.CA-L.HI',
'R.CA-R.HI',
'VAR_FCD']

score_results_name = f'{MED_DIR}/{subject_id}_{ses}_{type_of_confunds}_{type_of_sweep}_medication_score.csv'

if os.path.isfile(score_results_name):
    print(f'{subject_id} already has {Paths.TYPE_OF_SWEEP} done')
    #sys.exit()

print(f'processing {subject_id}')

score_dfs = []

for medication in np.unique(med_df['medication']):
    print(medication)
    med_df_specific_med = med_df[med_df['medication']==medication] #1000x9
    med_df_specific_med = med_df_specific_med.sort_values(['med_zi'])
    med_df_specific_med.reset_index(inplace=True, drop=True)
    base_med_zi = med_df_specific_med['med_zi'].min() # select med_zi=0 as baseline (no medication effect) 50x9
    base_feat = med_df_specific_med[med_df_specific_med['med_zi']==base_med_zi][feat]
    base_feat_mean = base_feat.mean()
    score_i = [] # score computed using the +1 strategy comparing medication simulations vs baseline simulations
    score_i_diff = [] # difference of datafeatures subtracting medication simulations vs baseline simulations
    score_emp = []  # score computed using the +1 strategy comparing medication simulations vs baseline empirical features
    score_emp_diff = [] # difference of datafeatures subtracting medication simulations vs baseline empirical features
    score_emp_med = [] # score computed using the +1 strategy comparing emp medication vs emp baseline
    score_emp_med_diff = [] # difference computed comparing emp medication vs emp baseline
    emp_med_diff = [] # difference between medication simulations and empirical medication

    med_zi_i = []

    for i in med_df_specific_med.index:
        score_i.append(np.sum(med_df_specific_med.loc[i, feat] > base_feat_mean)) # score computed using the +1 strategy comparing medication simulations vs baseline simulations
        score_i_diff.append(np.sum(med_df_specific_med.loc[i, feat] - base_feat_mean)) # difference of datafeatures subtracting medication simulations vs baseline simulations
        score_emp.append(np.sum(med_df_specific_med.loc[i, feat] > emp_pid_data[feat])) # score computed using the +1 strategy comparing medication simulations vs baseline empirical features
        score_emp_diff.append(np.sum(med_df_specific_med.loc[i, feat] - emp_pid_data[feat])) # difference of datafeatures subtracting medication simulations vs baseline empirical features
        score_emp_med.append(np.sum(emp_pid_fup_data[feat] > emp_pid_data[feat]))  # score computed using the +1 strategy comparing emp medication vs emp baseline
        score_emp_med_diff.append(np.sum(emp_pid_fup_data[feat] - emp_pid_data[feat])) # difference computed comparing emp medication vs emp baseline
        emp_med_diff.append(np.sum(med_df_specific_med.loc[i, feat] - emp_pid_fup_data[feat])) # difference between medication simulations and empirical medication
        med_zi_i.append(med_df_specific_med.loc[i, 'med_zi'])

    score_df = pd.DataFrame({'sim_med-sim_base_score': score_i, 'sim_med-sim_base_diff': score_i_diff, 
    'sim_med-emp_base_score': score_emp, 'sim_med-emp_base_diff': score_emp_diff, 
    'emp_med-emp_base_score': score_emp_med, 'emp_med-emp_base_diff': score_emp_med_diff,
    'sim_med-emp_med_diff': emp_med_diff,
    'medication': [medication]*len(score_i), 'med_zi': med_zi_i})
    score_dfs.append(score_df)
    print(np.unique(score_df['medication']))

result_df = pd.concat(score_dfs, axis=0)
result_df['med_zi_norm'] = result_df.groupby('medication')['med_zi'].transform(lambda x: x - x.min())
result_df.to_csv(score_results_name)
print(np.unique(result_df['medication']))
print(f'{subject_id} done')
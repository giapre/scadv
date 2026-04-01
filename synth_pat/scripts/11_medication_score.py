import pandas as pd
import numpy as np
import os
from synth_pat.paths import Paths
from synth_pat.scripts.plot_utils import plot_med_results

feat = ['L.PU-L.CACG',
'L.PU-L.RACG',
'R.PU-R.CACG',
'R.PU-R.RACG',
'L.PU-L.IN',
'R.PU-R.IN',
'L.CA-L.HI',
'R.CA-R.HI',
'VAR_FCD']

for pid in os.listdir(Paths.RESULTS):

    med_results_file = f'{Paths.RESULTS}/{pid}/{Paths.TYPE_OF_SWEEP}_medication_extracted_features.csv'
    if not os.path.isfile(med_results_file):
        print(f'{pid} does not have {Paths.TYPE_OF_SWEEP}')
        continue
        
    med_score_results_dir = f'{Paths.RESULTS}/{pid}/medication'
    os.makedirs(med_score_results_dir, exist_ok=True)
    score_results_name = f'{med_score_results_dir}/{Paths.TYPE_OF_SWEEP}_medication_score.csv'

    if os.path.isfile(score_results_name):
        print(f'{pid} already has {Paths.TYPE_OF_SWEEP} done')
        continue

    print(f'processing {pid}')
    med_results = pd.read_csv(med_results_file, index_col=0)

    score_dfs = []

    for medication in np.unique(med_results['medication']):
        med_results_specific_med = med_results[med_results['medication']==medication]
        med_results_specific_med = med_results_specific_med.sort_values(['med_zi'])
        med_results_specific_med.reset_index(inplace=True)
        base_med_zi = med_results_specific_med['med_zi'].min()
        base_feat = med_results_specific_med[med_results_specific_med['med_zi']==base_med_zi][feat]
        score_i = []
        score_i_diff = []
        med_zi_i = []

        for i in med_results_specific_med.index:
            for j in base_feat.index:
                score_i.append(np.sum(med_results_specific_med.loc[i, feat] > base_feat.loc[j, feat]))
                score_i_diff.append(np.sum(med_results_specific_med.loc[i, feat] - base_feat.loc[j, feat]))
                med_zi_i.append(med_results_specific_med.loc[i, 'med_zi'])

        score_df = pd.DataFrame({'score': score_i, 'difference':score_i_diff, 'medication': [medication]*len(score_i), 'med_zi': med_zi_i})
        score_dfs.append(score_df)

    result_df = pd.concat(score_dfs, axis=0)
    result_df['med_zi_norm'] = result_df.groupby('medication')['med_zi'].transform(lambda x: x - x.min())
    result_df.to_csv(score_results_name)
    save_path = f'{med_score_results_dir}/{Paths.TYPE_OF_SWEEP}_medication_score.png'
    plot_med_results('score', result_df, pid, save_path)
    save_path = f'{med_score_results_dir}/{Paths.TYPE_OF_SWEEP}_medication_difference.png'
    plot_med_results('difference', result_df, pid, save_path)
    print(f'{pid} done')
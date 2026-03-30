import pandas as pd
import numpy as np
from synth_pat.paths import Paths

feat = ['L.PU-L.CACG',
'L.PU-L.RACG',
'R.PU-R.CACG',
'R.PU-R.RACG',
'L.PU-L.IN',
'R.PU-R.IN',
'L.CA-L.HI',
'R.CA-R.HI',
'VAR_FCD']

med_results = pd.read_csv(f'{Paths.RESULTS}/{Paths.TYPE_OF_SWEEP}.csv', index_col = 0)

base_feat = med_results.loc[0, feat]
score = []

for i in med_results.index:
    score_i = 0
    for f in feat:
        if med_results.loc[i, f] > med_results[f]:
            score_i += 1
    score.append(score_i)
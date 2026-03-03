import pandas as pd 
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt

from synth_pat.paths import Paths

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from synth_pat.scripts.analysis_utils import drop_high_corr_features

import matplotlib.pyplot as plt
import numpy as np

daniela = pd.read_csv(f'{Paths.DATA}/ALL_daniela_full_extracted_features.csv', index_col = 'pid')
demo = pd.read_csv('/home/prior/gp_home/scadv/data/full_demographics.csv', index_col='SubjectID')
daniela['diagnosis'] = demo['diagnosis']

all_synth_df = []
for pid in os.listdir(Paths.RESULTS):
    daniela_pid = daniela.loc[pid]
    pid_idx = pid.split('-')[1]
    diagnosis = daniela.loc[pid_idx, 'diagnosis']
    file_dir = f'{Paths.RESULTS}/{pid}/bold_sweep_extracted_features.csv'
    synth_res_df = pd.read_csv(file_dir, index_col=0)
    #all_synth_df.append(df)

    #synth_res_df = pd.concat(all_synth_df, axis=0)
    feat_df_reduced = drop_high_corr_features(synth_res_df)
    params = ['ws', 'njdopa_ctx', 'njdopa_str']
    X_train_df = feat_df_reduced.drop(columns=params)
    X_train = StandardScaler().fit_transform(X_train_df.values)
    pca = PCA(n_components=5)
    X_train_r = pca.fit_transform(X_train)

    X_test = StandardScaler().fit_transform(daniela_pid[X_train_df.columns].values)
    X_test_r = pca.transform(X_test)

    
    fig, axes = plt.subplots(1, 3, figsize=(7,6))
    for i, param in enumerate(params):


        # --- TRAIN (simulations) ---
        axes[i].scatter(
            X_train_r[:,0],
            X_train_r[:,1],
            c=np.log(synth_res_df[param]),
            alpha=0.6
        )

        plt.colorbar(axes[i], label=param)

        # HC subjects
        axes[i].scatter(
            X_test_r[:,0],
            X_test_r[:,1],
            c='red',
            marker='x',
            s=120,
            label='HC'
        )

        plt.title(f'Colored by {param}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.savef()
        plt.show()
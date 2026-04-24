from paths import Paths
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
MED_DIR = f"{RESULTS_DIR}/medication"
type_of_sweep =  Paths.TYPE_OF_SWEEP
type_of_confunds = Paths.TYPE_OF_CONFOUNDS
demo = pd.read_csv(Paths.DEMO, index_col='PSN')
subject_id_idx = int(subject_id.split('-')[1])
remission = demo.loc[subject_id_idx, 'Remission']
print(f'doing {subject_id}, {type_of_sweep}, {type_of_confunds}')

sweep_file = f"{RESULTS_DIR}/simulations/{subject_id}_{ses}_{type_of_sweep}_extracted_features.csv"
med_file = f"{MED_DIR}/{subject_id}_{ses}_{type_of_confunds}_{type_of_sweep}_medication_effect_extracted_features.csv"
base_emp_file = f"{RESULTS_DIR}/{subject_id}_ses-t0_{type_of_confunds}_extracted_emp_features.csv"
fup_emp_file = f"{RESULTS_DIR}/{subject_id}_ses-t1_{type_of_confunds}_extracted_emp_features.csv"
outpath = f'{MED_DIR}/figures/{type_of_confunds}_{type_of_sweep}/'
os.makedirs(outpath, exist_ok=True)

sweep_df = pd.read_csv(sweep_file, index_col=0)
med_df = pd.read_csv(med_file, index_col=0)
base_emp_df = pd.read_csv(base_emp_file, index_col=0)
fup_emp_df = pd.read_csv(fup_emp_file, index_col=0)
emp_pid_data = base_emp_df.iloc[0]
emp_pid_fup_data = fup_emp_df.iloc[0]

params =['ws', 'njdopa_ctx', 'njdopa_str']
p1_name, p2_name, p3_name = params
vars_x_dic = {'ws': 'serotonin', 'njdopa_ctx': 'cortical dopamine', 'njdopa_str': 'striatal dopamine'}
sweep_df[params] = np.log10(sweep_df[params])
med_df[params] = np.log10(med_df[params])

feat_dic = {'L.PU-L.CACG': 'Left Putamen - Caudal Anterior Cingulum',
'L.PU-L.RACG': 'Left Putamen - Rostral Anterior Cingulum',
'R.PU-R.CACG': 'Right Putamen - Caudal Anterior Cingulum',
'R.PU-R.RACG': 'Right Putamen - Rostral Anterior Cingulum',
'L.PU-L.IN': 'Left Putamen - Insula',
'R.PU-R.IN': 'Right Putamen - Insula',
'L.CA-L.HI': 'Left Caudate - Hippocampus',
'R.CA-R.HI': 'Right Caudate - Hippocampus',
'VAR_FCD': 'Fluidity',
'GBC': 'Global Brain Coupling'}

for p in feat_dic.keys():
    print(f'\n ========= {p} ========= \n')
    p2 = p#-R.HI'
    p1 = 'L.PU-L.IN'
    #p1 ='GBC'
    #p2 = 'VAR_FCD'

    viridis = cm.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7,6))

    # --- Sweep background points ---
    ax.scatter(
        sweep_df[p1],
        sweep_df[p2],
        color=viridis(0.15),   # light blue from viridis
        alpha=0.2,
        s=15,
        label="Parameter sweep"
    )

    #ax.scatter(
    #    ppc_df[p1],
    #    ppc_df[p2],
    #    color='r',   # light blue from viridis
    #    alpha=0.2,
    #    s=15,
    #    label="Parameter sweep"
    #)

    # --- Medicated subjects colored by Z_D2 ---
    # Get unique medication types
    med_types = med_df['medication'].unique()

    # Create color map
    cmap = cm.get_cmap("viridis", len(med_types))

    # Loop over medication types
    for i, med in enumerate(med_types):
        sub_df = med_df[med_df['medication'] == med]

        ax.scatter(
            sub_df[p1],
            sub_df[p2],
            color=cmap(i),
            s=20,
            alpha=0.5,
            edgecolor="white",
            label=med
        )

    # --- Healthy condition ---
    ax.scatter(
        emp_pid_fup_data[p1],
        emp_pid_fup_data[p2],
        color=viridis(0.9),   # dark purple
        s=120,
        #edgecolor="white",
        label="Medicated condition"
    )

    # --- Unmedicated condition ---
    ax.scatter(
        emp_pid_data[p1],
        emp_pid_data[p2],
        color=viridis(0.05),   # dark purple
        s=120,
        edgecolor="white",
        label="Unmedicated pathological condition"
    )

    # --- Labels ---
    ax.set_xlabel(feat_dic[p1])
    ax.set_ylabel(feat_dic[p2])

    # --- Style ---
    ax.grid(True, color="lightgray", linewidth=0.6, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("lightgray")

    ax.legend(frameon=False)

    plt.tight_layout()
    plt.title(f'{subject_id}, remission: {remission}')
    plt.savefig(f'{outpath}/{p}_med_effect.png')
    plt.close()

print('Done!')
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
fig_outpath = f"{MED_DIR}/figures"
print(f'doing {subject_id}, {type_of_sweep}, {type_of_confunds}')

score_results_name = f'{MED_DIR}/{subject_id}_{ses}_{type_of_confunds}_{type_of_sweep}_medication_score.csv'

if not os.path.isfile(score_results_name):
    print(f'{subject_id} does not have the scores yet!')
    sys.exit()

score_results_df = pd.read_csv(score_results_name)
print(f'processing {subject_id}')

#emp_med_emp_base_score = score_results_df.loc[0, 'emp_med-emp_base_score']
#emp_med_emp_base_diff = score_results_df.loc[0, 'emp_med-emp_base_diff']

# Group by medication + med_zi
summary = (
    score_results_df.groupby(["medication", "med_zi"])
      .agg(
          score_mean=("sim_med-sim_base_score", "mean"),
          score_std=("sim_med-sim_base_score", "std"),
          diff_mean=("sim_med-sim_base_diff", "mean"),
          diff_std=("sim_med-sim_base_diff", "std"),
      )
      .reset_index()
)

score_threshold = 5.5 - 4.5  # <-- adjust depending on your question
summary["passes_score_threshold"] = summary["score_mean"] > score_threshold

diff_threshold = 0.1
summary["passes_diff_threshold"] = summary["diff_mean"] > diff_threshold

score_crossing = (
    summary[summary["passes_score_threshold"]]
    .groupby("medication")["med_zi"]
    .min()
)

diff_crossing = (
    summary[summary["passes_diff_threshold"]]
    .groupby("medication")["med_zi"]
    .min()
)

print("Score crossing:\n", score_crossing)
print("Diff crossing:\n", diff_crossing)

def plot_threshold(summary, thing_to_plot, threshold, remission, save_path):
    import matplotlib.pyplot as plt

    cmap = plt.cm.viridis
    purple = cmap(0.1)   # dark purple
    green = cmap(0.6)    # green
    lightgreen = cmap(0.8)  # yellow

    medications = summary["medication"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, med in enumerate(medications):
        ax = axes[i]
        
        sub = summary[summary["medication"] == med]
        sub["med_zi"] = sub["med_zi"] + 0.5
        sub = sub.sort_values("med_zi")
        
        # Plot mean
        ax.plot(sub["med_zi"], sub[f"{thing_to_plot}_mean"], color=green)
        
        # Fill std
        ax.fill_between(
            sub["med_zi"],
            sub[f"{thing_to_plot}_mean"] - sub[f"{thing_to_plot}_std"],
            sub[f"{thing_to_plot}_mean"] + sub[f"{thing_to_plot}_std"],
            alpha=0.2, 
            color=green, 
        )
        
        # Threshold line
        ax.axhline(threshold)
        #ax.axhline(emp_thres, color='orange')
        
        # Highlight points above threshold
        #passed = sub[sub[f"passes_{thing_to_plot}_threshold"]]
        #ax.scatter(passed["med_zi"], passed[f"{thing_to_plot}_mean"], color=green)
        
        ax.set_title(f'{med}, remission: {remission}')
        ax.set_xlabel("med_zi")
        ax.set_ylabel(f"Mean {thing_to_plot}")
        ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

thing_to_plot = 'score'
summary['score_mean'] = summary['score_mean'] - 4.5
save_path = f'{fig_outpath}/{type_of_confunds}_{type_of_sweep}/{subject_id}_{ses}_{type_of_confunds}_{type_of_sweep}_{thing_to_plot}_threshold.png'
plot_threshold(summary, thing_to_plot, score_threshold, remission, save_path)

thing_to_plot = 'diff'
save_path = f'{fig_outpath}/{type_of_confunds}_{type_of_sweep}/{subject_id}_{ses}_{type_of_confunds}_{type_of_sweep}_{thing_to_plot}_threshold.png'
plot_threshold(summary, thing_to_plot, diff_threshold, remission, save_path)

print('Done')
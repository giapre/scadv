import numpy as np
import pandas as pd
import json
import os
from nilearn.signal import clean
from nilearn.input_data import NiftiLabelsMasker
import matplotlib.pyplot as plt


def compute_fcd(ts, window_length=20, overlap=19):
    n_samples, n_regions = ts.shape
    #    if n_samples < n_regions:
    #        print('ts transposed')
    #        ts=ts.T
    #        n_samples, n_regions = ts.shape

    window_steps_size = window_length - overlap
    n_windows = int(np.floor((n_samples - window_length) / window_steps_size + 1))

    # upper triangle indices
    Isupdiag = np.triu_indices(n_regions, 1)    

    #compute FC for each window
    FC_t = np.zeros((int(n_regions*(n_regions-1)/2),n_windows))
    for i in range(n_windows):
        FCtemp = np.corrcoef(ts[window_steps_size*i:window_length+window_steps_size*i,:].T)
        #FCtemp = np.nan_to_num(FCtemp, nan=0)
        FC_t[:,i] = FCtemp[Isupdiag]


    # compute FCD by correlating the FCs with each other
    FCD = np.corrcoef(FC_t.T)

    return FCD

def fcd_variance_excluding_overlap(sim_fcd, window_length, overlap):
    import numpy as np
    step_size = window_length - overlap
    W = sim_fcd.shape[0]
    min_sep = int(np.ceil(window_length / step_size))
    
    idx = np.arange(W)
    dist = np.abs(idx[:, None] - idx[None, :])
    
    mask = dist >= min_sep
    
    vals = sim_fcd[mask]
    
    return np.var(vals, axis=0)

def get_repetition_time(json_file):
    # Get the repetition time
    with open(json_file, "r") as f:
        data = json.load(f)
    tr = data.get("RepetitionTime", None)
    return tr

def plot_signal_and_matrices(pid, ses, combination, filtered_bold, fcd, var_fcd, fc, mean_fc, path):

    plt.figure(figsize=(12, 5))
    plt.subplot(131)
    plt.plot(2*filtered_bold/filtered_bold.max(axis=0) + np.arange(filtered_bold.shape[1]), linewidth=0.5)
    plt.title("Filtered BOLD signals")
    plt.subplot(132)
    plt.imshow(fcd, cmap='viridis')
    plt.title(f"FCD, VAR={var_fcd:.5f}")
    plt.subplot(133)
    plt.imshow(fc, cmap='viridis')
    plt.title(f"FC, GBC={mean_fc:.5f}")
    plt.suptitle(f"Subject {pid}, session {ses}, strategy: {combination}")
    plt.savefig(f"{path}/{pid}_{ses}_{combination}_signal_matrices.png")
    plt.close()

def compute_basic_metrics(filtered_bold, tr):
    window_length = int(20//tr)
    overlap = window_length - 1
    fcd = compute_fcd(filtered_bold, window_length=window_length, overlap=overlap)
    var_fcd = fcd_variance_excluding_overlap(fcd, window_length=window_length, overlap=overlap) #np.var(emp_fcd,axis=0)
    fc = np.corrcoef(filtered_bold.T)
    triu_idx = np.triu_indices(fc.shape[0], k=1)
    gbc = np.mean(fc[triu_idx])

    filtered_bold = (filtered_bold-np.min(filtered_bold))/(np.max(filtered_bold)-np.min(filtered_bold))

    return fc, gbc, fcd, var_fcd

def plot_basic_metrics(df, pid, ses):
    """
    Plot basic metrics from dataframe
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.bar(df['Strategy'], df['VAR_FCD'])
    plt.title('Variance of FCD')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 2)
    plt.bar(df['Strategy'], df['GBC'])
    plt.title('Global Brain Connectivity')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    plt.bar(df['Strategy'], df['MEAN_ALFF'])
    plt.title('Mean ALFF')
    plt.xticks(rotation=45)

    plt.suptitle(f'Basic Metrics for {pid} {ses}')
    plt.tight_layout()
    #os.makedirs(path_config.pipeline_postproc / "metrics", exist_ok=True)
    #plt.savefig(path_config.pipeline_postproc / "metrics" / f"{pid}_{ses}_basic_metrics_summary.png")

def get_combined_top5_compcor(confounds_json_file):
    with open(confounds_json_file) as f:
        meta = json.load(f)

    comps = []
    for name, info in meta.items():
        if name.startswith(("a_comp_cor")):
            var = info.get("VarianceExplained", 0)
            comps.append((name, var))

    # sort by variance explained globally
    comps = sorted(comps, key=lambda x: x[1], reverse=True)

    # pick top 5 total
    top5 = [name for name, _ in comps[:2]]

    return top5

def get_compcor_50pct(confounds_json_file):
    """
    Returns:
    - csf_comps: list of c_comp_cor_* explaining ≥50% variance
    - wm_comps:  list of w_comp_cor_* explaining ≥50% variance
    """

    with open(confounds_json_file, "r") as f:
        meta = json.load(f)

    csf = []
    wm = []

    # Collect components and their variance explained
    for name, info in meta.items():
        if name.startswith("c_comp_cor"):
            csf.append((name, info.get("VarianceExplained", 0)))
        elif name.startswith("w_comp_cor"):
            wm.append((name, info.get("VarianceExplained", 0)))

    def select_until_50pct(components):
        # sort by variance explained (descending)
        components = sorted(components, key=lambda x: x[1], reverse=True)
        selected = []
        cumvar = 0.0
        for name, var in components:
            selected.append(name)
            cumvar += var
            if cumvar >= 0.5:
                break
        return selected, cumvar

    csf_sel, csf_var = select_until_50pct(csf)
    wm_sel, wm_var = select_until_50pct(wm)

    return csf_sel, wm_sel

def get_top5_compcor(confounds_json_file):
    """
    Returns:
    - csf_comps: list of top 5 c_comp_cor_* components
    - wm_comps:  list of top 5 w_comp_cor_* components
    """

    with open(confounds_json_file, "r") as f:
        meta = json.load(f)

    csf = []
    wm = []

    # Collect components and their variance explained
    c_count = 0
    w_count = 0 

    for name, _ in meta.items():
        if name.startswith("c_comp_cor"):
            c_count += 1
            csf.append(name)
        if c_count >= 5:
            break

    for name, _ in meta.items():         
        if name.startswith("w_comp_cor"):
            w_count += 1
            wm.append(name)
        if w_count >= 5:
            break

    return csf, wm
import numpy as np
import pandas as pd
import json
import os
from nilearn.signal import clean
from nilearn.input_data import NiftiLabelsMasker
import matplotlib.pyplot as plt
from synth_pat.paths import Paths
from synth_pat.scripts.analysis_utils import compute_fcd

RESOURCES_DIR = Paths.RESOURCES

def read_lut_and_fs_default(atlas):
    """
    Get the FreeSurferColorLUT and the fs_default dataframe
    """
    if atlas.lower() == "aal2":
        default = pd.read_csv(RESOURCES_DIR / "aal2_default.txt",
                              sep=r"\s+",  # whitespace separator
                              comment="#",  # skip lines starting with #
                              header=None,  # no header in file
                              engine="python")  # use python engine for regex separator

        default.columns = ["Index", "FullName"]

        lut = pd.read_csv(RESOURCES_DIR / "aal2_lut.txt",
                          sep=r"\s+",
                          comment="#",
                          header=None,
                          engine="python")

        lut.columns = ["Index", "FullName"]

    elif atlas.lower() == "dk":
        # --- Desikan-Killiany : fichiers type FreeSurferColorLUT ---
        default = pd.read_csv(RESOURCES_DIR / "fs_default.txt",
                              sep=r"\s+", comment="#", header=None, engine="python")
        default.columns = ["Index", "Abbrev", "FullName", "R", "G", "B", "A"]

        lut = pd.read_csv(RESOURCES_DIR / "FreeSurferColorLUT.txt",
                          sep=r"\s+", comment="#", header=None, engine="python")
        lut.columns = ["Index", "FullName", "R", "G", "B", "A"]

    return default, lut


def get_lut_indices_corresponding_to_fs_default(atlas):
    lut_idx = []
    default, lut = read_lut_and_fs_default(atlas)
    for region in default["FullName"]:  #
        lut_idx.append(int(lut[lut["FullName"] == region]["Index"]))
    return lut_idx, list(default["Abbrev"])


def old_adjust_fmri_nodes(bold_img, aparc_img, RESOURCES_DIR, atlas='dk'):
    """
    Extract only the signal of the regions you have in fs_default. 
    """
    atlas_labels = np.unique(aparc_img.get_fdata()) # Indices of the labels of the parcellation according to the LUT
    masker = NiftiLabelsMasker(labels_img=aparc_img, standardize=True)
    masker.fit(bold_img)
    used_labels = masker.labels_ # Same but in the BOLD mask
    LOGGER.info("All atlas labels:", len(atlas_labels))
    LOGGER.info("Labels used by masker:", len(used_labels))
    dropped_labels = list(set(atlas_labels) - set(used_labels)) # Sometimes due to registration misalignement a label can be dropped
    dropped_labels = np.append(dropped_labels, 0)
    LOGGER.info("Dropped labels:", set(atlas_labels) - set(used_labels))
    unique_labels = np.setdiff1d(atlas_labels, list(dropped_labels))
    unique_labels_list =list(unique_labels) # All labels actually in the coregistered BOLD-parcellation
    
    lut_idx = []
    default, lut = read_lut_and_fs_default(RESOURCES_DIR, atlas)
    for region in default["FullName"]:
        lut_idx.append(int(lut[lut["FullName"]==region]["Index"])) 

    if atlas=='dk':
        lut_idx.pop(0)
        idx_to_keep = [unique_labels_list.index(i) for i in lut_idx]
        LOGGER.info(f"You will have {len(idx_to_keep)} nodes in you BOLD")
        idx_to_keep = [i-1 for i in idx_to_keep]
        labels = list(default["FullName"])
        labels.pop(0) # Remove "Unkown"
    return idx_to_keep, labels

def adjust_fmri_nodes(masker, bold_img, RESOURCES_DIR, atlas="dk"):
    """
    Returns:
    - idx_to_keep: column indices in BOLD time series
    - region_names: anatomical labels (ordered)
    - fs_labels: FreeSurfer numeric labels (ordered)
    """

    masker.fit(bold_img)
    used_labels = masker.labels_

    default, lut = read_lut_and_fs_default(RESOURCES_DIR, atlas)

    # Select atlas labels (drop background / Unknown)
    atlas_names = list(default["FullName"])
    if atlas == "dk" and "Unknown" in atlas_names:
        atlas_names.remove("Unknown")

    lut_df = lut[lut["FullName"].isin(atlas_names)]
    fs_labels = lut_df["Index"].astype(int).values

    # Map FS label -> column index
    label_to_col = {lab: i for i, lab in enumerate(used_labels)}

    idx_to_keep = []
    region_names = []
    fs_labels_kept = []

    for fs_lab, name in zip(fs_labels, lut_df["FullName"]):
        if fs_lab in label_to_col:
            idx_to_keep.append(label_to_col[fs_lab])
            region_names.append(name)
            fs_labels_kept.append(fs_lab)

    LOGGER.info(f"Final number of ROIs: {len(idx_to_keep)}")

    return idx_to_keep, region_names, fs_labels_kept


def bandpass_nilearn(bold, tr=2.0, low_pass=0.198, high_pass=0.01, confounds=None, standardize=True):
    """
    bold: array (T, R)
    tr: repetition time in seconds
    low_pass: upper frequency (Hz)
    high_pass: lower frequency (Hz)
    confounds: array (T, n_conf) or None
    returns: filtered array (T, R)
    """
    # nilearn expects shape (T, n_signals)
    filtered = clean(signals=bold,
                     confounds=confounds,
                     detrend=True,
                     standardize=standardize,
                     low_pass=low_pass,
                     high_pass=high_pass,
                     t_r=tr)
    return filtered

def get_repetition_time(json_file):
    # Get the repetition time
    with open(json_file, "r") as f:
        data = json.load(f)
    tr = data.get("RepetitionTime", None)
    return tr

def plot_signal_and_matrices(pid, ses, combination, filtered_bold, fcd, fc):

    plt.figure(figsize=(12, 5))
    plt.subplot(131)
    plt.plot(5*filtered_bold/filtered_bold.max() + np.arange(filtered_bold.shape[1]), linewidth=0.5)
    plt.title("Filtered BOLD signals")
    plt.subplot(132)
    plt.imshow(fcd, cmap='viridis')
    plt.title(f"FCD, VAR={np.var(fcd)}")
    plt.subplot(133)
    plt.imshow(fc, cmap='viridis')
    plt.title(f"FC, GBC={np.mean(np.triu(fc, k=1))}")
    plt.suptitle(f"Subject {pid}, session {ses}, strategy: {combination}")
    plt.show()
    #plt.savefig(path_config.pipeline_postproc / f"{pid}_{combination}_signal_matrices.png")

def compute_basic_metrics(filtered_bold, tr):
    window_length = int(20//tr)
    overlap = window_length - 1
    fcd = compute_fcd(filtered_bold, window_length=window_length, overlap=overlap)
    var_fcd = np.var(fcd)
    fc = np.corrcoef(filtered_bold.T)
    gbc = np.mean(np.triu(fc, k=1))

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
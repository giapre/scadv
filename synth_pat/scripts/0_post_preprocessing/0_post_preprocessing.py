import os
import sys

import nibabel as nib
from nilearn import image
from nilearn.signal import clean
from nilearn.input_data import NiftiLabelsMasker

import pandas as pd 
import numpy as np 
import json

import matplotlib.pyplot as plt

from postproc_utils import *

# Set-up directories 
pid = sys.argv[1]
ses = sys.argv[2]
print(f"Searching for {pid}_{ses} data")

DATASET_DIR = "/data/core-psy-archive/data/PRONIA/"
json_file = os.path.join(DATASET_DIR, f"{pid}/{ses}/func/{pid}_{ses}_task-rest_bold.json") #json file of the raw func image
DERIVATIVES_DIR = os.path.join(DATASET_DIR, "test_vbt_pipe/vbt_derivatives") # where preproc outputs are
FMRI_DIR = os.path.join(DERIVATIVES_DIR, pid, ses, 'func')
FS_DIR = os.path.join(DERIVATIVES_DIR, f'freesurfer/{pid}_{ses}') 
if not os.path.exists(FS_DIR):
    FS_DIR = os.path.join(DERIVATIVES_DIR, f"freesurfer/{pid}_ses-t0")

# Extract relevant info

tr = get_repetition_time(json_file) # time repetition
confounds_data = pd.read_csv(f"{FMRI_DIR}/{pid}_{ses}_task-rest_desc-confounds_timeseries.tsv", sep='\t') # confounds
confounds_json = f"{FMRI_DIR}/{pid}_{ses}_task-rest_desc-confounds_timeseries.json"
c_comp_cor_top5, w_comp_cor_top5 = get_top5_compcor(confounds_json)
c_comp_cor_50, w_comp_cor_50 = get_compcor_50pct(confounds_json)
motion_correction = ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']
conf_dic = {'daniela': motion_correction + ['white_matter', 'csf'],
    'traditional': motion_correction + ['global_signal', 'white_matter', 'csf'],
    'aCompCor': motion_correction + c_comp_cor_top5 + w_comp_cor_top5,
    'aCompCor50': motion_correction + c_comp_cor_50 + w_comp_cor_50
    }

bold_file = f"{FMRI_DIR}/{pid}_{ses}_task-rest_space-T1w_desc-preproc_bold.nii.gz"
aparc_file = f"{FS_DIR}/mri/aparc+aseg.nii.gz"

confound = conf_dic
# Load images
bold_img = nib.load(bold_file)
aparc_img = nib.load(aparc_file)
masker = NiftiLabelsMasker(
    labels_img=aparc_img,
    standardize=False,
    detrend=True,
    low_pass=0.198,
    high_pass=0.01,
    t_r=tr
)

results = []
emp_bold_path_dict = {}
for combination in conf_dic:
    selected_confunds_columns = conf_dic[combination]
    selected_confunds = confounds_data[selected_confunds_columns]
    time_series = masker.fit_transform(bold_img, confounds=selected_confunds.values)
    print(f"Bold has shape: {time_series.shape}")

    # print(f"Reshaped BOLD has dimension {time_series[:, idx].shape}")

    filtered_bold = time_series[40:,:] #postproc_utils.bandpass_nilearn(time_series, tr=tr)
    emp_bold_path_dict[combination] = f"{FS_DIR}/pipe/{pid}_{ses}_{combination}_filtered_bold.npz"
    np.savez(emp_bold_path_dict[combination], bold=filtered_bold, labels=masker.labels_, TimeRepetition=tr)
    fc, gbc, fcd, var_fcd = compute_basic_metrics(filtered_bold, tr)
    #plot_signal_and_matrices(pid, ses, combination, filtered_bold, fcd, var_fcd, fc, gbc, f"{FS_DIR}/pipe/")

print(f'\n {pid} done! \n')
    #results.append([combination, var_fcd, gbc])

#df = pd.DataFrame(data=results, columns=['Strategy', 'VAR_FCD', 'GBC',])
#df.to_csv(f"{pid}_{ses}_basic_metrics.csv", index=False)



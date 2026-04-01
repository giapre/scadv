from paths import Paths
import numpy as np
import pandas as pd
import glob
import os
import sys
from analysis_utils import compute_features, make_roi_alff_df, make_roi_fc_mean_df, make_roi_fc_couples_df, fcd_variance_excluding_overlap
from utils import prepare_fs_default
from plot_utils import plot_signal_and_matrices

type_of_filter = Paths.TYPE_OF_CONFOUNDS
subject_id = sys.argv[1]
ses = sys.argv[2]

PID_RESULTS = os.path.join(Paths.DERIVATIVES, f'freesurfer/{subject_id}_{ses}/pipe') 
if not os.path.exists(PID_RESULTS):
    PID_RESULTS = os.path.join(Paths.DERIVATIVES, f"freesurfer/{subject_id}_ses-t0/pipe")

matches = glob.glob(os.path.join(PID_RESULTS, f"*{ses}_{type_of_filter}_filtered_bold.npz"))

if not matches:
    # no file → skip
    print(f"Skipping {subject_id}: no filtered bold file in {matches}")
else:

    empirical_data = np.load(matches[0])
    empirical_bold = empirical_data['bold']
    empirical_labels = list(empirical_data['labels'])
    empirical_tr = empirical_data['TimeRepetition']
    # fMRIPrep includes background label (0) but no signal so I remove it
    if empirical_labels[0] == 0:
        empirical_labels.pop(0)

    assert len(empirical_labels) == empirical_bold.shape[1]
    lut = pd.read_csv(
        f"{Paths.RESOURCES}/FreeSurferColorLUT.txt",
        sep=r"\s+",
        comment="#",
        names=["No", "Region", "R", "G", "B", "A"],
        index_col="No",
    )

    # Load fs_default (target order)
    fs = prepare_fs_default()     
    fs_regions = fs["Region"].tolist()
    # Map fMRIPrep labels to regions
    # LUT rows corresponding to fMRIPrep labels
    empirical_lut = lut.loc[empirical_labels]
    # Build column index mapping, label number to column index in empirical_bold
    label_to_col = {label: i for i, label in enumerate(empirical_labels)}

    ordered_cols = []
    ordered_regions = []

    for region in fs_regions:
        rows = empirical_lut[empirical_lut["Region"] == region]

        if len(rows) == 0:
            continue  # region not present in empirical BOLD

        label_no = rows.index[0]              # FreeSurfer label number
        col_idx = label_to_col[label_no]      # column in empirical_bold

        ordered_cols.append(col_idx)
        ordered_regions.append(region)

    empirical_bold_to_keep = empirical_bold[10:, ordered_cols]
    dt = empirical_tr * 1000
    window_size = int(20 // empirical_tr)
    overlap = window_size - 1
    emp_fc, emp_fcd, emp_zscored_ALFF, emp_fALFF = compute_features(empirical_bold_to_keep[:,:,None], dt, window_size, overlap)
    emp_zscored_ALFF = np.array(emp_zscored_ALFF)

    output_name = f'{PID_RESULTS}/{subject_id}_{ses}_{type_of_filter}_full_emp_results.npz'

    np.savez(output_name, 
            FC=emp_fc,
            FCD=emp_fcd,
            ALFF=emp_zscored_ALFF,
            fALFF=emp_fALFF,
            ordered_regions=ordered_regions,
            time_repetition=empirical_tr)

    triu_idx = np.triu_indices(emp_fc.shape[0], k=1)
    emp_gbc = np.mean(emp_fc[triu_idx])
    window_length = int(20//empirical_tr)
    overlap = window_length - 1
    emp_var_fcd = fcd_variance_excluding_overlap(emp_fcd, window_length=window_length, overlap=overlap) #np.var(emp_fcd,axis=0)

    emp_df = pd.DataFrame({'subject_id': subject_id, 'GBC': emp_gbc, 'VAR_FCD': emp_var_fcd}) #'we': np.round(params[:,0],4), 'sigma': np.round(params[:,1],4)})#'we': np.round(params[:,0],4), 'wd': np.round(params[:,1],4), 'ws': np.round(params[:,2],4)})
    plot_signal_and_matrices(subject_id, ses, type_of_filter, empirical_bold_to_keep, emp_fcd[:,:,0], emp_var_fcd[0], emp_fc[:,:,0], emp_gbc, PID_RESULTS)

    region_labels = fs['Label'].tolist()
    mask = pd.read_csv(f'{Paths.RESOURCES}/Masks/dk_sero_dopa_mask.csv', index_col=0)
    assert region_labels == mask.columns.to_list()[:84] # asserting the same order of my simulations
    print(region_labels)
    fc_regions = region_labels#['PU', 'CA', 'HI', 'STG', 'CER', 'CACG', 'RACG', 'IN', 'PCG', 'POP', 'POR', 'PTR']
    h_fc_regions = region_labels#['L.'+region for region in fc_regions] + ['R.'+region for region in fc_regions]

    fc_mean_df = make_roi_fc_mean_df(emp_fc, h_fc_regions)
    alff_df = make_roi_alff_df(emp_zscored_ALFF, h_fc_regions)

    fc_combinations = [['PU', 'RACG'], 
                    ['PU', 'CACG'],
                    ['PU', 'IN'],
                    ['PU', 'CER'],
                    ['CA', 'RACG'],
                    ['CA', 'CACG'],
                    ['CA', 'IN'],
                    ['CA', 'PCG'],
                    ['CA', 'HI'],
                    ['CA', 'CER'],
                    ['HI', 'IN'],]

    h_fc_combinations = []
    for combination in fc_combinations:
        for hemi in ['L.', 'R.']:
            r0 = hemi+combination[0]
            r1 = hemi+combination[1]
            h_fc_combinations.append([r0, r1])

    fc_couples_df = make_roi_fc_couples_df(emp_fc, h_fc_combinations)

    final_df = pd.concat([emp_df, fc_mean_df, fc_couples_df, alff_df], axis=1)
    final_df.to_csv(f'{PID_RESULTS}/{subject_id}_{ses}_{type_of_filter}_extracted_emp_features.csv')

    print(f'Results saved at {PID_RESULTS}/{subject_id}_{ses}_{type_of_filter}_extracted_emp_features.csv')

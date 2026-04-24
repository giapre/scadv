from analysis_utils import compute_features, make_roi_alff_df, make_roi_fc_mean_df, make_roi_fc_couples_df, fcd_variance_excluding_overlap, do_pca
import pandas as pd
import numpy as np
import os
import sys
from paths import Paths

type_of_sweep = Paths.TYPE_OF_SWEEP
type_of_confounds = Paths.TYPE_OF_CONFOUNDS

subject_id = sys.argv[1]
ses = sys.argv[2]
GEN_RESULTS_DIR = f"/data/core-psy-archive/data/PRONIA/test_vbt_pipe/vbt_derivatives/freesurfer/{subject_id}_{ses}/pipe"
if not os.path.exists(GEN_RESULTS_DIR):
    GEN_RESULTS_DIR = f"/data/core-psy-archive/data/PRONIA/test_vbt_pipe/vbt_derivatives/freesurfer/{subject_id}_ses-t0/pipe"

RESULTS_DIR = f"{GEN_RESULTS_DIR}/medication/"

print(f'Processing {subject_id}')

input_file = os.path.join(RESULTS_DIR, f'{subject_id}_{ses}_{type_of_confounds}_{type_of_sweep}_medication_effect.npz')
output_file = f'{RESULTS_DIR}/{subject_id}_{ses}_{type_of_confounds}_{type_of_sweep}_medication_effect_extracted_features.csv'
if os.path.isfile(output_file):
    print(f'{output_file} already exists! Terminating here')
    #sys.exit()

if os.path.isfile(input_file):
    print ("File exists")

    data = np.load(input_file)
    bold_all = data["bold"]
    bold_all = bold_all[:,:84,:] # removing midbrain structures (SN, RF, VTA)
    medication = data["medication"]
    med_zi = data["med_zi"]
    med_params = data["med_params"]
    med_param_names = data["med_param_names"]
    est_params = data["est_params"]
    est_param_names = data["est_param_names"]

    print(bold_all.shape)
    print(med_params.shape)
    print(med_param_names)
    print(est_params.shape)
    print(est_param_names)

    sim_fc, sim_fcd, sim_alff, sim_falff = compute_features(bold_all, 1000, 20, 19)
    
    R = sim_fc.shape[0]
    triu_idx = np.triu_indices(R, k=1)
    sim_gbc = np.mean(sim_fc[triu_idx[0], triu_idx[1], :], axis=0)
    sim_var_fcd = fcd_variance_excluding_overlap(sim_fcd, window_length=20, overlap=19) #np.var(sim_fcd,axis=0)

    sim_df = pd.DataFrame({'GBC': sim_gbc, 'VAR_FCD': sim_var_fcd, 
    'medication': medication, 'med_zi': med_zi,
    med_param_names[0]: np.round(med_params[:,0],4), med_param_names[1]: np.round(med_params[:,1],4), med_param_names[2]: np.round(med_params[:,2],4),
    est_param_names[0]: np.round(est_params[:,0],4), est_param_names[1]: np.round(est_params[:,1],4), est_param_names[2]: np.round(est_params[:,2],4)})

    fc_regions = ['PU', 'CA', 'HI', 'STG', 'CER', 'CACG', 'RACG', 'IN', 'PCG', 'POP', 'POR', 'PTR']
    h_fc_regions = ['L.'+region for region in fc_regions] + ['R.'+region for region in fc_regions]

    fc_mean_df = make_roi_fc_mean_df(sim_fc, h_fc_regions)
    alff_df = make_roi_alff_df(sim_alff, h_fc_regions)

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

    fc_couples_df = make_roi_fc_couples_df(sim_fc, h_fc_combinations)

    final_df = pd.concat([sim_df, fc_mean_df, fc_couples_df, alff_df], axis=1)
    final_df.to_csv(output_file)

    print(f"Features saved at {output_file}")
    print('Now printing the images')

    from plot_utils import plot_signal_and_matrices
    fig_path = f'{RESULTS_DIR}/figures/'
    os.makedirs(fig_path, exist_ok=True)
    for i in range(bold_all.shape[2])[::10]:
        med_name = medication[i]
        zi = med_zi[i]
        combination = f'{med_name}_{zi}'
        plot_signal_and_matrices(pid=subject_id, ses=f'{ses} {type_of_confounds}', combination=combination, filtered_bold=bold_all[:,:,i], fcd=sim_fcd[:,:,i], var_fcd=sim_var_fcd[i], fc=sim_fc[:,:,i], mean_fc=sim_gbc[i], path=fig_path)
    
    print(f"Figures saved at {fig_path}")

else:
    print('Input file does not exist!')
    


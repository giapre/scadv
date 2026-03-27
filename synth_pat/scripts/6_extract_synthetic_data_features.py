from analysis_utils import compute_features, make_roi_alff_df, make_roi_fc_mean_df, make_roi_fc_couples_df, fcd_variance_excluding_overlap
import pandas as pd
import numpy as np
import os
import sys
from paths import Paths

type_of_sweep = Paths.TYPE_OF_SWEEP
subject_id = sys.argv[1]
RESULTS_DIR = f"/data/core-psy-archive/data/PRONIA/test_vbt_pipe/vbt_derivatives/freesurfer/{subject_id}/pipe/simulations"

if subject_id.startswith('sub-'):
    print(f'Processing {subject_id}')

    file = os.path.join(RESULTS_DIR, f'{subject_id}_{type_of_sweep}.npz')

    if os.path.isfile(file):
        print ("File exists")
        data = np.load(file)
        bold_all = data["bold"]
        bold_all = bold_all[:,:84,:] # removing midbrain structures (SN, RF, VTA)
        params = data["params"]
        param_names = data["param_names"]

        sim_fc, sim_fcd, sim_alff, sim_falff = compute_features(bold_all, 1000, 20, 19)
        
        R = sim_fc.shape[0]
        triu_idx = np.triu_indices(R, k=1)
        sim_gbc = np.mean(sim_fc[triu_idx[0], triu_idx[1], :], axis=0)
        sim_var_fcd = fcd_variance_excluding_overlap(sim_fcd, window_length=20, overlap=19) #np.var(sim_fcd,axis=0)

        #if 'jdopa' in type_of_sweep:
        sim_df = pd.DataFrame({'GBC': sim_gbc, 'VAR_FCD': sim_var_fcd, param_names[0]: np.round(params[:,0],4), param_names[1]: np.round(params[:,1],4), param_names[2]: np.round(params[:,2],4)})
        #else:
        #    sim_df = pd.DataFrame({'GBC': sim_gbc, 'VAR_FCD': sim_var_fcd, 'we': np.round(params[:,0],4), 'wd': np.round(params[:,1],4), 'ws': np.round(params[:,2],4)})#'we': np.round(params[:,0],4), 'sigma': np.round(params[:,1],4)})#'we': np.round(params[:,0],4), 'wd': np.round(params[:,1],4), 'ws': np.round(params[:,2],4)})

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
        final_df.to_csv(f'{RESULTS_DIR}/{subject_id}_{type_of_sweep}_extracted_features.csv')

        print(f"Features saved at {RESULTS_DIR}/{subject_id}_{type_of_sweep}_extracted_features.csv")
        print('Now printing the images')

        from plot_utils import plot_signal_and_matrices
        fig_path = f'{RESULTS_DIR}/figures/'
        os.makedirs(fig_path, exist_ok=True)
        pid = subject_id.split('_')[0]
        ses = subject_id.split('_')[1]
        for i in range(bold_all.shape[2]):
            combination = f'{param_names[0]}:{np.round(params[i,0],4)}__{param_names[1]}:{np.round(params[i,1],4)}__{param_names[2]}:{np.round(params[i,2],4)}'
            plot_signal_and_matrices(pid=pid, ses=ses, combination=combination, filtered_bold=bold_all[:,:,i], fcd=sim_fcd[:,:,i], var_fcd=sim_var_fcd[i], fc=sim_fc[:,:,i], mean_fc=sim_gbc[i], path=fig_path)
        
        print(f"Figures saved at {fig_path}")
        


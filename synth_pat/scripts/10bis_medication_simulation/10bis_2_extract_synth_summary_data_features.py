from synth_pat.scripts.analysis_utils import compute_features, fcd_variance_excluding_overlap, make_roi_fc_mean_df, make_roi_alff_df, make_roi_fc_couples_df
import pandas as pd
from synth_pat.scripts.plot_utils import plot_signal_and_matrices
import numpy as np
import os
from synth_pat.paths import Paths

type_of_sweep = Paths.TYPE_OF_SWEEP
ses = 'run-01'

for pid in os.listdir(Paths.RESULTS):
    if pid.startswith('sub-'):
        print(f'Processing {pid}')
        save_name = f'{Paths.RESULTS}/{pid}/{type_of_sweep}_medication_extracted_features.csv'
        if os.path.exists(save_name):
            print(f'Results already exist at {save_name}')
            continue

        input_dir = f'{Paths.RESULTS}/{pid}'
        file = os.path.join(input_dir, f'{type_of_sweep}_medication.npz')

        if os.path.isfile(file):
            print ("File exists")
            data = np.load(file)
            bold_all = data["bold"]
            bold_all = bold_all[:,:84,:] # removing midbrain structures (SN, RF, VTA)
            med_params = data['med_params']
            medication_name = data['medication_name']
            est_params = data['est_params']
            med_param_names = ['Z_D1', 'Z_D2', 'Z_S']#data['med_param_names']
            est_param_names = data['est_param_names']
            med_zi = data['med_zi']

            sim_fc, sim_fcd, sim_alff, sim_fALFF = compute_features(bold_all, 1000, 20, 19)

            R = sim_fc.shape[0]
            triu_idx = np.triu_indices(R, k=1)
            sim_gbc = np.mean(sim_fc[triu_idx[0], triu_idx[1], :], axis=0)
            sim_var_fcd = fcd_variance_excluding_overlap(sim_fcd, window_length=20, overlap=19) #np.var(sim_fcd,axis=0)

            sim_df = pd.DataFrame({'GBC': sim_gbc, 'VAR_FCD': sim_var_fcd, 
                                   med_param_names[0]: np.round(med_params[:,0],4), med_param_names[1]: np.round(med_params[:,1],4), med_param_names[2]: np.round(med_params[:,2],4),
                                   est_param_names[0]: np.round(est_params[:,0],4), est_param_names[1]: np.round(est_params[:,1],4), est_param_names[2]: np.round(est_params[:,2],4), 
                                   'med_zi': med_zi, 'medication': medication_name})
            
            #FIG_DIR = f'{Paths.RESULTS}/{pid}/sim_bold_ts'
            #os.makedirs(FIG_DIR, exist_ok=True)
            #for i in range(sim_gbc.shape[0])[::90]:
            #    plot_signal_and_matrices(pid, f'{param_names[1]}: {np.round(params[:,0],4)}, {param_names[1]}: {np.round(params[:,1],4)}, {param_names[2]}: {np.round(params[:,2],4)}', type_of_sweep, bold_all[:,:,i], sim_fcd[:,:,i], sim_var_fcd[i], sim_fc[:,:,i], sim_gbc[i], FIG_DIR)

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
            final_df.to_csv(f'{save_name}')
        



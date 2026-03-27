import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from sbi.inference import SNPE, DirectPosterior

from sbi.analysis import pairplot
import sbi.utils as utils

from paths import Paths
from plot_utils import plot_sbi_violin_estimated_params, plot_sbi_kde_distr

subject_id = sys.argv[1]
ses = sys.argv[2]
print(f'Running {subject_id} {ses}')
type_of_confounds = Paths.TYPE_OF_CONFOUNDS
type_of_sweep = Paths.TYPE_OF_SWEEP
PID_RESULTS = os.path.join(Paths.DERIVATIVES, f'freesurfer/{subject_id}_{ses}/pipe') 
if not os.path.exists(PID_RESULTS):
    PID_RESULTS = os.path.join(Paths.DERIVATIVES, f"freesurfer/{subject_id}_ses-t0/pipe")
SBI_RESULTS = f'{PID_RESULTS}/sbi'
os.makedirs(SBI_RESULTS, exist_ok=True)

if os.path.isfile(f'{SBI_RESULTS}/{ses}_{type_of_sweep}_{type_of_confounds}_params_post_distr.npz'):
     print(f'Sbi for {subject_id} {ses} {type_of_sweep}_{type_of_confounds}, already done!')

else:
     # Getting the synthetic datafeatures and parameters
     theta_and_features_df = pd.read_csv(f'{PID_RESULTS}/simulations/{subject_id}_ses-t0_{type_of_sweep}_extracted_features.csv', index_col=0)
     print(theta_and_features_df.shape)
     selected_params = ['ws', 'njdopa_ctx', 'njdopa_str'] # put the params_names instead of hard coding this 
     theta_and_features_df[selected_params] = np.log10(theta_and_features_df[selected_params])

     print(np.any(theta_and_features_df.values == np.inf))
     print(np.any(theta_and_features_df.values == np.nan))

     # Dividing features and parameters dataframes and preparing torch arrays
     datafeat_df = theta_and_features_df.drop(columns=selected_params)
     datafeat = datafeat_df.values

     theta_df = theta_and_features_df[selected_params]
     theta = theta_df.values

     x = np.array(datafeat, dtype='float32')
     x = torch.as_tensor(x)

     theta = np.array(theta, dtype='float32')
     theta = theta.reshape(theta.shape[0],len(selected_params))
     theta = torch.as_tensor(theta)

     print( 'theta shape:',theta.shape,flush=True)
     print('data feature shape:', x.shape,flush=True)
     print(theta.isnan().any())
     print(x.isnan().any())

     # Preparing the prior distribution with min and max range of parameters
     prior_min = [theta_and_features_df['ws'].min(), theta_and_features_df['njdopa_ctx'].min(), theta_and_features_df['njdopa_str'].min()]
     prior_max = [theta_and_features_df['ws'].max(), theta_and_features_df['njdopa_ctx'].max(), theta_and_features_df['njdopa_str'].max()]
     prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)) # perhaps different distributions log or gaussian
     num_params=prior.sample().shape[0]
     print(f'Prior has {num_params} types of paramters')

     # Run the inference algo
     inference = SNPE(prior, density_estimator='maf', device='cpu')
     start_time = time.time()
     posterior_estimator = inference.append_simulations(theta, x).train()
     print ("-"*60)
     print("---training took:  %s seconds ---" % (time.time() - start_time))
     posterior = DirectPosterior(posterior_estimator, prior,) 

     # Getting the empirical data
     all_pid_obs = pd.read_csv(f'{PID_RESULTS}/{subject_id}_{ses}_{type_of_confounds}_extracted_emp_features.csv')
     x_obs_summary_statistics = all_pid_obs.loc[0, datafeat_df.columns].values

     # Infering the posterior distribution of parameters
     start_time = time.time()

     posterior_samples = posterior.sample((1000,), x_obs_summary_statistics).numpy()

     print ("-"*60)
     print("--- posterior sampling took: %s seconds ---" % (time.time() - start_time))

     ws_est=posterior_samples[:,0]
     njdopa_ctx_est=posterior_samples[:,1]
     njdopa_str_est=posterior_samples[:,2]

     print("njdopa_ctx_est=", njdopa_ctx_est.mean())
     print("njdopa_str_est=", njdopa_str_est.mean())
     print("ws_est=", ws_est.mean())

     post_results_output = f'{SBI_RESULTS}/{ses}_{type_of_sweep}_{type_of_confounds}_params_post_distr'
     np.savez(post_results_output, njdopa_ctx=njdopa_ctx_est, njdopa_str=njdopa_str_est, ws=ws_est)

     params_label = np.array(selected_params)
     plot_sbi_violin_estimated_params(params_label, (ws_est, njdopa_ctx_est, njdopa_str_est), SBI_RESULTS, ses, type_of_sweep, type_of_confounds)
     plot_sbi_kde_distr(params_label, prior, posterior_samples, SBI_RESULTS, ses, type_of_sweep, type_of_confounds)

print(f"{subject_id} {ses} Done!")
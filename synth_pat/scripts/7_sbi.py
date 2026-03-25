import os
import time

import numpy as np
import pandas as pd
import torch

from sbi.inference import SNPE, DirectPosterior

from sbi.analysis import pairplot
import sbi.utils as utils

from synth_pat.paths import Paths
from synth_pat.scripts.plot_utils import plot_sbi_violin_estimated_params, plot_sbi_kde_distr

def mode(x):
     values, counts = np.unique(x, return_counts=True)
     m = counts.argmax()
     return values[m]

type_of_extraction = 'daniela'
type_of_sweep = Paths.TYPE_OF_SWEEP
all_pid_obs = pd.read_csv(f'{Paths.DATA}/ALL_{type_of_extraction}_full_extracted_features.csv', index_col='pid')

list1 = ['sub-2015060902', 'sub-2015121001', 'sub-2019012101', 'sub-2019052301', 'sub-2015060401', 'sub-2015061901', 'sub-2015111402', 'sub-2019012402', 'sub-2019052302']
list2 = ['sub-2015052501', 'sub-2015120501', 'sub-2015120401']
list3 = [pid for pid in os.listdir(f'{Paths.RESULTS}') if pid not in list1+list2]

for pid in list1+list2:#all_pid_obs.index:
     #pid = f'sub-{pid_id}'
     print(f'Running {pid}')

     if os.path.isfile(f'{Paths.RESULTS}/{pid}/{type_of_sweep}_{type_of_extraction}_posterior_distr.npz'):
          print(f'Sbi for {pid} {type_of_sweep}_{type_of_extraction}, already done, going to next patient')
          continue
     else:
          pid_idx = int(pid.split('-')[1])
          # Getting the parameters
          theta_and_features_df = pd.read_csv(f'{Paths.RESULTS}/{pid}/{type_of_sweep}_extracted_features.csv', index_col=0)
          print(theta_and_features_df.shape)
          selected_params = np.array(['ws', 'njdopa_ctx', 'njdopa_str'])
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

          all_pid_obs = pd.read_csv(f'{Paths.DATA}/ALL_{type_of_extraction}_full_extracted_features.csv', index_col='pid')
          x_obs_summary_statistics = all_pid_obs.loc[pid_idx, datafeat_df.columns] 

          start_time = time.time()

          posterior_samples = posterior.sample((1000,), x_obs_summary_statistics).numpy()

          print ("-"*60)
          print("--- posterior sampling took: %s seconds ---" % (time.time() - start_time))

          filename = f"{Paths.RESULTS}/{pid}/{type_of_sweep}_{type_of_extraction}_posterior_distr"
          np.savez(filename, est_params = posterior_samples, param_names=selected_params)
          print(f"Saved posterior samples to {filename}")

          ws_est=posterior_samples[:,0]
          njdopa_ctx_est=posterior_samples[:,1]
          njdopa_str_est=posterior_samples[:,2]

          print("njdopa_ctx_est=", njdopa_ctx_est.mean())
          print("njdopa_str_est=", njdopa_str_est.mean())
          print("ws_est=", ws_est.mean())

          #mean_results_output = f'{Paths.RESULTS}/{pid}/{type_of_sweep}_{type_of_extraction}_estimated_mean_params'
          #mode_results_output = f'{Paths.RESULTS}/{pid}/{type_of_sweep}_{type_of_extraction}_estimated_mode_params'
          #np.savez(mean_results_output, njdopa_ctx=njdopa_ctx_est.mean(), njdopa_str=njdopa_str_est.mean(), ws=ws_est.mean())
          #np.savez(mode_results_output, njdopa_ctx=mode(njdopa_ctx_est), njdopa_str=mode(njdopa_str_est), ws=mode(ws_est))

          #params_label = np.array(selected_params)
          plot_sbi_violin_estimated_params(selected_params, (ws_est, njdopa_ctx_est, njdopa_str_est), pid, type_of_sweep, type_of_extraction)
          plot_sbi_kde_distr(selected_params, prior, posterior_samples, pid, type_of_sweep, type_of_extraction)
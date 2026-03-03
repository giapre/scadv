from synth_pat.scripts.analysis_utils import compute_features
import pandas as pd
import numpy as np
import os
from synth_pat.paths import Paths

type_of_sweep = Paths.TYPE_OF_SWEEP

for pid in os.listdir(Paths.RESULTS):
    if pid.startswith('sub-'):
        print(f'Processing {pid}')

        input_dir = f'{Paths.RESULTS}/{pid}'
        file = os.path.join(input_dir, f'{type_of_sweep}.npz')

        if os.path.isfile(file):
            print ("File exists")
            data = np.load(file)
            bold_all = data["bold"]
            bold_all = bold_all[:,:84,:] # removing midbrain structures (SN, RF, VTA)
            params = data["params"]

            fc_ut, fcd_ut, zscored_ALFF, fALFF = compute_features(bold_all, 1000, 20, 19)

            ## SAVE THE REUSLTS
            output_name = f'{Paths.RESULTS}/{pid}/{type_of_sweep}_features.npz'

            np.savez(output_name, 
                        FC=fc_ut,
                        FCD=fcd_ut,
                        ALFF=zscored_ALFF,
                        fALFF=fALFF,
                        params=params)

            assert os.path.exists(output_name), "Save failed!"
            print(f'Data features from simulations saved at {output_name}')
        else:
            print (f"File does not exist, skipping {pid}")



import os
import re
import glob
import numpy as np
from synth_pat.paths import Paths

# With this script I extract the single-stored signals and put them in a single file so I can use the 4, 5, 6 scripts inside scripts

SNAKE_RESULT_DIR = f'{Paths.SNAKEMAKE}/results/'

#type_of_sweep = 'jdopa_ws_sweep'
for pid in os.listdir(SNAKE_RESULT_DIR):
    results_path = f'{Paths.SNAKEMAKE}/results/{pid}'

    pattern = re.compile(
        r"ws(?P<ws>[^_]+)_ctx(?P<ctx>[^_]+)_str(?P<str>.+)\.npz"
    )

    files = glob.glob(os.path.join(results_path, "*.npz"))

    if len(files) != 1331:
        print(f'Skipping {pid}, sweep still incomplete')

    elif len(files) == 1331:    

        ws = np.empty(len(files))
        njdopa_ctx = np.empty(len(files))
        njdopa_str = np.empty(len(files))

        bolds = []

        for i, file in enumerate(files):
            match = pattern.search(os.path.basename(file))
            ws[i] = float(match.group("ws"))
            njdopa_ctx[i] = float(match.group("ctx"))
            njdopa_str[i] = float(match.group("str"))

            data = np.load(file, mmap_mode="r") 
            bolds.append(data["bold"])

        bolds = np.array(bolds)
        bolds = bolds.squeeze(-1).transpose(1, 2, 0)

        params = np.vstack((ws, njdopa_ctx, njdopa_str)).T

        # ------------------------
        # Save merged file
        # ------------------------

        save_path = f'{Paths.RESULTS}/{pid}/'
        os.makedirs(save_path, exist_ok = True )
        save_name = f'{save_path}/bigger_we_bold_sweep.npz'
        np.savez(save_name, bold=bolds, params=params)

        print("Saved merged file:", save_path)
        print(params.shape)

        # ------------------------
        # Verify file integrity
        # ------------------------

        test = np.load(save_name)
        assert test["bold"].shape == bolds.shape
        assert test["params"].shape == params.shape

        print("Verification successful. Deleting single files...")

        # ------------------------
        # Delete single files
        # ------------------------

        for file in files:
            os.remove(file)

        print(f"Deleted {len(files)} individual files.")

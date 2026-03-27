import os
import sys
import numpy as np

# ------------------------
# CONFIG
# ------------------------


subject_id   = sys.argv[1] 

base_dir = f"/data/core-psy-archive/data/PRONIA/test_vbt_pipe/vbt_derivatives/freesurfer/{subject_id}/pipe/simulations"

ws_values = np.array([
    0.001, 0.003, 0.01, 0.03, 0.1,
    0.3, 1.0, 3.0, 10.0, 30.0, 100.0
])

DELETE_AFTER = True  # set False if you want to keep original files

# ------------------------
# STORAGE
# ------------------------

all_bold = []
all_params = []
files_to_delete = []

print(f"Processing simulations in: {base_dir}")

# ------------------------
# LOOP OVER FILES
# ------------------------

for fname in sorted(os.listdir(base_dir)):

    if not fname.endswith(".npz"):
        continue

    file_path = os.path.join(base_dir, fname)

    # Skip final merged file if script re-run
    if "ALL_simulations" in fname:
        continue

    # ------------------------
    # Parse parameters from filename
    # ------------------------

    try:
        parts = fname.replace(".npz", "").split("_")
        njdopa_ctx = float(parts[0].split("=")[1])
        njdopa_str = float(parts[1].split("=")[1])
    except Exception as e:
        print(f"Skipping malformed filename: {fname}")
        continue

    # ------------------------
    # Load data
    # ------------------------

    data = np.load(file_path)
    bold = data["bold"]   # expected shape: (time, regions, n_ws)

    print(f"{fname} -> shape {bold.shape}")

    # ------------------------
    # Expand over ws dimension
    # ------------------------

    n_ws = bold.shape[2]

    if n_ws != len(ws_values):
        print(f"WARNING: ws mismatch in {fname} (found {n_ws})")

    for i, ws in enumerate(ws_values[:n_ws]):
        sim = bold[:, :, i]      # (time, regions)
        sim = sim[:, :, None]    # (time, regions, 1)

        all_bold.append(sim)
        all_params.append([ws, njdopa_ctx, njdopa_str])

    files_to_delete.append(file_path)

# ------------------------
# STACK DATA
# ------------------------

print("Stacking all simulations...")

all_bold = np.concatenate(all_bold, axis=2)   # (time, regions, N)
all_bold = all_bold.astype(np.float32)        # reduce memory

all_params = np.array(all_params)

print("Final shapes:")
print("bold:", all_bold.shape)
print("params:", all_params.shape)

# ------------------------
# SAVE FINAL FILE
# ------------------------

out_file = os.path.join(base_dir, f"{subject_id}_sweep_simulations.npz")

np.savez(
    out_file,
    bold=all_bold,
    params=all_params,
    param_names=np.array(["ws", "njdopa_ctx", "njdopa_str"])
)

print(f"\nSaved merged dataset to:\n{out_file}")

# ------------------------
# DELETE ORIGINAL FILES
# ------------------------

if DELETE_AFTER:
    print("\nDeleting individual simulation files...")

    for f in files_to_delete:
        try:
            os.remove(f)
            print(f"Deleted {f}")
        except Exception as e:
            print(f"Could not delete {f}: {e}")

    print("Cleanup complete.")

else:
    print("\nDELETE_AFTER=False → original files kept.")
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

subj = sys.argv[1]

freesurfer_dir = '/data/core-psy-archive/data/PRONIA/test_vbt_pipe/vbt_derivatives/freesurfer'
SUBJECT_DIR = f'{freesurfer_dir}/{subj}/pipe'

# Load data
W = pd.read_csv(f"{SUBJECT_DIR}/dk_weights_with_sero_and_dopa.csv", index_col=0)
L = pd.read_csv(f"{SUBJECT_DIR}/dk_lengths_with_sero_and_dopa.csv", index_col=0)
R = np.load(f"{SUBJECT_DIR}/Receptors.npy").reshape(3, 90)
Ja = np.load(f"{SUBJECT_DIR}/Ja.npy").reshape(90)

# Create figure layout
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 3)

# --- Top row ---
# W heatmap
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(W.values, vmax=0.25, aspect="equal")
ax1.set_title("Weights")
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.outline.set_visible(False)

# L heatmap
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(L.values, aspect="equal")
ax2.set_title("Lengths (in mm)")
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.outline.set_visible(False)

# Ja barplot
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(np.arange(len(Ja)), Ja)
ax3.set_title("Self excitation (adjusted by cortical thickness)")
ax3.set_xlabel("Brain region")
ax3.set_ylabel("Value")

# --- Bottom row ---
# R[0]
ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(np.arange(len(R[0])), R[0])
ax4.set_title("D1 Receptors")
ax4.set_xlabel("Brain region")

# R[1]
ax5 = fig.add_subplot(gs[1, 1])
ax5.bar(np.arange(len(R[1])), R[1])
ax5.set_title("D2 Receptors")
ax5.set_xlabel("Brain region")

# R[2]
ax6 = fig.add_subplot(gs[1, 2])
ax6.bar(np.arange(len(R[2])), R[2])
ax6.set_title("5HT2A Receptors")
ax6.set_xlabel("Brain region")

plt.tight_layout()

# Save figure
plt.savefig(f"{SUBJECT_DIR}/connectivity_receptors_overview.png", dpi=300)
plt.close()

print("Figure saved as connectivity_receptors_overview.png")
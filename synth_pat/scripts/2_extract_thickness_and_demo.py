#!/usr/bin/env python3
import os
import sys
import pandas as pd

freesurfer_dir = "/data/core-psy-archive/data/PRONIA/test_vbt_pipe/vbt_derivatives/freesurfer"
demo_dir = '/data/core-psy-archive/projects/VBT_SCZ/demo.csv'

all_data = []
demo_data = pd.read_csv(demo_dir, index_col='PSN')

# Get subject from command line
if len(sys.argv) < 2:
    print("Usage: python extract_thickness.py <sub-XXXX>")
    sys.exit(1)

subj = sys.argv[1]
output_csv = os.path.join(freesurfer_dir, subj, "pipe/gray_matter_thickness.csv")

if not subj.startswith('sub-'):
    print(f"Skipping {subj}: not a patient")
    sys.exit(0)

subj_idx = int(subj.split('-')[1].split('_')[0])
if subj_idx not in demo_data.index:
    print(f"Skipping {subj}: not in the demo list")
    sys.exit(0)

if os.path.exists(output_csv):
    print(f"Skipping {subj}: file already exists")
    sys.exit(0)

subj_path = os.path.join(freesurfer_dir, subj)
stats_path = os.path.join(subj_path, "stats")
if not os.path.isdir(stats_path):
    print(f"Skipping {subj}: no stats folder in {stats_path}")
    sys.exit(0)

subj_dict = {"SubjectID": subj}
print(f'Running patient {subj}')

for hemi in ["lh", "rh"]:
    stats_file = os.path.join(stats_path, f"{hemi}.aparc.stats")
    if not os.path.exists(stats_file):
        print(f"Skipping {subj} {hemi}: stats file not found")
        continue
    
    with open(stats_file, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            region_name = parts[0]
            thick_avg = parts[4]
            subj_dict[f"{hemi}_{region_name}"] = float(thick_avg)

# Add demographics
subj_dict['AGE_T0'] = demo_data.loc[subj_idx, 'AGE_T0']
subj_dict['SEX'] = demo_data.loc[subj_idx, 'SEX']
subj_dict['Remission'] = demo_data.loc[subj_idx, 'Remission']

all_data.append(subj_dict)

# Convert to DataFrame and save
df = pd.DataFrame(all_data)
cols = ["SubjectID"] + sorted([c for c in df.columns if c != "SubjectID"])
df = df[cols]
df.to_csv(output_csv, index=False)
print(f"Saved thickness data to {output_csv}")
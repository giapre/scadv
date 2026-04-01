import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyvista as pv
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(PROJECT_DIR)
from utils import prepare_FreeSurferColorLUT, rename_to_fs_lut_region

def rename_to_fs_from_lut(df, lut_path):
    """
    Rename dataframe indices from MRtrix-style labels (e.g. L.CMFG)
    to FreeSurfer-style labels (e.g. ctx-lh-caudalmiddlefrontal)
    using a LUT file.
    
    Parameters:
    - df: pandas DataFrame (regions in index)
    - lut_path: path to LUT file (the one you pasted)
    
    Returns:
    - df with renamed index
    """

    mapping = {}

    with open(lut_path, 'r') as f:
        for line in f:
            line = line.strip()

            # skip comments or empty lines
            if not line or line.startswith('#'):
                continue

            parts = line.split()

            # need at least 3 columns: index, short name, fs name
            if len(parts) < 3:
                continue

            short_name = parts[1]   # e.g. L.CMFG
            fs_name = parts[2]      # e.g. ctx-lh-caudalmiddlefrontal

            mapping[short_name] = fs_name

    # rename index
    df = df.copy()
    df.columns = df.columns.map(lambda x: mapping.get(x, x))
    print(df)

    return df

def dk_plot_annotation_surface(
    title,
    df_annot_dir,
    pat_name,
    data_dir,
    hemisphere,
    colormap
):
    
    """
    Plot cortical surface annotation data from FreeSurfer using PyVista.

    Parameters:
    - data_dir: Path to subjects folder (e.g. ".../Patients/")
    - pat_name: Name of the patient
    - annt_to_plot: name of the annotation to visualize (e.g. "receptor_number", "receptor_density", "GM Vol.")
    - hemisphere: 'lh' or 'rh'
    - colormap: Matplotlib colormap name (e.g. 'plasma', 'cividis')
    """

    # Prepare the df with the annotation to plot
    #lut_df = prepare_FreeSurferColorLUT()
    df_annot = pd.read_csv(df_annot_dir, index_col=0)
    if 'ALFF' in df_annot.columns[0]:
        df_annot.columns = df_annot.columns.str.replace('_ALFF', '', regex=False)
    print(df_annot.columns[0])
    #df_annot = rename_to_fs_lut_region(df_annot, lut_df)
    df_annot = rename_to_fs_from_lut(df_annot, '/Users/giacomopreti/Desktop/VBT/scadv/resources/fs_default.txt')
    df_annot = df_annot.T

    # Load surface and annotation
    annot_file = f"{data_dir}{pat_name}/label/{hemisphere}.aparc.annot"
    surface_file = f"{data_dir}{pat_name}/surf/{hemisphere}.pial"
    labels, ctab, names = nib.freesurfer.read_annot(annot_file)
    coords, faces = nib.freesurfer.io.read_geometry(surface_file)

    # Decode names and build mapping
    region_names = [name.decode('utf-8') for name in names]
    label_to_name = {i: region_names[i] for i in range(len(region_names))}

    # Map each face to a region label
    def get_face_region(face, labels):
        face_labels = labels[face]
        valid_labels = [label for label in face_labels if label != -1]
        return max(set(valid_labels), key=valid_labels.count) if valid_labels else -1

    face_region_ids = np.array([get_face_region(face, labels) for face in faces])
    face_region_names = np.array([label_to_name.get(fr, 'Unknown') for fr in face_region_ids])

    # Map values to regions
    label_to_value = dict(zip(df_annot.index, df_annot[0]))
    volumes_of_faces = [label_to_value.get(f'ctx-{hemisphere}-{name}', 0) for name in face_region_names]
    weights = np.array(volumes_of_faces)
    print(face_region_names[:10])
    print(list(label_to_value.keys())[:10])

    # Create mesh for PyVista
    faces_pyvista = np.hstack([
        np.full((faces.shape[0], 1), 3),  # leading 3 for triangle
        faces
    ]).ravel()
    mesh = pv.PolyData(coords, faces_pyvista)
    mesh.cell_data["Annotation"] = weights

    # Plotting
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=False, scalars="Annotation", cmap=colormap, opacity=0.6, scalar_bar_args={
        "title": title,
        "vertical": True,
        "title_font_size": 12,
        "label_font_size": 10,
    })
    plotter.add_title(title, font_size=13)
    plotter.show()

# === Script execution ===
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot DK annotation surface values")
    parser.add_argument("--title", type=str, default='Normalized ALFF across regions',
                        help="Title of the plot")
    parser.add_argument("--df_annot_dir", type=str, default='/Users/giacomopreti/Desktop/VBT/synthetic-patient/results/alff_for_figure.csv',
                        help="The path to the annotation df")
    parser.add_argument("--pat_name", type=str, default='sub-001',
                        help="Subject name (e.g., 'sub-001')")
    parser.add_argument("--data_dir", type=str, default='/Users/giacomopreti/Desktop/VBT/sub-001/processed/',
                        help="Path to the base data directory")
    parser.add_argument("--hemisphere", type=str, default='rh',
                        help="Hemisphere: 'lh' or 'rh'")
    parser.add_argument("--colormap", type=str, default='viridis',
                        help="Colormap to use for visualization")

    args = parser.parse_args()

    dk_plot_annotation_surface(
        title=args.title,
        df_annot_dir=args.df_annot_dir,
        pat_name=args.pat_name,
        data_dir=args.data_dir,
        hemisphere=args.hemisphere,
        colormap=args.colormap
    )
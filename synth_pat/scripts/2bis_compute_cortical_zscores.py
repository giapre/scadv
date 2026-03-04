import json
import sys
import os
import numpy as np
import pandas as pd
from paths import Paths

freesurfer_dir = "/data/core-psy-archive/data/PRONIA/test_vbt_pipe/vbt_derivatives/freesurfer"
resources_dir = Paths.RESOURCES

with open(f"{Paths.RESOURCES}/cortical_GAMLSS_coeffs.json", "r") as f:
    models = json.load(f)
print(models)

# Function to get BCTo parameters
def bcto_parameters(sex, subregion, age):
    coefs = models[sex][subregion]
    
    # Handle missing coefficients 
    def safe_exp(val):
        return np.exp(val) if val is not None else np.nan
    
    def safe_val(val):
        return val if val is not None else np.nan
    
    mu    = safe_exp(coefs['mu_intercept']) + safe_val(coefs['mu_age']) * age if coefs['mu_intercept'] is not None else np.nan
    sigma = safe_exp(coefs['sigma_intercept']) + safe_val(coefs['sigma_age']) * age if coefs['sigma_intercept'] is not None else np.nan
    nu    = safe_val(coefs['nu_intercept']) + safe_val(coefs['nu_age']) * age
    tau   = safe_exp(coefs['tau_intercept']) + safe_val(coefs['tau_age']) * age if coefs['tau_intercept'] is not None else np.nan
    
    return mu, sigma, nu, tau

# Function to compute Z-score
def bcto_z(y, sex, subregion, age):
    mu, sigma, nu, tau = bcto_parameters(sex, subregion, age)
    
    if np.isnan(mu) or np.isnan(sigma) or np.isnan(nu):
        return np.nan  # Cannot compute z-score if parameters missing
    
    if abs(nu) > 1e-6:
        z = ((y / mu)**nu - 1) / (nu * sigma)
    else:
        z = np.log(y / mu) / sigma
    return z

# Example usage
# z_score = bcto_z(y=2.5, sex='male', subregion='Region_1', age=25)
# print(z_score)

subj = sys.argv[1]
pipe_dir = f'{freesurfer_dir}/{subj}/pipe'
thickess_file = f'{pipe_dir}/gray_matter_thickness.csv'

if not os.path.exists(thickess_file):
    print(f"Skipping {subj}: no cortical thickess available at {thickess_file}!")
    sys.exit(0)

demo_and_thick = pd.read_csv(thickess_file, index_col='SubjectID')

output_name = f'{freesurfer_dir}/{subj}/pipe/cortical_thick_zscores.csv'

if os.path.exists(output_name):
    print(f"Skipping {subj}: file already exists")
    sys.exit(0)

# Compute the z score for each brain region
sex = 'Male' if demo_and_thick.loc[subj, 'SEX'] == 1 else 'Female'
age = demo_and_thick.loc[subj, 'AGE_T0']
remission = demo_and_thick.loc[subj, 'Remission']
demo_and_thick.drop(columns=['SEX', 'AGE_T0', 'Remission'], inplace=True)
reg_z = {}
for col in demo_and_thick.columns:
    y = demo_and_thick.loc[subj, col]
    reg_z.update({col: bcto_z(y, sex, col, age)})

print(reg_z)
reg_z_df = pd.DataFrame([reg_z])
reg_z_df['AGE_T0'] = age
reg_z_df['SEX'] = sex
reg_z_df['Remission'] = remission
reg_z_df['SubjectID'] = subj
reg_z_df.to_csv(output_name, index=False)
print(f'File saved at {output_name}')


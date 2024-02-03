import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

dataset = nc.Dataset('./ECCO_ocean_bolus/ECCO_OCEAN_BOLUS_VELOCITY_SNIPPET.nc')

for var_name, variable in dataset.variables.items():
    print(f"Variable: {var_name}")
    print(f"Shape: {variable.shape}")
    print(f"Attributes: {variable.__dict__}")
    print('------------------------\n\n')

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

dataset = nc.Dataset('./YoMaHa_currents/yomaha_currents.nc')

# for var_name, variable in dataset.variables.items():
#     print(f"Variable: {var_name}")
#     print(f"Shape: {variable.shape}")
#     print(f"Attributes: {variable.__dict__}")
#     print('------------------------\n\n')

latitude = dataset.variables['LATITUDE'][:]
longitude = dataset.variables['LONGITUDE'][:]
u_component = dataset.variables['U'][:]
v_component = dataset.variables['V'][:]

for i in range(len(u_component)):
  for j in range(len(u_component[i])):
    if u_component[i][j] == 99999.0:
      u_component[i][j] = 0

for i in range(len(v_component)):
  for j in range(len(v_component[i])):
    if v_component[i][j] == 99999.0:
      v_component[i][j] = 0

# Close the dataset
dataset.close()

# Create a meshgrid for the latitude and longitude
lon, lat = np.meshgrid(longitude, latitude)

# Plot the vector field using quiver
plt.figure(figsize=(10, 6))
plt.quiver(lon, lat, u_component, v_component,  color='blue', width=0.002)
plt.title('Vector Field Plot')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
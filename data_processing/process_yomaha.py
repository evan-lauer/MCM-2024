import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import csv

dataset = nc.Dataset('./YoMaHa_currents/yomaha_currents.nc')

for var_name, variable in dataset.variables.items():
    print(f"Variable: {var_name}")
    print(f"Shape: {variable.shape}")
    print(f"Attributes: {variable.__dict__}")
    print('------------------------\n\n')

latitude = dataset.variables['LATITUDE'][:]
longitude = dataset.variables['LONGITUDE'][:]
u_component = dataset.variables['U'][:]
v_component = dataset.variables['V'][:]

u_unc = dataset.variables['U_Unc'][:]
v_unc = dataset.variables['V_Unc'][:]

print(u_component)

for i in range(len(u_component)):
  for j in range(len(u_component[i])):
    if u_component[i][j] == 99999.0:
      u_component[i][j] = None

for i in range(len(v_component)):
  for j in range(len(v_component[i])):
    if v_component[i][j] == 99999.0:
      v_component[i][j] = None


for i in range(len(u_unc)):
  for j in range(len(u_unc[i])):
    if u_unc[i][j] == 99999.0:
      u_unc[i][j] = None

for i in range(len(v_unc)):
  for j in range(len(v_unc[i])):
    if v_unc[i][j] == 99999.0:
      v_unc[i][j] = None

data = [['latitude','longitude','velocity_eastward','velocity_northward','uncertainty_eastward','uncertainty_northward']]



# for lat in range(len(latitude)):
#   for long in range(len(longitude)):
#     if u_component[lat][long] != None and u_component[lat][long] != 0.0:
#       data.append([latitude[lat], longitude[long], u_component[lat][long], v_component[lat][long], u_unc[lat][long], v_unc[lat][long]])

# with open('./data_processing/yomaha_uncertainty.csv','w', newline='') as file:
#   csv_writer = csv.writer(file)
#   csv_writer.writerows(data)


# Close the dataset
dataset.close()

# Create a meshgrid for the latitude and longitude
# lon, lat = np.meshgrid(longitude, latitude)

# # Plot the vector field using quiver
# plt.figure(figsize=(10, 6))
# plt.quiver(lon, lat, u_component, v_component,  color='blue', width=0.002)
# plt.title('Vector Field Plot')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()
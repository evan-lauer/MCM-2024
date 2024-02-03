import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import csv

horiz_currents_filename = 'cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_1706987874202.nc'
vert_currents_filename = 'cmems_mod_glo_phy-wcur_anfc_0.083deg_P1D-m_1706987903452.nc'

horiz_dataset = nc.Dataset('./Copernicus_currents/' + horiz_currents_filename)
vert_dataset = nc.Dataset('./Copernicus_currents/' + vert_currents_filename)


eastward_component = horiz_dataset.variables['uo'][:][0]
northward_component = horiz_dataset.variables['vo'][:][0]
vertical_component = vert_dataset.variables['wo'][:][0]

depth_array = horiz_dataset.variables['depth'][:]
longitude_array = horiz_dataset.variables['longitude'][:]
latitude_array = horiz_dataset.variables['latitude'][:]


# for var_name, variable in vert_dataset.variables.items():
#   print(f"Variable: {var_name}")
#   print(f"Shape: {variable.shape}")
#   print(f"Attributes: {variable.__dict__}")

# for var_name, variable in horiz_dataset.variables.items():
#   print(f"Variable: {var_name}")
#   print(f"Shape: {variable.shape}")
#   print(f"Attributes: {variable.__dict__}")


def slice_along_longitude_line(longitude, vector_component):
  # Find the index of the given longitude in the longitude array
  longitude_index = get_nearest_index(longitude, longitude_array)
  # Return array[depth][latitude]
  slice = []
  for d in range(len(depth_array)):
    layer = []
    for l in range(len(latitude_array)):
      layer.append(vector_component[d][l][longitude_index])
    slice.append(layer)
  return slice
  
# Assumes that array is sorted in ascending order
def get_nearest_index(value, array):
  if value <= array[0]:
    return 0
  for i in range(len(array)-1):
    if array[i] <= value and value <= array[i+1]:
      return i
  return len(array) - 1

print(get_nearest_index(2000, depth_array))
print(depth_array[get_nearest_index(2000,depth_array)])

print('------------------\n\n\n\n\n')

eastward_slice = eastward_component[39]
northward_slice = northward_component[39]
vertical_slice = vertical_component[39]


magnitude_slice = [[0, *longitude_array]]


for i in range(len(latitude_array)):
  latitude_line = [latitude_array[i]]
  for j in range(len(longitude_array)):
    latitude_line.append(np.linalg.norm([eastward_slice[i][j],northward_slice[i][j],vertical_slice[i][j]]))
  magnitude_slice.append(latitude_line)

print(magnitude_slice)


with open('./data_processing/copernicus_2000m_currents.csv','w', newline='') as file:
  csv_writer = csv.writer(file)
  csv_writer.writerows(magnitude_slice)


# # Create a meshgrid for the latitude and longitude
# lon, lat = np.meshgrid(longitude_array, latitude_array)

# # Plot the vector field for the surface currents at depth ~2000m
# plt.figure(figsize=(10, 6))
# plt.quiver(lon, lat, eastward_component[39], northward_component[39],  color='blue', width=0.002)
# plt.title('Vector Field Plot')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()


# Create a meshgrid for the latitude and longitude
# dep, lat = np.meshgrid(depth_array, latitude_array)

# # Plot the vector field for the surface currents at depth 0
# plt.figure(figsize=(10, 6))
# plt.quiver(dep, lat, slice_along_longitude_line(21, vertical_component), slice_along_longitude_line(21, northward_component),  color='blue', width=0.002)
# plt.title('Vector Field Plot')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()

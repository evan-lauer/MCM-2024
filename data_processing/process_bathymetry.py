import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

# This file has data for elevation as a function of latitude and longitude
# If (latitude, longitude) is on land, it will be a positive number.
# If (latitude, longitude) is in the ocean, it will be a negative number.
# Resolution := 450 meters (15 arc seconds)
#
# Elevation refers to the elevation (in meters) at the center of each grid cell.
FILENAME = 'target_area_bathymetry.nc'

dataset = nc.Dataset('./oceanographic_data/GEBCO_bathymetry/' + FILENAME)

elevation_array = dataset.variables['elevation'][:]

latitude_array = dataset.variables['lat'][:]
longitude_array = dataset.variables['lon'][:]

def plot_ocean_floor():
  # Create a 2D meshgrid for latitude and longitude 
  lon, lat = np.meshgrid(longitude_array, latitude_array)

  # Plot the contour plot
  plt.figure(figsize=(10, 8))
  contour = plt.contourf(lon, lat, elevation_array, cmap='viridis')
  plt.colorbar(contour, label='Elevation (meters)')
  plt.title('Ocean Floor Elevation')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.show()

# Expects lat and long in degrees
def calculate_average_depth(start_lat, end_lat, start_long, end_long):
  num_data_points = 0
  avg = 0
  for i in range(len(latitude_array)):
     for j in range(len(longitude_array)):
        if start_lat <= latitude_array[i] and latitude_array[i] <= end_lat and start_long <= longitude_array[j] and longitude_array[j] <= end_long:
          if elevation_array[i][j] < 0: # Only include points that are underwater
            num_data_points += 1
            avg += (elevation_array[i][j] - avg) / num_data_points
            
  return avg

def calculate_land_proportion(start_lat, end_lat, start_long, end_long):
  num_land_points = 0
  total_points = 0
  for i in range(len(latitude_array)):
     for j in range(len(longitude_array)):
      if start_lat <= latitude_array[i] and latitude_array[i] <= end_lat and start_long <= longitude_array[j] and longitude_array[j] <= end_long:
        total_points += 1
        if elevation_array[i][j] >= 0: # Only include points that are on land
          num_land_points += 1
  return num_land_points / total_points


# print("Depth and land proportion of target region:")
# print(calculate_average_depth(37.5,39,20,21.5))
# print(calculate_land_proportion(37.5,39,20,21.5))


# print("Depth and land proportion of Korea region:")
# print(calculate_average_depth(36.8,38.3,124.5,126))
# print(calculate_land_proportion(36.8,38.3,124.5,126))

# print()
# print(min(latitude_array))
# print(max(latitude_array))
# print()
# print(min(longitude_array))
# print(max(longitude_array))



# Print the variables and shapes of each variable
# for var_name, variable in dataset.variables.items():
#     print(f"Variable: {var_name}")
#     print(f"Shape: {variable.shape}")
#     print(f"Attributes: {variable.__dict__}")

plot_ocean_floor()
# Close the NetCDF file when done
dataset.close()

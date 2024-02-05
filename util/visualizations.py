import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

# Assumes that array is sorted in ascending order
def get_nearest_index_hi_or_lo(value, array, smaller):
  if value <= array[0]:
    return 0
  for i in range(len(array)-1):
    if array[i] <= value and value <= array[i+1]:
      if smaller: # If smaller, then return the index of the last smaller element
        return i
      else:
        return i+1 # If larger, then return the index of the first larger element
  return len(array) - 1

# Assumes that array is sorted in ascending order
def get_nearest_index_closest_val(value, array):
  if value <= array[0]:
    return 0
  for i in range(len(array)-1):
    if array[i] <= value and value <= array[i+1]:
      if value - array[i] < array[i+1] - value:
        return i
      else:
        return i+1
  return len(array) - 1

def plot_ocean_floor(lat_start, lat_end, long_start, long_end):
  # This file has data for elevation as a function of latitude and longitude
  # If (latitude, longitude) is on land, it will be a positive number.
  # If (latitude, longitude) is in the ocean, it will be a negative number.
  # Resolution := 450 meters (15 arc seconds)
  #
  # Elevation refers to the elevation (in meters) at the center of each grid cell.
  FILENAME = 'target_area_bathymetry.nc'
  dataset = nc.Dataset('./GEBCO_bathymetry/' + FILENAME)
  # Create a 2D meshgrid for latitude and longitude
  latitude_array = dataset.variables['lat'][:]
  longitude_array = dataset.variables['lon'][:]
  # Define the indices of our target area
  lat_lower_bound = get_nearest_index_hi_or_lo(lat_start, latitude_array, smaller=True)
  lat_upper_bound = get_nearest_index_hi_or_lo(lat_end, latitude_array, smaller=False)
  long_lower_bound = get_nearest_index_hi_or_lo(long_start, longitude_array, smaller=True)
  long_upper_bound = get_nearest_index_hi_or_lo(long_end, longitude_array, smaller=False)

  elevation_array = dataset.variables['elevation'][lat_lower_bound:lat_upper_bound, long_lower_bound:long_upper_bound]

  lon, lat = np.meshgrid(longitude_array[long_lower_bound:long_upper_bound], latitude_array[lat_lower_bound:lat_upper_bound])

  # Plot the contour plot
  plt.figure(figsize=(10, 8))
  contour = plt.contourf(lon, lat, elevation_array, cmap='viridis')
  plt.colorbar(contour, label='Elevation (meters)')
  plt.title('Ocean Floor Elevation')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.show()

# Example usage, for plotting the whole target area
# plot_ocean_floor(37,40,17.5,21.5)
# Example usage, for plotting a smaller subset of the target area
# plot_ocean_floor(38,38.25, 18,18.25)
  

def plot_current_at_depth(depth, lat_start, lat_end, long_start, long_end, directory):

  horiz_currents_filename = 'cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_1706987874202.nc'

  horiz_dataset = nc.Dataset('./Copernicus_currents/' + horiz_currents_filename)


  depth_array = horiz_dataset.variables['depth'][:]
  longitude_array = horiz_dataset.variables['longitude'][:]
  latitude_array = horiz_dataset.variables['latitude'][:]

  lat_lower_bound = get_nearest_index_hi_or_lo(lat_start, latitude_array, smaller=True)
  lat_upper_bound = get_nearest_index_hi_or_lo(lat_end, latitude_array, smaller=False)
  long_lower_bound = get_nearest_index_hi_or_lo(long_start, longitude_array, smaller=True)
  long_upper_bound = get_nearest_index_hi_or_lo(long_end, longitude_array, smaller=False)

  
  depth_index = get_nearest_index_closest_val(depth, depth_array)

  eastward_component = horiz_dataset.variables['uo'][:,:,lat_lower_bound:lat_upper_bound,long_lower_bound:long_upper_bound][0][depth_index]
  northward_component = horiz_dataset.variables['vo'][:,:,lat_lower_bound:lat_upper_bound,long_lower_bound:long_upper_bound][0][depth_index]
  
  # Create a meshgrid for the latitude and longitude
  lon, lat = np.meshgrid(longitude_array[long_lower_bound:long_upper_bound], latitude_array[lat_lower_bound:lat_upper_bound])

  # Plot the vector field for the surface currents at depth ~2000m
  plt.figure(figsize=(10, 6))
  plt.quiver(lon, lat, eastward_component, northward_component,  color='blue', width=0.002)
  plt.title('Horizontal Currents at Depth = ' + str(depth_array[depth_index]) + 'm')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  # plt.show()
  plt.savefig('./trial_data/' + directory + '/current_slice.png', bbox_inches='tight')

  

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from util.visualizations import get_nearest_index_closest_val


# This file has data for elevation as a function of latitude and longitude
# If (latitude, longitude) is on land, it will be a positive number.
# If (latitude, longitude) is in the ocean, it will be a negative number.
# Resolution := 450 meters (15 arc seconds)
#
# Elevation refers to the elevation (in meters) at the center of each grid cell.
FILENAME = 'target_area_bathymetry.nc'

dataset = nc.Dataset('./GEBCO_bathymetry/' + FILENAME)

elevation_array = dataset.variables['elevation'][:]

latitude_array = dataset.variables['lat'][:]
longitude_array = dataset.variables['lon'][:]


# Watch out! depth > 0 means we are in the ocean. elevation > 0 means we are on land
def is_valid_ocean(latitude, longitude, depth):
  if latitude < 37 or 40 < latitude or longitude < 17.5 or longitude > 21.5:
    raise ValueError("Error, current location is outside the target area")
    
  lat_i = get_nearest_index_closest_val(latitude, latitude_array)
  long_i = get_nearest_index_closest_val(longitude, longitude_array)
  # print(latitude_array[lat_i]) 
  # print(longitude_array[long_i])
  if elevation_array[lat_i][long_i] >= 0:
    return False # If lat, long is on land, then we cannot be in a valid ocean
  elif elevation_array[lat_i][long_i] <= (-1 * depth):
    return True # If the elevation is lower than the depth, return true
  return False

# print(is_valid_ocean(39.1, 18.6, 1420))
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

# This file has data for elevation as a function of latitude and longitude
# If (latitude, longitude) is on land, it will be a positive number.
# If (latitude, longitude) is in the ocean, it will be a negative number.
# Resolution := 450 meters (15 arc seconds)
#
# Elevation refers to the elevation (in meters) at the center of each grid cell.
dataset = nc.Dataset('./GEBCO_bathymetry/GEBCO_bathymetry.nc')

elevation_array = dataset.variables['elevation'][:]

# Create a 2D meshgrid for latitude and longitude
latitude_array = dataset.variables['lat'][:]
longitude_array = dataset.variables['lon'][:]
lon, lat = np.meshgrid(longitude_array, latitude_array)

# Plot the contour plot
plt.figure(figsize=(10, 8))
contour = plt.contourf(lon, lat, elevation_array, cmap='viridis')
plt.colorbar(contour, label='Elevation (meters)')
plt.title('Ocean Floor Elevation')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Close the NetCDF file when done
dataset.close()

# Print the variables and shapes of each variable
# for var_name, variable in dataset.variables.items():
#     print(f"Variable: {var_name}")
#     print(f"Shape: {variable.shape}")
#     print(f"Attributes: {variable.__dict__}")
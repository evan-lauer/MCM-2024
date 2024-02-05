import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter


def get_info_from_csv(path):
  data_array = np.genfromtxt(path, delimiter=',', skip_header=1)

  num_trials = round(max(data_array[:,0]) + 1)
  num_steps = data_array[-1,1] + 1

  slices = np.ndarray(shape=(5, num_trials, 3)) # 5 slices, # trials, 3 vector components

  # We want 5 slices, one for each day that has passed.
  # So we want [0-23] hours, [24-47] hours, [48-71] hours, [72-95] hours, and [96-119] hours
  for row in data_array:
    trial_val = round(row[0])
    hour_val = round(row[1])
    if hour_val == 23:
      slices[0][trial_val] = row[2:]
    elif hour_val == 47:
      slices[1][trial_val] = row[2:]
    elif hour_val == 71:
      slices[2][trial_val] = row[2:]
    elif hour_val == 95:
      slices[3][trial_val] = row[2:]
    elif hour_val == 119:
      slices[4][trial_val] = row[2:]
  return tuple([slices, num_trials, num_steps])

# Return a heatmap and indices into each bin, for each slice
def get_heatmap_and_indices(slices, save_figures):
  # Specify the desired bin width for latitude and longitude
  bin_width_lat = 0.005
  bin_width_lon = 0.005

  heatmaps = []
  heatmap_indices = []

  for i in range(5):
    # Calculate the bin edges dynamically
    lat_min, lat_max = np.min(slices[i][:, 0]), np.max(slices[i][:, 0])
    lon_min, lon_max = np.min(slices[i][:, 1]), np.max(slices[i][:, 1])

    lat_edges = np.arange(lat_min, lat_max + bin_width_lat, bin_width_lat)
    lon_edges = np.arange(lon_min, lon_max + bin_width_lon, bin_width_lon)

    # Create 2D histogram with dynamically determined bin edges
    hist, _, _ = np.histogram2d(slices[i][:, 0], slices[i][:, 1], bins=[lat_edges, lon_edges])

    # Normalize to get probabilities
    hist = hist / np.sum(hist)

    # Create a meshgrid for plotting
    X, Y = np.meshgrid(lat_edges, lon_edges)
    pcm = plt.pcolormesh(X, Y, hist.T, cmap='magma', norm=LogNorm())
    cbar = plt.colorbar(pcm, label='Probability')
    # Format colorbar labels to display in decimal format
    cbar.formatter = ScalarFormatter(useMathText=False)
    cbar.formatter.set_scientific(False)
    cbar.formatter.set_powerlimits((-1,1))
    cbar.update_ticks()


    # Plot the heat map
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Heat Map of Latitude and Longitude Data')

    heatmaps.append(pcm.get_array())

    # Find indices for each bin
    lat_indices = np.digitize(slices[i][:, 0], lat_edges)
    lon_indices = np.digitize(slices[i][:, 1], lon_edges)

    # Combine indices into a single array
    bin_indices = np.column_stack((lat_indices, lon_indices))
    heatmap_indices.append(bin_indices)
    if save_figures:
      plt.savefig('./slices/slice' + str(i + 1) + '.png')
  return tuple([heatmaps, heatmap_indices])

# TODO: Figure out the standard deviation of the depths
# within each bin. this is a proxy for how likely you
# are to find it if you look in that spot

# TODO: Make this work for all 5 slices

filepath_03 = './trial_data/30trials_38.0772144lat_19.8620142long_2200depth/data.csv'

slices, NUM_TRIALS, NUM_STEPS = get_info_from_csv(filepath_03)
heatmaps, heatmap_indices = get_heatmap_and_indices(slices, True)


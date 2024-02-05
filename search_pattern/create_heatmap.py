import numpy as np
import matplotlib.pyplot as plt
import math
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
  lat_edge_array = []
  lon_edge_array = []

  for i in range(5):
    # Calculate the bin edges dynamically
    lat_min, lat_max = np.min(slices[i][:, 0]), np.max(slices[i][:, 0])
    lon_min, lon_max = np.min(slices[i][:, 1]), np.max(slices[i][:, 1])

    lat_edges = np.arange(lat_min, lat_max + bin_width_lat, bin_width_lat)
    lon_edges = np.arange(lon_min, lon_max + bin_width_lon, bin_width_lon)

    # Create 2D histogram with dynamically determined bin edges
    hist, _, _ = np.histogram2d(slices[i][:, 1], slices[i][:, 0], bins=[lon_edges, lat_edges])

    # Normalize to get probabilities
    hist = hist / np.sum(hist)

    # Create a meshgrid for plotting
    X, Y = np.meshgrid(lon_edges,lat_edges)
    pcm = plt.pcolormesh(X, Y, hist.T, cmap='magma', norm=LogNorm())
    cbar = plt.colorbar(pcm, label='Probability')
    # Format colorbar labels to display in decimal format
    cbar.formatter = ScalarFormatter(useMathText=False)
    cbar.formatter.set_scientific(False)
    cbar.formatter.set_powerlimits((-1,1))
    cbar.update_ticks()


    # Plot the heat map
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Heat Map of Latitude and Longitude Data')

    heatmaps.append(pcm.get_array())

    lat_edge_array.append(lat_edges)
    lon_edge_array.append(lon_edges)

    # Find indices for each bin
    lat_indices = np.digitize(slices[i][:, 0], lat_edges)
    lon_indices = np.digitize(slices[i][:, 1], lon_edges)

    # Combine indices into a single array
    bin_indices = np.column_stack((lat_indices, lon_indices))
    heatmap_indices.append(bin_indices)
    if save_figures:
      plt.savefig('./slices/slice' + str(i + 1) + '.png')
    plt.close()
  return tuple([heatmaps, heatmap_indices, lat_edge_array, lon_edge_array])

def calculate_bucket_sds(heatmap_indices, heatmaps, slices):
  std_deviations = []
  for s in range(5):
    # For every slice, we need a new array the same shape as the heatmap
    # at that slice
    std_deviations_slice = np.full(shape=(len(heatmaps[s]),len(heatmaps[s][0])),fill_value=None)
    
    for i in range(len(heatmap_indices[s])):
      heatmap_index = heatmap_indices[s][i]
      depth_at_index = slices[s][i][2]
      if std_deviations_slice[heatmap_index[0]-1, heatmap_index[1]-1] is None:
        std_deviations_slice[heatmap_index[0]-1, heatmap_index[1]-1] = [depth_at_index]
      else:
        std_deviations_slice[heatmap_index[0]-1, heatmap_index[1]-1].append(depth_at_index)
    for i in range(len(std_deviations_slice)):
      for j in range(len(std_deviations_slice[0])):
        if std_deviations_slice[i][j] is not None:
          std_deviations_slice[i][j] = np.std(std_deviations_slice[i][j])
    std_deviations.append(std_deviations_slice)
  return std_deviations

# Calculate Euclidean distance super fast
def eudis5(v1, v2):
  dist = [(a - b)**2 for a, b in zip(v1, v2)]
  dist = math.sqrt(sum(dist))
  return dist

# MAIN:
filepath_03 = './trial_data/200trials_38.0772144lat_19.8620142long_2200depth/data.csv'  
DEPTH = 100
DEPTH_FACTOR = abs((DEPTH / 5000) - 1)
# Path history oldest -> newest
# './trial_data/30trials_38.0772144lat_19.8620142long_2200depth/data.csv'

slices, NUM_TRIALS, NUM_STEPS = get_info_from_csv(filepath_03)
heatmaps, heatmap_indices, lat_edge_array, lon_edge_array = get_heatmap_and_indices(slices, True)
std_deviations = calculate_bucket_sds(heatmap_indices, heatmaps, slices)


# Turn all occupied indices into a list
indices_list = []
for i in range(5):
  indices_list.append(np.array(list(set(map(tuple, heatmap_indices[i]-1)))))

std_deviations_list = []
# Turn all nonzero standard deviations into a list in that same order
for i in range(5):
  std_devs = np.ndarray(shape=len(indices_list[i]))
  for j in range(len(indices_list[i])):
    std_devs[j] = std_deviations[i][indices_list[i][j][0]][indices_list[i][j][1]]
  std_deviations_list.append(std_devs)
  
# Turn all visited cells into a list in that same order
visited_list = []
for i in range(5):
  visited_list.append(np.full(shape=len(indices_list[i]), fill_value=False))

# Turn all probabilities into a list in that same order
probabilities_list = []
for i in range(5):
  probabilities = np.ndarray(shape=len(indices_list[i]))
  for j in range(len(indices_list[i])):
    probabilities[j] = heatmaps[i][indices_list[i][j][0]][indices_list[i][j][1]]
  probabilities_list.append(probabilities)

def run_search_algorithm(indices_list, std_deviations_list, visited_list, probabilities_list):
  histograms = []
  for i in range(5):
    # Time remaining! Only one day to search!
    time_remaining = 24 # hours
    SEARCH_TIME = 2 # hours
    SEARCH_SPEED = 3704 # meters per hour (this is 10 knots)
    METERS_PER_BIN = 555
    # Get 2d index of the highest probability node
    histogram = []
    start_node_index = np.where(probabilities_list[i] == np.max(probabilities_list[i]))[0][0]
    current_node_index = start_node_index
    num_nodes_visited = 1
    visited_list[i][current_node_index] = True
    time_remaining -= SEARCH_TIME
    histogram.append([(indices_list[i][start_node_index][0],indices_list[i][start_node_index][1]), probabilities_list[i][start_node_index], max(0.7 * DEPTH_FACTOR - 0.007 * std_deviations_list[i][start_node_index], 0.2), time_remaining])
    # Until we're done:
    while num_nodes_visited < len(indices_list[i]) and time_remaining > 0:
      # For every neighbor (which is every node):
      best_candidate_value = -1
      best_candidate_index = -1
      best_candidate_travel_time = -1
      best_candidate_p_it_is_here = -1
      best_candidate_p_we_find_it = -1
      for neighbor in range(len(indices_list[i])):
        # If already visited, then skip
        if visited_list[i][neighbor]:
          continue

        # Distance to the neighbor in meters
        # Example: if the neighbor is 3 bins north, then this evaluates to ~1650 meters
        travel_distance = eudis5(indices_list[i][current_node_index], indices_list[i][neighbor]) * METERS_PER_BIN
        # Time required to travel to the neighbor, in hours
        travel_time = travel_distance / SEARCH_SPEED

        if time_remaining - travel_time - SEARCH_TIME < 0:
          # If traveling to & searching this neighbor would cause us to run out of time,
          # then we can skip it.
          continue
        
        # Otherwise calculate the proportion
        p_submarine_is_here = probabilities_list[i][neighbor]
        p_we_find_it = max(0.70 * DEPTH_FACTOR - 0.007 * std_deviations_list[i][neighbor], 0.2)
        
        # Value ratio
        value_ratio = (p_submarine_is_here * p_we_find_it) / travel_time

        if value_ratio > best_candidate_value:
          best_candidate_value = value_ratio
          best_candidate_index = neighbor
          best_candidate_travel_time = travel_time
          best_candidate_p_it_is_here = p_submarine_is_here
          best_candidate_p_we_find_it = p_we_find_it

      if best_candidate_index == -1:
        # If this is the case, then we can't visit any neighbors without running out of time.
        break
      
      # If we get here, we assume we've found the best candidate, so we travel to and
      # visit the candidate
      time_remaining -= best_candidate_travel_time
      time_remaining -= SEARCH_TIME
      visited_list[i][best_candidate_index] = True
      num_nodes_visited += 1
      histogram.append([(indices_list[i][best_candidate_index][0],indices_list[i][best_candidate_index][1]), best_candidate_p_it_is_here, best_candidate_p_we_find_it, time_remaining])

      current_node_index = best_candidate_index
    histograms.append(histogram)
  return histograms

def probability_of_find(histograms):
  p_never_found = 1
  for histogram in histograms:
    probability_of_not_finding_it = 1
    for entry in histogram:
      probability_of_not_finding_it *= (1 - entry[1] * entry[2])
    p_never_found *= probability_of_not_finding_it
  return 1 - p_never_found

def plot_search_path(histograms, i):
  coordinates = np.ndarray(shape=(len(histograms[i]), 2))
  for index in range(len(histograms[i])):
    coordinates[index][0] = lat_edge_array[i][histograms[i][index][0][0]] + .0025
    coordinates[index][1] = lon_edge_array[i][histograms[i][index][0][1]] + .0025
  # Create a heatmap for the probability array
  heatmap = plt.imshow(heatmaps[i], cmap='binary', origin='lower', extent=(min(lon_edge_array[i]), max(lon_edge_array[i]), min(lat_edge_array[i]), max(lat_edge_array[i])), alpha=0.8,vmin=0.00001)

  # Overlay scatter plot with lines connecting successive points
  plt.scatter(coordinates[:, 1], coordinates[:, 0], color='black', marker='', label='Coordinates',linewidths=1)

  # Connect successive points with lines
  for index in range(1, len(coordinates)):
    plt.plot([coordinates[index-1][1], coordinates[index][1]], [coordinates[index-1][0], coordinates[index][0]], color='black', label='_nolegend_')

  # Adding labels and title
  plt.xlabel('Latitude')
  plt.ylabel('Longitude')
  plt.title('Heatmap with Overlayed Scatter Plot')
  cbar = plt.colorbar(heatmap, label='Probability')


  # Display legend
  plt.legend()

  # Display the plot
  plt.show()


histograms = run_search_algorithm(indices_list,std_deviations_list,visited_list,probabilities_list)
# print(histograms[3])
# print(visited_list[3])
plot_search_path(histograms, 2)
print(probability_of_find(histograms))



import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import os
from util.convert_lat_long import add_meters_east, add_meters_north
from util.randomize_vector import compute_random_vector
from util.visualizations import get_nearest_index_closest_val, plot_current_at_depth
from util.record_trials import write_trial_data_to_csv
from util.determine_ocean_floor import is_valid_ocean


# Initialize our data arrays
#
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

np.random.seed(1)

# Initialize relevant constants
#
STEP_SIZE = 60 * 60              # Step size: seconds
SCALE_FACTOR = STEP_SIZE         # We need to scale up our m/s velocities by the number of steps
DURATION = 60 * 60 * 24 * 5      # Duration: seconds (make sure DURATION is divisible by STEP_SIZE)
NUM_STEPS = int(DURATION / STEP_SIZE) # Number of steps
NUM_TRIALS = 30                   # Number of submersibles to simulate
SAVE_PLOTS = True
SHOW_PLOTS = False
initial_coordinates = [37.8948704,20.4061121, 1000] # Latitude must lie within 37-40, longitude within 17.5-21.5
lat_long_variation = 0.0 # 1/125th of a degree latitude (approximately 900m)
depth_variation = 10
# Generate a series of random starting coordinates, varying uniformly from += lat_long variation
STARTING_LATITUDES = np.random.uniform(initial_coordinates[0] - lat_long_variation, initial_coordinates[0] + lat_long_variation, size=NUM_TRIALS) 
STARTING_LONGITUDES = np.random.uniform(initial_coordinates[1] - lat_long_variation, initial_coordinates[1] + lat_long_variation, size=NUM_TRIALS)
# Generate a series of random starting depths, varying uniformly from += depth_variation
STARTING_DEPTHS = np.random.uniform(initial_coordinates[2] - depth_variation, initial_coordinates[2] + depth_variation, size=NUM_TRIALS)  
# When indexing into the data, each trial will index slightly differently
# to represent the fact that, while the general pattern of currents is likely
# correct, it may have lateral variation  
OFFSETS = np.random.normal(0,.0833, size=NUM_TRIALS)    



# Initialize variables!
#
displacement_vectors = np.ndarray(shape=(NUM_STEPS, NUM_TRIALS, 3), dtype=np.float64) # for each submersible, there is a xyz component of displacement. Format: (m_north, m_east, m_vert)
displacement_vectors.fill(0)
coordinate_vectors = np.ndarray(shape=(NUM_STEPS, NUM_TRIALS, 3)) # for each time step, for each submersible, there is a lat and long. This is for graphing later

# For each trial, for each time step: 
#   Generate a random value. This is centered at 0 with SD of 1, so it represents a Z score
gaussian_sample_matrix = np.random.normal(0, 1, size=(NUM_STEPS, NUM_TRIALS))

# print(np.linalg.norm(compute_random_vector(.0002, 1)))

is_beached = np.full(shape=NUM_TRIALS, fill_value=False)

for s in range(NUM_STEPS):
  gaussian_sample_step = gaussian_sample_matrix[s]
  for t in range(NUM_TRIALS):
    if is_beached[t]:
      continue
    # For each trial:
    # Calculate the current lat and long based on the displacement
    latitude, longitude = add_meters_north(STARTING_LATITUDES[t], STARTING_LONGITUDES[t], displacement_vectors[s][t][0])
    latitude, longitude = add_meters_east(latitude, longitude, displacement_vectors[s][t][1])
    # Calculate the current depth based on the displacement
    depth = STARTING_DEPTHS[t] + displacement_vectors[s][t][2]
    # Record the current coordinate vector, BEFORE we update it
    coordinate_vectors[s][t] = np.array([latitude, longitude, depth])
    # Get indices of the current position
    i_lat = get_nearest_index_closest_val(latitude + OFFSETS[t], latitude_array)
    i_long = get_nearest_index_closest_val(longitude + OFFSETS[t], longitude_array)
    i_depth = get_nearest_index_closest_val(depth + OFFSETS[t], depth_array)
    # Get appropriate velocity components from the data arrays
    v_north = northward_component[i_depth][i_lat][i_long]
    v_east = eastward_component[i_depth][i_lat][i_long]
    # This needs a negative sign b/c we are using a system where positive depth means deeper)
    v_vertical = -vertical_component[i_depth][i_lat][i_long] 

    # Check if any of the data is invalid (this means we're close enough to land that the
    # nearest data points puts us inside of land)
    if np.ma.is_masked(v_north) or np.ma.is_masked(v_east) or np.ma.is_masked(v_vertical):
      # If this is the case, then we need to determine (using high-res bathymetry) if we're 
      # actually in valid ocean
      if is_valid_ocean(latitude, longitude, depth) and s > 0:
        # If we are, then go back and reconstruct the velocity vector from the last time step
        # and apply it again
        if s == NUM_STEPS - 1:
          continue
        else:
          displacement_vectors[s+1][t] = displacement_vectors[s][t] + (displacement_vectors[s][t] - displacement_vectors[s-1][t])
        continue
      else:
        # If we aren't, then beach us
        is_beached[t] = True
        for remaining in range(s, NUM_STEPS):
          coordinate_vectors[remaining][t] = np.array([latitude, longitude, depth])
        # coordinate_vectors[s:NUM_STEPS - 1][t] = np.array([latitude, longitude, depth])
        # if s != NUM_STEPS - 1:
        #   displacement_vectors[s+1][t] = displacement_vectors[s][t]
        continue
    
    # Compute random vector, "scaled" to the magnitude of the velocity vector
    r_v_north, r_v_east, r_v_vertical = compute_random_vector(np.linalg.norm(np.array([v_north, v_east, v_vertical])), gaussian_sample_step[t])

    # print([v_north,v_east,v_vertical])
    # print([r_v_north,r_v_east,r_v_vertical])

    # Make sure depth cannot be negative
    new_displacement = displacement_vectors[s][t] + SCALE_FACTOR * np.array([v_north, v_east, v_vertical]) + SCALE_FACTOR * np.array([r_v_north, r_v_east, r_v_vertical])
    if STARTING_DEPTHS[t] + new_displacement[2] < 0:
      new_displacement[2] = 0
    if s == NUM_STEPS - 1:
      continue
    else:
      displacement_vectors[s+1][t] = new_displacement


def plot_from_above(coordinate_vectors):
  # Set up the map plot
  fig, ax = plt.subplots()
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.title('Simulation Positions')

  # Plot positions for each simulation
  for i in range(NUM_TRIALS):
    latitudes = coordinate_vectors[:, i, 0]
    longitudes = coordinate_vectors[:, i, 1]

    # Connect positions with lines
    ax.plot(longitudes, latitudes, marker='o', markersize=1, linewidth=1, label=f'Simulation {i+1}')

  # Add legend
  ax.legend()

  plt.show()
  

def plot_in_3d(coordinate_vectors, directory):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for i in range(NUM_TRIALS):
    latitudes = coordinate_vectors[:, i, 0]
    longitudes = coordinate_vectors[:, i, 1]
    depths = coordinate_vectors[:, i, 2]

    # Plot the trajectory line
    ax.plot(longitudes, latitudes, depths, label=f'Submersible {i + 1}')

    # Mark the start and end points with markers
    ax.scatter(longitudes[0], latitudes[0], depths[0], c='green', marker='o')  # Start point
    ax.scatter(longitudes[-1], latitudes[-1], depths[-1], c='red', marker='x')  # End point

  ax.invert_zaxis()
  # Set labels and title
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  ax.set_zlabel('Depth')

  plt.tick_params(axis='x', labelsize=8)
  plt.tick_params(axis='y', labelsize=8)
  plt.tick_params(axis='z', labelsize=8)

  

  if SHOW_PLOTS:
    plt.show()
    
  if SAVE_PLOTS:
    if not os.path.exists('./trial_data/' + DIRECTORY):
      os.makedirs('./trial_data/' + DIRECTORY)
    ax.view_init(elev=5, azim=90)
    plt.savefig('./trial_data/' + DIRECTORY + '/frontview.png', bbox_inches='tight')
    ax.view_init(elev=5, azim=0)
    plt.savefig('./trial_data/' + DIRECTORY + '/leftview.png', bbox_inches='tight')
    ax.view_init(elev=45, azim=45)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    plt.savefig('./trial_data/' + DIRECTORY + '/cornerview.png')
    ax.view_init(elev=90, azim=90)
    plt.savefig('./trial_data/' + DIRECTORY + '/topview.png', bbox_inches='tight')

DIRECTORY = str(NUM_TRIALS) + 'trials_' + str(initial_coordinates[0]) + 'lat_' + str(initial_coordinates[1]) + 'long_' + str(initial_coordinates[2]) + 'depth'
if SAVE_PLOTS:
  write_trial_data_to_csv(coordinate_vectors, NUM_TRIALS, NUM_STEPS, initial_coordinates)
plot_in_3d(coordinate_vectors, DIRECTORY)
if SAVE_PLOTS:
  plot_current_at_depth(initial_coordinates[2], 37,40,17.5,21.5, DIRECTORY)

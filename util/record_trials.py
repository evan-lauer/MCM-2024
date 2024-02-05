import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import csv
import os


def write_trial_data_to_csv(coordinate_vectors, NUM_TRIALS, NUM_STEPS, initial_coordinates):
  # This creates a unique file name depending on the starting parameters of the run
  DIRECTORY = str(NUM_TRIALS) + 'trials_' + str(initial_coordinates[0]) + 'lat_' + str(initial_coordinates[1]) + 'long_' + str(initial_coordinates[2]) + 'depth'

  rows = [['trial_number','time_elapsed','latitude','longitude','depth']]
  for trial in range(NUM_TRIALS):
    for step in range(NUM_STEPS):
      rows.append([trial, step, coordinate_vectors[step][trial][0],coordinate_vectors[step][trial][1],coordinate_vectors[step][trial][2]])

  if not os.path.exists('./trial_data/' + DIRECTORY):
    os.makedirs('./trial_data/' + DIRECTORY)
  with open('./trial_data/' + DIRECTORY + '/data.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(rows)

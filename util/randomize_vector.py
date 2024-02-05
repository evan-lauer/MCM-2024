import numpy as np

def generate_vector(magnitude):
    # Generate random values for x, y, and z
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    z = np.random.uniform(-.05, .05)

    # Create a vector from the random values
    vector = np.array([x, y, z])

    # Normalize the vector to have the specified magnitude
    normalized_vector = (magnitude / np.linalg.norm(vector)) * vector

    return tuple(normalized_vector)

# Define a few important helper functions
def get_vector_magnitude(vector):
  return np.sqrt(np.dot(vector,vector))

# Take the z score and calculate the actual random value according to the correct distribution
def compute_random_vector(velocity_magnitude, z_score):
  if velocity_magnitude <= 0:
     velocity_magnitude = .000001
  approximate_magnitude = 0.3 * velocity_magnitude - .3 * (velocity_magnitude ** 2)
  if approximate_magnitude <= 0:
     print("uh oh, zero vector")
     return tuple([0,0,0])
  randomized_magnitude = (approximate_magnitude / 2) + (approximate_magnitude) * z_score
  return generate_vector(randomized_magnitude)
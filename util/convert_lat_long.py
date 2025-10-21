import math


def add_meters_north(latitude, longitude, north_south_distance):
    # Earth's radius (approximately spherical Earth)
    earth_radius = 6378000  # in meters

    # Convert north/south distance to angular distance
    angular_distance_north_south = north_south_distance / earth_radius

    # Calculate new latitude
    new_latitude = latitude + math.degrees(angular_distance_north_south)

    return new_latitude, longitude


def add_meters_east(latitude, longitude, east_west_distance):
    # Earth's radius (approximately spherical Earth)
    earth_radius = 6378000  # in meters

    # Convert east/west distance to angular distance
    angular_distance_east_west = east_west_distance / \
        (earth_radius * math.cos(math.radians(latitude)))

    # Calculate new longitude
    new_longitude = longitude + math.degrees(angular_distance_east_west)

    return latitude, new_longitude

# Example usage:
# original_latitude = 40.7128  # Replace with the actual latitude of your object
# original_longitude = -74.006  # Replace with the actual longitude of your object

# new_latitude, new_longitude = add_meters_north(original_latitude, original_longitude, 766666)
# new_latitude, new_longitude = add_meters_east(new_latitude, new_longitude, -3799420)

# print(f"Original Latitude: {original_latitude}, Original Longitude: {original_longitude}")
# print(f"New Latitude: {new_latitude}, New Longitude: {new_longitude}")

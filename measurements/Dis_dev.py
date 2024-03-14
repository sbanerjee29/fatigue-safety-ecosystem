import numpy as np
from scipy.integrate import quad

# Convert km/h to m/s
def kmh_to_ms(speed_kmh):
    return speed_kmh * (1000 / 3600)

def deceleration_distance(S0, a):
    # Calculate the distance covered during deceleration
    integrand = lambda t: S0 * np.exp(-a * t)
    result, _ = quad(integrand, 0, np.inf)
    return result

def total_distance_steering_change(steering_angle, initial_speed_kmh, deceleration, distance_barrier_truck):
    # Convert initial speed to m/s
    initial_speed_ms = kmh_to_ms(initial_speed_kmh)

    # Convert steering angle to radians
    theta = np.radians(steering_angle)

    # Calculate lateral distance
    D_lane = distance_barrier_truck * np.tan(theta)

    # Calculate distance covered during deceleration
    D_deceleration = deceleration_distance(initial_speed_ms, deceleration)

    # Calculate total distance
    D_total = D_lane + D_deceleration

    return D_total

# Example usage
initial_speed_kmh = 70.0  # Replace with actual initial speed in km/h
deceleration = 0.1       # Replace with actual deceleration in m/s^2
distance_barrier_truck = 10.0  # Replace with actual distance in meters
steering_angles = [5, 40]  # Replace with actual steering angles in degrees

total_distances = []

for steering_angle in steering_angles:
    total_distance = total_distance_steering_change(steering_angle, initial_speed_kmh, deceleration, distance_barrier_truck)
    total_distances.append(total_distance)

print("Total distances:", total_distances)

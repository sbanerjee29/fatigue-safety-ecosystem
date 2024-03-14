import math
decay_rate = 0.1  


for initial_speed_mps in range(20, 2, -1):
    initial_speed_kph = initial_speed_mps * 3.6 

    def calculate_maximum_steering_angle(wheel_base, friction_coefficient, current_speed):    
        g = 9.81
        tan_theta = friction_coefficient * g * wheel_base / current_speed**2
        max_steering_angle = math.atan(tan_theta)
        return math.degrees(max_steering_angle)

    def calculate_horizontal_distance(distance_to_barrier, angle_degrees):
        angle_radians = math.radians(angle_degrees)
        horizontal_distance = distance_to_barrier * math.tan(angle_radians)
        return horizontal_distance

    wheel_base = 11.0  
    friction_coefficient = 0.8  

    time_elapsed = 0.0  

    while time_elapsed <= 10.0:  
        current_speed_kph = initial_speed_kph * math.exp(-decay_rate * time_elapsed)
        current_speed_mps = current_speed_kph / 3.6
        max_steering_angle = calculate_maximum_steering_angle(wheel_base, friction_coefficient, current_speed_mps)
        max_steering_angle=max_steering_angle*50/100
        time_elapsed += 1.0  
        print(f"At time {time_elapsed} seconds, Steering Angle for Autonomous Parking with Speed {current_speed_kph:.2f} km/h: {max_steering_angle:.2f} degrees")


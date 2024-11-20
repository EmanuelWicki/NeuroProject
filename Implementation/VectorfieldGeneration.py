import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev

json_file_path = "C:/Users/Emanuel Wicki/Documents/Neuro Project/NeuroProject/Implementation/example.json"

# Function to create an irregular shape (ventricle-like) with more points for smoother calculations
def generate_ventricle_shape():
    t = np.linspace(0, 2 * np.pi, 200)
    r = 0.4 + 0.15 * np.sin(3 * t) + 0.1 * np.cos(5 * t)
    x = r * np.cos(t) + 0.08
    y = r * np.sin(t)
    return x, y

# Function to create the white matter boundary
def generate_white_matter_boundary():
    t = np.linspace(0, 2 * np.pi, 800)
    r = (0.9 + 0.1 * np.sin(7 * t) + 0.15 * np.cos(11 * t) +
         0.1 * np.sin(13 * t + 1.5) + 0.05 * np.cos(17 * t + 0.5))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y

# Function to generate evenly distributed points on a closed curve
def generate_evenly_distributed_points(x, y, num_points):
    points = np.vstack((x, y)).T
    tck, u = splprep([x, y], s=0)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    return np.vstack((x_new, y_new)).T

# Function to follow the vector field
def follow_vector_field(start_point, vector_tree, vector_directions, white_matter_boundary, max_steps=500, step_size=0.01, tolerance=0.02):
    current_point = np.array(start_point)
    path = [current_point]
    for _ in range(max_steps):
        # Find the nearest vector
        _, idx = vector_tree.query(current_point)
        direction = vector_directions[idx]
        
        # Move along the vector field
        next_point = current_point + step_size * direction
        path.append(next_point)
        
        # Check if the point is close enough to the white matter boundary
        distance_to_boundary = np.min(np.linalg.norm(white_matter_boundary - next_point, axis=1))
        if distance_to_boundary <= tolerance:
            break
        
        current_point = next_point
    
    return np.array(path)

# Load vector field from JSON
with open(json_file_path, 'r') as f:
    vector_data = json.load(f)

# Convert vector data into numpy arrays
vector_positions = np.array([[entry['start_x'], entry['start_y']] for entry in vector_data])
vector_directions = np.array([[entry['vector_x'], entry['vector_y']] for entry in vector_data])

# Create a KDTree for efficient nearest-neighbor search
vector_tree = KDTree(vector_positions)

# Generate initial ventricle shape and white matter boundary
ventricle_x, ventricle_y = generate_ventricle_shape()
white_matter_x, white_matter_y = generate_white_matter_boundary()

# Prepare white matter boundary for distance calculations
white_matter_boundary = np.vstack((white_matter_x, white_matter_y)).T

# Generate evenly distributed points on the initial ventricle
num_points = 400
ventricle_points = generate_evenly_distributed_points(ventricle_x, ventricle_y, num_points)

# Generate paths for each initial point
paths = []
for point in ventricle_points:
    path = follow_vector_field(point, vector_tree, vector_directions, white_matter_boundary)
    paths.append(path)

# Plot the results
plt.figure(figsize=(8, 8))
plt.plot(ventricle_x, ventricle_y, label='Initial Ventricle', color='red')
plt.plot(white_matter_x, white_matter_y, label='White Matter Boundary', color='blue')
for path in paths:
    plt.plot(path[:, 0], path[:, 1], color='green', alpha=0.6)
plt.legend()
plt.axis('equal')
plt.title("Vector Field Paths Connecting Ventricle to White Matter Boundary")
plt.show()

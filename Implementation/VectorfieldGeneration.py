import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree

# File paths
json_file_path = "C:/Users/Emanuel Wicki/Documents/Neuro Project/NeuroProject/Implementation/example.json"
averaged_grid_file = "C:/Users/Emanuel Wicki/Documents/Neuro Project/NeuroProject/Implementation/averaged_vector_grid.json"

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
    tck, u = splprep([x, y], s=0, per=True)  # `per=True` ensures the curve is treated as closed
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    return np.vstack((x_new, y_new)).T

# Function to create a grid and average vectors in each cell
def create_and_average_vector_grid(vector_data, grid_size):
    grid = {}
    for entry in vector_data:
        start_x, start_y = entry['start_x'], entry['start_y']
        vector_x, vector_y = entry['vector_x'], entry['vector_y']
        
        # Determine grid cell
        cell_x = np.floor(start_x / grid_size) * grid_size
        cell_y = np.floor(start_y / grid_size) * grid_size
        cell_key = (cell_x, cell_y)
        
        if cell_key not in grid:
            grid[cell_key] = {'vectors': [], 'count': 0}
        
        grid[cell_key]['vectors'].append((vector_x, vector_y))
        grid[cell_key]['count'] += 1
    
    # Compute average vector for each cell
    averaged_grid = {}
    for cell, data in grid.items():
        avg_vector = np.mean(data['vectors'], axis=0)
        averaged_grid[cell] = {'vector_x': avg_vector[0], 'vector_y': avg_vector[1]}
    
    return averaged_grid

# Function to save the averaged grid to a JSON file
def save_averaged_grid(grid, file_path):
    # Convert tuple keys to strings
    grid_with_str_keys = {f"{key[0]},{key[1]}": value for key, value in grid.items()}
    with open(file_path, 'w') as f:
        json.dump(grid_with_str_keys, f, indent=4)

# Function to load the averaged grid from a JSON file
def load_averaged_grid(file_path):
    with open(file_path, 'r') as f:
        grid_with_str_keys = json.load(f)
    # Convert string keys back to tuples
    return {tuple(map(float, key.split(','))): value for key, value in grid_with_str_keys.items()}

# Function to follow the vector grid
def follow_averaged_vector_grid(start_point, vector_grid, grid_size, white_matter_boundary, max_steps=800, step_size=0.01, tolerance=0.05):
    current_point = np.array(start_point)
    path = [current_point]
    for _ in range(max_steps):
        # Determine the grid cell
        cell_x = np.floor(current_point[0] / grid_size) * grid_size
        cell_y = np.floor(current_point[1] / grid_size) * grid_size
        cell_key = (cell_x, cell_y)
        
        if cell_key not in vector_grid:
            break  # Stop if there's no vector in this grid cell
        
        # Get the average vector for the grid cell
        vector_x, vector_y = vector_grid[cell_key]['vector_x'], vector_grid[cell_key]['vector_y']
        direction = np.array([vector_x, vector_y])
        
        # Move along the vector field
        next_point = current_point + step_size * direction
        path.append(next_point)
        
        # Check if the point is close enough to the white matter boundary
        distance_to_boundary = np.min(np.linalg.norm(white_matter_boundary - next_point, axis=1))
        if distance_to_boundary <= tolerance:
            break
        
        current_point = next_point
    
    return np.array(path)

# Load raw vector data from JSON
with open(json_file_path, 'r') as f:
    vector_data = json.load(f)

# Define grid size
grid_size = 0.02

# Create averaged vector grid
averaged_vector_grid = create_and_average_vector_grid(vector_data, grid_size)

# Save the averaged grid to a JSON file
save_averaged_grid(averaged_vector_grid, averaged_grid_file)

# Load the averaged grid for further use
averaged_vector_grid = load_averaged_grid(averaged_grid_file)

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
    path = follow_averaged_vector_grid(point, averaged_vector_grid, grid_size, white_matter_boundary)
    paths.append(path)

# Plot the results
plt.figure(figsize=(8, 8))
plt.plot(ventricle_x, ventricle_y, label='Initial Ventricle', color='red')
plt.plot(white_matter_x, white_matter_y, label='White Matter Boundary', color='blue')

# Plot averaged vector grid
for cell, data in averaged_vector_grid.items():
    cell_x, cell_y = cell
    vector_x, vector_y = data['vector_x'], data['vector_y']
    plt.arrow(cell_x, cell_y, vector_x * 0.0, vector_y * 0.0, head_width=0.0, head_length=0.0, fc='purple', ec='purple', alpha=0.8)
    # plt.arrow(cell_x, cell_y, vector_x * 0.02, vector_y * 0.02, head_width=0.01, head_length=0.01, fc='purple', ec='purple', alpha=0.8)

# Plot paths
for path in paths:
    plt.plot(path[:, 0], path[:, 1], color='green', linewidth=0.5, alpha=1)

plt.legend()
plt.axis('equal')
plt.title("Averaged Vector Field Paths Connecting Ventricle to White Matter Boundary")
plt.show()

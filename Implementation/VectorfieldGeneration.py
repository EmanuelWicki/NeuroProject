import json
import numpy as np
import matplotlib.pyplot as plt

# File paths
json_file_path = "C:/Users/Emanuel Wicki/Documents/Neuro Project/NeuroProject/Implementation/vector_field2.json"

# Function to create the ventricular surface
def generate_ventricle_shape():
    t = np.linspace(0, 2 * np.pi, 300)
    r = 0.4 + 0.15 * np.sin(3 * t) + 0.1 * np.cos(5 * t)
    x = r * np.cos(t) + 0.08
    y = r * np.sin(t)
    return x, y

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

# Function to follow the vector grid
def follow_averaged_vector_grid(start_point, vector_grid, grid_size, max_steps=800, step_size=0.015):
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
        
        current_point = next_point
    
    return np.array(path)

# Function to detect divergence using a sliding window
def detect_divergence_points_window(vector_grid, grid_size, window_size=7, divergence_threshold=3):
    divergence_points = []
    half_window = window_size // 2
    grid_keys = np.array(list(vector_grid.keys()))
    
    for cell in vector_grid:
        cx, cy = cell
        vectors_in_window = []
        
        # Collect vectors within the window
        for dx in range(-half_window, half_window + 1):
            for dy in range(-half_window, half_window + 1):
                neighbor_cell = (cx + dx * grid_size, cy + dy * grid_size)
                if neighbor_cell in vector_grid:
                    vectors_in_window.append(
                        np.array([vector_grid[neighbor_cell]['vector_x'], vector_grid[neighbor_cell]['vector_y']])
                    )
        
        # Compute average vector in the window
        if len(vectors_in_window) > 1:
            avg_vector = np.mean(vectors_in_window, axis=0)
            avg_vector /= np.linalg.norm(avg_vector)  # Normalize
            
            # Check divergence of individual vectors in the window
            for vector in vectors_in_window:
                angle = np.degrees(np.arccos(np.clip(np.dot(avg_vector, vector / np.linalg.norm(vector)), -1.0, 1.0)))
                if angle > divergence_threshold:
                    divergence_points.append(cell)
                    break
    
    return divergence_points

# Function to save paths to a JSON file
def save_paths_to_json(paths, output_file):
    paths_data = {"paths": []}
    for path in paths:
        path_points = [{"x": float(point[0]), "y": float(point[1])} for point in path]
        paths_data["paths"].append(path_points)
    
    with open(output_file, 'w') as f:
        json.dump(paths_data, f, indent=4)
    print(f"Paths saved to {output_file}")

# Load raw vector data from JSON
with open(json_file_path, 'r') as f:
    vector_data = json.load(f)

# Define grid size
grid_size = 0.015

# Create averaged vector grid
averaged_vector_grid = create_and_average_vector_grid(vector_data, grid_size)

# Generate initial ventricle shape
ventricle_x, ventricle_y = generate_ventricle_shape()
ventricle_points = np.vstack((ventricle_x, ventricle_y)).T

# Detect divergence points using a sliding window
all_divergence_points = detect_divergence_points_window(averaged_vector_grid, grid_size)

# Generate side branches from divergence points
side_branches = []
for point in all_divergence_points:
    side_branch = follow_averaged_vector_grid(point, averaged_vector_grid, grid_size)
    side_branches.append(side_branch)

# Save the generated paths to a JSON file
output_file_path = "C:/Users/Emanuel Wicki/Documents/Neuro Project/NeuroProject/Implementation/paths.json"
save_paths_to_json(side_branches, output_file_path)

# Plot the results
plt.figure(figsize=(8, 8))

# Plot vector grid
for cell, data in averaged_vector_grid.items():
    cell_x, cell_y = cell
    vector_x, vector_y = data['vector_x'], data['vector_y']
    plt.arrow(cell_x, cell_y, vector_x * 0.0, vector_y * 0.0, head_width=0.0, head_length=0.00, fc='purple', ec='purple', alpha=0.6)
    # plt.arrow(cell_x, cell_y, vector_x * 0.01, vector_y * 0.01, head_width=0.005, head_length=0.005, fc='purple', ec='purple', alpha=0.6)

# Plot ventricular surface
plt.plot(ventricle_x, ventricle_y, label='Ventricular Surface', color='red')

# Plot side branches
for branch in side_branches:
    plt.plot(branch[:, 0], branch[:, 1], color='blue', linewidth=0.2, alpha=0.5)

plt.legend()
plt.axis('equal')
plt.title("Branching Paths from Ventricular Surface")
plt.show()

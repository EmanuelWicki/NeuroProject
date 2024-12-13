import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# File paths
json_file_path = "expansion_vectors.json"  # Replace with the path to your JSON file

# Function to create a grid and average vectors in each cell
def create_and_average_vector_grid_3D(vector_data, grid_size):
    grid = {}
    for step_data in vector_data:
        for entry in step_data['vectors']:
            start_x, start_y, start_z = entry['position']
            vector_x, vector_y, vector_z = entry['vector']
            
            # Determine grid cell
            cell_x = np.floor(start_x / grid_size) * grid_size
            cell_y = np.floor(start_y / grid_size) * grid_size
            cell_z = np.floor(start_z / grid_size) * grid_size
            cell_key = (cell_x, cell_y, cell_z)
            
            if cell_key not in grid:
                grid[cell_key] = {'vectors': [], 'count': 0}
            
            grid[cell_key]['vectors'].append((vector_x, vector_y, vector_z))
            grid[cell_key]['count'] += 1
    
    # Compute average vector for each cell
    averaged_grid = {}
    for cell, data in grid.items():
        avg_vector = np.mean(data['vectors'], axis=0)
        averaged_grid[cell] = {'vector_x': avg_vector[0], 'vector_y': avg_vector[1], 'vector_z': avg_vector[2]}
    
    return averaged_grid

# Function to follow the averaged vector grid in 3D space
def follow_averaged_vector_grid_3D(start_point, vector_grid, grid_size, max_steps=500, step_size=0.01):
    current_point = np.array(start_point)
    path = [current_point]
    for _ in range(max_steps):
        # Determine the grid cell
        cell_x = np.floor(current_point[0] / grid_size) * grid_size
        cell_y = np.floor(current_point[1] / grid_size) * grid_size
        cell_z = np.floor(current_point[2] / grid_size) * grid_size
        cell_key = (cell_x, cell_y, cell_z)
        
        if cell_key not in vector_grid:
            break  # Stop if there's no vector in this grid cell
        
        # Get the average vector for the grid cell
        vector_x, vector_y, vector_z = vector_grid[cell_key]['vector_x'], vector_grid[cell_key]['vector_y'], vector_grid[cell_key]['vector_z']
        direction = np.array([vector_x, vector_y, vector_z])
        
        # Move along the vector field
        next_point = current_point + step_size * direction
        path.append(next_point)
        
        current_point = next_point
    
    return np.array(path)

# Function to save paths to a JSON file
def save_paths_to_json(paths, output_file):
    paths_data = {"paths": []}
    for path in paths:
        path_points = [{"x": float(point[0]), "y": float(point[1]), "z": float(point[2])} for point in path]
        paths_data["paths"].append(path_points)
    
    with open(output_file, 'w') as f:
        json.dump(paths_data, f, indent=4)
    print(f"Paths saved to {output_file}")

# Load raw vector data from JSON
with open(json_file_path, 'r') as f:
    vector_data = json.load(f)

# Define grid size
grid_size = 0.07

# Create averaged vector grid
averaged_vector_grid = create_and_average_vector_grid_3D(vector_data, grid_size)

# Define starting points for paths (e.g., the positions in the first step)
starting_points = [entry['position'] for entry in vector_data[0]['vectors']]  # Start with the first step's positions

# Generate paths by following the averaged vector grid
paths = []
for start_point in starting_points:
    path = follow_averaged_vector_grid_3D(start_point, averaged_vector_grid, grid_size)
    paths.append(path)

# Save the generated paths to a JSON file
output_file_path = "3D_paths.json"
save_paths_to_json(paths, output_file_path)

# Plot the results in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the vector grid
for cell, data in averaged_vector_grid.items():
    cell_x, cell_y, cell_z = cell
    vector_x, vector_y, vector_z = data['vector_x'], data['vector_y'], data['vector_z']
    ax.quiver(cell_x, cell_y, cell_z, vector_x, vector_y, vector_z, length=0.01, color='purple', alpha=0.5)

# Plot the paths
for path in paths:
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color='blue', linewidth=0.5, alpha=0.7)

ax.set_title("3D Vector Field and Paths")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.show()

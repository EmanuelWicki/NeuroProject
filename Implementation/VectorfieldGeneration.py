import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree

# File paths
json_file_path = "C:/Users/Emanuel Wicki/Documents/Neuro Project/NeuroProject/Implementation/example.json"
averaged_grid_file = "C:/Users/Emanuel Wicki/Documents/Neuro Project/NeuroProject/Implementation/averaged_vector_grid.json"

# Function to compute angular difference between vectors
def compute_angular_difference(vec1, vec2):
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vec1, unit_vec2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

# Function to detect divergence using a sliding window
def detect_diverging_vectors_window(vector_grid, grid_size, window_size=3, divergence_threshold=25):
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
                angle = compute_angular_difference(vector, avg_vector)
                if angle > divergence_threshold:
                    divergence_points.append(cell)
                    break
    
    return divergence_points

# Function to trace paths from diverging points
def trace_branch_paths(start_points, vector_grid, grid_size, white_matter_boundary, max_steps=800, step_size=0.01, tolerance=0.05):
    paths = []
    for start_point in start_points:
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
        paths.append(np.array(path))
    return paths

# Load vector grid from JSON
with open(averaged_grid_file, 'r') as f:
    vector_grid = json.load(f)

# Convert string keys back to tuples
vector_grid = {eval(k): v for k, v in vector_grid.items()}

# Define grid size
grid_size = 0.02

# Generate white matter boundary
def generate_white_matter_boundary():
    t = np.linspace(0, 2 * np.pi, 800)
    r = (0.9 + 0.1 * np.sin(7 * t) + 0.15 * np.cos(11 * t) +
         0.1 * np.sin(13 * t + 1.5) + 0.05 * np.cos(17 * t + 0.5))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.vstack((x, y)).T

white_matter_boundary = generate_white_matter_boundary()

# Detect diverging vectors using a sliding window
diverging_points = detect_diverging_vectors_window(vector_grid, grid_size, window_size=5)

# Trace paths from diverging points
branch_paths = trace_branch_paths(diverging_points, vector_grid, grid_size, white_matter_boundary)

# Save paths to JSON
with open("branch_paths.json", "w") as f:
    json.dump([path.tolist() for path in branch_paths], f, indent=4)

# Plot the results
plt.figure(figsize=(8, 8))
plt.plot(white_matter_boundary[:, 0], white_matter_boundary[:, 1], label='White Matter Boundary', color='blue')
for path in branch_paths:
    plt.plot(path[:, 0], path[:, 1], color='green', alpha=0.6)

plt.legend()
plt.axis('equal')
plt.title("Branch Paths with Divergence Detection")
plt.show()

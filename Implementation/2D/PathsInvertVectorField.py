import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev


#####################################################################################################################################
##################  This Script takes JSON file of expansion vector as input and generates a inversed vectorfield  ##################
##################  Inverse Vector Field to guide paths from the white matter to the ventricle (they grow the other way around ######
#####################################################################################################################################

# Function to generate the ventricular shape
def generate_ventricle_shape():
    t = np.linspace(0, 2 * np.pi, 2000)
    r = 0.4 + 0.15 * np.sin(3 * t) + 0.1 * np.cos(5 * t)
    x = r * np.cos(t) + 0.08
    y = r * np.sin(t)
    return x, y

# Function to create the white matter boundary
def generate_white_matter_boundary():
    t = np.linspace(0, 2 * np.pi, 800)
    r = (0.9 + 0.1 * np.sin(7 * t) + 0.15 * np.cos(11 * t) + 0.1 * np.sin(13 * t + 1.5) + 0.05 * np.cos(17 * t + 0.5))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y


# Function to load and invert the vector field from JSON file
def load_and_invert_vector_field(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract and invert vector field components
    start_x = np.array([item["start_x"] for item in data])
    start_y = np.array([item["start_y"] for item in data])
    vector_x = np.array([-item["vector_x"] for item in data])  # Invert x component
    vector_y = np.array([-item["vector_y"] for item in data])  # Invert y component

    # Normalize the vectors for visualization
    magnitude = np.sqrt(vector_x**2 + vector_y**2)
    vector_x /= magnitude
    vector_y /= magnitude
    
    return start_x, start_y, vector_x, vector_y

# Load normal vector field of the expansion
def load_vector_field(json_file_path):
    """
    Loads the vector field from a JSON file without inverting the vectors.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract vector field components
    start_x = np.array([item["start_x"] for item in data])
    start_y = np.array([item["start_y"] for item in data])
    vector_x = np.array([item["vector_x"] for item in data])  # Normal vectors
    vector_y = np.array([item["vector_y"] for item in data])  # Normal vectors
    
    return start_x, start_y, vector_x, vector_y

def select_evenly_distributed_points(x, y, num_points):
    """
    Selects evenly spaced points along a curve defined by (x, y) based on arc length.
    """
    # Calculate cumulative arc length
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)  # Euclidean distances between consecutive points
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)  # Cumulative sum of distances, starting at 0

    # Generate evenly spaced distances
    total_length = cumulative_dist[-1]
    desired_distances = np.linspace(0, total_length, num_points)

    # Interpolate to find the corresponding (x, y) points
    evenly_spaced_x = np.interp(desired_distances, cumulative_dist, x)
    evenly_spaced_y = np.interp(desired_distances, cumulative_dist, y)

    return evenly_spaced_x, evenly_spaced_y


# Function to trace a point through the inverted vector field
def trace_point(x_start, y_start, start_x, start_y, vector_x, vector_y, ventricle_x, ventricle_y, 
                steps=500, step_size=0.001, stop_threshold=0.01, hole_threshold=1e-4):
    """
    Traces a point through the inverted vector field and stops when:
    1. The point reaches the ventricular surface (distance threshold).
    2. The vector magnitude is too small (indicating the hole).
    """
    x, y = x_start, y_start
    path_x, path_y = [x], [y]
    ventricle_coords = np.column_stack((ventricle_x, ventricle_y))
    vector_coords = np.column_stack((start_x, start_y))

    for _ in range(steps):
        # Calculate distance to the ventricular shape
        current_point = np.array([[x, y]])
        distances = cdist(current_point, ventricle_coords)
        min_distance = np.min(distances)
        
        # Stop if close to the ventricular boundary
        if min_distance <= stop_threshold:
            print(f"Stopping: Close to ventricular boundary at ({x:.3f}, {y:.3f}).")
            break

        # Find the nearest vector in the vector field
        distances_to_vectors = cdist(current_point, vector_coords)
        idx = np.argmin(distances_to_vectors)
        
        # Get the corresponding vector
        vec_x, vec_y = vector_x[idx], vector_y[idx]
        
        # Stop if vector magnitude is too small (indicating the hole)
        magnitude = np.sqrt(vec_x**2 + vec_y**2)
        if magnitude < hole_threshold:
            print(f"Stopping: Entering hole at ({x:.3f}, {y:.3f}).")
            break

        # Dynamically reduce step size as the hole is approached
        dynamic_step_size = min(step_size, magnitude / 2)

        # Update position using the vector direction
        x += vec_x * dynamic_step_size
        y += vec_y * dynamic_step_size
        
        path_x.append(x)
        path_y.append(y)
    
    return path_x, path_y

# Trace paths from the ventricle to the white matter
def trace_point_normal(x_start, y_start, start_x, start_y, vector_x, vector_y, white_matter_x, white_matter_y, 
                       steps=1000, step_size=0.01, stop_threshold=0.02):
    """
    Traces a point through the normal vector field and stops when:
    1. The point reaches the white matter surface (distance threshold).
    2. The point exits the vector field boundary.
    """
    x, y = x_start, y_start
    path_x, path_y = [x], [y]
    white_matter_coords = np.column_stack((white_matter_x, white_matter_y))
    vector_coords = np.column_stack((start_x, start_y))

    # Define bounds of the vector field
    min_x, max_x = np.min(start_x), np.max(start_x)
    min_y, max_y = np.min(start_y), np.max(start_y)

    for _ in range(steps):
        # Calculate distance to the white matter boundary
        current_point = np.array([[x, y]])
        distances = cdist(current_point, white_matter_coords)
        min_distance = np.min(distances)
        
        # Stop if close to the white matter boundary
        if min_distance <= stop_threshold:
            print(f"Stopping: Close to white matter boundary at ({x:.3f}, {y:.3f}).")
            break

        # Stop if the point moves outside the vector field bounds
        if x < min_x or x > max_x or y < min_y or y > max_y:
            print(f"Stopping: Exiting vector field boundary at ({x:.3f}, {y:.3f}).")
            break

        # Find the nearest vector in the vector field
        distances_to_vectors = cdist(current_point, vector_coords)
        idx = np.argmin(distances_to_vectors)
        
        # Get the corresponding vector
        vec_x, vec_y = vector_x[idx], vector_y[idx]
        
        # Dynamically adjust step size to avoid overshooting
        dynamic_step_size = min(step_size, min_distance / 2)

        # Update position using the vector direction
        x += vec_x * dynamic_step_size
        y += vec_y * dynamic_step_size
        
        path_x.append(x)
        path_y.append(y)
    
    return path_x, path_y


# Main plotting function
def main():
    # Generate shapes
    ventricle_x, ventricle_y = generate_ventricle_shape()
    white_matter_x, white_matter_y = generate_white_matter_boundary()
    
    # Load the inverted vector field
    json_file_path = "vector_field3.json"  # Replace with your actual vector field file
    start_x, start_y, vector_x_inverted, vector_y_inverted = load_and_invert_vector_field(json_file_path)
    
    # Load the normal vector field
    start_x, start_y, vector_x_normal, vector_y_normal = load_vector_field(json_file_path)

    # Select number of points separately
    num_points_inverted = 2000  # Number of points to trace for the inverted vector field
    num_points_normal = 300    # Number of points to trace for the normal vector field
    
    # Select starting points
    white_matter_points = np.linspace(0, len(white_matter_x) - 1, num_points_inverted, dtype=int)
    ventricle_points = np.linspace(0, len(ventricle_x) - 1, num_points_normal, dtype=int)
    
    # Plotting
    plt.figure(figsize=(8, 8))
    plt.fill(ventricle_x, ventricle_y, color='gray', alpha=0.5, label='Ventricular Shape')
    plt.plot(white_matter_x, white_matter_y, color='blue', linewidth=1, label='White Matter Boundary')
    
    # Select evenly distributed points along the white matter boundary
    evenly_spaced_wm_x, evenly_spaced_wm_y = select_evenly_distributed_points(white_matter_x, white_matter_y, num_points_inverted)

    # Trace inward paths (from white matter to ventricular surface)
    for i in range(num_points_inverted):
        start_x_wm, start_y_wm = evenly_spaced_wm_x[i], evenly_spaced_wm_y[i]
        path_x, path_y = trace_point(
            start_x_wm, start_y_wm, start_x, start_y, vector_x_inverted, vector_y_inverted, ventricle_x, ventricle_y
        )
        plt.plot(path_x, path_y, color='blue', linewidth=0.5, alpha=0.7, label="Inward Path" if i == 0 else "")


    # Trace outward paths (from ventricular surface to white matter boundary)
    for idx in ventricle_points:
        start_x_v, start_y_v = ventricle_x[idx], ventricle_y[idx]
        path_x, path_y = trace_point_normal(
            start_x_v, start_y_v, start_x, start_y, vector_x_normal, vector_y_normal, white_matter_x, white_matter_y
        )
        plt.plot(path_x, path_y, color='blue', linewidth=0.5, alpha=0.7, label="Outward Path" if idx == ventricle_points[0] else "")
    
    plt.title("Traced Paths: Inward and Outward through Vector Field")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

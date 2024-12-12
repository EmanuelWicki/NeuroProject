import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance import cdist

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

# Function to plot the vector grid
def plot_vector_field(json_file_path):
    # Load the inverted vector field
    start_x, start_y, vector_x, vector_y = load_and_invert_vector_field(json_file_path)
    
    # Plot the vector field
    plt.figure(figsize=(8, 8))
    plt.quiver(start_x, start_y, vector_x, vector_y, angles='xy', scale_units='xy', scale=0.1, color='red')
    
    plt.title("Inverted Vector Field Visualization (Normalized)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

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

# Main plotting function
def main():
    # Generate shapes
    ventricle_x, ventricle_y = generate_ventricle_shape()
    white_matter_x, white_matter_y = generate_white_matter_boundary()
    
    # Load and invert the vector field
    json_file_path = "vector_field3.json"  # Replace with your actual file path
    start_x, start_y, vector_x, vector_y = load_and_invert_vector_field(json_file_path)
    
    # Select starting points on the white matter boundary for tracing
    num_points = 2500  # Number of points to trace
    white_matter_points = np.linspace(0, len(white_matter_x) - 1, num_points, dtype=int)
    
    # Plotting
    plt.figure(figsize=(8, 8))
    
    # Plot the ventricular shape
    plt.fill(ventricle_x, ventricle_y, color='gray', alpha=0.5, label='Ventricular Shape')
    
    # Plot the white matter boundary
    plt.plot(white_matter_x, white_matter_y, color='blue', linewidth=1, label='White Matter Boundary')
    
    # Trace and plot paths
    for idx in white_matter_points:
        start_x_wm, start_y_wm = white_matter_x[idx], white_matter_y[idx]
        # Pass all required arguments explicitly
        path_x, path_y = trace_point(
            start_x_wm, start_y_wm, start_x, start_y, vector_x, vector_y, ventricle_x, ventricle_y
        )
        plt.plot(path_x, path_y, color='blue', linewidth=0.3, alpha=0.7)
    
    plt.title("Traced Paths through Inverted Vector Field to Ventricular Surface")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
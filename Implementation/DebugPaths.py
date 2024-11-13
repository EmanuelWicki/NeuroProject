import numpy as np
import matplotlib.pyplot as plt

# Function to create an irregular shape (ventricle-like) with more points for smoother calculations
def generate_ventricle_shape():
    t = np.linspace(0, 2 * np.pi, 300)  # Increase number of points for smoother surface
    r = 0.2 + 0.08 * np.sin(3 * t) + 0.05 * np.cos(5 * t)  # Smaller ventricle shape
    x = r * np.cos(t) - 0.1  # Centered at (-0.1, 0)
    y = r * np.sin(t)
    return x, y

# Function to compute outward normals of a closed 2D curve
def compute_outward_normals(x, y):
    normals = []
    centroid = np.array([np.mean(x), np.mean(y)])
    for i in range(len(x)):
        # Compute tangent vector as difference between neighboring points
        next_idx = (i + 1) % len(x)
        prev_idx = (i - 1) % len(x)
        tangent = np.array([x[next_idx] - x[prev_idx], y[next_idx] - y[prev_idx]])
        tangent /= np.linalg.norm(tangent)  # Normalize tangent
        # Normal is perpendicular to tangent
        normal = np.array([-tangent[1], tangent[0]])  # Rotate tangent by 90 degrees to get normal
        
        # Ensure the normal points outwards
        point = np.array([x[i], y[i]])
        direction_to_centroid = centroid - point
        if np.dot(normal, direction_to_centroid) > 0:
            normal = -normal  # Reverse if pointing towards centroid

        normals.append(normal)
    normals = np.array(normals)
    return normals

# Function to check if two line segments intersect
def check_intersection(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

# Function to generate the adjusted expanded surface by wrapping a line around the normal vectors
# and removing points that are enclosed due to concave intersections
def generate_adjusted_expanded_surface(x, y, normals, expansion_length=0.05):
    expanded_points = []
    global skip_indices
    skip_indices = set()

    # Generate all expanded points
    for i in range(len(x)):
        if i in skip_indices:
            continue
        point = np.array([x[i], y[i]])
        normal = normals[i]
        expanded_point = point + expansion_length * normal
        expanded_points.append(expanded_point)

        # Check for intersections with previous normals
        for j in range(i):
            if j in skip_indices:
                continue
            prev_point = np.array([x[j], y[j]])
            prev_expanded_point = prev_point + expansion_length * normals[j]
            if check_intersection(point, expanded_point, prev_point, prev_expanded_point):
                # If intersection is found, mark the indices in between as skipped
                start_idx = min(i, j)
                end_idx = max(i, j)
                skip_indices.update(range(start_idx + 1, end_idx))
                break

    expanded_points = np.array(expanded_points)
    return expanded_points[:, 0], expanded_points[:, 1]

# Function to fit a line that connects the tips of the valid blue vectors without intersections
def fit_direct_curve(expanded_x, expanded_y, skip_indices):
    # Only consider points that were not skipped
    valid_points = [(expanded_x[i], expanded_y[i]) for i in range(len(expanded_x)) if i not in skip_indices]
    valid_points = np.array(valid_points)
    return valid_points[:, 0], valid_points[:, 1]

# Plotting function to visualize the adjusted expanded surface
def plot_expanded_surface(initial_x, initial_y, expanded_x, expanded_y, normals, expansion_length, skip_indices):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Plot initial surface
    ax.fill(initial_x, initial_y, color='gray', alpha=0.4, label='Ventricle Surface')
    
    # Plot expanded surface
    ax.plot(expanded_x, expanded_y, color='red', linewidth=2, linestyle='-', label='Adjusted Expanded Surface')
    
    # Plot normal vectors
    for i in range(len(initial_x)):
        point = np.array([initial_x[i], initial_y[i]])
        normal = normals[i]
        expanded_point = point + expansion_length * normal
        if i not in skip_indices:
            ax.plot([point[0], expanded_point[0]], [point[1], expanded_point[1]], color='blue', linestyle='--', linewidth=0.5)
    
    # Fit and plot the direct curve to the tips of the blue vectors
    fitted_x, fitted_y = fit_direct_curve(expanded_x, expanded_y, skip_indices)
    ax.plot(fitted_x, fitted_y, color='green', linewidth=2, linestyle='-', label='Fitted Direct Curve')
    
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect('equal')
    ax.set_title('Adjusted Expanded Surface Without Internal Intersections')
    plt.legend()
    plt.show()

# Main execution
ventricle_x, ventricle_y = generate_ventricle_shape()

# Compute normals for the ventricle shape
normals = compute_outward_normals(ventricle_x, ventricle_y)

# Generate the adjusted expanded surface without internal intersections
expansion_length = 0.05
expanded_x, expanded_y = generate_adjusted_expanded_surface(ventricle_x, ventricle_y, normals, expansion_length)

# Plot the adjusted expanded surface
plot_expanded_surface(ventricle_x, ventricle_y, expanded_x, expanded_y, normals, expansion_length, skip_indices)

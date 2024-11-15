import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import splprep, splev
import os

# Function to create an irregular shape (ventricle-like) with more points for smoother calculations
def generate_ventricle_shape():
    t = np.linspace(0, 2 * np.pi, 300)  # Increase number of points for smoother surface
    r = 0.4 + 0.15 * np.sin(3 * t) + 0.1 * np.cos(5 * t)  # Randomized variations for larger initial shape
    x = r * np.cos(t) - 0.1  # Centered at (-0.1, 0)
    y = r * np.sin(t)
    return x, y

# Function to create the white matter boundary
def generate_white_matter_boundary():
    t = np.linspace(0, 2 * np.pi, 600)  # Increase the number of points for smoother and more detailed boundary
    r = (1.3 + 0.2 * np.sin(7 * t) + 0.15 * np.cos(11 * t) + 0.1 * np.sin(13 * t + 1.5) +
         0.05 * np.cos(17 * t + 0.5))  # Add multiple sine and cosine components for complexity
    x = r * np.cos(t)
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

# Apply Laplacian smoothing to reduce sharp concavities without losing features
def adaptive_laplacian_smoothing(x, y, alpha=0.005):
    smoothed_x = x.copy()
    smoothed_y = y.copy()
    for i in range(len(x)):
        prev_idx = (i - 1) % len(x)
        next_idx = (i + 1) % len(x)
        smoothed_x[i] = (1 - alpha) * x[i] + alpha * 0.5 * (x[prev_idx] + x[next_idx])
        smoothed_y[i] = (1 - alpha) * y[i] + alpha * 0.5 * (y[prev_idx] + y[next_idx])
    return smoothed_x, smoothed_y

# Function to interpolate points using a spline
def interpolate_points(points, num_points=100):
    tck, u = splprep([points[:, 0], points[:, 1]], s=0)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    return np.vstack((x_new, y_new)).T

# Initialize the ventricle and white matter boundary
ventricle_x, ventricle_y = generate_ventricle_shape()
initial_ventricle_x, initial_ventricle_y = ventricle_x.copy(), ventricle_y.copy()  # Store initial structure for plotting
white_matter_x, white_matter_y = generate_white_matter_boundary()

# Convert white matter boundary to array for distance calculations
white_matter_points = np.vstack((white_matter_x, white_matter_y)).T

# Number of expansion iterations
num_iterations = 1000
step_size = 0.001  # Step size for each iteration

# Set the filename to save the PDF in the same directory as this script
pdf_filename = os.path.join(os.getcwd(), 'ventricle_expansion_simplified_surface.pdf')
paths_pdf_filename = os.path.join(os.getcwd(), 'ventricle_to_white_matter_paths.pdf')

# Initialize figure for plotting all steps
plt.figure(figsize=(10, 10))
ax = plt.gca()

# Plot the white matter boundary
ax.plot(white_matter_x, white_matter_y, color='blue', linewidth=2, label='White Matter Boundary')

# Plot the initial ventricle shape as a filled region
ax.fill(initial_ventricle_x, initial_ventricle_y, color='gray', alpha=0.4, label='Initial Ventricle')

# Save each step's expansion vectors
paths = [[] for _ in range(len(ventricle_x))]
for step in range(num_iterations):
    print(f"Processing step {step + 1}")  # Debug statement

    # Compute outward normals for ventricle shape
    normals = compute_outward_normals(ventricle_x, ventricle_y)

    # Convert ventricle points to array
    ventricle_points = np.vstack((ventricle_x, ventricle_y)).T

    # Expansion step for each ventricle point in the direction of the normal
    new_ventricle_points = []
    for i in range(len(ventricle_x)):
        point = ventricle_points[i]
        normal = normals[i]

        # Move point along the normal direction by a fixed step size
        new_point = point + step_size * normal
        new_ventricle_points.append(new_point)

        # Store the path for each point
        paths[i].append(new_point)

    new_ventricle_points = np.array(new_ventricle_points)

    # Ensure the expanded surface is closed by averaging the start and end point normals for the merged expansion
    new_ventricle_points[0] = (new_ventricle_points[0] + new_ventricle_points[-1]) / 2
    new_ventricle_points[-1] = new_ventricle_points[0]

    # Interpolate points to maintain a smooth curve
    simplified_points = interpolate_points(new_ventricle_points)

    # Apply adaptive Laplacian smoothing to reduce sharp concavities
    ventricle_x, ventricle_y = adaptive_laplacian_smoothing(simplified_points[:, 0], simplified_points[:, 1])

    # Plot the expansion vectors as a vector field for this step
    ax.quiver(ventricle_points[:, 0], ventricle_points[:, 1], normals[:, 0], normals[:, 1],
              angles='xy', scale_units='xy', scale=20, color='green', alpha=0.3)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.set_title('Ventricle Expansion Over Time')
plt.legend()
plt.savefig(pdf_filename)
plt.show()

print(f"PDF saved to: {pdf_filename}")

# Plot the paths from initial ventricle points to white matter boundary
plt.figure(figsize=(10, 10))
ax = plt.gca()

# Plot the white matter boundary
ax.plot(white_matter_x, white_matter_y, color='blue', linewidth=2, label='White Matter Boundary')

# Plot the initial ventricle shape as a filled region
ax.fill(initial_ventricle_x, initial_ventricle_y, color='gray', alpha=0.4, label='Initial Ventricle')

# Plot paths from initial ventricle points to white matter
for path in paths:
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], color='red', alpha=0.4)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.set_title('Paths from Ventricle to White Matter')
plt.legend()
plt.savefig(paths_pdf_filename)
plt.show()

print(f"Paths PDF saved to: {paths_pdf_filename}")

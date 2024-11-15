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

# Find the intersection of a ray with a closed 2D curve in the normal direction
def find_ray_intersection(point, direction, boundary_points):
    min_distance = float('inf')
    for i in range(len(boundary_points)):
        # Get two consecutive points on the boundary to form a line segment
        p1 = boundary_points[i]
        p2 = boundary_points[(i + 1) % len(boundary_points)]

        # Parameterize the line segment as p = p1 + t * (p2 - p1), with 0 <= t <= 1
        segment = p2 - p1

        # Solve for intersection using parametric equations
        A = np.array([segment, -direction]).T
        if np.linalg.det(A) == 0:
            continue  # No unique solution, parallel lines

        b = point - p1
        t, u = np.linalg.solve(A, b)

        # Check if intersection lies within the segment (0 <= t <= 1) and in the direction of the ray (u > 0)
        if 0 <= t <= 1 and u > 0:
            distance = u
            if distance < min_distance:
                min_distance = distance

    return min_distance if min_distance != float('inf') else None

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
num_iterations = 10
max_expansion_factor = 0.005  # Max expansion step for longer distances
min_expansion_factor = 0.001  # Minimum step for shorter distances

# Set the filename to save the PDF in the same directory as this script
pdf_filename = os.path.join(os.getcwd(), 'ventricle_expansion_simplified_surface.pdf')

# Save each step in a PDF
with PdfPages(pdf_filename) as pdf:
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

            # Find the intersection along the normal direction with the white matter boundary
            intersection_distance = find_ray_intersection(point, normal, white_matter_points)

            # If an intersection is found, expand by a fraction of the shortest distance
            if intersection_distance:
                # Dynamic expansion based on the distance to white matter
                expansion_distance = intersection_distance * 0.005
                if expansion_distance < min_expansion_factor:
                    expansion_distance = min_expansion_factor
                new_point = point + expansion_distance * normal
            else:
                new_point = point + min_expansion_factor * normal

            new_ventricle_points.append(new_point)

        new_ventricle_points = np.array(new_ventricle_points)

        # Ensure the expanded surface is closed by averaging the start and end point normals for the merged expansion
        new_ventricle_points[0] = (new_ventricle_points[0] + new_ventricle_points[-1]) / 2
        new_ventricle_points[-1] = new_ventricle_points[0]

        # Interpolate points to maintain a smooth curve
        simplified_points = interpolate_points(new_ventricle_points)

        # Apply adaptive Laplacian smoothing to reduce sharp concavities
        ventricle_x, ventricle_y = adaptive_laplacian_smoothing(simplified_points[:, 0], simplified_points[:, 1])

        # Plot and save every iteration with specified visualization
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # Plot the white matter boundary
        ax.plot(white_matter_x, white_matter_y, color='blue', linewidth=2, label='White Matter Boundary')

        # Plot the initial ventricle shape as a filled region
        ax.fill(initial_ventricle_x, initial_ventricle_y, color='gray', alpha=0.4, label='Initial Ventricle')

        # Plot the current ventricle boundary as a fine line
        ax.plot(ventricle_x, ventricle_y, color='red', linewidth=1, linestyle='-', label=f'Boundary at Step {step + 1}')

        # Plot the expansion vectors
        for i in range(len(ventricle_x)):
            if i == 0 or i == len(ventricle_x) - 1:
                if i == 0:
                    ax.arrow(ventricle_points[i, 0], ventricle_points[i, 1], normals[i, 0] * 0.02, normals[i, 1] * 0.02,
                             head_width=0.02, head_length=0.02, fc='green', ec='green', alpha=0.6)
            else:
                ax.arrow(ventricle_points[i, 0], ventricle_points[i, 1], normals[i, 0] * 0.02, normals[i, 1] * 0.02,
                         head_width=0.02, head_length=0.02, fc='green', ec='green', alpha=0.6)

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title(f'Step {step + 1}')
        plt.legend()
        pdf.savefig()  # Save current figure to the PDF
        plt.close()

print(f"PDF saved to: {pdf_filename}")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
    t = np.linspace(0, 2 * np.pi, 300)
    r = 1 + 0.1 * np.sin(5 * t)  # Variations for a slightly irregular surface
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y

# Function to compute outward normals of a closed 2D curve
def compute_outward_normals(x, y):
    normals = []
    centroid = np.array([np.mean(x), np.mean(y)])  # Centroid used for consistent normal direction
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

# Initialize the ventricle and white matter boundary
ventricle_x, ventricle_y = generate_ventricle_shape()
initial_ventricle_points = np.vstack((ventricle_x, ventricle_y)).T

# Convert white matter boundary to array for distance calculations
white_matter_points = np.vstack((generate_white_matter_boundary())).T

# Number of expansion iterations
num_iterations = 100
max_expansion_factor = 0.005  # Max expansion step for longer distances
min_expansion_factor = 0.001  # Minimum step for shorter distances

# Store paths for each point
paths = np.zeros((len(initial_ventricle_points), num_iterations + 1, 2))
paths[:, 0, :] = initial_ventricle_points

# Expansion process
for step in range(num_iterations):
    print(f"Processing step {step + 1}")  # Debug statement

    # Compute outward normals for ventricle shape
    normals = compute_outward_normals(paths[:, step, 0], paths[:, step, 1])

    # Expansion step for each ventricle point in the direction of the normal
    new_ventricle_points = []
    for i in range(len(paths)):
        point = paths[i, step]
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

    # Store the new points in paths
    for i in range(len(new_ventricle_points)):
        paths[i, step + 1] = new_ventricle_points[i]

# Set the filename to save the PDF in the same directory as this script
pdf_filename = os.path.join(os.getcwd(), 'ventricle_expansion_paths_corrected.pdf')

# Save each path plot in a PDF
with PdfPages(pdf_filename) as pdf:
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Plot the paths of each point
    step_interval = 10  # Only plot every 10th path to keep visualization clear
    for i in range(0, len(paths), step_interval):
        ax.plot(paths[i, :, 0], paths[i, :, 1], linestyle='-', linewidth=0.5, color='black')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Paths of Ventricle Surface Points to White Matter Boundary')
    pdf.savefig()  # Save current figure to the PDF
    plt.close()

print(f"PDF saved to: {pdf_filename}")

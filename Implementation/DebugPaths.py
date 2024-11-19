import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import splprep, splev
from PIL import Image
import os

# Function to create an irregular shape (ventricle-like) with more points for smoother calculations
def generate_ventricle_shape():
    t = np.linspace(0, 2 * np.pi, 300)
    r = 0.4 + 0.15 * np.sin(3 * t) + 0.1 * np.cos(5 * t)
    x = r * np.cos(t) + 0.08
    y = r * np.sin(t)
    return x, y

# Function to create the original white matter boundary
def generate_white_matter_boundary():
    t = np.linspace(0, 2 * np.pi, 600)
    r = (0.9 + 0.1 * np.sin(7 * t) + 0.15 * np.cos(11 * t) + 0.1 * np.sin(13 * t + 1.5) + 0.05 * np.cos(17 * t + 0.5))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y

# Function to compute outward normals of a closed 2D curve with a direction consistency check
def compute_outward_normals(x, y):
    normals = []
    centroid = np.array([np.mean(x), np.mean(y)])
    for i in range(len(x)):
        next_idx = (i + 1) % len(x)
        prev_idx = (i - 1) % len(x)
        tangent = np.array([x[next_idx] - x[prev_idx], y[next_idx] - y[prev_idx]])
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        point = np.array([x[i], y[i]])
        direction_to_centroid = centroid - point
        if np.dot(normal, direction_to_centroid) > 0:
            normal = -normal
        normals.append(normal)
    return np.array(normals)

# Create the offset white matter boundary
def create_offset_boundary(x, y, offset_distance=0.02):
    normals = compute_outward_normals(x, y)
    offset_x = x + offset_distance * normals[:, 0]
    offset_y = y + offset_distance * normals[:, 1]
    return offset_x, offset_y

# Find the intersection of a ray with a closed 2D curve in the normal direction
def find_ray_intersection(point, direction, boundary_points):
    min_distance = float('inf')
    for i in range(len(boundary_points)):
        p1 = boundary_points[i]
        p2 = boundary_points[(i + 1) % len(boundary_points)]
        segment = p2 - p1
        A = np.array([segment, -direction]).T
        if np.linalg.det(A) == 0:
            continue
        b = point - p1
        t, u = np.linalg.solve(A, b)
        if 0 <= t <= 1 and u > 0:
            distance = u
            if distance < min_distance:
                min_distance = distance
    return min_distance if min_distance != float('inf') else None


# Interpolate points using a spline
def interpolate_points(points, num_points=200):
    tck, u = splprep([points[:, 0], points[:, 1]], s=0)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    return np.vstack((x_new, y_new)).T

# Re-sample points to maintain a constant density along the expanded surface
def resample_points(points, density):
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_distances[-1]
    new_num_points = int(total_length * density)
    new_distances = np.linspace(0, total_length, new_num_points)
    x_new = np.interp(new_distances, cumulative_distances, points[:, 0])
    y_new = np.interp(new_distances, cumulative_distances, points[:, 1])
    return np.vstack((x_new, y_new)).T

# Initialize the ventricle and white matter boundaries
ventricle_x, ventricle_y = generate_ventricle_shape()
initial_ventricle_x, initial_ventricle_y = ventricle_x.copy(), ventricle_y.copy()
white_matter_x, white_matter_y = generate_white_matter_boundary()
offset_white_matter_x, offset_white_matter_y = create_offset_boundary(white_matter_x, white_matter_y)  # Offset boundary

white_matter_points = np.vstack((offset_white_matter_x, offset_white_matter_y)).T
pdf_filename = os.path.join(os.getcwd(), 'ventricle_expansion_vector_field.pdf')
gif_filename = os.path.join(os.getcwd(), 'ventricle_expansion_vector_field.gif')
frames = []

# Define initial density
density = len(ventricle_x) / np.sum(np.sqrt(np.sum(np.diff(np.vstack((ventricle_x, ventricle_y)).T, axis=0)**2, axis=1)))

# Define the number of steps for the initial phase
initial_phase_steps = 50

# Define cell size for vector field grid
cell_size = 0.05
vector_field = {}

with PdfPages(pdf_filename) as pdf:
    step = 0
    while True:
        step += 1
        print(f'Number of points at step {step}: {len(ventricle_x)}')
        normals = compute_outward_normals(ventricle_x, ventricle_y)
        ventricle_points = np.vstack((ventricle_x, ventricle_y)).T
        new_ventricle_points = []

        close_to_boundary_count = 0
        for i in range(len(ventricle_points)):
            normal = normals[i]
            point = ventricle_points[i]
            intersection_distance = find_ray_intersection(point, normal, white_matter_points)
            if intersection_distance:
                if intersection_distance <= 0.07:
                    # Maintain distance without moving closer to the boundary
                    point = point + 0 * normal
                    close_to_boundary_count += 1
                else:
                    # Dynamic expansion distance logic
                    if step <= initial_phase_steps:
                        expansion_distance = intersection_distance / 200  # Smaller expansion step in initial phase
                    else:
                        expansion_distance = intersection_distance / 50  # Larger expansion step in later phase
                    point = point + min(expansion_distance, 0.01) * normal

                    # Compute cell key for vector field
                    cell_x = np.floor(point[0] / cell_size) * cell_size + cell_size / 2
                    cell_y = np.floor(point[1] / cell_size) * cell_size + cell_size / 2
                    cell_key = (cell_x, cell_y)

                    # Store the vector in the cell or average if already present
                    if cell_key not in vector_field:
                        vector_field[cell_key] = (normal, 1)
                    else:
                        current_vector, count = vector_field[cell_key]
                        averaged_vector = (current_vector * count + normal) / (count + 1)
                        vector_field[cell_key] = (averaged_vector, count + 1)

            else:
                point = point + 0.005 * normal

            new_ventricle_points.append(point)

        new_ventricle_points = np.array(new_ventricle_points)

        # Resample points
        resampled_points = resample_points(new_ventricle_points, density)
        ventricle_x, ventricle_y = resampled_points[:, 0], resampled_points[:, 1]

        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.plot(white_matter_x, white_matter_y, color='blue', linewidth=2, label='Original White Matter Boundary')
        ax.plot(offset_white_matter_x, offset_white_matter_y, color='green', linestyle='--', linewidth=1.5, label='Offset White Matter Boundary')
        ax.fill(initial_ventricle_x, initial_ventricle_y, color='gray', alpha=0.4, label='Initial Ventricle')
        ax.plot(ventricle_x, ventricle_y, color='red', linewidth=1, linestyle='-', label=f'Boundary at Step {step}')

        # Plot vector field
        for (cx, cy), (vector, count) in vector_field.items():
            ax.arrow(cx, cy, vector[0] * 0.02, vector[1] * 0.02,  # Adjust scaling factor as needed
                     head_width=0.01, head_length=0.01, fc='purple', ec='purple', alpha=0.8)

        # Plot cell borders
        xmin, xmax, ymin, ymax = -1.5, 1.5, -1.5, 1.5  # Set bounds based on your coordinate system
        x_cells = np.arange(xmin, xmax, cell_size)
        y_cells = np.arange(ymin, ymax, cell_size)
        for xc in x_cells:
            ax.plot([xc, xc], [ymin, ymax], color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        for yc in y_cells:
            ax.plot([xmin, xmax], [yc, yc], color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'Step {step}')
        plt.legend()
        pdf.savefig()
        frame_filename = f'frame_{step}.png'
        plt.savefig(frame_filename)
        frames.append(Image.open(frame_filename))
        plt.close()

        # Stopping condition
        if close_to_boundary_count >= 0.9 * len(ventricle_points):
            break

if frames:
    frames[0].save(gif_filename, save_all=True, append_images=frames[1:], duration=max(100 // len(frames), 10), loop=0)

for frame in frames:
    os.remove(frame.filename)

print(f"PDF saved to: {pdf_filename}")
print(f"GIF saved to: {gif_filename}")

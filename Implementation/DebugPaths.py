import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import splprep, splev
from PIL import Image
import os
import json
import math

# Function to create an irregular shape (ventricle-like) with more points for smoother calculations
def generate_ventricle_shape():
    t = np.linspace(0, 2 * np.pi, 200)
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

# Function to compute the signed area of a closed curve (shoelace formula)
def compute_signed_area(x, y):
    x = np.append(x, x[0])  # Close the loop
    y = np.append(y, y[0])
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

# Ensure the curve is clockwise-oriented
def ensure_clockwise_orientation(x, y):
    signed_area = compute_signed_area(x, y)
    if signed_area > 0:  # Counterclockwise
        x = x[::-1]
        y = y[::-1]
    return x, y

# Compute normals pointing to the left of the tangent vector for a clockwise curve
def compute_outward_normals_cw(x, y):
    normals = []
    for i in range(len(x)):
        next_idx = (i + 1) % len(x)
        prev_idx = (i - 1) % len(x)

        # Compute tangent vector
        tangent = np.array([x[next_idx] - x[prev_idx], y[next_idx] - y[prev_idx]])
        tangent /= np.linalg.norm(tangent)

        # Compute normal vector (perpendicular to tangent, pointing left)
        normal = np.array([-tangent[1], tangent[0]])

        normals.append(normal)

    return np.array(normals)

# Find the intersection of a ray with a closed 2D curve in the normal direction
def find_ray_intersection(point, direction, boundary_points, angle_offsets=None):
    min_distance = float('inf')
    if angle_offsets is None:
        angle_offsets = [0]  # Default: no rotation (single direction)
    for angle in angle_offsets:
        # Rotate the normal direction by the current angle
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                                     [np.sin(angle), np.cos(angle)]])
        rotated_direction = np.dot(rotation_matrix, direction)

        for i in range(len(boundary_points)):
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % len(boundary_points)]
            segment = p2 - p1
            A = np.array([segment, -rotated_direction]).T
            if np.linalg.det(A) == 0:
                continue  # Skip parallel cases
            b = point - p1
            t, u = np.linalg.solve(A, b)
            if 0 <= t <= 1 and u > 0:  # Intersection is within the segment and along the ray
                distance = u
                if distance < min_distance:
                    min_distance = distance
    return min_distance if min_distance != float('inf') else None

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

# Initialize the ventricle and white matter boundary
ventricle_x, ventricle_y = generate_ventricle_shape()
ventricle_x, ventricle_y = ensure_clockwise_orientation(ventricle_x, ventricle_y)
initial_ventricle_x, initial_ventricle_y = ventricle_x.copy(), ventricle_y.copy()
white_matter_x, white_matter_y = generate_white_matter_boundary()
white_matter_points = np.vstack((white_matter_x, white_matter_y)).T
pdf_filename = os.path.join(os.getcwd(), 'ventricle_expansion_json_vector_field.pdf')
gif_filename = os.path.join(os.getcwd(), 'ventricle_expansion_json_vector_field.gif')
frames = []

# Define initial density
density = len(ventricle_x) / np.sum(np.sqrt(np.sum(np.diff(np.vstack((ventricle_x, ventricle_y)).T, axis=0)**2, axis=1)))

# Define vector field data
vector_data = []

# Define the number of steps for the initial phase
initial_phase_steps = 40

# Define initial intersection limit and the minimum limit
intersection_limit = 0.095
min_intersection_limit = 0.005
first_zone_reached = False  # Flag to track if the first zone is reached

# Tracking whether the required criterion is met
reached_zone = False

with PdfPages(pdf_filename) as pdf:
    step = 0
    while True:
        step += 1
        print(f'Number of points at step {step}: {len(ventricle_x)}')
        normals = compute_outward_normals_cw(ventricle_x, ventricle_y)
        ventricle_points = np.vstack((ventricle_x, ventricle_y)).T
        new_ventricle_points = []

        close_to_boundary_count = 0
        reached_stop_criterion = False

        # Gradually increase the required percentage of points in the threshold zone
        required_percentage = 0.8 + 0.2 * (1 - (intersection_limit - min_intersection_limit) / (0.095 - min_intersection_limit))
        required_percentage = min(required_percentage, 1.0)  # Cap at 100%

        for i in range(len(ventricle_points)):
            normal = normals[i]
            point = ventricle_points[i]

            intersection_distance = find_ray_intersection(point, normal, white_matter_points, angle_offsets=None)

            if intersection_distance:
                if intersection_distance <= intersection_limit:
                    if first_zone_reached:
                        angle_offsets = np.radians([-6, -3, 0, 3, 6])
                        intersection_distance = find_ray_intersection(point, normal, white_matter_points, angle_offsets)

                    point = point + 0 * normal
                    close_to_boundary_count += 1
                else:
                    if first_zone_reached:
                        expansion_distance = intersection_distance / 10
                    else:
                        expansion_distance = (
                            intersection_distance / 200 if step <= initial_phase_steps
                            else intersection_distance / 60
                        )
                    point = point + min(expansion_distance, 0.2) * normal

                    vector_data.append({
                        "start_x": ventricle_points[i, 0],
                        "start_y": ventricle_points[i, 1],
                        "end_x": point[0],
                        "end_y": point[1],
                        "vector_x": normal[0],
                        "vector_y": normal[1]
                    })
            else:
                point = point + 0.005 * normal

            if intersection_distance and intersection_distance <= min_intersection_limit:
                reached_stop_criterion = True

            new_ventricle_points.append(point)
        print(f'Close to boundary count: {close_to_boundary_count}')
        new_ventricle_points = np.array(new_ventricle_points)
        resampled_points = resample_points(new_ventricle_points, density)
        ventricle_x, ventricle_y = resampled_points[:, 0], resampled_points[:, 1]

        if not reached_zone and close_to_boundary_count >= required_percentage * len(ventricle_points):
            print(f"{int(required_percentage * 100)}% of points inside {intersection_limit:.2f} region at step {step}. Adjusting intersection limit.")
            intersection_limit = max(intersection_limit - 0.01, min_intersection_limit)
            first_zone_reached = True
            reached_zone = True

        if reached_zone and close_to_boundary_count < required_percentage * len(ventricle_points):
            reached_zone = False

        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.plot(white_matter_x, white_matter_y, color='blue', linewidth=2, label='White Matter Boundary')
        ax.fill(initial_ventricle_x, initial_ventricle_y, color='gray', alpha=0.4, label='Initial Ventricle')
        ax.plot(ventricle_x, ventricle_y, color='red', linewidth=1, linestyle='-', label=f'Boundary at Step {step}')
        for i in range(len(ventricle_points)):
            ax.arrow(ventricle_points[i, 0], ventricle_points[i, 1], normals[i, 0] * 0.02, normals[i, 1] * 0.02,
                     head_width=0.02, head_length=0.02, fc='green', ec='green', alpha=0.6)
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

        if reached_stop_criterion:
            print(f"Stopping criterion reached: a single intersection distance <= {min_intersection_limit:.2f}.")
            break

# Save vector data to JSON
with open('vector_field.json', 'w') as f:
    json.dump(vector_data, f)

if frames:
    frames[0].save(gif_filename, save_all=True, append_images=frames[1:], duration=max(100 // len(frames), 10), loop=0)

for frame in frames:
    os.remove(frame.filename)

print(f"PDF saved to: {pdf_filename}")
print(f"GIF saved to: {gif_filename}")

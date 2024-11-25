import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import json
from tqdm import tqdm

# Function to compute distances of points from the ventricle
def compute_distances_from_ventricle(ventricle_points, side_branches):
    distances = []
    for branch in tqdm(side_branches, desc="Computing distances"):
        branch_distances = [
            min(np.linalg.norm(point - ventricle_points, axis=1)) for point in branch
        ]
        distances.append(branch_distances)
    return distances

# Generate the white matter boundary
def generate_white_matter_boundary():
    t = np.linspace(0, 2 * np.pi, 800)
    r = (0.9 + 0.1 * np.sin(7 * t) + 0.15 * np.cos(11 * t) + 0.1 * np.sin(13 * t + 1.5) + 0.05 * np.cos(17 * t + 0.5))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y

# Original animation: paths grow sequentially
def create_growth_animation(
    ventricle_x, ventricle_y, white_matter_x, white_matter_y, side_branches, gif_path, duration=10, fps=20
):
    total_frames = duration * fps
    branch_frames = np.linspace(0, len(side_branches), total_frames, dtype=int)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.plot(ventricle_x, ventricle_y, label='Ventricular Surface', color='red')
    ax.plot(white_matter_x, white_matter_y, label='White Matter Surface', color='green')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')

    lines = []
    for _ in side_branches:
        line, = ax.plot([], [], color='blue', linewidth=0.2, alpha=0.8)
        lines.append(line)

    def update(frame):
        for i, line in enumerate(lines[:branch_frames[frame]]):
            line.set_data(side_branches[i][:frame, 0], side_branches[i][:frame, 1])
        return lines

    ani = FuncAnimation(fig, update, frames=total_frames, interval=1000 // fps, blit=True)
    writer = PillowWriter(fps=fps)
    print(f"Saving GIF to {gif_path}...")
    ani.save(gif_path, writer=writer)
    plt.close(fig)
    print(f"GIF saved successfully at {gif_path}.")

# Distance-based animation
def create_distance_based_animation(
    ventricle_x, ventricle_y, white_matter_x, white_matter_y, side_branches, distances, gif_path, duration=10, fps=20
):
    total_frames = duration * fps
    max_distance = max(max(d) for d in distances)
    distance_bins = np.linspace(0, max_distance, total_frames)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.plot(ventricle_x, ventricle_y, label='Ventricular Surface', color='red')
    ax.plot(white_matter_x, white_matter_y, label='White Matter Surface', color='green')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')

    lines = []
    for _ in side_branches:
        line, = ax.plot([], [], color='blue', linewidth=0.2, alpha=0.8)
        lines.append(line)

    def update(frame):
        current_bin = distance_bins[frame]
        next_bin = distance_bins[frame + 1] if frame + 1 < total_frames else max_distance

        for i, (line, branch, branch_distances) in enumerate(
            zip(lines, side_branches, distances)
        ):
            points_to_plot = branch[
                (current_bin <= np.array(branch_distances))
                & (np.array(branch_distances) < next_bin)
            ]
            existing_data = line.get_data()
            if points_to_plot.size > 0:
                new_data_x = np.hstack((existing_data[0], points_to_plot[:, 0]))
                new_data_y = np.hstack((existing_data[1], points_to_plot[:, 1]))
                line.set_data(new_data_x, new_data_y)

        return lines

    ani = FuncAnimation(fig, update, frames=total_frames, interval=1000 // fps, blit=True)
    writer = PillowWriter(fps=fps)
    print(f"Saving GIF to {gif_path}...")
    ani.save(gif_path, writer=writer)
    plt.close(fig)
    print(f"GIF saved successfully at {gif_path}.")

# Generate the ventricular surface
def generate_ventricle_shape():
    t = np.linspace(0, 2 * np.pi, 300)
    r = 0.4 + 0.15 * np.sin(3 * t) + 0.1 * np.cos(5 * t)
    x = r * np.cos(t) + 0.08
    y = r * np.sin(t)
    return x, y

# File paths
json_file_path = "C:/Users/Emanuel Wicki/Documents/Neuro Project/NeuroProject/Implementation/paths.json"
output_gif_path1 = "C:/Users/Emanuel Wicki/Documents/Neuro Project/NeuroProject/Implementation/path_growth.gif"
output_gif_path2 = "C:/Users/Emanuel Wicki/Documents/Neuro Project/NeuroProject/Implementation/path_growth_by_radius.gif"

# Load side branches from the saved JSON
with open(json_file_path, 'r') as f:
    paths_data = json.load(f)
side_branches = [np.array([[point["x"], point["y"]] for point in path]) for path in paths_data["paths"]]

# Generate the ventricular shape
ventricle_x, ventricle_y = generate_ventricle_shape()
ventricle_points = np.column_stack((ventricle_x, ventricle_y))

# Generate the white matter boundary
white_matter_x, white_matter_y = generate_white_matter_boundary()

# Compute distances of branch points from the ventricle
distances = compute_distances_from_ventricle(ventricle_points, side_branches)

# Create and save both animations
print("Creating original growth animation...")
create_growth_animation(ventricle_x, ventricle_y, white_matter_x, white_matter_y, side_branches, output_gif_path1, duration=10, fps=20)

print("Creating distance-based growth animation...")
create_distance_based_animation(
    ventricle_x, ventricle_y, white_matter_x, white_matter_y, side_branches, distances, output_gif_path2, duration=10, fps=20
)

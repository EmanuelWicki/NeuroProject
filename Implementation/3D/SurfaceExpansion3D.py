import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


# Generate a 3D sphere mesh using Trimesh
def generate_sphere(center, radius, resolution=40):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v)).ravel()
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v)).ravel()
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v)).ravel()
    vertices = np.vstack((x, y, z)).T

    # Create faces using Delaunay triangulation
    from scipy.spatial import Delaunay
    delaunay = Delaunay(vertices)
    faces = delaunay.simplices

    return trimesh.Trimesh(vertices=vertices, faces=faces)


# Perform multiple expansion steps
def expand_multiple_steps(ventricle, white_matter, steps=10, max_distance=0.05, save_dir="steps_output"):
    ventricle_mesh = ventricle  # Start with the initial ventricle mesh

    # Create directory to save the steps
    os.makedirs(save_dir, exist_ok=True)

    for step in range(steps):
        new_points = []
        normals = ventricle_mesh.vertex_normals  # Compute normals for ventricle surface

        print(f"\n--- Step {step + 1} ---")
        print(f"Number of vertices: {len(ventricle_mesh.vertices)}")

        for i, (point, normal) in enumerate(zip(ventricle_mesh.vertices, normals)):
            # Perform ray intersection along the normal direction
            locations, _, _ = white_matter.ray.intersects_location(
                ray_origins=[point], ray_directions=[normal]
            )

            # Use the closest intersection point
            if len(locations) > 0:
                closest_point = locations[0]
                new_point = point + np.clip(np.linalg.norm(closest_point - point), 0, max_distance) * normal
            else:
                new_point = point + max_distance * normal  # If no intersection, expand outward

            new_points.append(new_point)

            # Debugging output for the first few points
            if i < 5:
                print(f"Point {i}: Original {point}, Normal {normal}, New {new_point}")

        # Update the ventricle mesh for the next step
        ventricle_mesh = trimesh.Trimesh(vertices=np.array(new_points), faces=ventricle_mesh.faces)

        # Save the current step
        step_filename = os.path.join(save_dir, f"step_{step + 1}.obj")
        ventricle_mesh.export(step_filename)
        print(f"Step {step + 1} saved to {step_filename}")

    return ventricle_mesh


# Visualize the spheres after all expansion steps
def visualize_multiple_steps(ventricle, white_matter, expanded_ventricle):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot white matter boundary
    ax.scatter(
        white_matter.vertices[:, 0],
        white_matter.vertices[:, 1],
        white_matter.vertices[:, 2],
        c='blue', s=1, label='White Matter'
    )

    # Plot initial ventricle
    ax.scatter(
        ventricle.vertices[:, 0],
        ventricle.vertices[:, 1],
        ventricle.vertices[:, 2],
        c='red', s=1, label='Initial Ventricle'
    )

    # Plot expanded ventricle
    ax.scatter(
        expanded_ventricle.vertices[:, 0],
        expanded_ventricle.vertices[:, 1],
        expanded_ventricle.vertices[:, 2],
        c='green', s=1, label='Expanded Ventricle'
    )

    ax.legend()
    plt.title("Ventricle Expansion Across Steps")
    plt.show()


# Main workflow
ventricle = generate_sphere(center=(0, 0, 0), radius=0.3, resolution=40)
white_matter = generate_sphere(center=(0, 0, 0), radius=1.0, resolution=40)

# Perform multiple expansion steps and save each step
expanded_ventricle = expand_multiple_steps(ventricle, white_matter, steps=10, max_distance=0.05, save_dir="steps_output")

# Visualize results
visualize_multiple_steps(ventricle, white_matter, expanded_ventricle)

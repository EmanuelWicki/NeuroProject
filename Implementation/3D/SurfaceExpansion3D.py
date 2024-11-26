import numpy as np
import trimesh
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


# Generate a 3D sphere mesh using Trimesh
def generate_sphere(center, radius, resolution=40):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)
    x = center[0] + radius * np.sin(v) * np.cos(u)
    y = center[1] + radius * np.sin(v) * np.sin(u)
    z = center[2] + radius * np.cos(v)
    vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Create faces using Delaunay triangulation
    points2D = np.vstack((u.flatten(), v.flatten())).T  # Use parameter space for triangulation
    delaunay = Delaunay(points2D)
    faces = delaunay.simplices

    return trimesh.Trimesh(vertices=vertices, faces=faces)


# Generate a flower-shaped white matter boundary
def generate_flower_shape(center, radius, resolution=100, petal_amplitude=0.2, petal_frequency=5):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)

    # Modify the radius to create petals
    r = radius + petal_amplitude * np.sin(petal_frequency * v) * np.sin(petal_frequency * u)

    x = center[0] + r * np.sin(v) * np.cos(u)
    y = center[1] + r * np.sin(v) * np.sin(u)
    z = center[2] + r * np.cos(v)

    vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Create faces using Delaunay triangulation in parameter space
    points2D = np.vstack((u.flatten(), v.flatten())).T
    delaunay = Delaunay(points2D)
    faces = delaunay.simplices

    return trimesh.Trimesh(vertices=vertices, faces=faces)


# Expansion process with adaptive step size
def expand_ventricle(ventricle, white_matter, steps=20, fraction=0.2):
    ventricle_mesh = ventricle.copy()  # Start with the initial ventricle mesh
    meshes = [ventricle_mesh]  # Save meshes for animation
    paths = {i: [vertex] for i, vertex in enumerate(ventricle.vertices)}  # Track paths of each vertex

    for step in range(steps):
        new_points = []

        # Calculate normals as vectors pointing outward from the centroid
        normals = (ventricle_mesh.vertices - ventricle_mesh.centroid)
        normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize to unit length

        print(f"\n--- Step {step + 1} ---")
        print(f"Number of vertices: {len(ventricle_mesh.vertices)}")

        for i, (point, normal) in enumerate(zip(ventricle_mesh.vertices, normals)):
            # Perform ray intersection along the normal direction
            locations, _, _ = white_matter.ray.intersects_location(
                ray_origins=[point], ray_directions=[normal]
            )

            # Compute new position based on intersection distance
            if len(locations) > 0:
                closest_point = locations[0]
                distance = np.linalg.norm(closest_point - point)
                step_distance = distance * fraction  # Fraction of the intersection distance
                new_point = point + step_distance * normal
            else:
                # If no intersection, move outward with a small step
                new_point = point + 0.01 * normal

            new_points.append(new_point)

            # Track the path of each point
            paths[i].append(new_point)

        # Update the ventricle mesh with new points
        ventricle_mesh = trimesh.Trimesh(vertices=np.array(new_points), faces=ventricle_mesh.faces)
        meshes.append(ventricle_mesh)  # Save the updated mesh for animation

    return meshes, paths


# Create an animation of the expansion process
def animate_expansion(meshes):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Initialize the trisurf plot
    trisurf = None

    def update(frame):
        nonlocal trisurf
        ax.clear()  # Clear the plot
        mesh = meshes[frame]
        trisurf = ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            triangles=mesh.faces,
            color="green",
            alpha=0.6,
            edgecolor="grey",
            linewidth=0.2,
        )
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_title(f"Expansion Step {frame + 1}")

    ani = FuncAnimation(fig, update, frames=len(meshes), interval=200, repeat=True)
    plt.show()


# Main workflow
if __name__ == "__main__":
    # Generate initial ventricle (sphere)
    ventricle = generate_sphere(center=(0, 0, 0), radius=0.3, resolution=40)

    # Generate flower-shaped white matter boundary
    white_matter = generate_flower_shape(
        center=(0, 0, 0),
        radius=1.0,
        resolution=100,
        petal_amplitude=0.2,
        petal_frequency=5
    )

    # Perform expansion process
    meshes, paths = expand_ventricle(ventricle, white_matter, steps=50, fraction=0.02)

    # Animate the expansion process
    animate_expansion(meshes)

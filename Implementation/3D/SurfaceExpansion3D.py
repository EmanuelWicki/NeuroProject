import numpy as np
import trimesh
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os


# Generate a bumpy sphere mesh
def generate_bumpy_sphere(center, radius, resolution=40, bump_amplitude=0.03, bump_frequency=3, asymmetry=0.02):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)

    # Add sinusoidal bumps to the radius with asymmetry
    r = radius + bump_amplitude * np.sin(bump_frequency * v) * np.sin(bump_frequency * u)
    r += asymmetry * np.sin(v)  # Add asymmetry in the polar direction

    x = center[0] + r * np.sin(v) * np.cos(u)
    y = center[1] + r * np.sin(v) * np.sin(u)
    z = center[2] + r * np.cos(v)

    vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Create faces using Delaunay triangulation in parameter space
    points2D = np.vstack((u.flatten(), v.flatten())).T
    delaunay = Delaunay(points2D)
    faces = delaunay.simplices

    return trimesh.Trimesh(vertices=vertices, faces=faces)


# Generate a flower-shaped white matter boundary
def generate_flower_shape(center, radius, resolution=100, petal_amplitude=1, petal_frequency=5):
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


# Expansion process with curvature
def expand_ventricle_with_curvature(ventricle, white_matter, steps=20, fraction=0.2):
    ventricle_mesh = ventricle.copy()  # Start with the initial ventricle mesh
    meshes = [ventricle_mesh]  # Save meshes for animation
    paths = {i: [vertex] for i, vertex in enumerate(ventricle.vertices)}  # Track paths of each vertex

    for step in range(steps):
        new_points = []

        # Calculate normals as vectors pointing outward from the centroid
        normals = (ventricle_mesh.vertices - ventricle_mesh.centroid)
        normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize to unit length

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
                direction = normal * step_distance  # Compute step direction
            else:
                # If no intersection, move outward with a small step
                direction = normal * 0.0

            # Add the new step to the cumulative path
            new_point = point + direction
            paths[i].append(new_point)

            new_points.append(new_point)
            print(f'Step: {step}/{steps}, points: {len(new_points)}')

        # Update the ventricle mesh with new points
        ventricle_mesh = trimesh.Trimesh(vertices=np.array(new_points), faces=ventricle_mesh.faces)
        meshes.append(ventricle_mesh)  # Save the updated mesh for animation

    return meshes, paths

# Plot the paths of vertices
def plot_vertex_paths(paths):
    """
    Plot the paths of the vertices during the expansion process.

    Parameters:
    - paths: Dictionary containing the paths of each vertex.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each vertex's path
    for path in paths.values():
        path = np.array(path)  # Convert to a numpy array for easier plotting
        ax.plot(path[:, 0], path[:, 1], path[:, 2], alpha=0.7, linewidth=0.8, color="orange")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_title("Vertex Paths During Expansion")
    plt.show()


# Create an animation of the expansion process with the white matter boundary
def animate_expansion_with_white_matter(meshes, white_matter, initial_surface_frames=20, save_as_gif=False, gif_filename="expansion_with_boundary.gif"):
    """
    Animate the expansion process, starting with the initial ventricular surface and including the white matter boundary.

    Parameters:
    - meshes: List of trimesh objects representing the expansion steps.
    - white_matter: Trimesh object representing the white matter boundary.
    - initial_surface_frames: Number of frames to show the initial ventricular surface.
    - save_as_gif: Boolean, whether to save the animation as a GIF.
    - gif_filename: Filename for the saved GIF.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()  # Clear the plot

        # Plot the white matter boundary as a transparent surface
        ax.plot_trisurf(
            white_matter.vertices[:, 0],
            white_matter.vertices[:, 1],
            white_matter.vertices[:, 2],
            triangles=white_matter.faces,
            color="blue",
            alpha=0.1,  # Set transparency
            edgecolor="grey",
            linewidth=0.1
        )

        # Display the initial surface for a few frames
        if frame < initial_surface_frames:
            mesh = meshes[0]
            ax.set_title("Initial Ventricle Surface")
        else:
            mesh = meshes[frame - initial_surface_frames]
            ax.set_title(f"Expansion Step {frame - initial_surface_frames + 1}")

        # Plot the expanding ventricle surface
        ax.plot_trisurf(
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

    ani = FuncAnimation(fig, update, frames=len(meshes) + initial_surface_frames, interval=200, repeat=True)

    if save_as_gif:
        writer = PillowWriter(fps=10)
        ani.save(gif_filename, writer=writer)
        print(f"Animation saved as {gif_filename}")
    else:
        plt.show()


# Main workflow
if __name__ == "__main__":
    # Generate initial ventricular surface (asymmetric bumpy sphere)
    ventricle = generate_bumpy_sphere(
        center=(0, 0, 0), radius=0.3, resolution=40, bump_amplitude=0.03, bump_frequency=3, asymmetry=0.02
    )

    # Generate flower-shaped white matter boundary
    white_matter = generate_flower_shape(
        center=(0, 0, 0),
        radius=1.0,
        resolution=100,
        petal_amplitude=0.8,
        petal_frequency=5
    )

    # Perform expansion process
    meshes, paths = expand_ventricle_with_curvature(ventricle, white_matter, steps=200, fraction=0.02)

    # Animate the expansion process with the initial ventricular surface
    animate_expansion_with_white_matter(meshes, white_matter, initial_surface_frames=20, save_as_gif=True, gif_filename="expansion_with_boundary.gif")

    # Plot the vertex paths
    plot_vertex_paths(paths)

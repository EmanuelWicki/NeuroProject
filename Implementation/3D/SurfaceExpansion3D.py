import numpy as np
import trimesh
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# Generate a bumpy sphere mesh
def generate_bumpy_sphere(center, radius, resolution=100, bump_amplitude=0.03, bump_frequency=3, asymmetry=0.02):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)

    r = radius + bump_amplitude * np.sin(bump_frequency * v) * np.sin(bump_frequency * u)
    r += asymmetry * np.sin(v)

    x = center[0] + r * np.sin(v) * np.cos(u)
    y = center[1] + r * np.sin(v) * np.sin(u)
    z = center[2] + r * np.cos(v)

    vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    points2D = np.vstack((u.flatten(), v.flatten())).T
    delaunay = Delaunay(points2D)
    faces = delaunay.simplices

    return trimesh.Trimesh(vertices=vertices, faces=faces)


# Generate a flower-shaped white matter boundary
def generate_flower_shape(center, radius, resolution=100, petal_amplitude=1, petal_frequency=5):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)

    r = radius + petal_amplitude * np.sin(petal_frequency * v) * np.sin(petal_frequency * u)

    x = center[0] + r * np.sin(v) * np.cos(u)
    y = center[1] + r * np.sin(v) * np.sin(u)
    z = center[2] + r * np.cos(v)

    vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    points2D = np.vstack((u.flatten(), v.flatten())).T
    delaunay = Delaunay(points2D)
    faces = delaunay.simplices

    return trimesh.Trimesh(vertices=vertices, faces=faces)


# Expansion process with curvature and voxelized vector field
def expand_ventricle_with_vector_field(ventricle, white_matter, steps=20, fraction=0.2, voxel_size=0.1):
    ventricle_mesh = ventricle.copy()
    meshes = [ventricle_mesh]
    paths = {i: [vertex] for i, vertex in enumerate(ventricle.vertices)}
    vectors = {}

    print(f'Expansion Started')

    for step in range(steps):
        new_points = []
        normals = (ventricle_mesh.vertices - ventricle_mesh.centroid)
        normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
        print(f'step: {step}/{steps}')

        for i, (point, normal) in enumerate(zip(ventricle_mesh.vertices, normals)):
            locations, _, _ = white_matter.ray.intersects_location(ray_origins=[point], ray_directions=[normal])

            if len(locations) > 0:
                closest_point = locations[0]
                distance = np.linalg.norm(closest_point - point)
                step_distance = distance * fraction
                direction = normal * step_distance
            else:
                direction = normal * 0.01

            new_point = point + direction
            paths[i].append(new_point)

            voxel = tuple((new_point // voxel_size).astype(int))
            vector = direction

            if voxel not in vectors:
                vectors[voxel] = [vector]
            else:
                vectors[voxel].append(vector)

            new_points.append(new_point)

        ventricle_mesh = trimesh.Trimesh(vertices=np.array(new_points), faces=ventricle_mesh.faces)
        meshes.append(ventricle_mesh)
        print(f'Expansion points: {len(new_points)}')

    averaged_vectors = {voxel: np.mean(vectors[voxel], axis=0) for voxel in vectors}

    return meshes, paths, averaged_vectors


# Create an animation of the expansion process with the white matter boundary
def animate_expansion_with_white_matter(meshes, white_matter, initial_surface_frames=20, save_as_gif=False, gif_filename="expansion_with_boundary.gif"):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()

        # Plot the white matter boundary as a transparent surface
        ax.plot_trisurf(
            white_matter.vertices[:, 0],
            white_matter.vertices[:, 1],
            white_matter.vertices[:, 2],
            triangles=white_matter.faces,
            color="blue",
            alpha=0.1,
            edgecolor="grey",
            linewidth=0.1,
        )

        # Display the initial surface for a few frames
        if frame < initial_surface_frames:
            mesh = meshes[0]
            ax.set_title("Initial Ventricle Surface")
        else:
            mesh = meshes[frame - initial_surface_frames]
            ax.set_title(f"Expansion Step {frame - initial_surface_frames + 1}")

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


# Plot the voxelized vector field with better-scaled arrows
def plot_vector_field(vectors, voxel_size):
    """
    Plot the voxelized vector field as arrows with improved scaling and visibility.

    Parameters:
    - vectors: Dictionary containing the voxel positions and their averaged vectors.
    - voxel_size: Float, size of each voxel in the grid.
    """
    print('Start of Vectorfield Generation')
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    for voxel, vector in vectors.items():
        origin = np.array(voxel) * voxel_size  # Compute the origin of the vector
        
        # Normalize the vector for consistent scaling
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm  # Normalize direction
        
        ax.quiver(
            origin[0], origin[1], origin[2],  # Origin of the vector
            vector[0], vector[1], vector[2],  # Vector components
            length=0.1,                      # Adjusted length for better representation
            color="green",
            alpha=0.8,                        # Higher alpha for better visibility
            linewidth=1.5,                    # Slightly thicker arrows
            arrow_length_ratio=0.3            # Adjust arrowhead size
        )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_box_aspect([1, 1, 1])  # Equal scaling for x, y, z axes
    ax.set_title("Voxelized Vector Field with Arrows")
    plt.show()



# Plot the paths of vertices
def plot_vertex_paths(paths):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    for path in paths.values():
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], alpha=0.7, linewidth=0.8, color="orange")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_title("Vertex Paths During Expansion")
    plt.show()


# Main workflow
if __name__ == "__main__":
    ventricle = generate_bumpy_sphere(center=(0, 0, 0), radius=0.3)
    white_matter = generate_flower_shape(center=(0, 0, 0), radius=1.0)

    meshes, paths, vectors = expand_ventricle_with_vector_field(
        ventricle, white_matter, steps=50, fraction=0.02, voxel_size=0.1
    )

    animate_expansion_with_white_matter(meshes, white_matter, initial_surface_frames=20, save_as_gif=True)
    # plot_vertex_paths(paths)
    plot_vector_field(vectors, voxel_size=0.1)

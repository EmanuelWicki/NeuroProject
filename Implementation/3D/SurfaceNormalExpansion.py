import numpy as np
import trimesh
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# Generate a bumpy sphere mesh
def generate_bumpy_sphere(center, radius, resolution=40, bump_amplitude=0.03, bump_frequency=3, asymmetry=0.02):
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
def generate_flower_shape(center, radius, resolution=150, petal_amplitude=1, petal_frequency=5):
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


# Calculate face normals
def calculate_face_normals(vertices, faces):
    tris = vertices[faces]
    edge1 = tris[:, 1] - tris[:, 0]
    edge2 = tris[:, 2] - tris[:, 0]
    face_normals = np.cross(edge1, edge2)
    lengths = np.linalg.norm(face_normals, axis=1)

    nonzero_mask = lengths > 1e-6
    face_normals[nonzero_mask] /= lengths[nonzero_mask][:, np.newaxis]
    face_normals[~nonzero_mask] = 0
    return face_normals


# Calculate corrected vertex normals
def calculate_corrected_vertex_normals(vertices, faces):
    centroid = np.mean(vertices, axis=0)

    face_normals = calculate_face_normals(vertices, faces)
    vertex_normals = np.zeros_like(vertices)

    for i, face in enumerate(faces):
        for j in range(3):
            vertex_normals[face[j]] += face_normals[i]

    lengths = np.linalg.norm(vertex_normals, axis=1)
    nonzero_mask = lengths > 1e-6
    vertex_normals[nonzero_mask] /= lengths[nonzero_mask][:, np.newaxis]
    vertex_normals[~nonzero_mask] = 0

    for i, (vertex, normal) in enumerate(zip(vertices, vertex_normals)):
        to_centroid = vertex - centroid
        if np.dot(normal, to_centroid) < 0:
            vertex_normals[i] = -normal

    return vertex_normals


def laplacian_smoothing(mesh, iterations=2, lambda_factor=0.9):
    """
    Perform Laplacian smoothing on a mesh.

    Parameters:
    - mesh: A trimesh object.
    - iterations: Number of smoothing iterations.
    - lambda_factor: Smoothing factor (0 < lambda < 1).

    Returns:
    - Smoothed trimesh object.
    """
    vertices = mesh.vertices.copy()
    edges = mesh.edges_unique

    adjacency_dict = {i: [] for i in range(len(vertices))}
    for edge in edges:
        adjacency_dict[edge[0]].append(edge[1])
        adjacency_dict[edge[1]].append(edge[0])

    for _ in range(iterations):
        new_vertices = vertices.copy()
        
        for vertex_index, neighbors in adjacency_dict.items():
            if neighbors:
                neighbor_positions = vertices[neighbors]
                centroid = neighbor_positions.mean(axis=0)
                new_vertices[vertex_index] = (
                    (1 - lambda_factor) * vertices[vertex_index] + lambda_factor * centroid
                )
        
        vertices = new_vertices

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)



# Expand ventricle with smoothing
def expand_ventricle_fixed_iterations(ventricle, white_matter, fraction=0.02, voxel_size=0.1, max_iterations=10):
    ventricle_mesh = ventricle.copy()
    meshes = [ventricle_mesh]
    paths = {i: [vertex] for i, vertex in enumerate(ventricle.vertices)}
    vectors_per_step = []

    print("Expansion Started")

    for step in range(max_iterations):
        new_points = []
        step_vectors = []

        normals = calculate_corrected_vertex_normals(ventricle_mesh.vertices, ventricle_mesh.faces)

        print(f"Step: {step + 1}, Expanding...")

        for i, (point, normal) in enumerate(zip(ventricle_mesh.vertices, normals)):
            locations, _, _ = white_matter.ray.intersects_location(ray_origins=[point], ray_directions=[normal])

            if len(locations) > 0:
                closest_point = locations[0]
                distance = np.linalg.norm(closest_point - point)

                step_distance = distance * fraction
                direction = normal * step_distance
            else:
                direction = normal * 0.0

            new_point = point + direction
            paths[i].append(new_point)

            step_vectors.append((point, direction))
            new_points.append(new_point)

        vectors_per_step.append(step_vectors)
        ventricle_mesh = trimesh.Trimesh(vertices=np.array(new_points), faces=ventricle_mesh.faces)

        # Apply Laplacian smoothing
        ventricle_mesh = laplacian_smoothing(ventricle_mesh, iterations=3, lambda_factor=0.7)
        meshes.append(ventricle_mesh)

        print(f"Step {step + 1} completed. Points updated: {len(new_points)}")

    return meshes, paths, vectors_per_step


# Main workflow
if __name__ == "__main__":
    ventricle = generate_bumpy_sphere(center=(0, 0, 0), radius=0.3)
    white_matter = generate_flower_shape(center=(0, 0, 0), radius=1.0)

    meshes, paths, vectors_per_step = expand_ventricle_fixed_iterations(
        ventricle, white_matter, fraction=0.1, voxel_size=0.05, max_iterations=30
    )

    # Plot results for verification
    for step, mesh in enumerate(meshes):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            triangles=mesh.faces,
            color="green",
            alpha=0.6,
            edgecolor="grey",
            linewidth=0.03,
        )
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_title(f"Expansion Step {step}")
        plt.savefig(f"expansion_step_{step}.png")
        plt.close()

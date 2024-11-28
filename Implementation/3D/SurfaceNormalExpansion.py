import numpy as np
import trimesh
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import pickle
import logging

# Generate a bumpy sphere mesh
def generate_bumpy_sphere(center, radius, resolution=20, bump_amplitude=0.03, bump_frequency=3, asymmetry=0.02):
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


def laplacian_smoothing(mesh, iterations=3, lambda_factor=0.8):
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


# Resample the mesh to maintain consistent density
def resample_mesh(mesh, target_edge_length):
    vertices, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=target_edge_length)
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def save_mesh(mesh, step):
    """
    Save the current mesh to a file.
    """
    filename = f"results/mesh_step_{step}.obj"
    mesh.export(filename)
    logging.info(f"Mesh saved: {filename}")

# Expand ventricle with smoothing and resampling
def expand_ventricle_fixed_iterations(ventricle, white_matter, fraction=0.02, target_edge_length=0.05, max_iterations=10):
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
            if i in paths:
                paths[i].append(new_point)
            else:
                paths[i] = [new_point]

            step_vectors.append((point, direction))
            new_points.append(new_point)

        vectors_per_step.append(step_vectors)
        ventricle_mesh = trimesh.Trimesh(vertices=np.array(new_points), faces=ventricle_mesh.faces)

        # Apply Laplacian smoothing
        ventricle_mesh = laplacian_smoothing(ventricle_mesh, iterations=3, lambda_factor=0.8)

        # Resample the mesh to maintain density
        resampled_mesh = resample_mesh(ventricle_mesh, target_edge_length)

        # Update paths for resampled vertices
        new_paths = {}
        for i, vertex in enumerate(resampled_mesh.vertices):
            if i < len(paths):
                new_paths[i] = paths[i]
            else:
                new_paths[i] = [vertex]  # Initialize new vertex paths

        paths = new_paths
        ventricle_mesh = resampled_mesh
        meshes.append(ventricle_mesh)
        
        save_mesh(ventricle_mesh, step + 1)

        print(f"Step {step + 1} completed. Points updated: {len(ventricle_mesh.vertices)}")

    return meshes, paths, vectors_per_step


# Function to get total displacement vectors
def get_total_displacement_vectors(paths):
    """
    Get the total displacement vectors for each vertex.

    Parameters:
    - paths: Dictionary where keys are vertex indices and values are lists of positions over time.

    Returns:
    - positions: (N, 3) array of initial positions.
    - displacement_vectors: (N, 3) array of displacement vectors.
    """
    positions = []
    displacement_vectors = []
    for i in paths:
        initial_position = paths[i][0]
        final_position = paths[i][-1]
        displacement_vector = final_position - initial_position
        positions.append(initial_position)
        displacement_vectors.append(displacement_vector)
    positions = np.array(positions)
    displacement_vectors = np.array(displacement_vectors)
    return positions, displacement_vectors


# Function to generate vector field
def generate_vector_field(positions, displacement_vectors, grid_resolution=50):
    """
    Generate a vector field from the displacement vectors of the mesh vertices.

    Parameters:
    - positions: (N, 3) array of initial positions.
    - displacement_vectors: (N, 3) array of displacement vectors.
    - grid_resolution: Number of grid points along each axis.

    Returns:
    - grid_points: (M, 3) array of grid point coordinates.
    - vectors: (M, 3) array of vectors at the grid points.
    """
    # Define the grid covering the space
    min_coords = positions.min(axis=0)
    max_coords = positions.max(axis=0)

    xi = np.linspace(min_coords[0], max_coords[0], grid_resolution)
    yi = np.linspace(min_coords[1], max_coords[1], grid_resolution)
    zi = np.linspace(min_coords[2], max_coords[2], grid_resolution)
    X, Y, Z = np.meshgrid(xi, yi, zi, indexing='ij')

    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Interpolate the displacement vectors onto the grid
    from scipy.interpolate import griddata
    vectors_x = griddata(positions, displacement_vectors[:, 0], grid_points, method='linear', fill_value=0)
    vectors_y = griddata(positions, displacement_vectors[:, 1], grid_points, method='linear', fill_value=0)
    vectors_z = griddata(positions, displacement_vectors[:, 2], grid_points, method='linear', fill_value=0)

    vectors = np.vstack([vectors_x, vectors_y, vectors_z]).T

    return grid_points, vectors

# Visualize the 3D vector field
def visualize_3D_vector_field(grid_points, vectors, voxel_size=0.1):
    """
    Visualize the 3D vector field with vectors confined within their voxels.

    Parameters:
    - grid_points: (M, 3) array of grid point coordinates.
    - vectors: (M, 3) array of vectors at the grid points.
    - voxel_size: The size of each voxel to scale vector length.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize vectors for visualization within voxels
    vector_magnitudes = np.linalg.norm(vectors, axis=1)
    vector_magnitudes[vector_magnitudes == 0] = 1  # Avoid division by zero
    normalized_vectors = (vectors.T / vector_magnitudes).T * voxel_size

    # Add quiver for vector visualization
    ax.quiver(
        grid_points[:, 0], grid_points[:, 1], grid_points[:, 2],
        normalized_vectors[:, 0], normalized_vectors[:, 1], normalized_vectors[:, 2],
        length=1.0, normalize=False, color='blue', linewidth=0.5, arrow_length_ratio=0.2
    )

    # Set axis labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Vector Field Visualization")
    plt.show()


# Main workflow
if __name__ == "__main__":
    ventricle = generate_bumpy_sphere(center=(0, 0, 0), radius=0.3)
    white_matter = generate_flower_shape(center=(0, 0, 0), radius=1.0)

    # Measure initial average edge length
    initial_avg_edge_length = np.mean([np.linalg.norm(ventricle.vertices[edge[0]] - ventricle.vertices[edge[1]]) for edge in ventricle.edges_unique])

    meshes, paths, vectors_per_step = expand_ventricle_fixed_iterations(
        ventricle, white_matter, fraction=0.1, target_edge_length=initial_avg_edge_length, max_iterations=10
    )

    # Collect total displacement vectors
    positions, displacement_vectors = get_total_displacement_vectors(paths)

    # Generate vector field
    grid_points, vectors = generate_vector_field(positions, displacement_vectors, grid_resolution=50)

    # Visualize the 3D vector field
    # visualize_3D_vector_field(grid_points, vectors, voxel_size=0.02)
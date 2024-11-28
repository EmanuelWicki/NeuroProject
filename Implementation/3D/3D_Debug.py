import numpy as np
import trimesh
import random
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


# Generate a bumpy sphere mesh
def generate_bumpy_sphere(center, radius, resolution=10, bump_amplitude=0.03, bump_frequency=3, asymmetry=0.02):
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


# Laplacian smoothing
def laplacian_smoothing(mesh, iterations=3, lambda_factor=0.5):
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


# Visualization for a single vertex
def visualize_single_vertex_expansion_as_mesh(ventricular_mesh, white_matter_mesh, vertex_index, step_vector, intersection_point, step):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot ventricular mesh as a surface
    ax.plot_trisurf(
        ventricular_mesh.vertices[:, 0],
        ventricular_mesh.vertices[:, 1],
        ventricular_mesh.vertices[:, 2],
        triangles=ventricular_mesh.faces,
        color="blue",
        alpha=0.4,
        edgecolor="gray",
        linewidth=0.5,
        label="Ventricular Mesh"
    )

    # Plot white matter mesh as a surface with alpha=0.3
    ax.plot_trisurf(
        white_matter_mesh.vertices[:, 0],
        white_matter_mesh.vertices[:, 1],
        white_matter_mesh.vertices[:, 2],
        triangles=white_matter_mesh.faces,
        color="green",
        alpha=0.05,
        edgecolor="gray",
        linewidth=0.5,
        label="White Matter Mesh (Outer)"
    )

    # Highlight the selected vertex on the ventricular mesh
    selected_vertex = ventricular_mesh.vertices[vertex_index]
    ax.scatter(
        selected_vertex[0], selected_vertex[1], selected_vertex[2],
        color="red",
        s=50,
        label="Selected Vertex"
    )

    # Highlight the intersection point on the white matter mesh
    ax.scatter(
        intersection_point[0], intersection_point[1], intersection_point[2],
        color="yellow",
        s=50,
        label="Intersection Point"
    )

    # Plot the direction vector for expansion
    ax.quiver(
        selected_vertex[0],
        selected_vertex[1],
        selected_vertex[2],
        step_vector[0],
        step_vector[1],
        step_vector[2],
        color="red",
        linewidth=2,
        label="Expansion Direction Vector"
    )

    # Plot ray to the intersection point
    ray_vector = intersection_point - selected_vertex
    ax.quiver(
        selected_vertex[0],
        selected_vertex[1],
        selected_vertex[2],
        ray_vector[0],
        ray_vector[1],
        ray_vector[2],
        color="purple",
        linestyle="dotted",
        label="Ray to Intersection"
    )

    # Set plot title and labels
    ax.set_title(f"Step {step} - Vertex Expansion Visualization")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend(loc="upper right")
    plt.show()


# Expand ventricle
def expand_ventricle_fixed_iterations(ventricle, white_matter, fraction=0.02, target_edge_length=0.05, max_iterations=10):
    ventricle_mesh = ventricle.copy()
    paths = {i: [vertex] for i, vertex in enumerate(ventricle.vertices)}

    # Select 10 random unique vertex indices
    random_vertex_indices = random.sample(range(len(ventricle_mesh.vertices)), 5)

    for step in range(max_iterations):
        normals = calculate_corrected_vertex_normals(ventricle_mesh.vertices, ventricle_mesh.faces)
        new_points = []
        step_vectors = []

        for i, (point, normal) in enumerate(zip(ventricle_mesh.vertices, normals)):
            locations, _, _ = white_matter.ray.intersects_location(ray_origins=[point], ray_directions=[normal])

            if len(locations) > 0:
                closest_point = locations[0]
                distance = np.linalg.norm(closest_point - point)
                step_distance = distance * fraction
                direction = normal * step_distance
            else:
                closest_point = point
                direction = normal * 0.0

            new_point = point + direction
            new_points.append(new_point)
            step_vectors.append(direction)

            # Visualize for the selected random vertices in the first step
            if step == 0 and i in random_vertex_indices:
                visualize_single_vertex_expansion_as_mesh(
                    ventricle_mesh,
                    white_matter,
                    vertex_index=i,
                    step_vector=direction,
                    intersection_point=closest_point,
                    step=step + 1
                )

        ventricle_mesh = trimesh.Trimesh(vertices=np.array(new_points), faces=ventricle_mesh.faces)

    return ventricle_mesh


# Main workflow
if __name__ == "__main__":
    ventricle = generate_bumpy_sphere(center=(0, 0, 0), radius=0.3)
    white_matter = generate_flower_shape(center=(0, 0, 0), radius=1.0)

    expand_ventricle_fixed_iterations(
        ventricle=ventricle,
        white_matter=white_matter,
        fraction=0.02,
        target_edge_length=0.05,
        max_iterations=1
    )

import numpy as np
import os
from scipy.spatial import Delaunay
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from trimesh.ray.ray_triangle import RayMeshIntersector

def generate_flower_shape(center, radius, resolution=200, petal_amplitude=1, petal_frequency=3):
    """
    Generate a flower-shaped mesh.
    """
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

def plot_mesh_with_expanded_face(mesh, expanded_mesh, face_index):
    """
    Plot the original mesh, highlight one expanded face, and split it into smaller triangles using centroid.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original mesh
    ax.plot_trisurf(
        mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
        triangles=mesh.faces, color='blue', alpha=0.5, label='Original Mesh'
    )

    # Plot the expanded face in red
    face_vertices = expanded_mesh.vertices[expanded_mesh.faces[face_index]]
    original_face_vertices = mesh.vertices[mesh.faces[face_index]]

    # Highlight the expanded face vertices
    ax.scatter(
        face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2],
        color='red', s=50, label='Expanded Face Vertices'
    )

    # Calculate the centroid of the face
    centroid = np.mean(face_vertices, axis=0)
    ax.scatter(
        centroid[0], centroid[1], centroid[2],
        color='green', s=100, label='Centroid'
    )

    # Split the face into smaller triangles using the centroid
    split_faces = []
    for i in range(len(face_vertices)):
        next_index = (i + 1) % len(face_vertices)
        triangle = [face_vertices[i], face_vertices[next_index], centroid]
        split_faces.append(triangle)

    # Plot all smaller triangles to fully cover the expanded face
    for triangle in split_faces:
        ax.add_collection3d(Poly3DCollection([triangle], color='orange', alpha=0.7))

    # Overlay the original face for clarity
    ax.add_collection3d(Poly3DCollection([original_face_vertices], color='blue', alpha=0.3, linewidths=1, edgecolors='black', label='Original Face'))

    # Connect original vertices to expanded vertices
    for i in range(len(face_vertices)):
        line = np.vstack((original_face_vertices[i], face_vertices[i]))
        ax.add_collection3d(Line3DCollection([line], colors='black', linewidths=1))

    # Adjust view to focus on the expanded face
    center = np.mean(face_vertices, axis=0)
    ax.set_xlim(center[0] - 0.4, center[0] + 0.4)
    ax.set_ylim(center[1] - 0.4, center[1] + 0.4)
    ax.set_zlim(center[2] - 0.4, center[2] + 0.4)
    ax.view_init(elev=20, azim=45)

    # Axes labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

def expand_face(mesh, face_index, expansion_factor=0.05):
    """
    Expand one face of the mesh in the normal direction.
    """
    # Get the face vertices
    face = mesh.faces[face_index]
    face_vertices = mesh.vertices[face]

    # Calculate the face normal
    face_normal = np.cross(face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0])
    face_normal = face_normal / np.linalg.norm(face_normal)

    # Expand each vertex along the normal
    expanded_vertices = face_vertices + expansion_factor * face_normal

    # Create a copy of the mesh with the expanded face
    new_vertices = mesh.vertices.copy()
    new_vertices[face] = expanded_vertices

    return trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces)

def main():
    # Parameters
    center = (0, 0, 0)
    sphere_radius = 1
    resolution = 20
    bump_amplitude = 0.0
    bump_frequency = 0

    # Generate the bumpy sphere mesh
    bumpy_sphere = generate_bumpy_sphere(center, sphere_radius, resolution, bump_amplitude, bump_frequency)

    # Select a face to expand
    face_index = 500  # Index of the face to expand

    # Expand the selected face
    expanded_sphere = expand_face(bumpy_sphere, face_index, expansion_factor=-0.1)

    # Plot the original mesh and the expanded face
    plot_mesh_with_expanded_face(bumpy_sphere, expanded_sphere, face_index)

def generate_bumpy_sphere(center, radius, resolution=20, bump_amplitude=0.001, bump_frequency=2):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)
    r = radius + bump_amplitude * (np.sin(bump_frequency * u) + np.sin(bump_frequency * v)) / 2

    x = center[0] + r * np.sin(v) * np.cos(u)
    y = center[1] + r * np.sin(v) * np.sin(u)
    z = center[2] + r * np.cos(v)

    vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    points2D = np.vstack((u.flatten(), v.flatten())).T
    delaunay = Delaunay(points2D)
    faces = delaunay.simplices

    return trimesh.Trimesh(vertices=vertices, faces=faces)

if __name__ == "__main__":
    main()

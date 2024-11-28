import numpy as np
import trimesh
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


# Generate a bumpy sphere mesh
def generate_bumpy_sphere(center, radius, resolution=20, bump_amplitude=0.03, bump_frequency=3, asymmetry=0.02):
    """
    Generate a bumpy sphere mesh and ensure it is watertight.

    Parameters:
        - center: Center of the sphere (tuple).
        - radius: Radius of the sphere.
        - resolution: Resolution of the mesh.
        - bump_amplitude: Amplitude of the bumps.
        - bump_frequency: Frequency of the bumps.
        - asymmetry: Amplitude of the asymmetry.

    Returns:
        - mesh: Watertight and manifold Trimesh object.
    """
    # Generate sphere vertices and faces
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

    # Create the Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Ensure the mesh is watertight
    print("Cleaning and processing the generated mesh...")
    mesh.process()  # Fix normals and ensure proper connectivity
    mesh.fill_holes()  # Fill any gaps

    # Check and fix manifold issues
    if not mesh.is_watertight:
        print("The mesh is still not watertight. Attempting further fixes...")
        mesh.remove_duplicate_faces()  # Remove duplicate faces
        mesh.remove_degenerate_faces()  # Remove tiny or broken faces
        mesh.remove_unreferenced_vertices()  # Clean up unused vertices
        mesh.fill_holes()  # Fill remaining holes again

    # Final validation
    print(f"Final watertight status: {mesh.is_watertight}")
    return mesh


# Generate a flower-shaped white matter boundary
def generate_flower_shape(center, radius, resolution=200, petal_amplitude=1, petal_frequency=5):
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


# Calculate corrected vertex normals
def calculate_corrected_vertex_normals(vertices, faces):
    """
    Calculate corrected vertex normals for a mesh.

    Parameters:
        - vertices: Vertex positions of the mesh (Nx3 array).
        - faces: Face indices of the mesh (Mx3 array).

    Returns:
        - vertex_normals: Corrected vertex normals for the mesh.
    """
    # Calculate face normals
    tris = vertices[faces]
    edge1 = tris[:, 1] - tris[:, 0]
    edge2 = tris[:, 2] - tris[:, 0]
    face_normals = np.cross(edge1, edge2)
    face_normals /= np.linalg.norm(face_normals, axis=1)[:, np.newaxis]

    # Initialize vertex normals
    vertex_normals = np.zeros_like(vertices)

    # Accumulate face normals for each vertex
    for i, face in enumerate(faces):
        for vertex_index in face:
            vertex_normals[vertex_index] += face_normals[i]

    # Normalize vertex normals
    vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]

    # Correct direction to ensure normals point outward
    centroid = vertices.mean(axis=0)
    for i, (vertex, normal) in enumerate(zip(vertices, vertex_normals)):
        direction = vertex - centroid
        if np.dot(normal, direction) < 0:
            vertex_normals[i] = -normal

    return vertex_normals


# Remesh the mesh to maintain constant vertex density
def remesh_to_constant_face_area(mesh, max_face_area):
    """
    Remesh the given mesh to maintain a constant maximum face area.

    Parameters:
        - mesh: Trimesh object to remesh.
        - max_face_area: Target maximum face area for the mesh.

    Returns:
        - remeshed_mesh: Trimesh object with updated faces and vertices.
    """
    print(f"Remeshing based on max_face_area: {max_face_area}")
    new_faces = []
    new_vertices = mesh.vertices.tolist()  # Start with existing vertices

    for face in mesh.faces:
        # Get the vertices of the face
        v0, v1, v2 = mesh.vertices[face]
        # Compute the area of the triangle
        face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        if face_area > max_face_area:
            # Add a new vertex at the centroid of the face
            centroid = (v0 + v1 + v2) / 3
            centroid_index = len(new_vertices)
            new_vertices.append(centroid)

            # Split the triangle into three smaller triangles
            new_faces.append([face[0], face[1], centroid_index])
            new_faces.append([face[1], face[2], centroid_index])
            new_faces.append([face[2], face[0], centroid_index])
        else:
            new_faces.append(face)

    # Create a new mesh with the updated vertices and faces
    remeshed_mesh = trimesh.Trimesh(vertices=np.array(new_vertices), faces=np.array(new_faces))
    print(f"Remeshing complete. New vertex count: {len(remeshed_mesh.vertices)}")
    print(f"New face count: {len(remeshed_mesh.faces)}")

    return remeshed_mesh


# Laplacian smoothing
def laplacian_smoothing(mesh, iterations=2, lambda_factor=0.2):
    """
    Perform Laplacian smoothing on a mesh.

    Parameters:
        - mesh: Trimesh object to smooth.
        - iterations: Number of smoothing iterations.
        - lambda_factor: Smoothing factor.

    Returns:
        - Smoothed mesh.
    """
    vertices = mesh.vertices.copy()
    edges = mesh.edges_unique
    adjacency = {i: [] for i in range(len(vertices))}
    for edge in edges:
        adjacency[edge[0]].append(edge[1])
        adjacency[edge[1]].append(edge[0])

    for _ in range(iterations):
        new_vertices = vertices.copy()
        for vertex_index, neighbors in adjacency.items():
            if neighbors:
                neighbor_positions = vertices[neighbors]
                centroid = neighbor_positions.mean(axis=0)
                new_vertices[vertex_index] = (
                    (1 - lambda_factor) * vertices[vertex_index] + lambda_factor * centroid
                )
        vertices = new_vertices

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


# Visualize the expansion process
def visualize_expansion_process(ventricle_mesh, white_matter_mesh, step, output_dir="visualization_steps"):
    """
    Save the ventricular mesh and white matter boundary visualization at a given step.

    Parameters:
        - ventricle_mesh: Trimesh object for the ventricular mesh.
        - white_matter_mesh: Trimesh object for the white matter boundary.
        - step: Current step number (can be integer or string like 'Final').
        - output_dir: Directory to save visualization images.
    """
    import os

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Format the step as part of the filename
    if isinstance(step, int):
        step_str = f"{step:03d}"
    else:
        step_str = str(step)

    # Plot the meshes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Ventricular mesh
    ax.plot_trisurf(
        ventricle_mesh.vertices[:, 0],
        ventricle_mesh.vertices[:, 1],
        ventricle_mesh.vertices[:, 2],
        triangles=ventricle_mesh.faces,
        color="blue",
        alpha=0.6,
        edgecolor="gray",
        linewidth=0.2,
        label="Ventricular Mesh"
    )

    # White matter mesh
    ax.plot_trisurf(
        white_matter_mesh.vertices[:, 0],
        white_matter_mesh.vertices[:, 1],
        white_matter_mesh.vertices[:, 2],
        triangles=white_matter_mesh.faces,
        color="green",
        alpha=0.1,
        edgecolor="gray",
        linewidth=0.2,
        label="White Matter Mesh"
    )

    # Set plot details
    ax.set_title(f"Mesh at Step {step}")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend(loc="upper right")

    # Save the figure as an image
    output_path = os.path.join(output_dir, f"step_{step_str}.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)  # Close the figure to avoid display
    print(f"Visualization saved at {output_path}")


def generate_growth_gif(output_dir="visualization_steps", gif_name="growth_animation.gif"):
    """
    Generate a GIF from the saved visualization images.

    Parameters:
        - output_dir: Directory where visualization images are stored.
        - gif_name: Name of the output GIF file.
    """
    import os
    from PIL import Image

    # Get all PNG files in the output directory
    image_files = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")]
    )

    # Load images and create the GIF
    images = [Image.open(file) for file in image_files]
    gif_path = os.path.join(output_dir, gif_name)
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=100,  # Duration between frames in milliseconds
        loop=0  # Infinite loop
    )

    print(f"GIF saved at: {gif_path}")


def compare_vertex_displacements(previous_vertices, current_vertices, step):
    """
    Compare the vertex displacements between two steps, accounting for changes in vertex count.

    Parameters:
        - previous_vertices: Vertices from the previous step (Nx3 array).
        - current_vertices: Vertices from the current step (Mx3 array).
        - step: Current step number.

    Returns:
        - total_displacement: The total absolute displacement of all shared vertices.
    """
    # Ensure only the common vertices are compared
    min_vertex_count = min(len(previous_vertices), len(current_vertices))
    previous_trimmed = previous_vertices[:min_vertex_count]
    current_trimmed = current_vertices[:min_vertex_count]

    # Calculate displacements
    displacements = np.linalg.norm(current_trimmed - previous_trimmed, axis=1)

    # Compute statistics
    total_displacement = np.sum(displacements)
    max_displacement = np.max(displacements)
    min_displacement = np.min(displacements)
    mean_displacement = np.mean(displacements)

    # Print statistics
    print(f"\nStep {step}: Vertex Displacement Comparison")
    print(f"  Total Displacement: {total_displacement:.6f}")
    print(f"  Max Displacement: {max_displacement:.6f}")
    print(f"  Min Displacement: {min_displacement:.6f}")
    print(f"  Mean Displacement: {mean_displacement:.6f}")

    return total_displacement



# Expand the ventricle mesh
def expand_ventricle(ventricle, white_matter, steps=10, fraction=0.05, max_face_area=0.002):
    previous_vertices = ventricle.vertices.copy()  # Save initial vertices

    for step in range(steps):
        print(f"\nStep {step + 1}/{steps}")

        # Compute normals for the current vertices
        normals = calculate_corrected_vertex_normals(ventricle.vertices, ventricle.faces)
        new_vertices = []

        for vertex, normal in zip(ventricle.vertices, normals):
            # Cast a ray to find the intersection point with the white matter
            locations, _, _ = white_matter.ray.intersects_location(
                ray_origins=[vertex], ray_directions=[normal]
            )

            if locations.size > 0:
                closest_point = locations[0]
                move_vector = (closest_point - vertex) * fraction
                # print(f"move vector: {move_vector}")
                new_vertex = vertex + move_vector
            else:
                new_vertex = vertex

            new_vertices.append(new_vertex)

        # Update the mesh with the new vertex positions
        ventricle = trimesh.Trimesh(vertices=np.array(new_vertices), faces=ventricle.faces)

        # Compare vertex displacements starting from step 2
        if step > 0:
            compare_vertex_displacements(previous_vertices, ventricle.vertices, step + 1)

        # Save current vertices for the next step comparison
        previous_vertices = ventricle.vertices.copy()

        # Remesh using face area
        ventricle = remesh_to_constant_face_area(ventricle, max_face_area)

        # Apply smoothing
        ventricle = laplacian_smoothing(ventricle, iterations=3, lambda_factor=0.4)

        # Visualize every few steps
        if step % 1 == 0 or step == steps - 1:
            visualize_expansion_process(ventricle, white_matter, step + 1)

        # After the loop, generate the GIF
        generate_growth_gif(output_dir="visualization_steps", gif_name="growth_animation.gif")

    return ventricle



# Main function
if __name__ == "__main__":
    ventricle = generate_bumpy_sphere(center=(0, 0, 0), radius=0.3, resolution=20)
    white_matter = generate_flower_shape(center=(0, 0, 0), radius=1.0, resolution=40)

    white_matter_face = np.mean(white_matter.area_faces)
    print(f"Average face area of the white matter mesh: {white_matter_face}")

    expanded_ventricle = expand_ventricle(
        ventricle=ventricle,
        white_matter=white_matter,
        steps=200,                # Number of expansion steps
        fraction=0.05,            # Fraction of the intersection distance to expand
        max_face_area=0.01      # Target maximum face area for remeshing
    )

    print("\nFinal Expanded Ventricle Mesh:")
    print(f"Vertex count: {len(expanded_ventricle.vertices)}")
    print(f"Edge count: {len(expanded_ventricle.edges)}")
    print(f"Face count: {len(expanded_ventricle.faces)}")
    print(f"Is watertight: {expanded_ventricle.is_watertight}")

    visualize_expansion_process(expanded_ventricle, white_matter, step="Final")

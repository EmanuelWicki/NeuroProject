import numpy as np
import trimesh
import os
import matplotlib.pyplot as plt


# Generate a bumpy sphere mesh with quad faces
def generate_bumpy_sphere(center, radius, resolution=50, bump_amplitude=0.001, bump_frequency=2, output_dir="ventricle_steps"):
    """
    Generate a bumpy sphere quad mesh with a watertight surface.

    Parameters:
        - center: Center of the sphere (tuple of 3 floats).
        - radius: Radius of the sphere.
        - resolution: Number of subdivisions along latitude and longitude.
        - bump_amplitude: Amplitude of the bumps.
        - bump_frequency: Frequency of the bumps.
        - output_dir: Directory to save the initial mesh.

    Returns:
        - mesh: Smooth, watertight Trimesh object with quadrilateral faces.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate parametric sphere coordinates
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)

    # Apply bumps to the radius
    r = radius + bump_amplitude * (np.sin(bump_frequency * u) + np.sin(bump_frequency * v)) / 2

    # Spherical coordinate conversion
    x = center[0] + r * np.sin(v) * np.cos(u)
    y = center[1] + r * np.sin(v) * np.sin(u)
    z = center[2] + r * np.cos(v)

    # Flatten vertices into a 2D array
    vertices = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

    # Create quad faces from grid
    quads = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v0 = i * resolution + j
            v1 = v0 + 1
            v2 = v0 + resolution
            v3 = v2 + 1
            quads.append([v0, v1, v3, v2])  # Quad face indices

    quads = np.array(quads)

    # Create the Trimesh object with quad faces
    mesh = trimesh.Trimesh(vertices=vertices, faces=quads, process=False)

    # Check for watertightness
    print("Processing and cleaning the bumpy sphere mesh...")
    mesh.process(validate=True)

    if not mesh.is_watertight:
        print("Warning: Mesh is not watertight after processing.")
    else:
        print("Bumpy sphere quad mesh is watertight.")

    # Save the mesh as .obj
    obj_path = os.path.join(output_dir, "ventricle_step_0.obj")
    mesh.export(obj_path)
    print(f"Saved initial bumpy sphere quad mesh to {obj_path}")

    return mesh


# Generate a flower-shaped white matter boundary
def generate_flower_shape(center, radius, resolution=50, petal_amplitude=1, petal_frequency=5):
    """
    Generate a flower-shaped quad mesh for the white matter boundary.

    Parameters:
        - center: Center of the flower (tuple of 3 floats).
        - radius: Radius of the flower shape.
        - resolution: Resolution of the parametric grid.
        - petal_amplitude: Amplitude of the petal deformation.
        - petal_frequency: Frequency of the petal deformation.

    Returns:
        - mesh: Trimesh object with quadrilateral faces.
    """
    # Generate parametric coordinates
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)

    # Apply flower-shaped deformation to the radius
    r = radius + petal_amplitude * np.sin(petal_frequency * v) * np.sin(petal_frequency * u)

    # Convert to Cartesian coordinates
    x = center[0] + r * np.sin(v) * np.cos(u)
    y = center[1] + r * np.sin(v) * np.sin(u)
    z = center[2] + r * np.cos(v)

    # Flatten vertices into a 2D array
    vertices = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

    # Create quad faces
    quads = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v0 = i * resolution + j
            v1 = v0 + 1
            v2 = v0 + resolution
            v3 = v2 + 1
            quads.append([v0, v1, v3, v2])

    quads = np.array(quads)

    # Create the Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=quads, process=False)
    return mesh


# Enhanced Laplacian smoothing for quad mesh
def laplacian_smoothing(mesh, iterations=5, lambda_factor=0.5):
    """
    Perform Laplacian smoothing on a quad mesh.

    Parameters:
        - mesh: Trimesh object to smooth.
        - iterations: Number of smoothing iterations.
        - lambda_factor: Smoothing factor (controls how much vertices move).

    Returns:
        - Smoothed mesh (Trimesh object).
    """
    vertices = mesh.vertices.copy()
    adjacency = {i: [] for i in range(len(vertices))}

    # Build adjacency list
    for face in mesh.faces:
        for i in range(len(face)):
            v0 = face[i]
            v1 = face[(i + 1) % len(face)]
            adjacency[v0].append(v1)
            adjacency[v1].append(v0)

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

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)


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
    os.makedirs(output_dir, exist_ok=True)
    step_str = f"{step:03d}" if isinstance(step, int) else str(step)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(*ventricle_mesh.vertices.T, triangles=ventricle_mesh.faces, color="blue", alpha=0.6)
    ax.plot_trisurf(*white_matter_mesh.vertices.T, triangles=white_matter_mesh.faces, color="green", alpha=0.3)
    output_path = os.path.join(output_dir, f"step_{step_str}.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Visualization saved at {output_path}")

def remesh_to_constant_face_area(mesh, max_face_area):
    """
    Remesh the given mesh to maintain a constant maximum face area for quad and triangular meshes.

    Parameters:
        - mesh: Trimesh object to remesh.
        - max_face_area: Target maximum face area for the mesh.

    Returns:
        - remeshed_mesh: Trimesh object with updated faces and vertices.
    """
    new_faces = []
    new_vertices = mesh.vertices.tolist()  # Start with existing vertices

    for face in mesh.faces:
        # Check if the face is a triangle (3 vertices) or a quad (4 vertices)
        if len(face) == 3:
            # Triangle: Decompose into three smaller triangles if necessary
            v0, v1, v2 = mesh.vertices[face]

            # Compute the area of the triangle
            face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

            if face_area > max_face_area:
                # Add a new vertex at the centroid
                centroid = (v0 + v1 + v2) / 3
                centroid_index = len(new_vertices)
                new_vertices.append(centroid)

                # Split into three smaller triangles
                new_faces.append([face[0], face[1], centroid_index])
                new_faces.append([face[1], face[2], centroid_index])
                new_faces.append([face[2], face[0], centroid_index])
            else:
                new_faces.append(face)
        elif len(face) == 4:
            # Quad: Decompose into four smaller quads if necessary
            v0, v1, v2, v3 = mesh.vertices[face]

            # Compute the area of the quad (split into two triangles)
            area1 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            area2 = 0.5 * np.linalg.norm(np.cross(v3 - v2, v0 - v3))
            face_area = area1 + area2

            if face_area > max_face_area:
                # Add a new vertex at the centroid of the quad
                centroid = (v0 + v1 + v2 + v3) / 4
                centroid_index = len(new_vertices)
                new_vertices.append(centroid)

                # Split into four smaller quads
                new_faces.append([face[0], face[1], centroid_index, face[3]])
                new_faces.append([face[1], face[2], centroid_index, face[0]])
                new_faces.append([face[2], face[3], centroid_index, face[1]])
                new_faces.append([face[3], face[0], centroid_index, face[2]])
            else:
                new_faces.append(face)

    # Create a new mesh with the updated vertices and faces
    remeshed_mesh = trimesh.Trimesh(vertices=np.array(new_vertices), faces=np.array(new_faces), process=False)
    print(f"Remeshing complete. New vertex count: {len(remeshed_mesh.vertices)}, New face count: {len(remeshed_mesh.faces)}")
    return remeshed_mesh


# Expand the ventricle mesh toward the white matter boundary
def expand_ventricle(ventricle, white_matter, steps=10, fraction=0.1, max_face_area=0.002, output_dir="ventricle_steps"):
    """
    Expand the ventricle mesh toward the white matter boundary by incrementally
    moving vertices along their normal directions.

    Parameters:
        - ventricle: Trimesh object representing the ventricle.
        - white_matter: Trimesh object representing the white matter boundary.
        - steps: Number of expansion steps.
        - fraction: Fraction of the intersection distance to move.
        - max_face_area: Target maximum face area for remeshing.
        - output_dir: Directory to save ventricular meshes and visualizations.

    Returns:
        - ventricle: Updated Trimesh object after expansion.
    """
    os.makedirs(output_dir, exist_ok=True)

    for step in range(steps):
        print(f"\nStep {step + 1}/{steps}")

        # Compute vertex normals
        normals = ventricle.vertex_normals

        # Move vertices outward along their normals
        new_vertices = []
        for vertex, normal in zip(ventricle.vertices, normals):
            # Ray intersection logic
            locations, _, _ = white_matter.ray.intersects_location(
                ray_origins=[vertex], ray_directions=[normal]
            )

            if len(locations) > 0:
                # Move a fraction of the distance to the first intersection
                intersection_distance = np.linalg.norm(locations[0] - vertex)
                move_distance = fraction * intersection_distance
                new_vertex = vertex + normal * move_distance
            else:
                # If no intersection, keep the vertex in place
                new_vertex = vertex

            new_vertices.append(new_vertex)

        # Update ventricle vertices
        ventricle.vertices = np.array(new_vertices)

        # Remesh to maintain a uniform quad structure
        ventricle = remesh_to_constant_face_area(ventricle, max_face_area)

        # Smooth the mesh
        ventricle = laplacian_smoothing(ventricle, iterations=3, lambda_factor=0.5)

        # Visualize the current step
        visualize_expansion_process(ventricle, white_matter, step + 1, output_dir=output_dir)

        # Save the ventricle mesh as an .obj file
        step_filename = os.path.join(output_dir, f"ventricle_step_{step + 1}.obj")
        ventricle.export(step_filename)
        print(f"Saved ventricular mesh at step {step + 1} to {step_filename}")

    return ventricle



if __name__ == "__main__":
    # Generate the initial meshes
    ventricle = generate_bumpy_sphere(center=(0, 0, 0), radius=0.3, resolution=40)
    white_matter = generate_flower_shape(center=(0, 0, 0), radius=1.0, resolution=200)

    # Expand the ventricle toward the white matter boundary
    expanded_ventricle = expand_ventricle(
        ventricle=ventricle,
        white_matter=white_matter,
        steps=10,
        fraction=0.2,
        max_face_area=0.002
    )

    # Print the final state of the ventricle mesh
    print("\nFinal Expanded Ventricle Mesh:")
    print(f"Vertex count: {len(expanded_ventricle.vertices)}")
    print(f"Face count: {len(expanded_ventricle.faces)}")
    print(f"Is watertight: {expanded_ventricle.is_watertight}")

    # Visualize the final mesh
    visualize_expansion_process(expanded_ventricle, white_matter, step="Final")

import numpy as np
import trimesh
import json
import os
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pymeshfix import MeshFix
from trimesh.remesh import subdivide
from trimesh.proximity import closest_point
from PIL import Image
import pyvista as pv

def FixMesh(mesh, enforce_convexity=False):
    mesh.process(validate=True)
    mesh.fill_holes()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    if enforce_convexity:
        mesh = trimesh.convex.convex_hull(mesh)
    
    # Check if the mesh is watertight
    if not mesh.is_watertight:
        print("Warning: The mesh is still not watertight after processing.")
    else:
        print("Bumpy sphere mesh is watertight.")

    mesh = laplacian_smoothing(mesh, iterations=5, lambda_factor=0.9)

    return mesh

# Generate a bumpy sphere mesh
def generate_bumpy_sphere(center, radius, resolution=50, bump_amplitude=0.001, bump_frequency=2, output_dir="ventricle_steps"):
    """
    Generate a bumpy sphere mesh with a watertight surface.

    Parameters:
        - center: Center of the sphere (tuple of 3 floats).
        - radius: Radius of the sphere.
        - resolution: Number of subdivisions along latitude and longitude.
        - bump_amplitude: Amplitude of the bumps.
        - bump_frequency: Frequency of the bumps.
        - output_dir: Directory to save the initial mesh.

    Returns:
        - mesh: Smooth, watertight Trimesh object with bumps.
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

    # Flatten the grid for triangulation
    vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    points2D = np.vstack((u.flatten(), v.flatten())).T

    # Perform Delaunay triangulation in 2D parameter space
    delaunay = Delaunay(points2D)
    faces = delaunay.simplices

    # Create the Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Ensure watertightness
    mesh = FixMesh(mesh, enforce_convexity=True)

    # Save the initial bumpy sphere as an .obj file
    step_filename = os.path.join(output_dir, f"ventricle_step_0.obj")
    mesh.export(step_filename)
    print(f"Saved initial bumpy sphere to {step_filename}")

    return mesh

def inward_offset_mesh(mesh, offset_distance):
    """
    Create an inward offset of the mesh by moving vertices along their normals.

    Parameters:
        - mesh: Trimesh object to offset.
        - offset_distance: Distance to move vertices inward.

    Returns:
        - offset_mesh: Trimesh object representing the offset mesh.
    """
    # Compute vertex normals
    normals = mesh.vertex_normals

    # Move vertices inward along their normals
    offset_vertices = mesh.vertices - normals * offset_distance

    # Create a new mesh with the offset vertices
    offset_mesh = trimesh.Trimesh(vertices=offset_vertices, faces=mesh.faces, process=False)

    # Remove degenerate faces
    offset_mesh.remove_degenerate_faces()
    offset_mesh.remove_unreferenced_vertices()

    return offset_mesh



def trimesh_to_pyvista(mesh):
    """
    Convert a Trimesh object to a PyVista-compatible PolyData object.
    
    Parameters:
        - mesh: Trimesh object to convert.
    
    Returns:
        - pyvista_mesh: PyVista PolyData object.
    """

    # Extract vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces

    # Convert faces to PyVista-compatible format
    n_faces = len(faces)
    pyvista_faces = np.empty((n_faces, 4), dtype=int)
    pyvista_faces[:, 0] = 3  # Number of vertices per face (triangles)
    pyvista_faces[:, 1:] = faces

    # Flatten the array to match PyVista's input format
    pyvista_faces = pyvista_faces.flatten()

    # Create and return a PyVista PolyData object
    return pv.PolyData(vertices, pyvista_faces)

def visualize_with_pyvista_live(ventricle, white_matter, plotter=None):
    """
    Visualize the ventricle and white matter meshes dynamically with PyVista.
    The plot is updated live without closing the window.
    
    Parameters:
        - ventricle: Trimesh object for the ventricle mesh.
        - white_matter: Trimesh object for the white matter mesh.
        - plotter: Optional PyVista Plotter object to allow live updates.
    """
    # Convert Trimesh objects to PyVista PolyData
    ventricle_pv = pv.PolyData(ventricle.vertices, np.hstack([np.full((len(ventricle.faces), 1), 3), ventricle.faces]).ravel())
    white_matter_pv = pv.PolyData(white_matter.vertices, np.hstack([np.full((len(white_matter.faces), 1), 3), white_matter.faces]).ravel())

    # Create a new Plotter if not provided
    if plotter is None:
        plotter = pv.Plotter()
        plotter.add_mesh(white_matter_pv, color="lightgray", opacity=0.2, label="White Matter Mesh")
        plotter.add_mesh(ventricle_pv, color="blue", label="Ventricle Mesh")
        plotter.add_legend()
        plotter.show_grid()
        plotter.camera_position = 'xy'
        plotter.show(interactive_update=True)  # Enable interactive updates

    # Update the existing meshes
    else:
        plotter.update_coordinates(ventricle_pv.points, render=False)
        plotter.update_coordinates(white_matter_pv.points, render=False)

    # Render the updated plot
    plotter.render()

    return plotter  # Return the Plotter for live updates

def save_growth_interactively(ventricle_steps, white_matter, output_filename="ventricle_growth"):
    """
    Save the growth process as individual .vtp files for interactive inspection.

    Parameters:
        - ventricle_steps: List of Trimesh objects representing the ventricle at different stages.
        - white_matter: Trimesh object for the white matter mesh (remains constant).
        - output_filename: Directory path to save the ventricle and white matter files.
    """

    # Ensure the output directory exists
    os.makedirs(output_filename, exist_ok=True)

    # Save the white matter as a static .vtp file
    white_matter_pv = pv.PolyData(
        white_matter.vertices,
        np.hstack([np.full((len(white_matter.faces), 1), 3), white_matter.faces]).ravel(),
    )
    white_matter_path = os.path.join(output_filename, "white_matter.vtp")
    white_matter_pv.save(white_matter_path)
    print(f"White matter saved to {white_matter_path}")

    # Save each ventricle step as a .vtp file
    for i, ventricle in enumerate(ventricle_steps):
        ventricle_pv = pv.PolyData(
            ventricle.vertices,
            np.hstack([np.full((len(ventricle.faces), 1), 3), ventricle.faces]).ravel(),
        )
        ventricle_path = os.path.join(output_filename, f"ventricle_step_{i}.vtp")
        ventricle_pv.save(ventricle_path)
        print(f"Ventricle step {i} saved to {ventricle_path}")


def export_combined_mesh_with_opacity(ventricle, white_matter, white_matter_offset, output_filename):
    """
    Export the ventricle and white matter meshes into a single .obj file.
    Assign transparency (opacity) to the white matter mesh via a custom .mtl file.

    Parameters:
        - ventricle: Trimesh object for the ventricle mesh.
        - white_matter: Trimesh object for the white matter mesh.
        - output_filename: Path to save the combined .obj file.
    """

    # Prepare directories and file paths
    base_filename = os.path.splitext(output_filename)[0]
    mtl_filename = base_filename + ".mtl"
    obj_filename = output_filename

    # Material names
    ventricle_material = "ventricle_material"
    white_matter_material = "white_matter_material"
    white_matter_offset_material = "white_matter_offset_material"

    # Write the .mtl file explicitly
    with open(mtl_filename, "w") as mtl_file:
        mtl_file.write(f"newmtl {ventricle_material}\n")
        mtl_file.write("Kd 1.0 0.0 0.0\n")  # Red color
        mtl_file.write("d 1.0\n")  # Fully opaque\n\n")

        mtl_file.write(f"newmtl {white_matter_material}\n")
        mtl_file.write("Kd 0.9 0.9 0.9\n")  # Light gray color
        mtl_file.write("d 0.2\n")  # 20% opacity

        mtl_file.write(f"newmtl {white_matter_offset_material}\n")
        mtl_file.write("Kd 0.9 0.9 0.9\n")  # Red color
        mtl_file.write("d 0.25\n")  # Fully opaque\n\n")

    # Combine meshes into a single .obj file
    with open(obj_filename, "w") as obj_file:
        obj_file.write(f"mtllib {os.path.basename(mtl_filename)}\n")

        # Ventricle Mesh
        obj_file.write(f"usemtl {ventricle_material}\n")
        for vertex in ventricle.vertices:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in ventricle.faces:
            obj_file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

        # White Matter Mesh
        obj_file.write(f"usemtl {white_matter_material}\n")
        offset = len(ventricle.vertices)  # Offset for vertex indexing
        for vertex in white_matter.vertices:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in white_matter.faces:
            obj_file.write(f"f {face[0] + 1 + offset} {face[1] + 1 + offset} {face[2] + 1 + offset}\n")

        # White Matter Mesh
        obj_file.write(f"usemtl {white_matter_offset_material}\n")
        offset = len(ventricle.vertices)  # Offset for vertex indexing
        for vertex in white_matter_offset.vertices:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in white_matter_offset.faces:
            obj_file.write(f"f {face[0] + 1 + offset} {face[1] + 1 + offset} {face[2] + 1 + offset}\n")

    print(f"Saved combined mesh with white matter opacity: {obj_filename}")



def generate_star_polygon(center, outer_radius, inner_radius, num_points=5, extrusion=0.1):
    """
    Generate a sharp-edged star mesh with symmetric spikes and visualize normals.
    
    Parameters:
        - center: Center of the star (x, y, z).
        - outer_radius: Radius of the star's outer vertices.
        - inner_radius: Radius of the star's inner vertices (controls sharpness).
        - num_points: Number of star points (default: 5).
        - extrusion: Extrusion height for 3D effect.
    
    Returns:
        - Trimesh star mesh.
    """
    vertices = []
    for i in range(2 * num_points):
        angle = i * np.pi / num_points
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2]
        vertices.append([x, y, z])
    
    # Top vertices (extrusion)
    top_vertices = [[v[0], v[1], center[2] + extrusion] for v in vertices]
    vertices.extend(top_vertices)

    # Faces (connect top and bottom layers)
    faces = []
    for i in range(2 * num_points):
        next_i = (i + 1) % (2 * num_points)
        
        # Bottom face
        faces.append([i, next_i, (2 * num_points) + next_i])
        faces.append([i, (2 * num_points) + next_i, (2 * num_points) + i])
    
    # Cap top and bottom faces
    bottom_center = [center[0], center[1], center[2]]
    top_center = [center[0], center[1], center[2] + extrusion]
    vertices.append(bottom_center)
    vertices.append(top_center)

    for i in range(2 * num_points):
        next_i = (i + 1) % (2 * num_points)
        faces.append([i, next_i, len(vertices) - 2])  # Bottom cap
        faces.append([(2 * num_points) + i, len(vertices) - 1, (2 * num_points) + next_i])  # Top cap

    # Create Trimesh object
    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
    
    # Subdivide to refine mesh
    iterations = 3
    for _ in range(iterations):
        mesh = mesh.subdivide()  # Subdivide all faces
    print(f"Mesh refined: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces.")

    # Visualize using PyVista with normals
    pv_mesh = pv.PolyData(mesh.vertices, np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces)))
    normals = mesh.vertex_normals

    # Start PyVista plotter
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, color="gold", show_edges=True, opacity=0.8)
    plotter.add_arrows(mesh.vertices, normals, mag=0.05, color="red")  # Add normals as arrows
    plotter.add_title("Sharp Star Mesh with Normals")
    plotter.show()

    return mesh



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
        - step: Continuous step number (int) or a string like "Final".
        - output_dir: Directory to save visualization images.
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Format the step as part of the filename
    if isinstance(step, int):  # If step is an integer, zero-pad it
        step_str = f"{step:03d}"
    else:  # If step is a string, use it directly
        step_str = step

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
        linewidth=0.2
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
        linewidth=0.2
    )

    # Set plot details
    ax.set_title(f"Mesh at Step {step}")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    # Save the figure as an image
    output_path = os.path.join(output_dir, f"step_{step_str}.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Visualization saved at {output_path}")


def generate_growth_gif(output_dir="visualization_steps", gif_name="growth_animation.gif"):
    """
    Generate a GIF from the saved visualization images.

    Parameters:
        - output_dir: Directory where visualization images are stored.
        - gif_name: Name of the output GIF file.
    """
    # Ensure proper handling of paths
    gif_path = os.path.join(output_dir, gif_name)

    # Get all PNG files in the output directory
    image_files = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")]
    )

    # Check if image files exist
    if not image_files:
        raise FileNotFoundError("No PNG files found in the specified output directory.")

    # Load images and create the GIF
    images = [Image.open(file) for file in image_files]
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


def refine_mesh_near_high_curvature(mesh, white_matter, step,
                                    white_matter_curvature_threshold=30, 
                                    distance_threshold=0.01, 
                                    max_iterations=2):
    """
    Refine mesh resolution in areas close to high-curvature regions of another mesh.

    Parameters:
        - mesh: Trimesh object to refine (ventricular mesh).
        - white_matter: Trimesh object (white matter mesh).
        - white_matter_curvature_threshold: Angle threshold (degrees) to identify sharp/high-curvature regions.
        - distance_threshold: Maximum distance to trigger subdivision near high-curvature regions.
        - max_iterations: Number of refinement iterations.

    Returns:
        - refined_mesh: Trimesh object with locally refined resolution.
    """
    print("Calculating high-curvature areas based on face normal differences...")

    # Step 1: Calculate face normal differences to identify high-curvature edges
    face_normals = white_matter.face_normals
    edge_faces = white_matter.face_adjacency  # Pairs of faces sharing an edge
    edge_vertices = white_matter.face_adjacency_edges  # Corresponding edge vertex pairs

    high_curvature_edges = []  # List of sharp edges
    high_curvature_vertices = set()  # Store connected vertices
    high_curvature_faces = set()  # Store connected faces

    for i, (face1, face2) in enumerate(edge_faces):
        normal1 = face_normals[face1]
        normal2 = face_normals[face2]
        angle = np.degrees(np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0)))
        
        if angle > white_matter_curvature_threshold:
            # Edge is sharp
            high_curvature_edges.append(edge_vertices[i])
            high_curvature_vertices.update(edge_vertices[i])

            # Collect the faces connected to this edge
            high_curvature_faces.add(face1)
            high_curvature_faces.add(face2)

    # Extract the 3D coordinates of high-curvature vertices
    high_curvature_vertices = np.array(list(high_curvature_vertices))  # Unique vertex indices
    high_curvature_positions = white_matter.vertices[high_curvature_vertices]  # Get their positions

    print(f"Found {len(high_curvature_edges)} high-curvature edges with angle > {white_matter_curvature_threshold} degrees.")
    print(f"Selected {len(high_curvature_faces)} faces connected to high-curvature edges.")

    if step == 1:
        visualize_high_curvature_edges(white_matter, high_curvature_edges)

    # Step 2: Build KDTree for high-curvature vertex positions
    high_curvature_tree = cKDTree(high_curvature_positions)

    # Step 3: Refine the ventricular mesh near high-curvature regions
    vertices = mesh.vertices
    faces = mesh.faces

    for iteration in range(max_iterations):
        print(f"\nRefinement iteration {iteration + 1}")

        # Find ventricular vertices near high-curvature vertices
        distances, indices = high_curvature_tree.query(vertices)
        close_vertices = np.where(distances < distance_threshold)[0]

        if len(close_vertices) == 0:
            print("No ventricular vertices found near high-curvature regions.")
            break

        print(f"Found {len(close_vertices)} ventricular vertices near high-curvature regions.")

        # Subdivide faces containing the close vertices
        new_vertices = vertices.tolist()
        new_faces = []

        for face in faces:
            if any(v in close_vertices for v in face):
                # Subdivide the face by adding a new vertex at the centroid
                v0, v1, v2 = face
                centroid = (vertices[v0] + vertices[v1] + vertices[v2]) / 3
                centroid_index = len(new_vertices)
                new_vertices.append(centroid)

                # Split into three smaller triangles
                new_faces.append([v0, v1, centroid_index])
                new_faces.append([v1, v2, centroid_index])
                new_faces.append([v2, v0, centroid_index])
            else:
                # Keep the face unchanged
                new_faces.append(face)

        # Update mesh
        vertices = np.array(new_vertices)
        faces = np.array(new_faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        print(f"Subdivision complete. New vertex count: {len(vertices)}, new face count: {len(faces)}")

    print("\nRefinement complete.")
    return mesh

def visualize_high_curvature_edges(mesh, high_curvature_edges):
    """
    Visualize high-curvature edges on the white matter mesh.

    Parameters:
        - mesh: Trimesh object (white matter mesh).
        - high_curvature_edges: List of high-curvature edges to highlight.
    """
    import pyvista as pv

    pv_faces = np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces)).ravel()
    pv_mesh = pv.PolyData(mesh.vertices, pv_faces)

    # Highlight high-curvature edges
    edge_lines = pv.PolyData()
    for edge in high_curvature_edges:
        line = pv.Line(mesh.vertices[edge[0]], mesh.vertices[edge[1]])
        edge_lines += line

    # Plot mesh and sharp edges
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, color="lightgray", opacity=0.6)
    plotter.add_mesh(edge_lines, color="red", line_width=3, label="High-Curvature Edges")
    plotter.add_legend()
    plotter.add_text("High-Curvature Edges on White Matter Mesh", font_size=12)
    plotter.show()


def expand_ventricle_dynamic_fraction_auto(
    ventricle,
    white_matter,
    steps=40,
    f_min=0.01,
    f_max=0.4,
    thresholds=[0.01, 0.02, 0.03],  # Threshold distances
    percentages=[0.9, 0.95, 1.0]    # Percentage of vertices required within each threshold
):
    """
    Expand the ventricle mesh dynamically in stages, ensuring vertices do not cross
    the white matter boundary. An inward offset mesh prevents overshooting.

    Parameters:
        - ventricle: Trimesh object representing the ventricle.
        - white_matter: Trimesh object representing the white matter.
        - steps: Number of expansion steps per stage.
        - f_min: Minimum fraction for expansion.
        - f_max: Maximum fraction for expansion.
        - thresholds: List of distances for each stage.
        - percentages: List of percentages of vertices required within each threshold.

    Returns:
        - ventricle: Expanded Trimesh object.
    """

    # Ensure output directory exists
    output_dir = "ventricle_obj_files"
    os.makedirs(output_dir, exist_ok=True)
    json_output_file = "expansion_vectors.json" 

    step_counter = 1
    ventricle_steps = []
    expansion_data = []  # To store all expansion vectors and positions

    print(f"Initial ventricle volume: {ventricle.volume:.4f}")
    print(f"White matter volume: {white_matter.volume:.4f}")

    for stage_idx, (threshold, percentage) in enumerate(zip(thresholds, percentages)):
        print(f"\nStarting Stage {stage_idx + 1}/{len(thresholds)} - Threshold: {threshold}, Required: {percentage * 100:.2f}%")

        # Create inward offset of the white matter mesh
        offset_white_matter = inward_offset_mesh(white_matter, threshold)
        print(f"Offset white matter mesh created with threshold: {threshold}")

        for step in range(steps):
            previous_vertices = ventricle.vertices.copy()
            normals = calculate_corrected_vertex_normals(previous_vertices, ventricle.faces)

            new_vertices = previous_vertices.copy()
            step_vectors = []

            # Measure distances to the original white matter surface
            distances = np.full(len(previous_vertices), np.inf)
            for i, (vertex, normal) in enumerate(zip(previous_vertices, normals)):
                if np.any(np.isnan(normal)) or np.linalg.norm(normal) == 0:
                    continue  # Skip invalid normals

                # Calculate intersection with the white matter
                locations, _, _ = white_matter.ray.intersects_location(
                    ray_origins=[vertex], ray_directions=[normal]
                )
                if locations.size > 0:
                    distances[i] = np.linalg.norm(locations[0] - vertex)

            # Identify vertices inside the no-movement zone
            vertices_in_offset = []
            for i, vertex in enumerate(previous_vertices):
                closest_point, _, _ = offset_white_matter.nearest.on_surface([vertex])
                if np.linalg.norm(closest_point - vertex) <= threshold:
                    vertices_in_offset.append(i)  # Track vertices in the offset zone

            # Adjust movement fraction dynamically
            current_volume = ventricle.volume
            volume_ratio = abs(current_volume / white_matter.volume)

            if volume_ratio <= 0.85:
                fraction = f_min + ((f_max - f_min) / (0.85 - 0.01)) * (volume_ratio - 0.01)
            else:
                fraction = f_max

            print(f"Step {step_counter}: Volume Ratio = {volume_ratio:.4f}, Expansion Fraction = {fraction:.4f}")

            # Calculate percentage of vertices within the threshold
            current_percentage = len(vertices_in_offset) / len(previous_vertices)
            print(f"Step {step_counter}: {current_percentage * 100:.2f}% of vertices within threshold.")

            if current_percentage >= percentage:
                print(f"Stage {stage_idx + 1} completed. Moving to the next stage.")
                break

            # Expand vertices: Only move vertices outside the offset zone
            new_vertices = previous_vertices.copy()
            for i, (vertex, normal) in enumerate(zip(previous_vertices, normals)):
                if i not in vertices_in_offset and distances[i] < np.inf:
                    move_vector = normal * min(distances[i] * fraction, distances[i])
                    new_vertices[i] = vertex + move_vector

                    # Store position and expansion vector
                    step_vectors.append({
                        "position": vertex.tolist(),
                        "vector": move_vector.tolist()
                    })
                else:
                    new_vertices[i] = vertex  # Keep vertex unchanged if within offset

            # Append step_vectors for the current step to expansion_data
            expansion_data.append({
                "step": step_counter,
                "vectors": step_vectors
            })
            # Update ventricle mesh
            ventricle = trimesh.Trimesh(vertices=new_vertices, faces=ventricle.faces)

            # Remesh and smooth
            ventricle = remesh_to_constant_face_area(ventricle, max_face_area=0.002)
            ventricle = laplacian_smoothing(ventricle, iterations=3, lambda_factor=0.6)
            # Refine mesh in high-curvature patches
            ventricle = refine_mesh_near_high_curvature(
                mesh=ventricle, 
                white_matter=white_matter, 
                step=step_counter, 
               white_matter_curvature_threshold=30, 
                distance_threshold=0.04, 
                max_iterations=1,
            )

            # Save the combined ventricular and white matter meshes
            obj_filename = os.path.join(output_dir, f"ventricle_step_{step_counter:04d}.obj")
            export_combined_mesh_with_opacity(ventricle, white_matter, white_matter, obj_filename)

            # Save visualization with continuous step numbering
            visualize_expansion_process(ventricle, white_matter, step=step_counter)
            ventricle_steps.append(ventricle.copy())

            step_counter += 1  # Increment step counter

    # Generate GIF from saved visualizations
    generate_growth_gif(output_dir="visualization_steps", gif_name="growth_animation.gif")
    print("\nAll stages completed.")

    # Save expansion data to JSON
    with open(json_output_file, "w") as json_file:
        json.dump(expansion_data, json_file, indent=4)
    print(f"Expansion vectors saved to {json_output_file}")

    return ventricle




# Main function
if __name__ == "__main__":
    ventricle = generate_bumpy_sphere(center=(0, 0, 0), radius=0.2, resolution=20)
    # white_matter = generate_pyramid(center=(0, 0, 0), base_length=1.0, height=1.5)
    # white_matter = refine_pyramid(white_matter, splits=4)
    white_matter = generate_star_polygon(center=(0, 0, -0.3), inner_radius=0.23, outer_radius=0.6, num_points=6, extrusion=0.6)

    white_matter_face = np.mean(white_matter.area_faces)
    print(f"Average face area of the white matter mesh: {white_matter_face}")

    expanded_ventricle = expand_ventricle_dynamic_fraction_auto(
        ventricle = ventricle, 
        white_matter = white_matter, 
        steps=1000, 
        f_min=0.15, 
        f_max=0.25,
        # thresholds=[0.15, 0.09, 0.02],  # Stages with thresholds
        thresholds=[0.08, 0.01],
        percentages=[0.6, 0.8, 0.95]    # Required percentages
        )

    print("\nFinal Expanded Ventricle Mesh:")
    print(f"Vertex count: {len(expanded_ventricle.vertices)}")
    print(f"Edge count: {len(expanded_ventricle.edges)}")
    print(f"Face count: {len(expanded_ventricle.faces)}")
    print(f"Is watertight: {expanded_ventricle.is_watertight}")

    visualize_expansion_process(expanded_ventricle, white_matter, step="Final")
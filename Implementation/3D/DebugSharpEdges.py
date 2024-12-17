import numpy as np
import trimesh
import json
import os
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from pymeshfix import MeshFix
from scipy.spatial import cKDTree
from trimesh.remesh import subdivide
from PIL import Image
import pyvista as pv
from trimesh.remesh import subdivide

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

def generate_flower_shape(center, radius, resolution=200, petal_amplitude=1, petal_frequency=3):
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

# Generate and visualize sharp star mesh
sharp_star = generate_star_polygon(center=(0, 0, 0), outer_radius=0.5, inner_radius=0.2, num_points=5, extrusion=0.3)


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

def detect_crossing_edges(ventricle_mesh, white_matter_mesh):
    """
    Detect edges in the ventricle mesh that cross the white matter mesh.
    
    Parameters:
        - ventricle_mesh: Trimesh object of the ventricle mesh.
        - white_matter_mesh: Trimesh object of the white matter mesh.
    
    Returns:
        - crossing_edges: List of edge indices that cross the white matter mesh.
    """
    crossing_edges = []

    # Extract vertices and edges
    vertices = ventricle_mesh.vertices
    edges = ventricle_mesh.edges_unique  # Unique edges

    # Loop through all edges
    for i, edge in enumerate(edges):
        v1 = vertices[edge[0]]
        v2 = vertices[edge[1]]
        direction = v2 - v1
        length = np.linalg.norm(direction)
        
        if length == 0:  # Skip degenerate edges
            continue

        direction /= length  # Normalize the direction vector

        # Cast a ray from v1 in the direction of v2
        locations, index_ray, index_tri = white_matter_mesh.ray.intersects_location(
            ray_origins=[v1], ray_directions=[direction]
        )

        # Check if any intersection occurs along the segment
        for loc in locations:
            if 0 < np.linalg.norm(loc - v1) < length:
                crossing_edges.append(i)
                break  # Move to next edge once a crossing is found

    print(f"Detected {len(crossing_edges)} edges crossing the white matter mesh.")
    return crossing_edges

def split_and_fix_crossing_edges(ventricle_mesh, white_matter_mesh):
    """
    Split edges crossing the white matter mesh, repair the mesh, and ensure watertightness.
    """
    crossing_edges = detect_crossing_edges(ventricle_mesh, white_matter_mesh)
    vertices = ventricle_mesh.vertices.copy()
    faces = ventricle_mesh.faces.copy()

    new_vertices = []  # New vertices (midpoints)
    new_faces = []     # New faces to replace affected ones

    midpoint_cache = {}  # Cache to store midpoints and avoid duplication

    for edge_idx in crossing_edges:
        v1_idx, v2_idx = ventricle_mesh.edges_unique[edge_idx]
        v1, v2 = vertices[v1_idx], vertices[v2_idx]

        # Calculate midpoint and check for duplicates
        edge_key = tuple(sorted([v1_idx, v2_idx]))
        if edge_key not in midpoint_cache:
            midpoint = (v1 + v2) / 2
            new_vertex_idx = len(vertices) + len(new_vertices)
            new_vertices.append(midpoint)
            midpoint_cache[edge_key] = new_vertex_idx
        else:
            new_vertex_idx = midpoint_cache[edge_key]

        # Find faces containing the edge
        face_indices = np.where(
            (faces == v1_idx).sum(axis=1) + (faces == v2_idx).sum(axis=1) == 2
        )[0]
        for face_idx in face_indices:
            face = faces[face_idx]
            remaining_vertex = face[(face != v1_idx) & (face != v2_idx)][0]

            # Replace the old face with two new faces using the midpoint
            new_faces.append([v1_idx, new_vertex_idx, remaining_vertex])
            new_faces.append([v2_idx, new_vertex_idx, remaining_vertex])

            # Mark the old face for removal
            faces[face_idx] = [-1, -1, -1]

    # Remove old faces and add new vertices/faces
    faces = faces[~np.all(faces == -1, axis=1)]  # Remove invalid faces
    vertices = np.vstack([vertices, new_vertices])
    faces = np.vstack([faces, np.array(new_faces)])

    # Rebuild the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Explicitly clean the mesh
    mesh.update_faces(mesh.nondegenerate_faces())  # Remove degenerate faces
    mesh.remove_unreferenced_vertices()            # Remove unused vertices

    # Fill holes safely without processing normals
    if not mesh.is_watertight:
        print("Mesh has holes. Attempting manual fill...")
        filled_mesh = mesh.fill_holes()
        if isinstance(filled_mesh, trimesh.Trimesh):
            mesh = filled_mesh

    # Validate vertices
    mesh.vertices = np.nan_to_num(mesh.vertices, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Added {len(new_vertices)} vertices and {len(new_faces)} new faces.")
    print(f"Mesh is watertight: {mesh.is_watertight}")
    return mesh

def correct_vertices_outside_mesh(mesh, boundary_mesh):
    """
    Correct vertices outside the boundary mesh by projecting them onto its surface.
    Ensures finite vertex positions and prevents duplicates.

    Parameters:
        - mesh: Trimesh object containing vertices to correct.
        - boundary_mesh: Trimesh object representing the boundary (white matter).

    Returns:
        - corrected_mesh: Trimesh object with updated vertices.
    """
    print("Correcting vertices outside the boundary mesh...")
    
    corrected_vertices = mesh.vertices.copy()
    boundary_tree = boundary_mesh.nearest  # KDTree for efficient nearest-surface queries

    # Project vertices onto the boundary surface
    for i, vertex in enumerate(corrected_vertices):
        closest_point, distance, _ = boundary_tree.on_surface([vertex])
        if not np.isfinite(vertex).all():  # Skip invalid vertices
            continue
        if distance > 1e-6:  # If vertex is outside the boundary (with small tolerance)
            corrected_vertices[i] = closest_point[0]  # Project vertex to boundary

    # Ensure no NaN or infinite values remain
    corrected_vertices = np.nan_to_num(corrected_vertices, nan=0.0, posinf=0.0, neginf=0.0)

    # Round to avoid floating-point artifacts
    corrected_vertices = np.round(corrected_vertices, decimals=8)

    # Rebuild the mesh with corrected vertices
    corrected_mesh = trimesh.Trimesh(vertices=corrected_vertices, faces=mesh.faces)
    print("Vertex correction complete. Checking for validity...")

    # Final check for validity
    if not np.all(np.isfinite(corrected_mesh.vertices)):
        raise ValueError("Invalid vertices detected after correction.")
    return corrected_mesh




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

def adaptive_laplacian_smoothing(vertices, faces, white_matter_vertices, vertex_normals,
                                 alpha=0.7, beta=0.3, gamma=0.5, lambda_smooth=0.5, iterations=3, skip_vertices=None):
    """
    Adaptive Laplacian smoothing with curvature, distance, and alignment weights.
    Skips vertices that are within the threshold zones.

    Parameters:
        - vertices: Vertex positions of the mesh (Nx3 array).
        - faces: Face indices of the mesh (Mx3 array).
        - white_matter_vertices: White matter mesh vertices (Px3 array).
        - vertex_normals: Precomputed vertex normals (Nx3 array).
        - alpha: Weight for curvature contribution.
        - beta: Weight for distance contribution.
        - gamma: Weight for directional alignment.
        - lambda_smooth: Smoothing factor for Laplacian smoothing.
        - iterations: Number of smoothing iterations.
        - skip_vertices: Indices of vertices to skip during smoothing.

    Returns:
        - Smoothed vertices (Nx3 array).
    """
    kdtree = cKDTree(white_matter_vertices)

    def compute_curvature(vertices, faces, normals):
        curvature = np.zeros(len(vertices))
        for face in faces:
            v1, v2, v3 = vertices[face]
            n1, n2, n3 = normals[face]
            curvature[face] += np.abs(np.dot(n2 - n1, v3 - v1) +
                                      np.dot(n3 - n2, v1 - v2) +
                                      np.dot(n1 - n3, v2 - v3))
        curvature /= curvature.max() + 1e-8
        return curvature

    white_matter_mean = white_matter_vertices.mean(axis=0)

    for _ in range(iterations):
        # Compute weights
        curvature = compute_curvature(vertices, faces, vertex_normals)
        distances, _ = kdtree.query(vertices)
        normalized_distances = distances / (distances.max() + 1e-8)
        weights = alpha * curvature + beta * (1 - normalized_distances) + gamma

        # Laplacian smoothing
        vertex_neighbors = {i: [] for i in range(len(vertices))}
        for face in faces:
            for v in face:
                vertex_neighbors[v].extend([u for u in face if u != v])

        new_vertices = vertices.copy()
        for i, neighbors in vertex_neighbors.items():
            if skip_vertices is not None and i in skip_vertices:
                continue  # Skip vertices that are within the threshold zones

            neighbors = list(set(neighbors))
            avg_position = np.mean(vertices[neighbors], axis=0)
            new_vertices[i] += weights[i] * lambda_smooth * (avg_position - vertices[i])

        vertices = new_vertices

    return vertices

def adaptive_refinement(mesh, white_matter_mesh, curvature_threshold=0.05, proximity_threshold=0.01):
    """
    Refine the mesh adaptively in regions with high curvature (concave areas) or near the white matter surface,
    and ensure the resulting mesh is watertight.

    Parameters:
        - mesh: Trimesh object (ventricular mesh).
        - white_matter_mesh: Trimesh object (white matter mesh).
        - curvature_threshold: Threshold for curvature to refine concave areas.
        - proximity_threshold: Distance threshold for proximity-based refinement.

    Returns:
        - watertight_mesh: Trimesh object with adaptively refined and watertight mesh.
    """
    from scipy.spatial import cKDTree

    # Step 1: Compute vertex normals and curvature
    vertex_normals = mesh.vertex_normals
    vertices = mesh.vertices
    faces = mesh.faces

    # Calculate curvature as the deviation of vertex normals
    curvature = np.zeros(len(vertices))
    adjacency = mesh.vertex_neighbors
    for i, neighbors in enumerate(adjacency):
        neighbor_normals = vertex_normals[neighbors]
        curvature[i] = np.linalg.norm(vertex_normals[i] - neighbor_normals.mean(axis=0))

    # Step 2: Identify vertices near the white matter surface
    kdtree = cKDTree(white_matter_mesh.vertices)
    distances, _ = kdtree.query(vertices)

    # Step 3: Mark triangles for refinement
    refine_faces = []
    for face_idx, face in enumerate(faces):
        face_curvature = curvature[face].mean()
        face_distance = distances[face].mean()
        
        if face_curvature > curvature_threshold or face_distance < proximity_threshold:
            refine_faces.append(face_idx)

    # Step 4: Subdivide marked triangles
    print(f"Refining {len(refine_faces)} triangles with high curvature or proximity...")
    refined_mesh = mesh.subdivide(face_index=refine_faces)

    # Step 5: Fix mesh connectivity and ensure watertightness
    print("Ensuring the mesh is watertight...")
    meshfix = MeshFix(refined_mesh.vertices, refined_mesh.faces)
    meshfix.repair(verbose=False, joincomp=True, remove_smallest_components=True)
    watertight_mesh = trimesh.Trimesh(vertices=meshfix.points, faces=meshfix.faces)

    # Validate watertightness
    if not watertight_mesh.is_watertight:
        print("Warning: The mesh is still not fully watertight after repair.")
    else:
        print("Mesh successfully refined and made watertight.")

    return watertight_mesh  


def enforce_boundary(vertices, white_matter_vertices, threshold=0.01):
    """
    Prevent vertices from overshooting the white matter boundary.

    Parameters:
        - vertices: Current mesh vertices (Nx3 array).
        - white_matter_vertices: White matter surface vertices (Px3 array).
        - threshold: Maximum allowable distance from the white matter surface.

    Returns:
        - Corrected vertices (Nx3 array).
    """
    from scipy.spatial import cKDTree

    kdtree = cKDTree(white_matter_vertices)
    distances, indices = kdtree.query(vertices)  # Find nearest points on white matter

    corrected_vertices = vertices.copy()
    for i, distance in enumerate(distances):
        if distance <= threshold:
            # Calculate direction back to the boundary
            direction = white_matter_vertices[indices[i]] - vertices[i]
            direction /= np.linalg.norm(direction) + 1e-8  # Normalize direction
            # Adjust vertex position to sit exactly on the boundary
            corrected_vertices[i] = white_matter_vertices[indices[i]]

    return corrected_vertices

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



def expand_ventricle_dynamic_fraction_auto(
    ventricle,
    white_matter,
    steps=40,
    f_min=0.01,
    f_max=0.4,
    thresholds=[0.01, 0.02, 0.03],
    percentages=[0.9, 0.95, 1.0]
):
    """
    Expand the ventricle mesh dynamically in stages, ensuring vertices do not cross
    the white matter boundary. Adaptive Laplacian smoothing improves mesh quality.

    Parameters:
        - ventricle: Trimesh object for the ventricle.
        - white_matter: Trimesh object for the white matter mesh.
        - steps: Number of expansion steps per stage.
        - f_min: Minimum expansion fraction.
        - f_max: Maximum expansion fraction.
        - thresholds: Distance thresholds for stages.
        - percentages: Percentage of vertices required within thresholds.

    Returns:
        - ventricle: Expanded and adaptively smoothed Trimesh object.
    """
    output_dir = "ventricle_obj_files"
    os.makedirs(output_dir, exist_ok=True)
    json_output_file = "expansion_vectors.json"

    step_counter = 1
    expansion_data = []  # Store all expansion vectors and positions
    ventricle_steps = []

    print(f"Initial ventricle volume: {ventricle.volume:.4f}")
    print(f"White matter volume: {white_matter.volume:.4f}")

    for stage_idx, (threshold, percentage) in enumerate(zip(thresholds, percentages)):
        print(f"\nStarting Stage {stage_idx + 1} - Threshold: {threshold}, Required: {percentage * 100:.2f}%")
        offset_white_matter = inward_offset_mesh(white_matter, threshold)
        print(f"Offset white matter mesh created with threshold: {threshold}")

        for step in range(steps):
            print(f"Starting Step {step}")  # Debug step value
            previous_vertices = ventricle.vertices.copy()
            normals = calculate_corrected_vertex_normals(previous_vertices, ventricle.faces)

            new_vertices = previous_vertices.copy()
            step_vectors = []

            # Measure distances to the white matter surface
            distances = np.full(len(previous_vertices), np.inf)
            for i, (vertex, normal) in enumerate(zip(previous_vertices, normals)):
                locations, _, _ = white_matter.ray.intersects_location(
                    ray_origins=[vertex], ray_directions=[normal]
                )
                if locations.size > 0:
                    distances[i] = np.linalg.norm(locations[0] - vertex)

            # Identify vertices inside the no-movement zone
            vertices_in_offset = []
            for i, vertex in enumerate(previous_vertices):
                # Distance to offset mesh
                offset_closest_point, _, _ = offset_white_matter.nearest.on_surface([vertex])
                offset_distance = np.linalg.norm(offset_closest_point - vertex)
                
                # Distance to white matter mesh
                wm_closest_point, _, _ = white_matter.nearest.on_surface([vertex])
                wm_distance = np.linalg.norm(wm_closest_point - vertex)

                # Check if vertex is between offset and white matter or beyond white matter
                if offset_distance <= threshold or wm_distance < offset_distance:
                    vertices_in_offset.append(i)

            # Adjust expansion fraction dynamically
            current_volume = ventricle.volume
            volume_ratio = abs(current_volume / white_matter.volume)
            fraction = f_min + (f_max - f_min) * min(1, (volume_ratio - 0.01) / 0.84)

            # Expand vertices
            for i, (vertex, normal) in enumerate(zip(previous_vertices, normals)):
                if i not in vertices_in_offset and distances[i] < np.inf:
                    # Calculate move vector
                    move_vector = normal * min(distances[i] * fraction, distances[i])
                    new_vertices[i] = vertex + move_vector

                    # Store position and expansion vector
                    step_vectors.append({
                        "position": vertex.tolist(),
                        "vector": move_vector.tolist()
                    })
            
            '''if step > 1:
                print(f"Correction done!")
                # Correct vertices that crossed the white matter surface
                new_vertices = correct_crossing_vertices(
                    vertices=previous_vertices,
                    expanded_vertices=new_vertices,
                    faces=ventricle.faces, 
                    white_matter_mesh=white_matter
                )'''

            ventricle = trimesh.Trimesh(vertices=new_vertices, faces=ventricle.faces)
            '''
            # Correct vertices outside the white matter boundary
            ventricle = correct_vertices_outside_mesh(ventricle, white_matter)

            # Fix edges that cross the white matter mesh
            crossing_edges = detect_crossing_edges(ventricle, white_matter)
            if crossing_edges:
                print(f"Fixing {len(crossing_edges)} crossing edges...")
                ventricle = split_and_fix_crossing_edges(ventricle, white_matter)
        '''
            # Store expansion data
            expansion_data.append({
                "step": step_counter,
                "vectors": step_vectors
            })

            ventricle = adaptive_refinement(ventricle, white_matter, curvature_threshold=0.3, proximity_threshold=0.01)

            # Remesh to constant face area
            ventricle = remesh_to_constant_face_area(ventricle, max_face_area=0.001)

            # Recalculate normals and apply adaptive smoothing
            updated_normals = calculate_corrected_vertex_normals(ventricle.vertices, ventricle.faces)
            ventricle.vertices = adaptive_laplacian_smoothing(
                vertices=ventricle.vertices,
                faces=ventricle.faces,
                white_matter_vertices=white_matter.vertices,
                vertex_normals=updated_normals,
                alpha=0.1, beta=0.1, gamma=0.2, lambda_smooth=0.6, iterations=4,
                skip_vertices=vertices_in_offset  # Skip vertices in the threshold zones
            )

            # Save the current step
            obj_filename = os.path.join(output_dir, f"ventricle_step_{step_counter:04d}.obj")
            export_combined_mesh_with_opacity(ventricle, white_matter, white_matter, obj_filename)
            visualize_expansion_process(ventricle, white_matter, step=step_counter)
            ventricle_steps.append(ventricle.copy())

            print(f"Step {step_counter}: Expansion and smoothing completed.")
            step_counter += 1

            # Check if enough vertices are within the threshold
            current_percentage = len(vertices_in_offset) / len(previous_vertices)
            print(f"Step {step_counter}: {current_percentage * 100:.2f}% of vertices within threshold: {threshold}")
            if current_percentage >= percentage:
                print(f"Stage {stage_idx + 1} completed. Moving to the next stage.")
                break

    # Generate a visualization GIF
    generate_growth_gif(output_dir="visualization_steps", gif_name="growth_animation.gif")
    print("\nAll stages completed.")

    # Save expansion data to JSON
    with open(json_output_file, "w") as json_file:
        json.dump(expansion_data, json_file, indent=4)
    print(f"Expansion vectors saved to {json_output_file}")

    return ventricle




# Main function
if __name__ == "__main__":
    ventricle = generate_bumpy_sphere(center=(0, 0, 0), radius=0.2, resolution=50)
    # white_matter = generate_pyramid(center=(0, 0, 0), base_length=1.0, height=1.5)
    # white_matter = refine_pyramid(white_matter, splits=4)
    white_matter = generate_star_polygon(center=(0, 0, -0.4), inner_radius=0.21, outer_radius=0.6, num_points=8, extrusion=0.8)
    # white_matter = generate_flower_shape(center=(0, 0, 0), radius=0.7, resolution=200, petal_amplitude=0.2, petal_frequency=3)

    white_matter_face = np.mean(white_matter.area_faces)
    print(f"Average face area of the white matter mesh: {white_matter_face}")

    expanded_ventricle = expand_ventricle_dynamic_fraction_auto(
        ventricle = ventricle, 
        white_matter = white_matter, 
        steps=20, 
        f_min=0.1, 
        f_max=0.11,
        # thresholds=[0.15, 0.09, 0.02],  # Stages with thresholds
        thresholds=[0.01, 0.005, 0.001],
        percentages=[0.5, 0.6, 0.95]    # Required percentages
        )

    print("\nFinal Expanded Ventricle Mesh:")
    print(f"Vertex count: {len(expanded_ventricle.vertices)}")
    print(f"Edge count: {len(expanded_ventricle.edges)}")
    print(f"Face count: {len(expanded_ventricle.faces)}")
    print(f"Is watertight: {expanded_ventricle.is_watertight}")

    visualize_expansion_process(expanded_ventricle, white_matter, step="Final")
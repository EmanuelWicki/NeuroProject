import numpy as np
import trimesh
import os
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from pymeshfix import MeshFix
from trimesh.remesh import subdivide
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
    thresholds=[0.01, 0.02, 0.03],  # Threshold distances
    percentages=[0.9, 0.95, 1.0]    # Percentage of vertices required within each threshold
):
    """
    Expand the ventricle mesh dynamically in stages, ensuring vertices do not cross
    the white matter boundary.

    Parameters:
        - ventricle: Trimesh object representing the ventricle.
        - white_matter: Trimesh object representing the white matter.
        - steps: Number of expansion steps.
        - f_min: Minimum fraction for expansion.
        - f_max: Maximum fraction for expansion.
        - thresholds: List of distances for each stage.
        - percentages: List of percentages of vertices required within each threshold.

    Returns:
        - ventricle: Expanded Trimesh object.
    """
    assert len(thresholds) == len(percentages), "Thresholds and percentages must have the same length."
    
    num_stages = len(thresholds)
    fixed_vertices = np.zeros(len(ventricle.vertices), dtype=bool)  # Track fixed vertices

    # PyVista plotter setup
    plotter = None  # To enable live updates
    ventricle_steps = []  # For time-series saving
    output_time_series="ventricle_growth.vtm"

    # Create a copy of the white matter mesh for visualization purposes
    initial_ventricle_volume  = ventricle.volume
    white_matter_visualization = white_matter.copy()
    white_matter_volume = white_matter_visualization.volume
    initial_ratio = abs(initial_ventricle_volume / white_matter_volume)

    print(f"Initial ventricle volume: {ventricle.volume:.4f}")
    print(f"White matter volume: {white_matter.volume:.4f}")

    for stage_idx, (threshold, percentage) in enumerate(zip(thresholds, percentages)):
        print(f"\nStarting Stage {stage_idx + 1}/{num_stages} - Threshold: {threshold}, Required Percentage: {percentage * 100}%")
        
        for step in range(steps):
            # Save current vertices before remeshing
            previous_vertices = ventricle.vertices.copy()

            # Calculate normals and distances
            normals = calculate_corrected_vertex_normals(previous_vertices, ventricle.faces)
            distances = np.full(len(previous_vertices), np.inf)  # Initialize distances

            for i, (vertex, normal) in enumerate(zip(previous_vertices, normals)):
                if fixed_vertices[i]:
                    continue  # Skip fixed vertices

                locations, _, _ = white_matter.ray.intersects_location(
                    ray_origins=[vertex], ray_directions=[normal]
                )
                if locations.size > 0:
                    closest_point = locations[0]
                    intersection_distance = np.linalg.norm(closest_point - vertex)
                    distances[i] = intersection_distance  # Record the distance

            # Dynamically adjust the fraction for movement
            current_volume = ventricle.volume
            volume_ratio = abs(current_volume / white_matter_volume)

            if volume_ratio <= 0.8:
                fraction = f_min + ((f_max - f_min) / (0.8 - initial_ratio)) * (volume_ratio - initial_ratio)
            else:
                fraction = f_max
            
            print(f"\nStep {step + 1}: Volume Ratio = {volume_ratio:.4f}, Expansion Fraction = {fraction:.4f}")

            # Mark vertices within the threshold as fixed
            within_threshold = distances <= threshold
            fixed_vertices |= within_threshold  # Update fixed vertices

            # Calculate the percentage of vertices within the threshold
            current_percentage = np.mean(within_threshold)
            print(f"Step {step + 1}: {current_percentage * 100:.2f}% of vertices within threshold.")

            if current_percentage >= percentage:
                print(f"Stage {stage_idx + 1} completed. Moving to the next stage.")
                break

            # Expand non-fixed vertices
            new_vertices = ventricle.vertices.copy()
            for i, (vertex, normal) in enumerate(zip(previous_vertices, normals)):
                if fixed_vertices[i]:
                    continue  # Skip fixed vertices

                if distances[i] < np.inf:  # If intersection is found
                    move_vector = normal * min(distances[i] * fraction, distances[i])
                    new_vertices[i] = vertex + move_vector

            # Update the ventricle mesh
            ventricle = trimesh.Trimesh(vertices=new_vertices, faces=ventricle.faces)

            # Remesh using face area
            ventricle = remesh_to_constant_face_area(ventricle, max_face_area=0.002)

            # Update fixed_vertices to match the new vertex count
            new_vertex_count = len(ventricle.vertices)
            if len(fixed_vertices) != new_vertex_count:
                print(f"Updating fixed vertices: Old count = {len(fixed_vertices)}, New count = {new_vertex_count}")
                new_fixed_vertices = np.zeros(new_vertex_count, dtype=bool)

                # Match fixed vertices to the new vertex array
                for old_idx, is_fixed in enumerate(fixed_vertices):
                    if is_fixed:
                        # Find the closest new vertex to the old vertex
                        old_vertex = previous_vertices[old_idx]
                        closest_new_idx = np.argmin(np.linalg.norm(ventricle.vertices - old_vertex, axis=1))
                        new_fixed_vertices[closest_new_idx] = True

                fixed_vertices = new_fixed_vertices

            # Apply smoothing
            ventricle = laplacian_smoothing(ventricle, iterations=3, lambda_factor=0.9)
            # Visualize Meshes
            # visualize_with_pyvista_live(ventricle, white_matter)

            # Visualize and save the current state
            visualize_expansion_process(ventricle, white_matter_visualization, step=f"Stage{stage_idx + 1}_Step{step + 1}")

            # Append the current ventricle step for time-series
            ventricle_steps.append(ventricle.copy())

    # After the loop, generate the GIF
    generate_growth_gif(output_dir="visualization_steps", gif_name="growth_animation.gif")
    print("\nAll stages completed.")

    # Save the time-series for interactive review
    save_growth_interactively(ventricle_steps, white_matter_visualization, output_filename=output_time_series)
    print(f"\nInteractive time-series saved to {output_time_series}")


    return ventricle



# Main function
if __name__ == "__main__":
    ventricle = generate_bumpy_sphere(center=(0, 0, 0), radius=0.2, resolution=30)
    # white_matter = generate_pyramid(center=(0, 0, 0), base_length=1.0, height=1.5)
    # white_matter = refine_pyramid(white_matter, splits=4)
    white_matter = generate_flower_shape(center=(0, 0, 0), radius=0.6, resolution=200, petal_amplitude=0.3, petal_frequency=3)

    white_matter_face = np.mean(white_matter.area_faces)
    print(f"Average face area of the white matter mesh: {white_matter_face}")

    expanded_ventricle = expand_ventricle_dynamic_fraction_auto(
        ventricle = ventricle, 
        white_matter = white_matter, 
        steps=2, 
        f_min=0.02, 
        f_max=0.4,
        thresholds=[0.06, 0.03, 0.02],  # Stages with thresholds
        percentages=[0.9, 0.95, 1.0]    # Required percentages
    )

    print("\nFinal Expanded Ventricle Mesh:")
    print(f"Vertex count: {len(expanded_ventricle.vertices)}")
    print(f"Edge count: {len(expanded_ventricle.edges)}")
    print(f"Face count: {len(expanded_ventricle.faces)}")
    print(f"Is watertight: {expanded_ventricle.is_watertight}")

    visualize_expansion_process(expanded_ventricle, white_matter, step="Final")
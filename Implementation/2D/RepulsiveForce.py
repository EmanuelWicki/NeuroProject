import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a half-pipe mesh
theta = np.linspace(-np.pi / 2, np.pi / 2, 20)
z = np.linspace(-1, 1, 10)
theta, z = np.meshgrid(theta, z)

# Radius with variations for an irregular surface
radius = 1 + 0.3 * np.sin(3 * theta)
x = radius * np.cos(theta)
y = radius * np.sin(theta)

# Compute vertex normals using cross product of neighboring vectors
vertex_normals = np.zeros((x.shape[0], x.shape[1], 3))
for i in range(1, x.shape[0] - 1):
    for j in range(1, x.shape[1] - 1):
        # Vertices defining the face
        v0 = np.array([x[i, j], y[i, j], z[i, j]])
        v1 = np.array([x[i, j + 1], y[i, j + 1], z[i, j + 1]])
        v2 = np.array([x[i + 1, j], y[i + 1, j], z[i + 1, j]])
        
        # Calculate edge vectors
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Normal calculation using cross product
        normal = np.cross(edge1, edge2)
        normal /= np.linalg.norm(normal)
        
        # Store the normal vector
        vertex_normals[i, j] = normal

# Reverse the normals for inward expansion
vertex_normals = -vertex_normals

# Ensure all normals point consistently inward by aligning with a reference direction
centroid = np.array([np.mean(x), np.mean(y), np.mean(z)])
for i in range(vertex_normals.shape[0]):
    for j in range(vertex_normals.shape[1]):
        direction_to_vertex = np.array([x[i, j], y[i, j], z[i, j]]) - centroid
        dot_product = np.dot(vertex_normals[i, j], direction_to_vertex)
        if dot_product > 0:  # If pointing outward, reverse it to point inward
            vertex_normals[i, j] = -vertex_normals[i, j]

# Expansion factor (inward direction)
expansion_factor = 0.1

# Expand vertices along reversed normals without repulsive force
expanded_vertices_no_repulsion = np.zeros_like(vertex_normals)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        expanded_vertices_no_repulsion[i, j] = np.array([x[i, j], y[i, j], z[i, j]]) + expansion_factor * vertex_normals[i, j]

# Create a copy of the expanded vertices for applying repulsive force
expanded_vertices_with_repulsion = np.copy(expanded_vertices_no_repulsion)

# Apply Repulsion Force
repulsion_coefficient = 0.05
repulsion_threshold = 0.2
p = 0.2  # Controls decay (as per your requirement)

# Store the repulsion forces for visualization
repulsion_forces = np.zeros_like(expanded_vertices_with_repulsion)

for i in range(expanded_vertices_with_repulsion.shape[0]):
    for j in range(expanded_vertices_with_repulsion.shape[1]):
        # Compare with neighbors
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if (di == 0 and dj == 0) or not (0 <= i + di < expanded_vertices_with_repulsion.shape[0]) or not (0 <= j + dj < expanded_vertices_with_repulsion.shape[1]):
                    continue
                
                v_i = expanded_vertices_with_repulsion[i, j]
                v_j = expanded_vertices_with_repulsion[i + di, j + dj]
                distance = np.linalg.norm(v_i - v_j)
                
                # Apply repulsion if vertices are too close
                if distance < repulsion_threshold:
                    force = -repulsion_coefficient * (v_i - v_j) / (distance ** p + 1e-8)  # Adding small value to prevent div by zero
                    repulsion_forces[i, j] += force
                    expanded_vertices_with_repulsion[i, j] += force  # Update the vertex position

# Plot the inward-expanded surface and visualize original vs repelled directions
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the inward-expanded surface (after applying repulsive force)
ax.plot_surface(expanded_vertices_with_repulsion[:, :, 0], expanded_vertices_with_repulsion[:, :, 1], expanded_vertices_with_repulsion[:, :, 2],
                 cmap='viridis', alpha=0.5, edgecolor='k')

# Plot original normal vectors without repulsive force (red arrows)
step = 1  # Step size to reduce the density of arrows for clarity
for i in range(0, expanded_vertices_no_repulsion.shape[0], step):
    for j in range(0, expanded_vertices_no_repulsion.shape[1], step):
        # Original inward normal direction without repulsion
        ax.quiver(expanded_vertices_no_repulsion[i, j, 0], expanded_vertices_no_repulsion[i, j, 1], expanded_vertices_no_repulsion[i, j, 2],
                  vertex_normals[i, j, 0], vertex_normals[i, j, 1], vertex_normals[i, j, 2],
                  length=0.1, color='red', alpha=0.6, label='Original Normal (Inward)' if i == 0 and j == 0 else "")

# Plot repulsion-adjusted directions (green arrows)
for i in range(0, expanded_vertices_with_repulsion.shape[0], step):
    for j in range(0, expanded_vertices_with_repulsion.shape[1], step):
        if np.linalg.norm(repulsion_forces[i, j]) > 1e-5:  # Plot only if a significant repulsion force is present
            # Adjusted direction after repulsion
            adjusted_direction = vertex_normals[i, j] + repulsion_forces[i, j]
            adjusted_direction /= np.linalg.norm(adjusted_direction)  # Normalize
            
            ax.quiver(expanded_vertices_with_repulsion[i, j, 0], expanded_vertices_with_repulsion[i, j, 1], expanded_vertices_with_repulsion[i, j, 2],
                      adjusted_direction[0], adjusted_direction[1], adjusted_direction[2],
                      length=0.1, color='green', alpha=0.8, label='Adjusted Direction (With Repulsion)' if i == 0 and j == 0 else "")

# Set labels and legend
ax.set_title('Inward Expansion with Original and Repulsion-Adjusted Directions')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Remove duplicate labels
ax.legend(by_label.values(), by_label.keys())
plt.show()

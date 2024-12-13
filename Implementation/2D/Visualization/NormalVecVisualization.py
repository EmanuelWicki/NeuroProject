import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define central vertex and neighboring vertices (non-planar)
vi = np.array([0, 0, 0])  # Central vertex
v_neighbors = [
    np.array([1, 0, 0.5]),
    np.array([0, 1, -0.5]),
    np.array([-1, 0, 0.2]),
    np.array([0, -1, -0.3])
]

# Calculate edges (vectors from vi to its neighbors)
edges = [v - vi for v in v_neighbors]

# Compute cross products of adjacent edges for normal estimation
cross_products = [np.cross(edges[i], edges[(i+1) % len(edges)]) for i in range(len(edges))]
normal_vector = np.mean(cross_products, axis=0)
normal_vector /= np.linalg.norm(normal_vector)

# Triangular faces between the central vertex and its neighbors
triangles = [
    [vi, v_neighbors[i], v_neighbors[(i + 1) % len(v_neighbors)]]
    for i in range(len(v_neighbors))
]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot central vertex
ax.scatter(vi[0], vi[1], vi[2], color='r', s=100, label='Vertex i (Central)')

# Plot neighboring vertices and edges
for idx, v in enumerate(v_neighbors):
    ax.scatter(v[0], v[1], v[2], color='b', s=50, label='Neighboring Vertex' if idx == 0 else "")
    # Plot edge vectors from vi to neighbors
    ax.quiver(vi[0], vi[1], vi[2], v[0] - vi[0], v[1] - vi[1], v[2] - vi[2], color='purple', linestyle='--', length=0.8, arrow_length_ratio=0.1)
    ax.text((vi[0] + v[0]) / 2, (vi[1] + v[1]) / 2, (vi[2] + v[2]) / 2, f'$\\mathbf{{e}}_{{i{idx}}}$', color='purple')

# Draw edges between central vertex and neighbors
for v in v_neighbors:
    ax.plot([vi[0], v[0]], [vi[1], v[1]], [vi[2], v[2]], color='gray')

# Draw triangular faces
ax.add_collection3d(Poly3DCollection(triangles, color="cyan", edgecolor="k", alpha=0.25, linewidths=1))

# Plot normal vector
ax.quiver(vi[0], vi[1], vi[2], normal_vector[0], normal_vector[1], normal_vector[2], length=0.8, color='g', label='Normal Vector')

# Labels and legend
ax.set_title('Normal Vector Computation for Vertex i with Non-Planar Mesh Triangles')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()

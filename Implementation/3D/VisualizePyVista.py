import pyvista as pv
import glob

def visualize_ventricle_growth_with_static_white_matter(ventricle_folder, white_matter_path):
    """
    Visualize ventricle growth with a static white matter mesh.

    Parameters:
        - ventricle_folder: Path to the folder containing ventricle .vtp files.
        - white_matter_path: Path to the white matter .vtp file.
    """
    # Load the white matter mesh
    white_matter = pv.read(white_matter_path)

    # Find all ventricle .vtp files
    ventricle_files = sorted(glob.glob(f"{ventricle_folder}/ventricle_step_*.vtp"))
    print("Found ventricle steps:", ventricle_files)

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Add the static white matter mesh with 20% opacity
    plotter.add_mesh(white_matter, color="white", opacity=0.2, label="White Matter")

    # Add each ventricle step to the plot
    for file in ventricle_files:
        print(f"Loading ventricle step: {file}")
        ventricle = pv.read(file)
        plotter.add_mesh(ventricle, label=file.split("\\")[-1])

    # Show the plot interactively
    plotter.add_legend()
    plotter.show()

# Usage
visualize_ventricle_growth_with_static_white_matter(
    ventricle_folder="C:/Users/Emanuel Wicki/Documents/Neuro Project/NeuroProject/ventricle_growth",
    white_matter_path="C:/Users/Emanuel Wicki/Documents/Neuro Project/NeuroProject/white_matter.vtp"
)

import numpy as np
import matplotlib.pyplot as plt
from xcgd_flume_classes import Starfish, XCGDAnalysis
from icecream import ic

if __name__ == "__main__":

    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"

    # Construct the XCGD analysis object
    mesh_length = 3.0
    # xcgd_analysis = XCGDAnalysis(lsf_tree=tree, mesh_length=mesh_length)
    xcgd_analysis = XCGDAnalysis(
        lsf_level=5,
        mesh_length=mesh_length,
        boundary_refinement=3,
        interior_refinement=1,
    )

    # Construct the starfish object
    ndvs = 7
    starfish = Starfish(
        xcgd_analysis=xcgd_analysis, obj_name="starfish", sub_analyses=[], ndvs=ndvs
    )

    # Generate and set a random set of dvs
    c0 = np.random.uniform(low=-0.2, high=0.2, size=ndvs)
    starfish.set_var_values(variables={"coeffs": c0})

    # Visualize each of the contributions to the LSF
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    ax = ax.flatten()

    # Construct the meshgrid that will be used to visualize the contributions to the parameterization
    npts = 100
    x = np.linspace(0.0, mesh_length, npts)
    y = np.linspace(0.0, mesh_length, npts)
    X, Y = np.meshgrid(x, y)

    x0 = y0 = mesh_length / 2

    theta_grid = np.arctan2(Y - y0, X - x0)

    points = np.array(xcgd_analysis.cut_mesh.get_cell_locations()).reshape(-1, 4, 2)

    for i in range(c0.size + 1):

        # If i == 0, plot the baseline LSF field
        if i == 0:
            # Evaluate the contour line for the circle
            contour_line = (
                (X - mesh_length / 2) ** 2
                + (Y - mesh_length / 2) ** 2
                - starfish.parameters["circle_radius"] ** 2
            )

            total = contour_line

        # elif i == c0.size + 1:
        #     contour_line = total

        else:
            # Get the harmonic index
            starting_harmonic = starfish.parameters["starting_harmonic"]
            harmonic = starting_harmonic + (i - 1)

            # Evaluate the contour for the current harmonic
            contour_line = c0[i - 1] * np.cos(harmonic * theta_grid)

            total += contour_line

        ax[i].set_title(rf"$n={i}$", fontweight="normal", fontsize=20)

        # Plot the current contour line
        ax[i].contour(X, Y, total, colors="#640391", linewidths=1.5, levels=[0])

        # Set the axis limits tightly around the mesh
        xmin = points[:, :, 0].min()
        xmax = points[:, :, 0].max()
        ymin = points[:, :, 1].min()
        ymax = points[:, :, 1].max()

        ax[i].set_xlim(xmin, xmax)
        ax[i].set_ylim(ymin, ymax)

        # Preserve physical geometry.
        ax[i].set_aspect("equal", adjustable="box")

        # Remove axes, ticks, and surrounding padding.
        ax[i].set_axis_off()

    # plt.show()
    fig.savefig(fname="Starfish_Parameterization.pdf", bbox_inches="tight", dpi=300)

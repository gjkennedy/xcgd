import xcgd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_mesh(
    filename,
    cut,
    linewidth=0.5,
    show_elements=True,
    show_stencil=True,
    zoom=False,
    length=3.0,
):
    """
    Plot a quadrilateral mesh.

    Parameters
    ----------
    X : ndarray, shape (num_elements, 8)
        Coordinates stored as
        [x0, y0, x1, y1, x2, y2, x3, y3],
        where the node ordering is
        [bottom-left, bottom-right, top-left, top-right].
    """

    X = cut.get_cell_locations()

    # Get the different elements
    interior = cut.get_interior_elements()
    exterior = cut.get_exterior_elements()
    interface = cut.get_interface_elements()

    X = np.asarray(X)

    fig, ax = plt.subplots()

    X_lsf, Y_lsf, lsf = get_lsf()
    ax.contour(X_lsf, Y_lsf, lsf, levels=[0], zorder=1)

    # Shape: (num_elements, 4, 2)
    points = X.reshape(-1, 4, 2)

    # Convert tensor-product ordering into boundary ordering:
    # bottom-left -> bottom-right -> top-right -> top-left
    boundary_order = [0, 1, 3, 2, 0]
    polygons = points[:, boundary_order, :]

    # Convert each polygon into its four line segments.
    segments = np.concatenate(
        [
            polygons[:, 0:2, :],
            polygons[:, 1:3, :],
            polygons[:, 2:4, :],
            polygons[:, 3:5, :],
        ],
        axis=0,
    )

    if show_elements:
        elem_maps = [interior, exterior, interface]
        colors = ["lightblue", "gray", "red"]

        for elem_map, color in zip(elem_maps, colors):
            collection = LineCollection(
                polygons[elem_map],
                facecolors=color,
                edgecolors="none",
                zorder=0,
                alpha=0.6,
            )
            ax.add_collection(collection)

    mesh_lines = LineCollection(
        segments,
        colors="black",
        linewidths=linewidth,
        zorder=2,
    )
    ax.add_collection(mesh_lines)

    if show_stencil:
        # Plot the stencil
        Xp = cut.get_node_locations()
        Xp = np.asarray(Xp)
        Xp = Xp.reshape(-1, 2)

        s = 5
        if zoom:
            s = 10

        interior_stencil = cut.get_interior_stencil()
        stencil = [value for sublist in interior_stencil for value in sublist]
        stencil = np.unique(np.asarray(stencil))
        ax.scatter(
            Xp[stencil, 0],
            Xp[stencil, 1],
            s=s,
            facecolors="white",
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )

        interior_interface_stencil = cut.get_interior_interface_stencil()
        stencil = [value for sublist in interior_interface_stencil for value in sublist]
        stencil = np.unique(np.asarray(stencil))
        ax.scatter(
            Xp[stencil, 0],
            Xp[stencil, 1],
            s=s,
            facecolors="white",
            edgecolors="black",
            linewidths=0.7,
            zorder=4,
        )

    if zoom:
        interior_quad = cut.get_interior_quadrature_points()
        quad_pts = [value for sublist in interior_quad for value in sublist]
        quad_pts = np.asarray(quad_pts).reshape(-1, 2)
        ax.scatter(
            quad_pts[:, 0],
            quad_pts[:, 1],
            s=3,
            facecolors="black",
            linewidths=0.7,
            zorder=3,
        )

        xmin = length * 0.25
        xmax = xmin + 0.25
        ymin = length * 5 / 16
        ymax = ymin + 0.25
    else:
        # Fit tightly around the complete mesh.
        xmin = points[:, :, 0].min()
        xmax = points[:, :, 0].max()
        ymin = points[:, :, 1].min()
        ymax = points[:, :, 1].max()

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Preserve physical geometry.
    ax.set_aspect("equal", adjustable="box")

    # Remove axes, ticks, and surrounding padding.
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fig.savefig(
        filename,
        dpi=500,
        bbox_inches="tight",
        pad_inches=0,
    )

    return fig, ax


def get_lsf(length=3.0, x0=1.5, y0=1.5, r0=1.0, c2=0.5):
    x = np.linspace(0.0, length, 200)
    y = np.linspace(0.0, length, 200)
    X, Y = np.meshgrid(x, y)

    theta = np.atan2(Y - y0, X - x0)
    lsf = (X - x0) ** 2 + (Y - y0) ** 2 - r0**2
    lsf += c2 * np.cos(5 * theta)

    return X, Y, lsf


def generate_cut_mesh(level=4, length=3.0, refine=True):
    tree = xcgd.Quadtree()
    tree.refine([level])
    tree.balance()

    source = tree.duplicate()
    lsf_mesh = xcgd.QuadtreeMesh(source, length)
    mesh = xcgd.QuadtreeMesh(tree, length)

    # Get the node locations and specify the level set function
    X = np.array(lsf_mesh.get_node_locations())

    cut_mesh = xcgd.QuadtreeCutMesh(mesh, lsf_mesh)
    lsf = cut_mesh.get_lsf()

    x = X[0::2]
    y = X[1::2]
    x0 = 1.5
    y0 = 1.5
    r0 = 1.0
    theta = np.atan2(y - y0, x - x0)
    c2 = 0.5

    lsf[:] = (x - x0) ** 2 + (y - y0) ** 2 - r0**2
    lsf[:] += c2 * np.cos(5 * theta)
    cut_mesh.update()

    if refine:
        interface_elems = cut_mesh.get_interface_elements()
        interior_elems = cut_mesh.get_interior_elements()

        refinement = np.zeros(tree.size(), dtype=np.int32)
        refinement[interface_elems] = 2
        refinement[interior_elems] = 1
        tree.refine(refinement)
        tree.balance()

    mesh.update()
    cut_mesh.update()

    return cut_mesh


if __name__ == "__main__":

    cut_mesh = generate_cut_mesh(level=4, refine=False)
    plot_mesh(
        "starfish_levelset.png",
        cut_mesh,
        show_elements=False,
        show_stencil=False,
        zoom=False,
    )

    cut_mesh = generate_cut_mesh(level=4)
    plot_mesh(
        "starfish_levelset_refine.png",
        cut_mesh,
        show_elements=False,
        show_stencil=False,
        zoom=False,
    )
    plot_mesh(
        "starfish_elements.png",
        cut_mesh,
        show_elements=True,
        show_stencil=False,
        zoom=False,
    )
    plot_mesh(
        "starfish_elements_and_stencil.png",
        cut_mesh,
        show_elements=True,
        show_stencil=True,
        zoom=False,
    )

    cut_mesh = generate_cut_mesh(level=5)
    plot_mesh(
        "starfish_quadrature_zoom.png",
        cut_mesh,
        show_elements=True,
        show_stencil=True,
        zoom=True,
    )

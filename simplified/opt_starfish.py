from flume.base_classes.system import System
from flume.interfaces.paropt_interface import FlumeParOptInterface
import numpy as np
from xcgd_flume_classes import (
    Starfish,
    XCGDAnalysis,
    XCGDArea,
    XCGDMinFrequency,
    XCGDPerimeter,
)
import xcgd
from math import pi
from paropt import ParOpt
import matplotlib.pyplot as plt
from icecream import ic
import shutil
import os
import argparse
from starfish import plot_mesh


class StarfishCallback:

    # Column width for design variable values (must fit e.g. "-1.234567e-01")
    COL_WIDTH = 16

    def __init__(
        self,
        starfish: Starfish,
        xcgd_analysis: XCGDAnalysis,
        output_prefix: str,
        perimeter,
        xcgd_frequency,
        perim_target,
        plot_every: int = 1,
        vtk_every: int = 5,
    ):

        # Store the starfish object
        self.starfish = starfish

        # # Store the quadtree for VTK output
        # self.tree = tree
        self.xcgd = xcgd_analysis

        # Store the output prefix
        self.output_prefix = output_prefix

        # Store the perimeter analysis object
        self.perimeter = perimeter

        # Store the frequency analysis object
        self.freq = xcgd_frequency

        # Store the target perimeter value for constraint violation reporting
        self.perim_target = perim_target

        # Determine number of DVs for header formatting
        self.ndvs = len(starfish.variables["coeffs"].value)

        # Store params to determine when to save png and write vtk files
        self.plot_every = plot_every
        self.vtk_every = vtk_every

        # Initialize the DV log file with the header
        self.dvs_log_path = os.path.join(self.output_prefix, "dvs_log.log")
        with open(self.dvs_log_path, "w") as f:
            f.write(self._format_header() + "\n")

    def _format_header(self):
        """Format the column header line."""
        col_w = self.COL_WIDTH
        header = f"{'iter':<6}"
        for i in range(self.ndvs):
            header += f"{'coeffs[' + str(i) + ']':>{col_w}}"
        return header

    def __call__(self, x, it_num):

        # Log design variables to file
        with open(self.dvs_log_path, "a") as f:
            # Write a header line every 10 iterations
            if it_num > 0 and it_num % 10 == 0:
                f.write(self._format_header() + "\n")

            col_w = self.COL_WIDTH
            line = f"{it_num:<6}"
            for val in np.array(x):
                line += f"{val:>{col_w}.8e}"
            f.write(line + "\n")

        if it_num % self.plot_every == 0:
            # Plot the figure for the current iteration
            # plot_mesh(filename=filename, cut=)

            # Plot the LSF contour
            fig = self.starfish.plot_lsf()

            # # Add iteration and perimeter constraint violation to the title
            # perim_val = self.perimeter.outputs["perimeter"].value
            # delta_p = (perim_val - self.perim_target) / (2 * pi)

            # # Also get the frequency value and add it to the figure title
            # omega_ks_min = self.freq.outputs["omega_ks_min"].value

            # fig.axes[0].set_title(
            #     rf"Iter = {it_num}: $\omega_{{ks}}$ = {omega_ks_min:.4f}, $\Delta P / P_0$ = {delta_p:.2e}"
            # )

            savename = os.path.join(
                self.output_prefix, "contours", f"lsf_contour_it_{it_num}.png"
            )
            fig.savefig(savename, bbox_inches="tight", dpi=400)

            plt.close(fig)

        if it_num % self.vtk_every == 0:
            # Save the mesh to VTK
            tree = self.xcgd.tree

            vtk_path = os.path.join(
                self.output_prefix, "meshes", f"mesh_it_{it_num}.vtk"
            )
            tree.to_vtk(vtk_path)

        if it_num % 10 == 0:
            print(f"Completed iteration {it_num}")

        return


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Starfish topology optimization with XCGD"
    )
    parser.add_argument(
        "--ndvs", type=int, default=5, help="Number of design variables (default: 5)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.2,
        help="Alpha value for perimeter constraint (default: 1.2)",
    )
    args = parser.parse_args()

    ndvs = args.ndvs
    alpha = args.alpha

    # Set output prefix based on ndvs and alpha
    output_prefix = f"output/output_ndvs_{ndvs}_alpha_{alpha:.2f}"

    # Create output directories
    if os.path.exists(os.path.join(output_prefix, "contours")):
        shutil.rmtree(os.path.join(output_prefix, "contours"))

    if os.path.exists(os.path.join(output_prefix, "meshes")):
        shutil.rmtree(os.path.join(output_prefix, "meshes"))

    os.makedirs(os.path.join(output_prefix, "contours"), exist_ok=True)
    os.makedirs(os.path.join(output_prefix, "meshes"), exist_ok=True)

    # Construct the XCGD analysis object
    mesh_length = 3.0
    # xcgd_analysis = XCGDAnalysis(lsf_tree=tree, mesh_length=mesh_length)
    xcgd_analysis = XCGDAnalysis(
        lsf_level=5,
        mesh_length=mesh_length,
        boundary_refinement=3,
        interior_refinement=1,
    )

    # Construct the design variable object
    starfish = Starfish(
        xcgd_analysis=xcgd_analysis, obj_name="starfish", sub_analyses=[], ndvs=ndvs
    )

    c0 = np.random.uniform(low=-0.2, high=0.2, size=ndvs)

    starfish.set_var_values(variables={"coeffs": c0})

    # Construct the XCGDFrequencyAnalysisObject
    ks_param = 50.0
    xcgd_freq = XCGDMinFrequency(
        xcgd_analysis=xcgd_analysis,
        obj_name="XCGD_Frequency",
        sub_analyses=[starfish],
        ks_param=ks_param,
    )

    # Construct the XCGDArea object
    area = XCGDArea(
        xcgd_analysis=xcgd_analysis, obj_name="XCGD_Area", sub_analyses=[starfish]
    )

    # Construct the XCGDPerimeter object
    perimeter = XCGDPerimeter(
        xcgd_analysis=xcgd_analysis, obj_name="XCGD_Perimeter", sub_analyses=[starfish]
    )

    # Define the System
    system = System(
        sys_name="XCGD_Starfish_Opt",
        top_level_analysis_list=[xcgd_freq, area, perimeter],
        log_name="flume.log",
        log_prefix=output_prefix,
    )

    # Define the design variables for the system
    coeffs_lb = -0.5
    coeffs_ub = 0.5

    system.declare_design_vars(
        global_var_name={"starfish.coeffs": {"lb": coeffs_lb, "ub": coeffs_ub}}
    )

    # Declare the objective for the system
    system.declare_objective(
        global_obj_name="XCGD_Frequency.omega_ks_min", obj_scale=-1.0
    )

    # Declare the constraints for the system (assuming area = pi is the equality constraint)
    area_val = pi
    perim_val = alpha * 2 * pi

    system.declare_constraints(
        global_con_name={
            "XCGD_Area.area": {"rhs": area_val, "direction": "both"},
            "XCGD_Perimeter.perimeter": {"rhs": perim_val, "direction": "both"},
        }
    )

    # Setup the FlumeParOptInterface
    starfish_callback = StarfishCallback(
        starfish=starfish,
        xcgd_analysis=xcgd_analysis,
        output_prefix=output_prefix,
        perimeter=perimeter,
        xcgd_frequency=xcgd_freq,
        perim_target=perim_val,
        plot_every=1,
        vtk_every=5,
    )

    interface = FlumeParOptInterface(flume_sys=system, callback=starfish_callback)

    # Create the ParOpt problem and get the options
    paroptprob = interface.construct_paropt_problem()

    # for i in range(2):
    #     paroptprob.checkGradients(1e-6)
    #     exit()

    maxit = 200
    options = interface.get_paropt_default_options(
        output_prefix=output_prefix, algorithm="mma", maxit=maxit
    )
    options["mma_move_limit"] = 0.25
    options["mma_init_asymptote_offset"] = 0.5
    options["mma_asymptote_relax"] = 1.5

    # Perform the optimization
    opt = ParOpt.Optimizer(paroptprob, options)
    opt.optimize()

    # Extract the optimized point
    x, z, zw, zl, zu = opt.getOptimizedPoint()

    # Write the optimized point to the json file
    x_opt = np.array(x)

    ic(x_opt)

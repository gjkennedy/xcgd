import os
import imageio.v3 as imageio
import argparse
from pathlib import Path
import re


def create_animation(files, savename, fps=10, final_pause_secs=2):
    # Get the file extension from the savename
    ext = os.path.splitext(savename)[1].lower()

    # Read the images contained within the files list
    frames = [imageio.imread(file) for file in files]

    # Construct the evolution based on the file extension
    if ext == ".gif":
        # Build per-frame durations: all frames use the normal fps-based duration
        # except the last frame, which holds for final_pause_secs seconds
        frame_duration_ms = 1000 / fps
        durations = [frame_duration_ms] * len(frames)
        durations[-1] = final_pause_secs * 1000

        # loop=0 means infinite loop
        imageio.imwrite(savename, frames, duration=durations, loop=0)
    elif ext in (".mp4", ".avi", ".mov", ".mkv"):
        imageio.imwrite(savename, frames, fps=fps, codec="libx264")
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    return


if __name__ == "__main__":
    # Define the argument parser from the command line
    parser = argparse.ArgumentParser(
        description="File that is used to generate the results/plots of interest for a topology optimized result."
    )

    # Add an input directory for the  as a command line argument
    parser.add_argument(
        "--results_dir",
        "-rdir",
        type=str,
        default=None,
        help="Path that defines the location of a JSON file that contains the initial optimal design variables to use for the optimization",
    )

    # Add an argument for the folder that contains the designs
    parser.add_argument(
        "--design-folder-name",
        "-folder",
        type=str,
        default=None,
        help="Name that defines the folder that contains the .png files that should be combined to visualize the design history",
    )

    # Add an argument that specifies the fps for the video
    parser.add_argument(
        "--frames-per-second",
        "-fps",
        type=int,
        default=10,
        help="Frames per second to use when creating the animation/video for the topology optimization history.",
    )

    # Add an argument that specifies the file format/extension for the video
    parser.add_argument(
        "--file-extension",
        "-ext",
        default="gif",
        type=str,
        help="File format to use for the topology evolution.",
    )

    # Add an argument that specifies the pause duration on the final frame
    parser.add_argument(
        "--final-pause",
        "-pause",
        type=float,
        default=2.0,
        help="Number of seconds to pause on the final frame before looping (default: 2).",
    )

    # Parse the args
    args = parser.parse_args()

    # Load in the information about the domain and mesh for the topology object from the results directory
    rdir = args.results_dir

    # Set the directory that contains the pngs to stitch together
    images_dir = os.path.join(rdir, args.design_folder_name)

    # Loop and plot the arc-length data and the deformed topology
    directory = Path(images_dir)

    pattern = re.compile(r"(\d+)")

    # Find and sort the files by the digit in the filename
    files = sorted(
        (f for f in directory.glob("*.png") if pattern.search(f.name)),
        key=lambda f: int(pattern.search(f.name).group(1)),
    )

    # Build the output filename by stripping the "output_" prefix from the
    # results directory folder name and appending the remainder to "lsf_evolution"
    folder_name = os.path.basename(os.path.normpath(args.results_dir))
    suffix = folder_name.replace("output_", "", 1)
    gif_savename = os.path.join(
        args.results_dir, f"lsf_evolution_{suffix}.{args.file_extension}"
    )

    # Create the animation for the topology evolution
    create_animation(
        files=files,
        savename=gif_savename,
        fps=args.frames_per_second,
        final_pause_secs=args.final_pause,
    )

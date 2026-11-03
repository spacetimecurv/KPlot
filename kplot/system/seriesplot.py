"""Plot a series of 2D slices (prefer the global plotter function)."""

# Import necessary standard libaries.
import os
import glob
import subprocess

# Import necessary third-party libraries.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches

# Import athplot utility from plot-tools.
from athplot.load_ath_bin import BinaryData
from athplot.image import Image

# Define the series plot class.
class SeriesPlot:
  """
  A class to handle plotting of series data from AthenaK bin output.
  """

  def __init__(self, series_path: str, pattern: str, extent: list, variable: str, label: str,
               slice: float | None = None, sliceplot: str | None = None, plotblocks: bool = True,
               lognorm: bool = False, vmin: float | None = None, vmax: float | None = None,
               blackhole: bool = False, magfields: bool = False, linedensity: float = 1.5,
               output_dir: str | None = None, tracker_path: str | None = None,
               horizon_path: str | None = None) -> None:
    """
    Initialize the SeriesPlot instance with the path to the series data.

    Parameters:
    series_path (str): The path to the series data file (bin folder).
    pattern (str): File pattern to be searched for (<jobname>.<id>.*.bin).
    extent (list): Range to plot inside.
    variable (str): Which variable from the bin file to plot (see athplot).
    label (str): Label to attach to the colorbar.
    slice (float): Which slice to plot on (if 3D data).
    sliceplot (str): Needed for magnetic field lines as flag for components.
    plotblocks (bool): Whether the MB bounds need to be plotted.
    lognorm (bool): Whether to use a colorbar in log-scale.
    vmin (float): Minimum value of the colorbar.
    vmax (float): Maximum value of the colorbar.
    blackhole (bool): Whether a black hole is present and needs to be plotted.
    magfields (bool): Whether the magnetic field lines need to be plotted.
    linedensity (float): Magnetic field line density for plotting.
    output_dir (str): Where to store the image series.
    tracker_path (str): Path to the black hole tracker file.
    horizon_path (str): Path to the horizon finder file.
    """
    self.series_path = series_path
    self.pattern = pattern
    self.extent = extent
    self.variable = variable
    self.slice = slice
    self.sliceplot = sliceplot
    self.plotblocks = plotblocks
    self.lognorm = lognorm
    self.vmin = vmin
    self.vmax = vmax
    self.blackhole = blackhole
    self.magfields = magfields
    self.linedensity = linedensity
    self.output_dir = output_dir
    self.label = label
    if blackhole and not (tracker_path or horizon_path):
      raise ValueError("Tracker path and horizon path must be provided if black hole is present.")

    self.tracker_path = tracker_path
    self.horizon_path = horizon_path

  def plot(self) -> None:
    """
    Load the series data and create the plots.

    Returns:
    None.
    """
    # Set the output directory.
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Load the series data.
    files = sorted(glob.glob(os.path.join(self.series_path, self.pattern)))
    extent = self.extent
    print(f"Found {len(files)} files. Plotting {len(files)} images...")

    if self.blackhole:
      # Load the tracker and horizon data.
      tracker_data = np.loadtxt(self.tracker_path, comments="#")
      horizon_data = np.loadtxt(self.horizon_path, comments="#")

    # Loop over the files.
    for file in files:
      # Extract the frame number for the filename.
      index = int(os.path.basename(file).split('.')[-2])

      # Extract filename for the output image.
      filename = os.path.basename(file)
      output_filename = filename.replace(".bin", ".png")
      output_path = os.path.join(self.output_dir, output_filename)
      if os.path.exists(output_path):
        print(f"Skipping existing file: {output_path}")
        continue

      # Initialize image and data
      print(f"Processing: {file}")
      image = Image(file, extent, self.variable, slice_loc=self.slice)
      print(f"Total number of blocks: {len(image.data.blocks)}")
      time = image.data.time
      data = image.make_image_data(256, 256)

      # Prepare magnetic fields if needed.
      if self.magfields:
        if self.sliceplot == "xy":
          Bx = Image(file, extent, "bcc1", slice_loc=self.slice)
          By = Image(file, extent, "bcc2", slice_loc=self.slice)
          x = np.linspace(extent[0], extent[1], 256)
          y = np.linspace(extent[2], extent[3], 256)
          X, Y = np.meshgrid(x, y)
          bx_data = Bx.make_image_data(256, 256)
          by_data = By.make_image_data(256, 256)
        elif self.sliceplot == "xz":
          Bx = Image(file, extent, "bcc1", slice_loc=self.slice)
          Bz = Image(file, extent, "bcc3", slice_loc=self.slice)
          x = np.linspace(extent[0], extent[1], 256)
          z = np.linspace(extent[2], extent[3], 256)
          X, Z = np.meshgrid(x, z)
          bx_data = Bx.make_image_data(256, 256)
          bz_data = Bz.make_image_data(256, 256)
        elif self.sliceplot == "yz":
          By = Image(file, extent, "bcc2", slice_loc=self.slice)
          Bz = Image(file, extent, "bcc3", slice_loc=self.slice)
          y = np.linspace(extent[0], extent[1], 256)
          z = np.linspace(extent[2], extent[3], 256)
          Y, Z = np.meshgrid(y, z)
          by_data = By.make_image_data(256, 256)
          bz_data = Bz.make_image_data(256, 256)
        else:
          raise ValueError("No valid slice for splot specified! (Options: xy,xz,yz)")

      # Creating the plots.
      plt.figure(figsize=(8, 6))
      if self.lognorm:
        pcm = plt.imshow(data, cmap='inferno',
                         norm=LogNorm(vmin=np.min(data),vmax=np.max(data)),
                         interpolation='nearest', origin='lower',
                         extent=image.extent)
      elif (self.vmin != None and self.vmax != None):
        if self.lognorm:
          pcm = plt.imshow(data, cmap='inferno',
                           norm=LogNorm(vmin=self.vmin,vmax=self.vmax),
                           interpolation='nearest', origin='lower',
                           extent=image.extent)
        else:
          pcm = plt.imshow(data, cmap='inferno',
                           vmin=self.vmin, vmax=self.vmax,
                           interpolation='nearest', origin='lower',
                           extent=image.extent)
      else:
        pcm = plt.imshow(data, cmap='inferno',
                         vmin=np.min(data), vmax=np.max(data),
                         interpolation='nearest', origin='lower',
                         extent=image.extent)

      if self.slice is not None:
        slice_dim, slice_pos = self.slice

      if self.plotblocks:
        for block in image.data.blocks:
          if self.slice is None:
            ext = block.get_extent()
            x1, x2, y1, y2 = ext[0], ext[1], ext[2], ext[3]

            # Optional: Only draw if the block is within the visual 'extent'
            if not (x2 < extent[0] or x1 > extent[1] or y2 < extent[2] or y1 > extent[3]):
              rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=0.5, edgecolor='cyan',
                                      facecolor='none', alpha=0.5)
              plt.gca().add_patch(rect)
          else:
            # Check if this specific block contains the slice plane
            if block.is_in_slice(slice_dim, slice_pos):
              ext = block.get_extent() # Get the 6-element physical boundaries

              # Map the 3D block extent to 2D plot coordinates based on the slice
              if slice_dim == 'z':
                x1, x2, y1, y2 = ext[0], ext[1], ext[2], ext[3]
              elif slice_dim == 'y':
                x1, x2, y1, y2 = ext[0], ext[1], ext[4], ext[5]
              else: # slice_dim == 'x'
                x1, x2, y1, y2 = ext[2], ext[3], ext[4], ext[5]

              # Optional: Only draw if the block is within the visual 'extent'
              if not (x2 < extent[0] or x1 > extent[1] or y2 < extent[2] or y1 > extent[3]):
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        linewidth=0.5, edgecolor='cyan',
                                        facecolor='none', alpha=0.5)
                plt.gca().add_patch(rect)

      # Plot the black hole if present.
      if self.blackhole:
        matchesh = np.isclose(horizon_data[:,1], time)
        matchest = np.isclose(tracker_data[:,1], time)
        idxh = np.where(matchesh)[0][0]
        idxt = np.where(matchest)[0][0]

        if self.sliceplot == "xy":
          circle = plt.Circle((tracker_data[idxt,2], tracker_data[idxt,3]), horizon_data[idxh,-1], edgecolor="cyan", facecolor="black", fill=True)
        elif self.sliceplot == "xz":
          circle = plt.Circle((tracker_data[idxt,2], tracker_data[idxt,4]), horizon_data[idxh,-1], edgecolor="cyan", facecolor="black", fill=True)
        else:
          circle = plt.Circle((tracker_data[idxt,3], tracker_data[idxt,4]), horizon_data[idxh,-1], edgecolor="cyan", facecolor="black", fill=True)
        plt.gca().add_patch(circle)

      # Plot the magnetic field lines if present.
      if self.magfields:
        if self.sliceplot == "xy":
          plt.streamplot(X, Y, bx_data, by_data, color="white", density=self.linedensity, linewidth=0.7)
        elif self.sliceplot == "xz":
          plt.streamplot(X, Z, bx_data, bz_data, color="white", density=self.linedensity, linewidth=0.7)
        else:
          plt.streamplot(Y, Z, by_data, bz_data, color="white", density=self.linedensity, linewidth=0.7)

      # Beautify.
      plt.xlim(extent[0:2])
      plt.ylim(extent[2:])
      plt.colorbar(pcm, extend='both', label=self.label)
      plt.title(fr'$t = {time:.2f}\ M_\odot$')

      plt.savefig(output_path, dpi=300)
      plt.close()

    print("Processing complete.")

  def video(self, framerate: int = 12, outname: str = "out.mp4") -> None:
    """
    Takes the images created with plot() to make a movie.

    Parameters:
    framerate (int): Frames per second.
    outname (str): File name of the movie inside output_dir.

    Returns:
    None.
    """
    # Ensure output directory exists.
    if not os.path.exists(self.output_dir):
      raise FileNotFoundError(f"Output directory not found: {self.output_dir}")

    # Prepare the images.
    prefix = self.pattern.split("*")[0]
    input_pattern = os.path.join(self.output_dir, prefix + "%05d.png")
    output_path = os.path.join(self.output_dir, outname)

    # Make the movie.
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(framerate),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        output_path
    ]

    try:
      subprocess.run(cmd, check=True)
    except FileNotFoundError:
      raise RuntimeError("ffmpeg is not installed or not found in PATH.")
    except subprocess.CalledProcessError as e:
      raise RuntimeError(f"ffmpeg failed: {e}")

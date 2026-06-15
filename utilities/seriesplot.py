#########################################################################
# File: seriesplot.py                                                   #
# Description: Series plot utilities for AthenaK bin output.            #
#########################################################################

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

  def __init__(self, series_path, pattern, extent, variable, label, slice=None, lognorm=False,
               vmin=None, vmax=None, blackhole=False, output_dir=None, tracker_path=None, horizon_path=None):
    """
    Initialize the SeriesPlot instance with the path to the series data.

    Parameters:
    series_path (str): The path to the series data file (bin folder).
    pattern (str): FIle pattern to be searched for (<jobname>.<id>.*.bin).
    extent (list): Range to plot inside.
    variable (str): Which variable from the bin file to plot (see athplot).
    label (str): Label to attach to the colorbar.
    slice: Which slice to plot on (if 3D data).
    lognorm: Whether to use a colorbar in log-scale.
    vmin: Minimum value of the colorbar.
    vmax: Maximum value of the colorbar.
    blackhole: Whether a black hole is present and needs to be plotted.
    output_dir: Where to store the image series.
    tracker_path: Path to the black hole tracker file.
    horizon_path: Path to the horizon finder file.
    """
    self.series_path = series_path
    self.pattern = pattern
    self.extent = extent
    self.variable = variable
    self.slice = slice
    self.lognorm = lognorm
    self.vmin = vmin
    self.vmax = vmax
    self.blackhole = blackhole
    self.output_dir = output_dir
    self.label = label
    if blackhole and not (tracker_path or horizon_path):
      raise ValueError("Tracker path and horizon path must be provided if black hole is present.")

    self.tracker_path = tracker_path
    self.horizon_path = horizon_path

  def plot(self):
    """
    Load the series data and create the plots.
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

        circle = plt.Circle((tracker_data[idxt,2], tracker_data[idxt,3]), horizon_data[idxh,-1], edgecolor="cyan", facecolor="black", fill=True)
        plt.gca().add_patch(circle)

      # Beautify.
      plt.xlim(extent[0:2])
      plt.ylim(extent[2:])
      plt.colorbar(pcm, extend='both', label=self.label)
      plt.title(fr'$t = {time:.2f}\ M_\odot$')

      plt.savefig(output_path, dpi=300)
      plt.close()

    print("Processing complete.")

  def video(self, framerate=12, outname="out.mp4"):
    """
    Takes the images created with plot() to make a movie.
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

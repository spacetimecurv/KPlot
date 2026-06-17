#########################################################################
# File: horizon.py                                                      #
# Description: Horizon utilities for the AthenaK horizon finder.        #
#########################################################################

# Import necessary standard libaries.
import re
import os

# Import necessary third-party libraries.
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["text.usetex"] = True

# Define the horizon finder class.
class HorizonFinder:
  """
  A class to handle output from the AthenaK horizon finder.
  """

  def __init__(self, horizon_path):
    """
    Initialize the HorizonFinder with the path to the horizon file.

    Parameters:
    horizon_path (str): The path to the horizon file.
    """
    self.horizon_path = horizon_path
    self.horizon_data = None

  def load_horizon_data(self):
    """
    Load the horizon data from the specified file path.

    Returns:
    Dictionary holding the horizon data.
    """
    if not os.path.exists(self.horizon_path):
      raise FileNotFoundError(f"Horizon file not found at: {self.horizon_path}")

    # Find the headers.
    with open(self.horizon_path) as f:
      header_line = f.readline().strip()

    headers = re.findall(r'\d+:([^\s]+)', header_line)

    # Read the data with numpy.
    data = np.loadtxt(self.horizon_path, comments='#')

    # Create a dictionary.
    data_dict = {name: data[:, i] for i, name in enumerate(headers)}
    self.horizon_data = data_dict

  def plot_horizon(self, variable=None, output_path=None, save=False,
                   xmin=None, xmax=None, ymin=None, ymax=None,
                   logx=False, logy=False, xlabel="", ylabel="", color="red"):
    """
    Simple function that plots a horizon variable.

    Parameters:
    variable (str): One of the keys of the horizon dict.
    output_path (str): Path where to store the plot.
    save (bool): Whether to store the plot or not.
    xmin (float): Minimum value on x.
    xmax (float): Maximum value on x.
    ymin (float): Minimum value on y.
    ymax (float): Maximum value on y.
    logx (bool): Whether to plot x-axis in log-scale.
    logy (bool): Whether to plot y-axis in log-scale.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    color (str): Color of the plot.
    """
    # Check if variable exists.
    if variable not in self.horizon_data.keys():
      raise ValueError("Specified variable does not exist in the horizon data.\n" \
                       "First load the data or check the headers.")

    # Plotting canvas.
    data = self.horizon_data
    fig, ax = plt.subplots(1,1,figsize=(8,4))
    ax.plot(data["time"], data[variable], color=color, linestyle="solid")
    if xmin != None and xmax != None:
      ax.set_xmin([xmin,xmax])
    else:
      ax.set_xlim([np.min(data["time"]),np.max(data["time"])])

    if ymin != None and ymax != None:
      ax.set_ymin([ymin,ymax])
    else:
      pass

    if logx:
      ax.set_xscale("log")

    if logy:
      ax.set_yscale("log")

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    fig.tight_layout()

    if save:
      plt.savefig(output_path, dpi=200)
    else:
      plt.show()

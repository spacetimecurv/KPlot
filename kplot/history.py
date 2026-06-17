#########################################################################
# File: history.py                                                      #
# Description: Utilities for the AthenaK history output.                #
#########################################################################

# Import necessary standard libaries.
import re
import os
import warnings

# Import third-party libraries.
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["text.usetex"] = True

# Define the History class.
class History:
  """
  A class to handle history output data from AthenaK.
  """

  def __init__(self, history_path):
    """
    Initialize the History class with path to the history output file.

    Parameters:
    history_path (str): Path to the history output file.
    """
    self.history_path = history_path
    self.history_data = None

  def load_history_data(self, raw=False):
    """
    Load the history data from the specified path.
    (Adapted from ~/athenak/vis/python/athena_read.py)

    Parameters:
    raw (bool): If True, do not prune file to remove stale data from
                previous runs.

    Returns:
    Dictionary holding the keys and data columns.
    """
    if not os.path.exists(self.history_path):
      raise FileNotFoundError(f"History file not found at: {self.history_path}")

    # Read data from the specified path.
    with open(self.history_path, 'r') as data_file:
      # Find headers.
      header_found = False
      multiple_headers = False
      header_location = None
      line = data_file.readline()
      while len(line) > 0:
        if line == '# Athena++ history data\n':
          if header_found:
            multiple_headers = True
          else:
            header_found = True
          header_location = data_file.tell()
        line = data_file.readline()
      if multiple_headers:
        warnings.warn('Multiple headers found; using most recent data')
      if header_location is None:
        raise RuntimeError('Could not find header! Check your file.')

      # Parse headers.
      data_file.seek(header_location)
      header = data_file.readline()
      data_names = re.findall(r'\[\d+\]=(\S+)', header)
      if len(data_names) == 0:
        raise RuntimeError('Could not parse header! Check your file.')

      # Prepare dictionary of results.
      data = {}
      for name in data_names:
        data[name] = []

      # Read data
      for line in data_file:
        for name, val in zip(data_names, line.split()):
          data[name].append(float(val))

    # Finalize data
    for key, val in data.items():
      data[key] = np.array(val)
    if not raw:
      branches_removed = False
      while not branches_removed:
        branches_removed = True
        for n in range(1, len(data['time'])):
          if data['time'][n] <= data['time'][n-1]:
            branch_index = np.where((data['time'][:n] >=
                                     data['time'][n]))[0][0]
            for key, val in data.items():
              data[key] = np.concatenate((val[:branch_index],
                                          val[n:]))
            branches_removed = False
            break

    self.history_data = data

  def plot_history(self, variable=None, output_path=None, save=False,
                   xmin=None, xmax=None, ymin=None, ymax=None,
                   logx=False, logy=False, xlabel="", ylabel="", color="red"):
    """
    Simple function that plots a history variable.

    Parameters:
    variable (str): One of the keys of the history dict.
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
    if variable not in self.history_data.keys():
      raise ValueError("Specified variable does not exist in the history data.\n" \
                       "First load the data or check the headers.")

    # Plotting canvas.
    data = self.history_data
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

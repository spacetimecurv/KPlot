#########################################################################
# File: waveforms.py                                                    #
# Description: Waveform utilities for the AthenaK waveform analysis.    #
#########################################################################

# Import necessary standard libaries.
import re
import os
from collections import defaultdict

# Import necessary third-party libraries.
import numpy as np

# Define the waveform reader class.
class Waveform:
  """
  A class to handle waveform data from AthenaK output files.
  """

  def __init__(self, waveform_path):
    """
    Initialize the Waveform class with the path to the waveform files.

    Parameters:
    waveform_path (str): The path to the waveform files.
    """
    self.waveform_path = waveform_path
    self.waveform_data = None

  def load_waveform_data(self):
    """
    Load the waveform data from the specified files path.

    Returns:
    Dictionary holding the waveform data.
    """
    if not os.path.exists(self.waveform_path):
      raise FileNotFoundError(f"Waveform file not found at: {self.waveform_path}")

    data = defaultdict(lambda: defaultdict(dict))

    # Read the files from the waveforms directory.
    regex_pattern = r"rpsi4_(real|imag)_(\d+)\.txt"
    for file in sorted(os.listdir(self.waveform_path)):
      # Extract the extraction radius and real/imag.
      regex = re.compile(regex_pattern)
      m = regex.search(file)
      if not m:
        continue
      kind, radius = m.groups()

      # Load the data.
      file_path = os.path.join(self.waveform_path, file)

      with open(file_path, "r") as f:
        header = [t.split(":")[1] for t in f.readline().lstrip("#").split()]

      data_array = np.loadtxt(file_path, comments="#")

      # Fill the dictionary.
      for i, name in enumerate(header[1:], start=1):
        data[radius][kind][name] = data_array[:,i]

      data[radius]["time"] = data_array[:, 0]

    self.waveform_data = data

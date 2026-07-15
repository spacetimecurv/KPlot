#########################################################################
# File: waveforms.py                                                    #
# Description: Waveform utilities for the AthenaK waveform analysis.    #
#########################################################################

# Import necessary standard libaries.
import re
import os
from collections import defaultdict
import math

# Import necessary third-party libraries.
import numpy as np
from watpy.wave.gwutils import ret_time
from watpy.wave.gwutils import fixed_freq_int_2
from watpy.utils.num import diff1

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

  def retarded_time(self, r, M=1.):
    """
    Calculates the retarded time on a Schwarzschild background in isotropic coordinates.

    Parameters:
    r: Extraction radius (in geometric units).
    M: Total gravitational mass of the system.

    Returns:
    Exchanges time column in waveform_data with retarded time.
    """
    if self.waveform_data is None:
      raise ValueError("Waveform data not loaded. Call load_waveform_data() first.")

    time_exists = any(
      "time" in self.waveform_data[radius]
      for radius in self.waveform_data.keys()
    )

    if not time_exists:
      raise ValueError(f"Time data not found for any extraction radius.")

    if (r == 1.0 or r == -1.0):
      rs = 0.0 # for the case, when r = -1 (extrapolated at infinity)
    else:
      r_areal = r * (1 + M / (2 * r))**2
      rs = r_areal + 2 * M * np.log(r_areal / (2 * M) - 1)

    for radius in self.waveform_data.keys():
      if "time" in self.waveform_data[radius]:
        self.waveform_data[radius]["time"] -= rs

  # Some waveform utility.
  @staticmethod
  def phase(signal):
    """
    Get the phase of the strain of a signal.
    Signal has to be (Re(psi4)+i*Im(psi4))/M,
      where M is the total gravitational mass of the system.

    Parameters:
    signal: (Re(psi4)+i*Im(psi4)) / M.
    """
    return -np.unwrap(np.angle(signal))

  # Taken from watpy wave/wave.py.
  @staticmethod
  def amplitude(signal):
    """
    Get the amplitude of the strain of a signal.
    Signal has to be (Re(psi4)+i*Im(psi4))/M,
      where M is the total gravitational mass of the system.

    Parameters:
    signal: (Re(psi4)+i*Im(psi4)) / M.
    """
    return np.abs(signal)

  @staticmethod
  def phi_dot(time, signal):
    """
    Get the angular frequency of the signal.
    Signal has to be (Re(psi4)+i*Im(psi4))/M,
      where M is the total gravitational mass of the system.

    Parameters:
    time: Retarded time.
    signal: (Re(psi4)+i*Im(psi4)) / M.
    """
    phase_to_diff = Waveform.phase(signal)
    return diff1(time, phase_to_diff, pad=True)

  # Compute the strain and frequency.
  @staticmethod
  def strain(time, Rerpsi4, Imrpsi4, radius, Omega, Mass):
    """
    Compute the strain and the frequency from the real and imaginary
      part of the psi4.

    Parameters:
    time: The simulation time.
    Rerpsi4: Real part of rpsi4.
    Imrpsi4: Imaginary part of rpsi4.
    radius: Extraction radius of the wave.
    Omega: Initial orbital frequency.
    Mass: Total gravitational mass of the system.
    """
    # Cutoff-frequency.
    f0 = Omega / np.pi
    strain_data = {}

    # Compute rpsi4, the retarded time and strain.
    rpsi4 = (Rerpsi4 + 1j * Imrpsi4) / Mass
    u = ret_time(time, radius, Mass)
    h = fixed_freq_int_2(rpsi4, cutoff=f0, dt=time[1]-time[0])

    strain_data["Amplitude"] = Waveform.amplitude(h)
    strain_data["Phase"] = Waveform.phase(h)
    strain_data["Strain"] = h
    strain_data["Momega"] = Waveform.phi_dot(u,h)
    strain_data["Ret. time"] = u
    strain_data["Sim. time"] = time

    return strain_data

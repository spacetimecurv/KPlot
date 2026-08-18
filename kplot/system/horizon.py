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

  def plot_horizon(self, output_path=None, save=False,
                   xmin=None, xmax=None, t_merger=None):
    """
    Simple function that plots irreducible mass, spin parameter, and minimum radius.

    Parameters:
    output_path (str): Path where to store the plot.
    save (bool): Whether to store the plot or not.
    xmin (float): Minimum value on x.
    xmax (float): Maximum value on x.
    t_merger (float): Time of merger.
    """
    if self.horizon_data is None:
      raise ValueError("Horizon data not loaded. Call load_horizon_data() first.")
    
    # Plot style.
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['lines.linestyle'] = 'solid'
    mpl.rcParams['axes.labelsize']  = 16
    mpl.rcParams['text.usetex']     = True
    mpl.rcParams['font.family']     = 'Computer Modern Serif'
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16

    # Load the data.
    hor = self.horizon_data
    t_lim = (xmin, xmax)
    colors = ['#F0A860', '#D55E00', '#009E73']
    labels = [r'$M_{\mathrm{BH,irr}}$', r'$a_{\mathrm{BH}}$', r'$\min{\left(r_{\mathrm{BH}}\right)}$']
    
    # Plot the data.
    fig, ax = plt.subplots(3,1,figsize=(8,10), sharex=True, gridspec_kw=dict(hspace=0.08))

    # Irreducible mass.
    mass = np.sqrt(hor['area'] / (16 * np.pi))
    ax[0].plot(hor['time'], mass, color=colors[0])

    # Spin parameter.
    spin = hor['S'] / hor['mass']**2
    ax[1].plot(hor['time'], spin, color=colors[1])

    # Minimum radius.
    ax[2].plot(hor['time'], hor['minradius'], color=colors[2])

    for i, a in enumerate(ax):
      if t_merger is not None:
        a.axvline(t_merger, color='grey', alpha=0.5)
      a.set_xlim(*t_lim)
      a.set_ylabel(labels[i])

      if i == 2:
        a.set_xlabel(r'$t\ [M_{\odot}]$')

    # Annotate.
    if t_merger is not None:
      ax[0].annotate('merger', xy=(t_merger, np.nanmax(mass)-0.05), xytext=(-4, -10), 
                     textcoords='offset points', ha='right', va='top', fontsize=11, color='grey')

    fig.subplots_adjust(left=0.115, right=0.965, top=0.96, bottom=0.085, hspace=0.06)
    if save:
      plt.savefig(output_path, dpi=150)
    else:
      plt.show()

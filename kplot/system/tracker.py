#########################################################################
# File: tracker.py                                                      #
# Description: Handle AthenaK's tracker files.                          #
#########################################################################

# Import necessary standard libaries.
import re
import os
from collections import defaultdict

# Import necessary third-party libraries.
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the horizon finder class.
class Tracker:
  """
  A class to handle AthenaK's tracker files.
  """

  def __init__(self, tracker_path_0, tracker_path_1):
    """
    Point the class to the tracker files, i.e. <job>.co_0/1.txt.

    Parameters:
    tracker_path_0 (str): The path to the first tracker file.
    tracker_path_1 (str): The path to the second tracker file.
    """
    self.tracker_path_0 = tracker_path_0
    self.tracker_path_1 = tracker_path_1
    self.tracker_data = None

  def load_tracker_data(self):
    """
    Load the tracker data from the specified file path.

    Returns:
    Dictionary holding the tracker data.
    """
    if not os.path.exists(self.tracker_path_0) or not os.path.exists(self.tracker_path_1):
      raise FileNotFoundError(f"Tracker files not found at: {self.tracker_path_0} or {self.tracker_path_1}")

    # Build a dictionary.
    dict_complete = defaultdict(dict)
    paths = [self.tracker_path_0, self.tracker_path_1]
    
    for j, path in enumerate(paths):
      # Find the headers.
      with open(path) as f:
        object = f.readline().lstrip("#").strip()
        header_line = f.readline().strip()

      headers = re.findall(r'\d+:([^\s]+)', header_line)

      # Read the data with numpy.
      data = np.loadtxt(path, comments='#')

      # Create a dictionary.
      data_dict = {name: data[:, i] for i, name in enumerate(headers)}
      dict_complete[j] = data_dict
      dict_complete[j]["object"] = object

    self.tracker_data = dict_complete

  def plot_tracker(self, output_path=None, save=False):
    """
    Simple function to compute the tracker points in time on the xy-plane.

    Parameters:
    output_path (str): Path where to store the plot.
    save (bool): Whether to store the plot or not.
    """
    if self.tracker_data is None:
      raise ValueError("Tracker data not loaded. Call load_tracker_data() first.")
    
    # Plot style.
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['lines.linestyle'] = 'solid'
    mpl.rcParams['axes.labelsize']  = 16
    mpl.rcParams['text.usetex']     = True
    mpl.rcParams['font.family']     = 'Computer Modern Serif'
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16

    # Load the data.
    trac = self.tracker_data

    colors = ['#D55E00', '#009E73']
    labels = [f'Tracker 0 ({trac[0]["object"]})', f'Tracker 1 ({trac[1]["object"]})']

    # Plot the data.
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.plot(trac[0]["x"], trac[0]["y"], color=colors[0], label=labels[0])
    ax.plot(trac[1]["x"], trac[1]["y"], color=colors[1], label=labels[1])
    ax.set_xlabel(r'$x$ [code units]')
    ax.set_ylabel(r'$y$ [code units]')
    ax.legend(frameon=False)
    fig.tight_layout()

    if save:
      plt.savefig(output_path, dpi=150)
    else:
      plt.show()

  def calc_eccentricity(self, t_fit_min, t_fit_max, t_min_plot, t_max_plot,
                        init_guess, bounds, output_path=None, save=False):
    """
    Compute the residual eccentricity from tracker data.

    Parameters:
    t_fit_min (float): Minimum time for the fit.
    t_fit_max (float): Maximum time for the fit.
    t_min_plot (float): Minimum time for the plot.
    t_max_plot (float): Maximum time for the plot.
    init_guess (list): Initial guess for the fit parameters.
    bounds (tuple): Bounds for the fit parameters.
    output_path (str): Path where to store the plot.
    save (bool): Whether to store the plot or not.
    """
    if self.tracker_data is None:
      raise ValueError("Tracker data not loaded. Call load_tracker_data() first.")
      
    # Define the fit formula.
    def tracker_fit(t, S0, A0, A1, B, of, phi):
      """
      Phenomenological fit function for the tracker data from https://arxiv.org/pdf/1507.07100.
      """
      return S0 + A0*t + 1/2*(A1*t**2) - B/of*np.cos(of*t + phi)

    # Prepare the data.
    co0 = self.tracker_data[0]
    co1 = self.tracker_data[1]

    t = co0["time"]
    d = np.sqrt((co0["x"] - co1["x"])**2 + (co0["y"] - co1["y"])**2 + (co0["z"] - co1["z"])**2)

    # Compute the fit.
    mask = (t >= t_fit_min) & (t <= t_fit_max)
    tf = t[mask]
    df = d[mask]

    # Initial guess for the fit parameters.
    p0 = init_guess
    bounds = bounds

    parameters, covariance = curve_fit(
      tracker_fit,
      tf,
      df,
      p0=p0,
      bounds=bounds,
      maxfev=100000
    )

    # Eccentricity.
    B = parameters[3]
    o = parameters[4]
    e = np.abs(B / (d[0]*o))
    print(f'The eccentricity is e = {e}')

    # Plot the data.
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    ax.plot(t, d, color='black', label='Data')
    ax.plot(tf, tracker_fit(tf, *parameters), ls='dashed', color='red', label='Fit')
    ax.text(0.05, 0.05, rf'$e=${e:.4f}', transform=ax.transAxes, fontsize=14,
            verticalalignment='bottom', bbox=props)
    ax.set_xlabel(r'$t\ [M_{\odot}]$')
    ax.set_ylabel(r'$d_{\mathrm{coord}}$')
    ax.set_xlim(t_min_plot, t_max_plot)
    ax.legend(frameon=False)

    fig.tight_layout()
    if save:
      plt.savefig(output_path, dpi=150)
    else:
      plt.show()

    return parameters, covariance, e
    
"""Utility for handling AthenaK apparent horizon output."""

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

  def __init__(self, horizon_path: str | None = None,
               shape_paths: list[str] | None = None) -> None:
    """
    Initialize the HorizonFinder with the path to the horizon file.

    Parameters:
    horizon_path (str): The path to the horizon summary file (None: shape only).
    shape_paths (list[str]): Paths to the <job>.horizon_shape_N.txt files, e.g. of
                             all output-XXXX segments, for the surface reconstruction.
    """
    self.horizon_path = horizon_path
    self.horizon_data = None
    self.shape_paths  = []
    self.shape_times  = None
    self._shape_locs  = None

    if horizon_path is not None:
      self.load_horizon_data()
    if shape_paths:
      self.load_shape_index(shape_paths)

  def load_horizon_data(self) -> None:
    """
    Load the horizon data from the specified file path into self.horizon_data.

    Returns:
    None.
    """
    if not os.path.exists(self.horizon_path):
      raise FileNotFoundError(f"Horizon file not found at: {self.horizon_path}")

    # Find the headers.
    with open(self.horizon_path) as f:
      header_line = f.readline().strip()

    headers = re.findall(r'\d+:([^\s]+)', header_line)

    # Read the data with numpy.
    data = np.loadtxt(self.horizon_path, comments='#', usecols=range(len(headers)))

    # Create a dictionary.
    data_dict = {name: data[:, i] for i, name in enumerate(headers)}
    self.horizon_data = data_dict

  def load_shape_index(self, shape_paths: list[str]) -> None:
    """
    Index the horizon shape files: the time and file position of every entry.

    Parameters:
    shape_paths (list[str]): Paths to the horizon shape files. On equal times
                             the entry of the earlier path is used.

    Returns:
    None.
    """
    times, files, offsets = [], [], []
    for i, path in enumerate(shape_paths):
      if not os.path.exists(path):
        raise FileNotFoundError(f"Horizon shape file not found at: {path}")
      with open(path, 'rb') as f:
        pos, t = 0, None
        for line in f:
          pos += len(line)
          if line.startswith(b'#'):
            t = float(line.split(b'Time =')[1])
          elif t is not None:
            times.append(t)
            files.append(i)
            offsets.append(pos - len(line))
            t = None

    if not times:
      raise ValueError("No horizon shape entries found.")
    self.shape_paths = list(shape_paths)
    self.shape_times = np.array(times)
    self._shape_locs = (np.array(files), np.array(offsets))

  def shape_coeffs(self, time: float) -> np.ndarray:
    """
    Y_lm coefficients of the horizon shape entry closest to time.

    Parameters:
    time (float): Target time [M_sun].

    Returns:
    The coefficients.
    """
    if self.shape_times is None:
      raise ValueError("Horizon shape not loaded. Call load_shape_index() first.")
    i = np.argmin(np.abs(self.shape_times - time))
    with open(self.shape_paths[self._shape_locs[0][i]], 'rb') as f:
      f.seek(self._shape_locs[1][i])
      return np.array(f.readline().split(), dtype=float)

  @staticmethod
  def shape_radius(coeffs: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Apparent-horizon radius r(theta, phi) about the BH puncture (AthenaK fast-flow
    finder), from the real Y_lm expansion of the shape file.

    Parameters:
    coeffs (np.ndarray): Y_lm coefficients from shape_coeffs.
    theta (np.ndarray): Polar angle.
    phi (np.ndarray): Azimuthal angle.

    Returns:
    The horizon radius.
    """
    from scipy.special import sph_harm_y
    a = iter(coeffs)
    lmax = int(round(np.sqrt(coeffs.size))) - 1
    r = np.zeros(np.broadcast(theta, phi).shape)
    for l in range(lmax + 1):
      r += next(a) * sph_harm_y(l, 0, theta, phi).real
      for m in range(1, l + 1):
        Y = np.sqrt(2.0) * sph_harm_y(l, m, theta, phi)
        r += next(a) * Y.real + next(a) * Y.imag
    return r

  def horizon_radius(self, time: float, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Apparent-horizon radius r(theta, phi) at the shape entry closest to time.

    Parameters:
    time (float): Target time [M_sun].
    theta (np.ndarray): Polar angle.
    phi (np.ndarray): Azimuthal angle.

    Returns:
    The horizon radius.
    """
    return self.shape_radius(self.shape_coeffs(time), theta, phi)

  def plot_horizon(self, output_path: str | None = None, save: bool = False,
                   xmin: float | None = None, xmax: float | None = None,
                   t_merger: float | None = None) -> None:
    """
    Simple function that plots irreducible mass, spin parameter, and minimum radius.

    Parameters:
    output_path (str): Path where to store the plot.
    save (bool): Whether to store the plot or not.
    xmin (float): Minimum value on x.
    xmax (float): Maximum value on x.
    t_merger (float): Time of merger.

    Returns:
    None.
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

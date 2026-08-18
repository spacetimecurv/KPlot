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
import matplotlib.pyplot as plt
import matplotlib as mpl
from watpy.wave.gwutils import ret_time
from watpy.wave.gwutils import fixed_freq_int_2
from watpy.wave.gwutils import fixed_freq_int_1
from watpy.utils.num import diff1
from scipy.integrate import cumulative_trapezoid

# Define the waveform reader class.
class Waveform:
  """
  A class to handle waveform data from AthenaK output files.
  """

  def __init__(self, waveform_path, radius_key, Mass, Omega):
    """
    Initialize the Waveform class with the path to the waveform files.

    Parameters:
    waveform_path (str): The path to the waveform files.
    radius_key    (str): Radius key for the psi4 data, e.g. '0300'.
    Mass        (float): Mass of the binary.
    Omega       (float): Orbital angular frequency.
    """
    self.waveform_path = waveform_path

    # Binary-specific parameters.
    self.radius_key    = radius_key
    self.radius        = int(self.radius_key)
    self.Mass          = Mass
    self.Omega         = Omega

    # Waveform-specific data.
    self.psi4_data     = None
    self.strain_data   = None
    self.strain_dot    = None
    self.mom_flux      = None

    # Kick intrinsic parameters. Computed downstream.
    self.vxy           = None
    self.vkick         = None
    self.v0            = None

  def load_psi4_data(self):
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

    self.psi4_data = data

    if self.psi4_data is not None:
      print("Psi4 data successfully loaded.")

  def retarded_time(self):
    """
    Calculates the retarded time on a Schwarzschild background in isotropic coordinates.

    Parameters:
    r: Extraction radius (in geometric units).
    M: Total gravitational mass of the system.

    Returns:
    Exchanges time column in psi4_data with retarded time.
    """
    if self.psi4_data is None:
      raise ValueError("Waveform data not loaded. Call load_psi4_data() first.")

    time_exists = any(
      "time" in self.psi4_data[radius]
      for radius in self.psi4_data.keys()
    )

    if not time_exists:
      raise ValueError(f"Time data not found for any extraction radius.")

    r = self.radius
    M = self.Mass
    if (r == 1.0 or r == -1.0):
      rs = 0.0 # for the case, when r = -1 (extrapolated at infinity)
    else:
      r_areal = r * (1 + M / (2 * r))**2
      rs = r_areal + 2 * M * np.log(r_areal / (2 * M) - 1)

    for radius in self.psi4_data.keys():
      if "time" in self.psi4_data[radius]:
        self.psi4_data[radius]["time"] -= rs

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
  def strain(self, mode_key='22', cutoff=None):
    """
    Compute the strain and the frequency from the real and imaginary
      part of the psi4.

    Parameters:
    mode_key: Which mode, e.g. '22' by default.
    cutoff: FFI cutoff frequency nu_0. Defaults to Omega/pi.
    """
    # Parameters
    Omega  = self.Omega
    Mass   = self.Mass
    radius = self.radius

    # Cutoff-frequency.
    f0 = Omega / np.pi if cutoff is None else cutoff
    strain_data = {}

    # Psi4 data.
    time    = self.psi4_data[self.radius_key]['time']
    Rerpsi4 = self.psi4_data[self.radius_key]['real'][mode_key]
    Imrpsi4 = self.psi4_data[self.radius_key]['imag'][mode_key]

    # Compute rpsi4, the retarded time and strain.
    rpsi4 = (Rerpsi4 + 1j * Imrpsi4) / Mass
    u = ret_time(time, radius, Mass)
    h = fixed_freq_int_2(rpsi4, cutoff=f0, dt=time[1]-time[0])

    strain_data["Amplitude"] = self.amplitude(h)
    strain_data["Phase"]     = self.phase(h)
    strain_data["Strain"]    = h
    strain_data["Momega"]    = self.phi_dot(u,h)
    strain_data["Ret. time"] = u
    strain_data["Sim. time"] = time
    strain_data["Cutoff"]    = f0

    self.strain_data = strain_data

  # Plot the strain.
  def plot_strain(self, tmin=0.0, tmax=None, output_dir=None):
    """
    Plot the strain.

    Parameters:
    tmin (float): Minimum time for the plotting window.
    tmax (float): Maximum time for the plotting window.
    output_dir (str): Output directory to store the plot.
    """
    if self.strain_data is None:
      raise ValueError("Strain data not computed. Call strain() first.")

    # Amplitude and retarded time.
    amp = np.abs(self.strain_data['Strain'])
    u_wave = self.strain_data['Ret. time']

    fig, ax = plt.subplots(1,1,figsize=(8,5))
    ax.plot(u_wave, self.strain_data["Strain"].real, color='purple',
            linewidth=0.5)
    ax.plot(u_wave, amp, color='black', ls='solid', lw=0.5)
    ax.plot(u_wave, -amp, color='black', ls='solid', lw=0.5)
    ax.set_xlim([tmin, tmax])
    ax.set_xlabel(r'$u/M$')
    ax.set_ylabel(r'$\Re h_{22}/M$')

    fig.tight_layout()
    if output_dir is not None:
      plt.savefig(os.path.join(output_dir, 'strain.png'), dpi=150)
    else:
      plt.show()

  # Mode-coupling coefficients (https://arxiv.org/pdf/0912.1285).
  @staticmethod
  def _a(l,m):
    """Equ. (57)."""
    if (abs(m) > l) or (abs(m + 1) > l):
      return 0.0
    else:
      return np.sqrt((l - m) * (l + m + 1)) / (l * (l + 1))

  @staticmethod
  def _b(l,m):
    """Equ. (58)."""
    if (l < 2) or (abs(m) > l):
      return 0.0
    else:
      return 0.5 / l * np.sqrt((l - 2) * (l + 2) * (l + m) * (l + m - 1) \
                        / ((2 * l - 1) * (2 * l + 1)))

  @staticmethod
  def _c(l,m):
    """Equ. (59)."""
    return 2.0 * m / (l * (l + 1))

  @staticmethod
  def _d(l,m):
    """Equ. (60)."""
    if (l < 2) or (abs(m) > l):
      return 0.0
    else:
      return 1.0 / l * np.sqrt((l - 2) * (l + 2) * (l - m) * (l + m) \
                        / ((2 * l - 1) * (2 * l + 1)))

  # Compute dh_lm/du.
  def hdot(self, cutoff, lmax=8):
    """
    dh_lm/du for R*h_lm/M.

    Returns:
    Time array and dictionary holding hdot for each (l,m).
    """
    if self.psi4_data is None:
      raise ValueError("Psi4 data not loaded. Call load_psi4_data() first.")

    time = self.psi4_data[self.radius_key]['time']
    dt   = time[1] - time[0]
    out  = {}
    for name in self.psi4_data[self.radius_key]['real']:
      l, m = int(name[0]), int(name[1:])
      if l > lmax:
          continue
      rpsi4 = (self.psi4_data[self.radius_key]['real'][name] + 1j * self.psi4_data[self.radius_key]['imag'][name]) / self.Mass
      out[(l, m)] = fixed_freq_int_1(rpsi4, cutoff=cutoff, dt=dt)

    self.strain_dot = out

  # Compute the momentum flux.
  def momentum_flux(self, lmax=8):
    """
    Eq. (12) of https://journals.aps.org/prd/pdf/10.1103/n7hg-cc2l written as Equ.(55)-(56) of
    https://arxiv.org/pdf/0912.1285.

    Returns:
    Px + iPy and Pz.
    """
    g = lambda l, m: self.strain_dot.get((l, m), 0.)
    Pxy, Pz = 0., 0.
    for l in range(2, lmax+1):
      for m in range(-l, l+1):
        hlm = g(l, m)
        if np.isscalar(hlm):
            continue
        Pxy = Pxy + hlm * (self._a(l,m)     * np.conj(g(l,m+1))
                         + self._b(l,-m)    * np.conj(g(l-1,m+1))
                         - self._b(l+1,m+1) * np.conj(g(l+1,m+1)))
        Pz  = Pz + hlm * (self._c(l,m)   * np.conj(g(l,m))
                        + self._d(l,m)   * np.conj(g(l-1,m))
                        + self._d(l+1,m) * np.conj(g(l+1,m)))

    self.mom_flux = [Pxy / (8 * np.pi), np.real(Pz) / (16 * np.pi)]

  # Compute the kick.
  def compute_kick(self, t_merge=None, insp_min=300.0, insp_max=-200.0, cutoff=None, hodograph=False,
                   output_dir=None):
    """
    Compute the kick based on Equs.(12)-(13) of https://journals.aps.org/prd/pdf/10.1103/n7hg-cc2l.

    Parameters:
    hodograph (bool): Whether to plot a hodograph or not.
    t_merge  (float): Merger time from psi4 data (use merger_time).
    insp_min (float): Minimum time of the v0 correction window.
    insp_max (float): Maximum time of the v0 correction window w.r.t. to merger time.
    cutoff   (float): Used for the FFT of hdot.
    output_dir (str): Where to store the hodograph.
    """
    CLIGHT_KMS = 299792.458

    # Compute hdot_lm.
    cut = self.Omega / np.pi if cutoff is None else cutoff
    self.hdot(cut)
    u_kick = ret_time(self.psi4_data[self.radius_key]['time'], self.radius, self.Mass)

    # Compute the momentum flux.
    self.momentum_flux()
    Pxy, Pz = self.mom_flux

    # Compute the velocity components.
    v_xy = -self.Mass * cumulative_trapezoid(Pxy, u_kick, initial=0.) # complex: v_x + i v_y
    v_z  = -self.Mass * cumulative_trapezoid(Pz,  u_kick, initial=0.)

    # Shift the merger time to retarded time.
    if t_merge is None:
      raise ValueError("Merger time has to be set.")

    u_merge = ret_time(t_merge, self.radius, self.Mass)

    # Integration constant v_0 (Equ. (24) / App. A.3.2).
    insp = (u_kick > insp_min) & (u_kick < u_merge + insp_max)
    v0   = -np.mean(v_xy[insp])
    v_xy = v_xy + v0

    # Print the results.
    v_kick = np.abs(v_xy[-1]) * CLIGHT_KMS
    print(f'v_kick^GW = {v_kick:.1f} km/s  (v_z = {v_z[-1]*CLIGHT_KMS:.2f} km/s)')

    self.vkick = v_kick
    self.vxy   = v_xy
    self.v0    = v0

    # Plot a hodograph.
    if hodograph:
      # Raw and corrected velocity.
      v_raw = (self.vxy - self.v0) * CLIGHT_KMS
      v_cor = self.vxy * CLIGHT_KMS

      i_merge = np.argmin(np.abs(u_kick - u_merge))

      fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
      for v_h, col, ls, lab in ((v_raw, 'black',  'dashed', r'$v_0=0$'),
                                (v_cor, 'purple', 'solid',  r'$v_0\neq0$')):
        ax1.plot(v_h.real[:i_merge+1], v_h.imag[:i_merge+1],
                 color=col, ls=ls, lw=0.8, label=lab)
        ax1.plot(v_h.real[i_merge:], v_h.imag[i_merge:],
                 color=col, ls=ls, lw=0.8, alpha=0.3)
        ax1.plot(v_h.real[i_merge], v_h.imag[i_merge], marker='o', color=col, ms=3)
        ax1.plot(v_h.real[-1], v_h.imag[-1], marker='*', color=col, ms=10)

      # The centre of the uncorrected spiral, i.e. the offset we are removing.
      ax1.plot(-self.v0.real * CLIGHT_KMS, -self.v0.imag * CLIGHT_KMS,
              marker='x', color='black', ms=7, ls='none', label='Spiral centre')
      ax1.axhline(0., color='grey', lw=0.4)
      ax1.axvline(0., color='grey', lw=0.4)
      ax1.set_aspect('equal')
      ax1.set_xlabel(r'$v_x\;[\mathrm{km\,s^{-1}}]$')
      ax1.set_ylabel(r'$v_y\;[\mathrm{km\,s^{-1}}]$')
      # Legend above the axes, so it cannot overlap the spiral for any run.
      ax1.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0),
                 ncol=3, columnspacing=1.2, handletextpad=0.4)

      # Magnitude against retarded time.
      ax2.plot(u_kick, np.abs(v_raw), color='black',  ls='dashed', lw=0.8, label=r'$v_0=0$')
      ax2.plot(u_kick, np.abs(v_cor), color='purple', ls='solid',  lw=0.8, label=r'$v_0\neq0$')
      ax2.axvline(u_merge, color='grey', lw=0.5, label='Merger')
      ax2.set_xlim([0.,u_merge-insp_max])
      ax2.set_xlabel(r'$u$')
      ax2.set_ylabel(r'$|v|\;[\mathrm{km\,s^{-1}}]$')
      ax2.set_title(rf'$v^{{\rm GW}}_{{\rm kick}}={self.vkick:.1f}\;\mathrm{{km\,s^{{-1}}}}$')
      ax2.legend(frameon=False)

      fig2.tight_layout()
      if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'hodograph.png'), dpi=150)
      else:
        plt.show()

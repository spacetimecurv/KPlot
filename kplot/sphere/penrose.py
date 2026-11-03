"""
Analysis of the generalized Penrose criteria after Lasota et al. 2014:

  https://arxiv.org/html/1310.7499v2

Take 'sph' output close to the horizon and reconstruct the
energy-momentum tensor, then quantify the Lasota criteria
for rotational energy extraction. Note: neglect radiation
contribution to the energy-momentum tensor.

Required output:

  <outputXX>
  file_type      = sph
  variable       = mhd_w_bcc
  dt             = 20.0
  radii          = 5.0, 10.0, 15.0
  ntheta         = 256
  center_tracker = 0

  <outputXX+1>
  file_type      = sph
  variable       = adm
  dt             = 20.0
  radii          = 5.0, 10.0, 15.0
  ntheta         = 256
  center_tracker = 0

  <outputXX+2>
  file_type      = sph
  variable       = z4c_alpha
  dt             = 20.0
  radii          = 5.0, 10.0, 15.0
  ntheta         = 256
  center_tracker = 0

  <outputXX+3-5>
  file_type      = sph
  variable       = z4c_beta<x,y,z>
  dt             = 20.0
  radii          = 5.0, 10.0, 15.0
  ntheta         = 256
  center_tracker = 0
"""

import os

import numpy as np

from ..utils.cli import require_sph, sphere_parser
from ..utils.eos import enthalpy, load_eos
from ..utils.grmhd import fluid, sqrt_det
from ..utils.parallel import default_workers, pool_map
from ..utils.plotting import savefig, time_axis
from ..utils.surface import Surface, run

# Defaults.
DEFAULT_RADII   = (5.0, 10.0)
DEFAULT_JOBNAME = "bns"

# Integrated fluxes (penrose_r{R}.txt) and phi-averaged theta profiles (penrose_profiles_r{R}.npz).
COLUMNS  = ('Mdot', 'Edot_EM', 'Edot_MA', 'Jdot_EM', 'Jdot_MA', 'f_ergo', 'f_neg_e', 'f_P2')
PROFILES = ('edot_EM', 'edot_MA', 'jdot_EM', 'jdot_MA', 'einf', 'gtt', 'P2', 'P2_EM',
            'fP2', 'fneg_e', 'omF_num', 'omF_den')


def _tag(radius: float) -> str:
  """
  Radius as used in the output file names.

  Parameters:
  radius (float): Extraction radius [M_sun].

  Returns:
  The radius formatted with :g.
  """
  return f'{radius:g}'


def _state(eos: dict, dens: np.ndarray, press: np.ndarray, ye: np.ndarray,
           temperature: np.ndarray, vel: list[np.ndarray], bcc: list[np.ndarray], g: tuple,
           alp: np.ndarray, bet: list[np.ndarray], pos: tuple) -> dict:
  r"""
  T^r_t, T^r_\phi, T^t_t, P^2, g_tt, u^r and \Omega_F pieces from the primitives.

  Parameters:
  eos (dict): EOS from load_eos.
  dens, press, ye, temperature (np.ndarray): Fluid primitives.
  vel (list[np.ndarray]): Velocity primitives.
  bcc (list[np.ndarray]): Magnetic field B^i.
  g (tuple): Spatial metric (gxx, gxy, gxz, gyy, gyz, gzz).
  alp (np.ndarray): Lapse.
  bet (list[np.ndarray]): Shift.
  pos (tuple): Cartesian position relative to the BH.

  Returns:
  The state quantities, keyed by name.
  """
  x, y, z = pos
  r = np.sqrt(x**2 + y**2 + z**2)
  varpi2 = x**2 + y**2
  f = fluid(g, alp, bet, vel, bcc)
  u_up, u_d, u_t, u_t_up = f['u_up'], f['u_d'], f['u_t'], f['u_t_up']
  b_up, b_d, b_t, b_t_up = f['b_up'], f['b_d'], f['b_t'], f['b_t_up']
  b2, g_tt = f['b2'], f['g_tt']

  # Radial and azimuthal components, xi = \partial_\phi = (-y, x, 0).
  with np.errstate(invalid='ignore', divide='ignore'):
    u_r_up = (x * u_up[0] + y * u_up[1] + z * u_up[2]) / r
    b_r_up = (x * b_up[0] + y * b_up[1] + z * b_up[2]) / r
    u_phi_up = (x * u_up[1] - y * u_up[0]) / varpi2
    b_phi_up = (x * b_up[1] - y * b_up[0]) / varpi2
  u_phi = x * u_d[1] - y * u_d[0]
  b_phi = x * b_d[1] - y * b_d[0]

  # T^\mu_\nu = (\rhoh + b^2) u^\mu u_\nu + (p + b^2/2) \delta^\mu_\nu - b^\mu b_\nu, split into EM and matter.
  rho_h = dens * enthalpy(eos, dens, ye, temperature)
  w, P = rho_h + b2, press + 0.5 * b2
  return {
    'sqrt_mg':  alp * sqrt_det(g),
    'g_tt':     g_tt,
    'b2':       b2,
    'u_r':      u_r_up,
    'Trt_EM':   b2 * u_r_up * u_t - b_r_up * b_t,
    'Trt_MA':   rho_h * u_r_up * u_t,
    'Trphi_EM': b2 * u_r_up * u_phi - b_r_up * b_phi,
    'Trphi_MA': rho_h * u_r_up * u_phi,
    'Ttt':      w * u_t_up * u_t + P - b_t_up * b_t,
    # Norm of the Noether current P^\mu = -T^\mu_t (> 0: spacelike).
    'P2':       (-w**2 * u_t**2 + 2 * w * P * u_t**2 + P**2 * g_tt
                 + b2 * b_t**2 - 2 * P * b_t**2),
    # \Omega_F = (u^\phi b^r - u^r b^\phi) / (u^t b^r - u^r b^t).
    'omF_num':  u_phi_up * b_r_up - u_r_up * b_phi_up,
    'omF_den':  u_t_up * b_r_up - u_r_up * b_t_up,
  }


def _compute(s: Surface, eos: dict) -> dict:
  """
  Lasota fluxes and profiles on one surface.

  Parameters:
  s (Surface): Extraction surface of one snapshot.
  eos (dict): EOS from load_eos.

  Returns:
  The surface-integrated fluxes and the phi-averaged profiles.
  """
  dens = np.maximum(s['dens'], 0.0)
  st = _state(eos, dens, np.maximum(s['press'], 0.0), s['s_00'], s['temperature'],
              s.vel, s.bcc, s.metric, s.alpha, s.beta, s.pos)
  sqrt_mg, g_tt, P2, Ttt = st['sqrt_mg'], st['g_tt'], st['P2'], st['Ttt']
  dA = s.dA
  area = dA.sum()

  def flux(f: np.ndarray) -> float:
    """
    Integral of sqrt(-g) f over the surface.

    Parameters:
    f (np.ndarray): Flux density.

    Returns:
    The integrated flux.
    """
    return (sqrt_mg * f * dA).sum()

  def phi_avg(f: np.ndarray) -> np.ndarray:
    """
    Average over phi.

    Parameters:
    f (np.ndarray): Field on the surface.

    Returns:
    The theta profile.
    """
    return f.mean(axis=0)

  return {
    'theta':    s.theta_1d,
    'Mdot':     -flux(dens * st['u_r']),
    'Edot_EM':  flux(st['Trt_EM']),
    'Edot_MA':  flux(st['Trt_MA']),
    'Jdot_EM':  -flux(st['Trphi_EM']),
    'Jdot_MA':  -flux(st['Trphi_MA']),
    'f_ergo':   dA[g_tt > 0].sum() / area,
    'f_neg_e':  dA[Ttt > 0].sum() / area,
    'f_P2':     dA[P2 > 0].sum() / area,
    # phi-averaged theta profiles, densitized by sqrt(-g) as the paper's dot(e)_H and dot(j)_H.
    'edot_EM':  phi_avg(sqrt_mg * st['Trt_EM']),
    'edot_MA':  phi_avg(sqrt_mg * st['Trt_MA']),
    'jdot_EM':  -phi_avg(sqrt_mg * st['Trphi_EM']),
    'jdot_MA':  -phi_avg(sqrt_mg * st['Trphi_MA']),
    'einf':     -phi_avg(sqrt_mg * Ttt),
    'gtt':      phi_avg(g_tt),
    'P2':       phi_avg(P2),
    'P2_EM':    phi_avg(0.25 * st['b2']**2 * g_tt),
    # Fraction of each theta ring where P is spacelike, and where its energy density is negative.
    'fP2':      phi_avg((P2 > 0).astype(float)),
    'fneg_e':   phi_avg((Ttt > 0).astype(float)),
    # The \Omega_F ratio is taken after averaging.
    'omF_num':  phi_avg(st['omF_num']),
    'omF_den':  phi_avg(st['omF_den']),
  }


def analyze(sph_dirs: list[str], eos_table: str, output_dir: str,
            radii: list[float] = DEFAULT_RADII, jobname: str = DEFAULT_JOBNAME,
            n_workers: int | None = None) -> None:
  """
  Run the Lasota generalized Penrose analysis over the snapshots in sph_dirs.

  Parameters:
  sph_dirs (list[str]): AthenaK output-XXXX/sph directories (all treated as one run).
  eos_table (str): PyCompOSE HDF5 EOS table matching the simulation EOS.
  output_dir (str): Directory the outputs are written to.
  radii (list[float]): SPH extraction radii [M_sun]; each must have mhd_w_bcc/adm/z4c
                       output. A snapshot is used at every radius it holds, so the
                       radii may cover different time spans.
  jobname (str): AthenaK job name prefixing the VTK files.
  n_workers (int): Worker processes for the snapshot loop (default: min(8, cpu_count())).

  Returns:
  None.
  """
  results = run(sph_dirs, jobname, radii, _compute, ctx=load_eos(eos_table),
                n_workers=n_workers)
  os.makedirs(output_dir, exist_ok=True)

  for radius, res in results.items():
    if not res:
      continue
    time = np.array([d['time'] for d in res])

    path = os.path.join(output_dir, f'penrose_r{_tag(radius)}.txt')
    np.savetxt(path, np.column_stack([time] + [[d[k] for d in res] for k in COLUMNS]),
               fmt='%.8e',
               header=f'r = {radius:g} M_sun, code units; Edot = int sqrt(-g) T^r_t dA '
                      '(> 0: absorbed), Jdot = -int sqrt(-g) T^r_phi dA, '
                      'f_* = area fractions\n'
                      'time  ' + '  '.join(COLUMNS))

    np.savez(os.path.join(output_dir, f'penrose_profiles_r{_tag(radius)}.npz'),
             time=time, theta=res[0]['theta'],
             center=np.array([d['center'] for d in res]),
             **{k: np.array([d[k] for d in res]) for k in PROFILES})
    print(f'r = {radius:g}: {len(res)} snapshots -> {path}')

  print(f'All outputs saved to {output_dir}')


def load_horizon(horizon_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Black hole mass and horizon angular velocity from an AthenaK horizon summary.

  Parameters:
  horizon_file (str): Path to the horizon summary file.

  Returns:
  (time, M, omega_H).
  """
  data = np.loadtxt(horizon_file, comments='#')
  t, first = np.unique(data[:, 1], return_index=True)
  M, S = data[first, 2], data[first, 6]
  a = S / M
  r_plus = M + np.sqrt(np.maximum(M**2 - a**2, 0.0))
  return t, M, a / (2.0 * M * r_plus)


def _symlog_linthresh(vals: np.ndarray, pct: float = 5.0) -> float:
  """
  Decade just below the pct-th percentile of |vals|.

  Parameters:
  vals (np.ndarray): Absolute values.
  pct (float): Percentile.

  Returns:
  The linear threshold.
  """
  vals = np.abs(np.asarray(vals, dtype=float)).ravel()
  vals = vals[np.isfinite(vals) & (vals > 0)]
  return 10.0**np.floor(np.log10(np.percentile(vals, pct))) if vals.size else 1e-12


def _auto_symlog(ax: "plt.Axes", pct: float = 5.0) -> None:
  """
  Automatic symlog y-axis.

  Parameters:
  ax (plt.Axes): Axes with the plotted lines.
  pct (float): Percentile for the linear threshold.

  Returns:
  None.
  """
  from matplotlib.ticker import FixedLocator, SymmetricalLogLocator
  lin = _symlog_linthresh(np.concatenate([np.asarray(l.get_ydata(), dtype=float)
                                          for l in ax.get_lines()] or [np.empty(0)]), pct)
  ax.set_yscale('symlog', linthresh=lin, linscale=0.5)
  ax.relim()
  ax.autoscale_view()

  ticks = SymmetricalLogLocator(base=100.0, linthresh=lin).tick_values(*ax.get_ylim())
  ax.yaxis.set_major_locator(FixedLocator([v for v in ticks if v == 0 or abs(v) > 10 * lin]))


# Plotter function.
def plot_penrose(output_dir: str, radii: list[float] = DEFAULT_RADII,
                 horizon_file: str | None = None, t_merger: float | None = None) -> None:
  """
  Fluxes vs time (fig_penrose_fluxes.png) and time-averaged theta profiles after
  t_merger (fig_penrose_profiles_r{R}.png).

  Parameters:
  output_dir (str): Analysis directory; the figures go there too.
  radii (list[float]): Extraction radii [M_sun].
  horizon_file (str): AthenaK horizon summary for M and omega_H.
  t_merger (float): Merger time [M_sun]; the time axis is then t - t_merger [ms].

  Returns:
  None.
  """
  from .plots import plt
  from matplotlib.lines import Line2D

  t_bh, _, om_bh = load_horizon(horizon_file)
  t_ms, xlabel = time_axis(t_merger)

  # Load the data.
  data = {}
  for radius in sorted(set(float(r) for r in radii)):
    path = os.path.join(output_dir, f'penrose_r{_tag(radius)}.txt')
    if not os.path.exists(path):
      print(f'  No {path}; skipping r = {radius:g}.')
      continue
    d = np.atleast_2d(np.loadtxt(path))
    ts = {'time': d[:, 0], **{k: d[:, i + 1] for i, k in enumerate(COLUMNS)}}
    prof = dict(np.load(os.path.join(output_dir, f'penrose_profiles_r{_tag(radius)}.npz')))
    data[radius] = (ts, prof)
  if not data:
    print('  Nothing to plot.')
    return

  # ------- FLUXES VS TIME -------
  fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.0), sharex=True)
  for radius, (ts, _) in data.items():
    t = t_ms(ts['time'])
    tmin, tmax = np.min(t), np.max(t)
    om_H = np.interp(ts['time'], t_bh, om_bh)
    Edot = ts['Edot_EM'] + ts['Edot_MA']
    Jdot = ts['Jdot_EM'] + ts['Jdot_MA']
    line, = axes[0].plot(t, Edot, lw=1.0, label=f'$r = {radius:g}\\,M_\\odot$')
    axes[0].plot(t, ts['Edot_EM'], color=line.get_color(), lw=0.8, ls=':')
    axes[1].plot(t, Jdot, color=line.get_color(), lw=1.0)
    axes[1].plot(t, ts['Jdot_EM'], color=line.get_color(), lw=0.8, ls=':')
    with np.errstate(invalid='ignore', divide='ignore'):
      axes[2].plot(t, (ts['Mdot'] - Edot) / ts['Mdot'], color=line.get_color(), lw=1.0)
    # NEC bound omega_H dot(J)_H <= dot(E)_H holds on the horizon.
    axes[0].plot(t, om_H * Jdot, color=line.get_color(), lw=0.6, ls='--')
  for ax in axes:
    ax.axhline(0.0, color='black', lw=0.6)
  axes[2].axhline(1.0, color='gray', lw=0.8, ls='--')

  # Labels.
  axes[-1].set_xlabel(xlabel)
  axes[0].set_ylabel(r'$\dot E_H$')
  axes[1].set_ylabel(r'$\dot J_H$')
  axes[2].set_ylabel(r'$\eta = (\dot M - \dot E_H)/\dot M$')

  # Legends.
  radius_handles, _ = axes[0].get_legend_handles_labels()
  style_handles = [
      Line2D([], [], color="black", linestyle="dotted", label="EM part"),
      Line2D([], [], color="black", linestyle="dashed", label=r"$\omega_H\dot{J}_H(\leq \dot{E}_H)$")
  ]
  axes[0].legend(handles=radius_handles + style_handles, ncol=2, fontsize=9,
                 frameon=False, loc='upper right')

  # Axes ticks.
  for ax in axes[:2]:
    _auto_symlog(ax)
  for a in axes:
    a.set_xlim(tmin, tmax)

  axes[0].set_title(r'Penrose process: $\dot E_H < 0$, $\dot J_H < 0$, $\eta > 1$',
                    fontsize=10)

  savefig(fig, os.path.join(output_dir, 'fig_penrose_fluxes.png'))

  # ------- TIME-AVERAGED theta PROFILES -------
  for radius, (_, prof) in data.items():
    keep = prof['time'] >= (t_merger if t_merger is not None else -np.inf)
    if not keep.any():
      continue
    avg = {k: prof[k][keep].mean(axis=0) for k in PROFILES if k in prof}
    om_H = np.interp(prof['time'][keep], t_bh, om_bh).mean()
    th = np.degrees(prof['theta'])
    thmin, thmax = np.min(th), np.max(th)
    ergo = avg['gtt'] > 0

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.0), sharex=True)
    ax = axes[0]
    ax.plot(th, avg['edot_EM'] + avg['edot_MA'], color='black', lw=1.2, label='total')
    ax.plot(th, avg['edot_EM'], color='tab:blue', lw=1.0, label='EM')
    ax.plot(th, avg['edot_MA'], color='tab:red', lw=1.0, label='matter')
    ax.set_ylabel(r'$\dot e = \sqrt{-g}\,T^r_{\ t}$')

    ax = axes[1]
    ax.plot(th, avg['jdot_EM'] + avg['jdot_MA'], color='black', lw=1.2)
    ax.plot(th, avg['jdot_EM'], color='tab:blue', lw=1.0)
    ax.plot(th, avg['jdot_MA'], color='tab:red', lw=1.0)
    ax.set_ylabel(r'$\dot\jmath = -\sqrt{-g}\,T^r_{\ \phi}$')

    ax = axes[2]
    with np.errstate(invalid='ignore', divide='ignore'):
      ax.plot(th, avg['omF_num'] / avg['omF_den'] / om_H, color='black', lw=1.2)
    ax.axhspan(0.0, 1.0, color='tab:green', alpha=0.12, lw=0, label=r"$0<\Omega_F<\omega_H$")
    ax.set_ylim(-0.5, 1.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_ylabel(r'$\Omega_F / \omega_H$')
    ax.set_xlabel(r'$\theta$ [deg]')

    for ax in axes[:2]:
      _auto_symlog(ax)

    for ax in axes:
      ax.axhline(0.0, color='black', lw=0.6)
      ax.fill_between(th, 0, 1, where=ergo, color='gray', alpha=0.12, lw=0,
                      transform=ax.get_xaxis_transform(), label=r"Ergoregion ($g_{tt}>0$)")
      ax.set_xlim(thmin, thmax)

    axes[0].legend(fontsize=9, frameon=False)
    axes[0].set_title(f'$r = {radius:g}\\,M_\\odot$, time-averaged, '
                      f'$\\omega_H = {om_H:.4f}$', fontsize=9)

    savefig(fig, os.path.join(output_dir, f'fig_penrose_profiles_r{_tag(radius)}.png'))

  _plot_noether(output_dir, data, t_merger)


def _plot_noether(output_dir: str, data: dict, t_merger: float | None) -> None:
  """
  Noether current P^mu = -T^mu_t (fig_penrose_noether_r{R}.png): its norm P^2 and
  where it is spacelike (P^2 > 0) or has negative energy (e_inf < 0).

  Parameters:
  output_dir (str): Output directory.
  data (dict): {radius: (time series, profiles)} as loaded in plot_penrose.
  t_merger (float): Merger time [M_sun]; profiles average t >= t_merger.

  Returns:
  None.
  """
  from .plots import plt

  t_ms, xlabel = time_axis(t_merger)
  for radius, (ts, prof) in data.items():
    if 'fP2' not in prof:
      print(f'  r = {radius:g}: no Noether profiles in the .npz; re-run the analysis.')
      continue
    keep = prof['time'] >= (t_merger if t_merger is not None else -np.inf)
    if not keep.any():
      continue
    avg = {k: prof[k][keep].mean(axis=0) for k in PROFILES}
    th = np.degrees(prof['theta'])
    ergo = avg['gtt'] > 0

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.5))
    ax = axes[0]
    ax.plot(th, avg['P2'], color='black', lw=1.2, label='total')
    ax.plot(th, avg['P2_EM'], color='tab:blue', lw=1.0, label=r'EM: $\frac{1}{4} b^4 g_{tt}$')
    ax.set_ylabel(r'$\langle P_\mu P^\mu \rangle_\phi$')
    _auto_symlog(ax)

    ax = axes[1]
    ax.plot(th, avg['fP2'], color='black', lw=1.2, label=r'spacelike, $P^2 > 0$')
    ax.plot(th, avg['fneg_e'], color='tab:red', lw=1.0, label=r'$e_\infty < 0$')
    ax.set_ylabel(r'fraction of $\phi$')
    ax.set_xlabel(r'$\theta$ [deg]')

    for ax in axes[:2]:
      ax.axhline(0.0, color='black', lw=0.6)
      ax.fill_between(th, 0, 1, where=ergo, color='gray', alpha=0.12, lw=0,
                      transform=ax.get_xaxis_transform(), label=r'ergoregion')
      ax.set_xlim(th.min(), th.max())
      ax.legend(fontsize=9, frameon=False)

    ax = axes[2]
    t = t_ms(ts['time'])
    ax.semilogy(t, ts['f_P2'], color='black', lw=1.0, label=r'$P^2 > 0$')
    ax.semilogy(t, ts['f_neg_e'], color='tab:red', lw=1.0, label=r'$e_\infty < 0$')
    ax.semilogy(t, ts['f_ergo'], color='gray', lw=1.0, ls='--', label='ergoregion')
    ax.set_ylabel('area fraction of sphere')
    ax.set_xlabel(xlabel)
    ax.set_xlim(t.min(), t.max())
    ax.legend(fontsize=9, frameon=False, ncol=3)

    axes[0].set_title(f'Noether current $P^\\mu = -T^\\mu_{{\\ t}}$, $r = {radius:g}\\,M_\\odot$', fontsize=9)
    savefig(fig, os.path.join(output_dir, f'fig_penrose_noether_r{_tag(radius)}.png'))


# Per-worker state of plot_noether_slice.
_W_EOS   = None
_W_FILES = None
_W_SLICE = None


def _init_slice_worker(eos: dict, files: dict, settings: tuple) -> None:
  """
  Pool initializer: store the EOS, the 3D dump paths and the slice settings.

  Parameters:
  eos (dict): EOS from load_eos.
  files (dict): {tag: {snapshot: path}} from find_snapshots.
  settings (tuple): (output_dir, half_width, npts).

  Returns:
  None.
  """
  global _W_EOS, _W_FILES, _W_SLICE
  _W_EOS, _W_FILES, _W_SLICE = eos, files, settings


def _slice_worker(task: tuple) -> None:
  """
  Plot P^2 on the x-z plane through the black hole for one 3D snapshot.

  Parameters:
  task (tuple): (snapshot, time, BH center, horizon Y_lm coefficients).

  Returns:
  None.
  """
  from matplotlib.colors import SymLogNorm
  from kplot.system.bin_convert import read_binary
  from kplot.system.horizon import HorizonFinder
  from ..utils.bin3d import slice_cells
  from ..utils.constants import MSUN_TO_MS
  from .plots import plt

  snap, time, c, coeffs = task
  output_dir, half_width, npts = _W_SLICE
  u = np.linspace(-half_width, half_width, npts)
  X, Z = np.meshgrid(u, u)

  def keep(g: tuple) -> bool:
    """
    Whether a meshblock intersects the slice window.

    Parameters:
    g (tuple): Meshblock geometry (x1min, x1max, x2min, x2max, x3min, x3max).

    Returns:
    True if the block is read.
    """
    return (g[2] <= c[1] < g[3] and g[0] < c[0] + half_width and g[1] > c[0] - half_width
            and g[4] < c[2] + half_width and g[5] > c[2] - half_width)

  d = {tag: read_binary(paths[snap], keep) for tag, paths in _W_FILES.items()}
  fd = d['mhd_w_bcc']
  nx = (fd['nx1_out_mb'], fd['nx2_out_mb'], fd['nx3_out_mb'])
  pos = slice_cells(fd['mb_geometry'], nx, np.arange(fd['n_mbs'] * np.prod(nx)),
                    (0, 2, 1), c[1], c[0] + u, c[2] + u)

  def get(tag: str, var: str) -> np.ndarray:
    """
    Sample a variable of one dump on the slice grid.

    Parameters:
    tag (str): Dump, e.g. 'adm'.
    var (str): Variable name.

    Returns:
    The values, NaN outside the read blocks.
    """
    v = np.asarray(d[tag]['mb_data'][var]).reshape(-1)[np.maximum(pos, 0)]
    return np.where(pos >= 0, v, np.nan)

  def mhd(var: str) -> np.ndarray:
    """
    Sample a variable of the mhd_w_bcc dump on the slice grid.

    Parameters:
    var (str): Variable name.

    Returns:
    The values, NaN outside the read blocks.
    """
    return get('mhd_w_bcc', var)

  s = _state(_W_EOS, np.maximum(mhd('dens'), 0.0), np.maximum(mhd('press'), 0.0), mhd('s_00'),
             mhd('temperature'), [mhd(f'vel{k}') for k in 'xyz'],
             [mhd(f'bcc{k}') for k in '123'],
             [get('adm', f'adm_g{k}') for k in ('xx', 'xy', 'xz', 'yy', 'yz', 'zz')],
             get('alpha', 'z4c_alpha'), [get(f'beta{k}', f'z4c_beta{k}') for k in 'xyz'],
             (X, np.zeros_like(X), Z))
  # The x-z plane is phi = 0 (x > 0) and phi = pi (x < 0) about the BH.
  theta, phi = np.arctan2(np.abs(X), Z), np.where(X >= 0, 0.0, np.pi)
  inside = np.hypot(X, Z) < HorizonFinder.shape_radius(coeffs, theta, phi)
  th = np.linspace(0.0, np.pi, 181)
  r_n, r_s = (HorizonFinder.shape_radius(coeffs, th, p) for p in (0.0, np.pi))
  ah_x = np.concatenate([r_n * np.sin(th), -r_s[::-1] * np.sin(th[::-1])])
  ah_z = np.concatenate([r_n * np.cos(th), r_s[::-1] * np.cos(th[::-1])])
  P2 = np.where(inside, np.nan, s['P2'])

  vmax = np.nanmax(np.abs(P2))
  fig, ax = plt.subplots(figsize=(6.5, 5.5))
  pc = ax.pcolormesh(X, Z, P2, cmap='PuOr_r', shading='auto',
                     norm=SymLogNorm(linthresh=_symlog_linthresh(P2), vmin=-vmax, vmax=vmax))
  ax.streamplot(u, u, mhd('bcc1'), mhd('bcc3'), color='k', linewidth=0.4, density=1.2,
                arrowsize=0.5)
  ax.contour(X, Z, np.where(inside, np.nan, s['g_tt']), [0.0], colors='cyan')
  ax.contour(X, Z, np.where(inside, np.nan, s['u_r']), [0.0], colors='orange')
  ax.fill(ah_x, ah_z, color='black')
  fig.colorbar(pc, ax=ax, label=r'$P_\mu P^\mu$  (> 0: spacelike)')
  # Streamlines stepping just past the grid would otherwise add an autoscale margin.
  ax.set_xlim(u[0], u[-1])
  ax.set_ylim(u[0], u[-1])
  ax.set_aspect('equal')
  ax.set_xlabel(r'$x - x_\mathrm{BH}$ [$M_\odot$]')
  ax.set_ylabel(r'$z - z_\mathrm{BH}$ [$M_\odot$]')
  ax.set_title(f'$t = {time * MSUN_TO_MS:.2f}$ ms; cyan: ergosphere, orange: $u^r = 0$',
               fontsize=9)
  savefig(fig, os.path.join(output_dir, f'fig_penrose_noether_slice_{snap:05d}.png'),
          tight=False)


def plot_noether_slice(simpath: str, eos_table: str, output_dir: str,
                       jobname: str = DEFAULT_JOBNAME, snapshots: list[int] | None = None,
                       half_width: float = 12.0, npts: int = 400,
                       n_workers: int | None = None) -> None:
  """
  P^2 on the x-z plane through the black hole from the 3D dumps, with the ergosphere
  (g_tt = 0), the stagnation surface (u^r = 0) and field lines.

  Parameters:
  simpath (str): Simulation directory with the output-XXXX segments.
  eos_table (str): PyCompOSE HDF5 EOS table matching the simulation EOS.
  output_dir (str): Output directory.
  jobname (str): AthenaK job name.
  snapshots (list[int]): 3D snapshot numbers (default: all with every dump).
  half_width (float): Half-width of the slice [M_sun].
  npts (int): Grid points per axis.
  n_workers (int): Worker processes for the snapshot loop (default: min(8, cpu_count())).

  Returns:
  None.
  """
  from glob import glob
  from kplot.system.horizon import HorizonFinder
  from ..utils.bin3d import find_snapshots

  tags = ('mhd_w_bcc', 'adm', 'alpha', 'betax', 'betay', 'betaz')
  files = {tag: find_snapshots(tag, simpath) for tag in tags}
  tracker = np.concatenate([np.loadtxt(f, comments='#', ndmin=2) for f in
                            glob(os.path.join(simpath, 'output-*', f'{jobname}.co_0.txt'))])
  horizon = HorizonFinder(shape_paths=sorted(glob(os.path.join(
      simpath, 'output-*', f'{jobname}.horizon_shape_0.txt'))))

  # Time, BH center and horizon shape of each snapshot.
  tasks = []
  if snapshots is None:
    snapshots = sorted(set.intersection(*(set(f) for f in files.values())))
  for snap in snapshots:
    with open(files['alpha'][snap], 'rb') as f:
      time = float([f.readline() for _ in range(3)][2].split(b'=')[1])
    i = np.argmin(np.abs(tracker[:, 1] - time))
    if abs(tracker[i, 1] - time) > 1.0:
      raise RuntimeError(f'No BH tracker entry near t = {time:g} in {simpath}/output-*.')
    tasks.append((snap, time, tracker[i, 2:5], horizon.shape_coeffs(time)))

  os.makedirs(output_dir, exist_ok=True)
  pool_map(_slice_worker, tasks, min(n_workers or default_workers(), len(tasks)),
           initializer=_init_slice_worker,
           initargs=(load_eos(eos_table), files, (output_dir, half_width, npts)))


def main(argv: list[str] | None = None) -> None:
  """
  Command-line entry point of the Penrose analysis.

  Parameters:
  argv (list[str]): Command-line arguments (default: sys.argv).

  Returns:
  None.
  """
  p = sphere_parser(__doc__, radii=DEFAULT_RADII, plot=True)
  p.add_argument("--eos-table", default=None,
                 help="PyCompOSE HDF5 EOS table matching the simulation EOS; "
                      "required unless --plot-only.")
  p.add_argument("--horizon-file", default=None,
                 help="AthenaK horizon summary (<job>.horizon_summary_0.txt).")
  p.add_argument("--slices", default=None, metavar="SIMPATH",
                 help="Also plot P^2 on the x-z plane through the BH from "
                      "the 3D dumps under SIMPATH; needs --eos-table.")
  p.add_argument("--slice-snapshot", type=int, action="append", default=None,
                 help="3D snapshot number for --slices (repeat for several). Default: all.")
  args = p.parse_args(argv)
  require_sph(p, args)
  radii = args.radii or list(DEFAULT_RADII)

  if not args.plot_only:
    if args.eos_table is None:
      p.error("--eos-table is required unless --plot-only is given")
    analyze(args.sph_dirs, args.eos_table, args.output_dir, radii=radii,
            jobname=args.jobname, n_workers=args.n_workers)
  if args.plot or args.plot_only:
    if args.horizon_file is None:
      p.error("plotting needs --horizon-file")
    plot_penrose(args.output_dir, radii=radii, horizon_file=args.horizon_file,
                 t_merger=args.t_merger)
  if args.slices:
    if args.eos_table is None:
      p.error("--slices needs --eos-table")
    plot_noether_slice(args.slices, args.eos_table, args.output_dir,
                       jobname=args.jobname, snapshots=args.slice_snapshot,
                       n_workers=args.n_workers)


if __name__ == '__main__':
  main()

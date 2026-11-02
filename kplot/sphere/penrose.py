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

# Built-in libraries.
import argparse
import os
from multiprocessing import Pool, cpu_count

# Third-party libraries.
import numpy as np

# KPlot utility.
from athplot.utils.units import cgs
from ._files import locate
from .ejecta import load_eos

# Defaults.
DEFAULT_RADII   = (5.0, 10.0)
DEFAULT_JOBNAME = "bns"

# The sph output variables a surface needs, in the order the worker unpacks them.
VARIABLES = ('mhd_w_bcc', 'adm', 'z4c_alpha', 'z4c_betax', 'z4c_betay', 'z4c_betaz')

# Integrated fluxes (penrose_r{R}.txt) and φ-averaged θ profiles (penrose_profiles_r{R}.npz).
COLUMNS  = ('Mdot', 'Edot_EM', 'Edot_MA', 'Jdot_EM', 'Jdot_MA', 'f_ergo', 'f_neg_e', 'f_P2')
PROFILES = ('edot_EM', 'edot_MA', 'jdot_EM', 'jdot_MA', 'einf', 'gtt', 'P2', 'P2_EM',
            'fP2', 'fneg_e', 'omF_num', 'omF_den')


def _tag(radius):
  return f'{radius:g}'

# EOS table and neutron mass, set once per worker by the Pool initializer.
_W_EOS     = None
_W_MN_GRAM = None


def _init_penrose_worker(eos, mn_gram):
  global _W_EOS, _W_MN_GRAM
  _W_EOS     = eos
  _W_MN_GRAM = mn_gram


# Per-snapshot worker function.
def _process_snapshot_penrose(args):
  """
  Per-snapshot worker: compute the Lasota metric on each surface.

  `args` is a list of ``(radius, files)``, `files` being the (path, surface
  index) of every entry of VARIABLES.  Each dump is parsed once, however many of
  the surfaces it holds.
  """
  # Runtime libraries.
  import numpy as _np
  import os as _os
  from athplot.load_sph_vtk import SphericalData as _SphericalData
  from ._files import Shell as _Shell, CENTER_TOL as _CENTER_TOL
  from athplot.utils.units import conv_dens as _conv_dens, cactus as _cactus, cgs as _cgs
  from ._integrate import (calc_ur as _calc_ur,
                            sqrt_det_metric_adm as _sqrt_det,
                            get_riemann_weights as _get_riemann_weights)

  # Specific enthalpy h = 1 + ε + p/ρ from the EOS table.
  def _enthalpy(dens, ye, temperature):
    eos = _W_EOS
    nb = _conv_dens(_cactus, _cgs, dens) * 1e-39 / _W_MN_GRAM
    pts = _np.column_stack((
      _np.log(_np.clip(nb.ravel(), eos['nb_min'], eos['nb_max'])),
      _np.clip(ye.ravel(), eos['ye_min'], eos['ye_max']),
      _np.log(_np.clip(temperature.ravel(), eos['t_min'], eos['t_max']))))
    h = 1.0 + eos['interp_e'](pts) + eos['interp_p_rho'](pts)
    return h.reshape(dens.shape)

  # Parse the file chunks.
  parsed = {}

  def shell(entry):
    path, iradius = entry
    if path not in parsed:
      parsed[path] = _SphericalData(path)
    return _Shell(path, iradius, data=parsed[path])

  # Iterate over the files.
  out = {}
  for radius, files in args:
    # Unpack the files.
    mhd, adm, alpha, betax, betay, betaz = shells = [shell(f) for f in files]

    time  = mhd.time
    r     = mhd.radius
    theta = mhd.theta
    phi   = mhd.phi

    centers = _np.array([sh.center for sh in shells])
    if not _np.allclose(centers, centers[0], rtol=0.0, atol=_CENTER_TOL):
      detail = '\n'.join(f'  {_os.path.basename(sh.path)}: '
                         f'({c[0]:.6g}, {c[1]:.6g}, {c[2]:.6g})'
                         for sh, c in zip(shells, centers))
      raise RuntimeError(
        f'Sphere centers disagree at t = {time:g} M_sun:\n{detail}\n'
        'Every sph output block feeding the MAD analysis must use the '
        'same center_tracker.')

    # Duplication correction.
    z4c_alpha_raw = alpha.shell('z4c_alpha')
    phi_zero      = (phi == phi.min())
    theta_north = (theta == theta.min())
    dup_corr = _np.ones(z4c_alpha_raw.shape)
    if z4c_alpha_raw[phi_zero & ~theta_north].max() > 1.0:
      dup_corr[phi_zero] = 0.5
    if z4c_alpha_raw[theta_north].max() > 1.0:
      dup_corr[theta_north] = 0.25

    dens  = _np.maximum(mhd.shell('dens') * dup_corr, 0.0)
    press = _np.maximum(mhd.shell('press') * dup_corr, 0.0)
    ye    = mhd.shell('s_00') * dup_corr
    temperature = mhd.shell('temperature') * dup_corr
    velx = mhd.shell('velx') * dup_corr
    vely = mhd.shell('vely') * dup_corr
    velz = mhd.shell('velz') * dup_corr
    bcc1 = mhd.shell('bcc1') * dup_corr
    bcc2 = mhd.shell('bcc2') * dup_corr
    bcc3 = mhd.shell('bcc3') * dup_corr

    gxx = adm.shell('adm_gxx') * dup_corr
    gyy = adm.shell('adm_gyy') * dup_corr
    gzz = adm.shell('adm_gzz') * dup_corr
    gxy = adm.shell('adm_gxy') * dup_corr
    gxz = adm.shell('adm_gxz') * dup_corr
    gyz = adm.shell('adm_gyz') * dup_corr
    z4c_alpha = z4c_alpha_raw * dup_corr
    z4c_betax = betax.shell('z4c_betax') * dup_corr
    z4c_betay = betay.shell('z4c_betay') * dup_corr
    z4c_betaz = betaz.shell('z4c_betaz') * dup_corr

    # Cartesian position relative to the sphere center and ξ = ∂_φ.
    x = r * _np.sin(theta) * _np.cos(phi)
    y = r * _np.sin(theta) * _np.sin(phi)
    xi_x, xi_y = -y, x
    varpi2 = x**2 + y**2

    alp = z4c_alpha
    bet = (z4c_betax, z4c_betay, z4c_betaz)
    g3 = ((gxx, gxy, gxz), (gxy, gyy, gyz), (gxz, gyz, gzz))
    sqrt_g = _sqrt_det(gxx, gyy, gzz, gxy, gxz, gyz, 1.0)

    def lower(v):
      return [sum(g3[i][j] * v[j] for j in range(3)) for i in range(3)]

    def dot(a, b):
      return sum(ai * bi for ai, bi in zip(a, b))

    # Fluid four-velocity from the Eulerian ũ^i = W v^i.
    util = (velx, vely, velz)
    u_d = lower(util)
    W = _np.sqrt(1.0 + dot(util, u_d))
    u_t_up = W / alp
    u_up = [util[i] - bet[i] * u_t_up for i in range(3)]
    u_t = -alp * W + dot(bet, u_d)

    # Magnetic four-vector b^μ; bcc is √γ B^i.
    B = [bcc1 / sqrt_g, bcc2 / sqrt_g, bcc3 / sqrt_g]
    bet_d = lower(bet)
    g_tt = -alp**2 + dot(bet, bet_d)
    b_t_up = dot(B, u_d) / alp
    b_up = [(B[i] + alp * b_t_up * u_up[i]) / u_t_up for i in range(3)]
    b_d = lower([b_up[i] + bet[i] * b_t_up for i in range(3)])
    b_t = g_tt * b_t_up + dot(bet_d, b_up)
    b2 = (dot(B, lower(B)) + (alp * b_t_up)**2) / W**2

    # Radial and azimuthal components.
    u_r_up = _calc_ur(*u_up, theta, phi)
    b_r_up = _calc_ur(*b_up, theta, phi)
    u_phi = xi_x * u_d[0] + xi_y * u_d[1]
    b_phi = xi_x * b_d[0] + xi_y * b_d[1]
    with _np.errstate(invalid='ignore', divide='ignore'):
      u_phi_up = (x * u_up[1] - y * u_up[0]) / varpi2
      b_phi_up = (x * b_up[1] - y * b_up[0]) / varpi2

    # T^μ_ν = (ρh + b²) u^μ u_ν + (p + b²/2) δ^μ_ν − b^μ b_ν, split into EM and matter.
    rho_h = dens * _enthalpy(dens, ye, temperature)
    Trt_EM = b2 * u_r_up * u_t - b_r_up * b_t
    Trt_MA = rho_h * u_r_up * u_t
    Trphi_EM = b2 * u_r_up * u_phi - b_r_up * b_phi
    Trphi_MA = rho_h * u_r_up * u_phi
    w, P = rho_h + b2, press + 0.5 * b2
    Ttt = w * u_t_up * u_t + P - b_t_up * b_t

    # Norm of the Noether current P^μ = −T^μ_t (> 0: spacelike).
    P2 = (-w**2 * u_t**2 + 2 * w * P * u_t**2 + P**2 * g_tt
          + b2 * b_t**2 - 2 * P * b_t**2)

    # Surface element √−g r² sinθ dθ dφ.
    sqrt_mg = alp * sqrt_g
    dA = _get_riemann_weights(dens, theta[0, :], phi[:, 0]) * _np.sin(theta) * r**2
    area = dA.sum()

    def flux(f):
      return (sqrt_mg * f * dA).sum()

    def phi_avg(f):
      return f.mean(axis=0)

    out[radius] = {
      'time':     time,
      'center':   centers[0],
      'theta':    theta[0, :],
      'Mdot':     -flux(dens * u_r_up),
      'Edot_EM':  flux(Trt_EM),
      'Edot_MA':  flux(Trt_MA),
      'Jdot_EM':  -flux(Trphi_EM),
      'Jdot_MA':  -flux(Trphi_MA),
      'f_ergo':   dA[g_tt > 0].sum() / area,
      'f_neg_e':  dA[Ttt > 0].sum() / area,
      'f_P2':     dA[P2 > 0].sum() / area,
      # φ-averaged θ profiles, densitized by √−g as the paper's ė_H and ȷ̇_H.
      'edot_EM':  phi_avg(sqrt_mg * Trt_EM),
      'edot_MA':  phi_avg(sqrt_mg * Trt_MA),
      'jdot_EM':  -phi_avg(sqrt_mg * Trphi_EM),
      'jdot_MA':  -phi_avg(sqrt_mg * Trphi_MA),
      'einf':     -phi_avg(sqrt_mg * Ttt),
      'gtt':      phi_avg(g_tt),
      'P2':       phi_avg(P2),
      'P2_EM':    phi_avg(0.25 * b2**2 * g_tt),
      # Fraction of each θ ring where P is spacelike, and where its energy density is negative.
      'fP2':      phi_avg((P2 > 0).astype(float)),
      'fneg_e':   phi_avg((Ttt > 0).astype(float)),
      # Ω_F = (u^φ b^r − u^r b^φ) / (u^t b^r − u^r b^t); the ratio is taken after averaging.
      'omF_num':  phi_avg(u_phi_up * b_r_up - u_r_up * b_phi_up),
      'omF_den':  phi_avg(u_t_up * b_r_up - u_r_up * b_t_up),
    }
  return out


# Analysis driver function.
def analyze(sph_dirs, eos_table, output_dir, radii=DEFAULT_RADII, jobname=DEFAULT_JOBNAME,
            n_workers=None):
  """
  Run the Lasota generalized Penrose analysis over the snapshots in
  `sph_dirs`.

  Parameters:
  sph_dirs (list/str): AthenaK ``output-XXXX/sph`` directories to scan
                       (all treated as one run).
  eos_table (str): PyCompOSE HDF5 EOS table matching the simulation EOS.
  output_dir (str): Directory the .txt outputs are written to.
  radii (sequence/float): SPH extraction radii [M_sun]; each must have mhd_w_bcc/adm/z4c output.
                          A snapshot is used at every radius it holds, so the radii may cover different
                          time spans.
  jobname (str): AthenaK job name prefixing the VTK files.
  n_workers (int): Worker processes for the snapshot loop.  Default: min(8, cpu_count()).
  """
  if n_workers is None:
    n_workers = min(8, cpu_count())
  radii = sorted(set(float(r) for r in radii))

  # ------------------------------------------------------------------
  # Locate the dumps holding each radius.
  # ------------------------------------------------------------------
  located  = {}
  complete = {}
  for radius in radii:
    located[radius] = {v: locate(sph_dirs, jobname, v, radius) for v in VARIABLES}
    for v in VARIABLES:
      print(f"  {located[radius][v].describe()}")
    index_sets = [set(located[radius][v].paths) for v in VARIABLES]
    complete[radius] = set.intersection(*index_sets)
    incomplete = len(set.union(*index_sets)) - len(complete[radius])
    if incomplete:
      print(f"  [skip] r = {radius:g}: {incomplete} snapshots missing one of "
            f"{', '.join(VARIABLES)}")
    print(f"Found {len(complete[radius])} snapshots at r = {radius:g}.")

  indices = sorted(set.union(*complete.values()))
  if not indices:
    raise RuntimeError(
      f"No snapshot in {list(sph_dirs)} has all of {', '.join(VARIABLES)} "
      f"at any of r = {', '.join(f'{r:g}' for r in radii)}.\n"
      "Check the sph directories, the job name and the extraction radii.")

  # ------------------------------------------------------------------
  # Build worker argument list: one task per snapshot, covering all radii.
  # ------------------------------------------------------------------
  worker_args = [
    [(radius, tuple(located[radius][v].shell(i) for v in VARIABLES))
     for radius in radii if i in complete[radius]]
    for i in indices
  ]
  print(f"Processing {len(worker_args)} snapshots with n_workers={n_workers}...")

  # ------------------------------------------------------------------
  # Parallel snapshot loop
  # ------------------------------------------------------------------
  eos = load_eos(eos_table)
  init_args = (eos, eos['mn_mev'] * cgs.eV * 1e6 / cgs.light_speed**2)
  if n_workers > 1:
    with Pool(n_workers, initializer=_init_penrose_worker, initargs=init_args) as pool:
      results = list(pool.imap_unordered(
        _process_snapshot_penrose, worker_args,
        chunksize=max(1, len(worker_args) // (n_workers * 4))))
  else:
    _init_penrose_worker(*init_args)
    results = [_process_snapshot_penrose(a) for a in worker_args]

  # ------------------------------------------------------------------
  # Unpack and save results, one file per radius
  # ------------------------------------------------------------------
  os.makedirs(output_dir, exist_ok=True)

  for radius in radii:
    res = sorted((r[radius] for r in results if radius in r), key=lambda d: d['time'])
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


def load_horizon(horizon_file):
  """(time, M, ω_H) from an AthenaK horizon summary."""
  data = np.loadtxt(horizon_file, comments='#')
  t, first = np.unique(data[:, 1], return_index=True)
  M, S = data[first, 2], data[first, 6]
  a = S / M
  r_plus = M + np.sqrt(np.maximum(M**2 - a**2, 0.0))
  return t, M, a / (2.0 * M * r_plus)


def _auto_symlog(ax, pct=5.0):
  """Automatic symlog y-axis."""
  from matplotlib.ticker import FixedLocator, SymmetricalLogLocator
  vals = np.concatenate([np.abs(np.asarray(l.get_ydata(), dtype=float))
                         for l in ax.get_lines()] or [np.empty(0)])
  vals = vals[np.isfinite(vals) & (vals > 0)]
  lin = 10.0**np.floor(np.log10(np.percentile(vals, pct))) if vals.size else 1e-12
  ax.set_yscale('symlog', linthresh=lin, linscale=0.5)
  ax.relim()
  ax.autoscale_view()

  ticks = SymmetricalLogLocator(base=100.0, linthresh=lin).tick_values(*ax.get_ylim())
  ax.yaxis.set_major_locator(FixedLocator([v for v in ticks if v == 0 or abs(v) > 10 * lin]))


# Plotter function.
def plot_penrose(output_dir, radii=DEFAULT_RADII, horizon_file=None,
                 t_merger=None):
  """Fluxes vs time (fig_penrose_fluxes.png) and time-averaged θ profiles after
  t_merger (fig_penrose_profiles_r{R}.png)."""
  from .plots import MSUN_TO_MS, plt
  from matplotlib.lines import Line2D

  t_bh, _, om_bh = load_horizon(horizon_file)
  t0 = 0.0 if t_merger is None else t_merger
  xlabel = r'$t$ [ms]' if t_merger is None else r'$t - t_\mathrm{merger}$ [ms]'

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
    t = (ts['time'] - t0) * MSUN_TO_MS
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
    # NEC bound ω_H J̇_H ≤ Ė_H holds on the horizon.
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

  fig.tight_layout()
  out = os.path.join(output_dir, 'fig_penrose_fluxes.png')
  fig.savefig(out, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved {out}')

  # ------- TIME-AVERAGED θ PROFILES -------
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

    fig.tight_layout()
    out = os.path.join(output_dir, f'fig_penrose_profiles_r{_tag(radius)}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

  _plot_noether(output_dir, data, t_merger, xlabel)


def _plot_noether(output_dir, data, t_merger, xlabel):
  """Noether current P^μ = -T^μ_t (fig_penrose_noether_r{R}.png): its norm P^2 and
  where it is spacelike (P^2 > 0) or has negative energy (e_inf < 0)."""
  from .plots import MSUN_TO_MS, plt

  t0 = 0.0 if t_merger is None else t_merger
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
    t = (ts['time'] - t0) * MSUN_TO_MS
    ax.semilogy(t, ts['f_P2'], color='black', lw=1.0, label=r'$P^2 > 0$')
    ax.semilogy(t, ts['f_neg_e'], color='tab:red', lw=1.0, label=r'$e_\infty < 0$')
    ax.semilogy(t, ts['f_ergo'], color='gray', lw=1.0, ls='--', label='ergoregion')
    ax.set_ylabel('area fraction of sphere')
    ax.set_xlabel(xlabel)
    ax.set_xlim(t.min(), t.max())
    ax.legend(fontsize=9, frameon=False, ncol=3)

    axes[0].set_title(f'Noether current $P^\\mu = -T^\\mu_{{\\ t}}$, $r = {radius:g}\\,M_\\odot$', fontsize=9)
    fig.tight_layout()
    out = os.path.join(output_dir, f'fig_penrose_noether_r{_tag(radius)}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# Main (argparser).
def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--sph-dir", action="append", default=[], dest="sph_dirs",
                 metavar="DIR",
                 help="AthenaK output-XXXX/sph directory (repeat for each segment); " \
                      "required unless --plot-only.")
  p.add_argument("--output-dir", required=True,
                 help="Directory for the .txt outputs and figures.")
  p.add_argument("--radius", type=float, action="append", default=[], dest="radii",
                 help="SPH extraction radius [M_sun] (repeat for several). "
                     f"Default: {', '.join(f'{r:g}' for r in DEFAULT_RADII)}.")
  p.add_argument("--eos-table", default=None,
                 help="PyCompOSE HDF5 EOS table matching the simulation EOS; "
                      "required unless --plot-only.")
  p.add_argument("--jobname", default=DEFAULT_JOBNAME,
                 help=f"AthenaK job name prefixing the VTK files. "
                      f"Default: {DEFAULT_JOBNAME}.")
  p.add_argument("--n-workers", type=int, default=None,
                 help="Worker processes for the snapshot loop. "
                      "Default: min(8, cpu_count()).")
  p.add_argument("--plot", action="store_true",
                 help="Plot some of the Lasota profiles.")
  p.add_argument("--plot-only", action="store_true",
                 help="Skip the analysis and plot the existing outputs in --output-dir.")
  p.add_argument("--horizon-file", default=None,
                 help="AthenaK horizon summary (<job>.horizon_summary_0.txt).")
  p.add_argument("--t-merger", type=float, default=None,
                 help="Merger time [M_sun]; the plot then uses t - t_merger. "
                      "Default: absolute time.")

  args = p.parse_args(argv)
  radii = args.radii or list(DEFAULT_RADII)

  if not args.plot_only:
    if not args.sph_dirs:
      p.error("--sph-dir is required unless --plot-only is given")
    if args.eos_table is None:
      p.error("--eos-table is required unless --plot-only is given")
    analyze(args.sph_dirs, args.eos_table, args.output_dir, radii=radii,
            jobname=args.jobname, n_workers=args.n_workers)
  if args.plot or args.plot_only:
    if args.horizon_file is None:
      p.error("plotting needs --horizon-file")
    plot_penrose(args.output_dir, radii=radii, horizon_file=args.horizon_file,
                 t_merger=args.t_merger)


if __name__ == '__main__':
  main()
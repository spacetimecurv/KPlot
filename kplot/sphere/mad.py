"""
SANE/MAD state and neutrino cooling efficiency of the post-merger disk, from
spherical extraction surfaces centred on the black hole.

Per snapshot and extraction radius R the analysis integrates over the sphere

    Mdot   = - ∮ α √γ ρ u^r r² dΩ           net accretion rate (> 0: inflow)
    Mdot_in, Mdot_out                        inflowing / outflowing parts alone
    Phi    = ½ ∮ |√γ B^r| r² dΩ             magnetic flux threading one hemisphere
    Phi_N, Phi_S = ∮_{N,S} √γ B^r r² dΩ      signed flux through each hemisphere

and the plot step forms, with window averages <.> over --t-avg,

    φ   = <Phi> / sqrt(<Mdot> r_g²)          dimensionless flux, r_g = M_BH
    η   = <L_ν>(t + R_ν - R) / (<Mdot> c²)  neutrino cooling efficiency

L_ν is the total (all species) luminosity of kplot.sphere.neutrinos, extracted at
R_ν, and is shifted back by the light travel time from R to R_ν.

The sphere is centred on the tracker given by the sph block's ``center_tracker``;
before that tracker is enabled the sphere sits at the origin, so the plot step
only uses times from --t-min on (default: the merger).

Required parfile output blocks (one block per variable, any number of radii):
    variable = mhd_w_bcc, adm, z4c_alpha, z4c_betax, z4c_betay, z4c_betaz
    center_tracker = <index of the black hole tracker>

Outputs (written to --output-dir):
    mad_r{R}.txt          time, Mdot, Mdot_in, Mdot_out, Phi, Phi_N, Phi_S  [code units]
    sphere_center_mad.txt t, center of the extraction surface                 [M_sun]
    mad_derived_r{R}.txt  (plot step) time, M_BH, <Mdot> [M_sun/s], φ_HL, φ_G,
                          φ_G from <Mdot_in>, <L_ν> [erg/s], η
    fig_mad.png           (plot step) <Mdot>, φ and η vs time, one line per radius
    fig_eta_mdot.png      (plot step) η vs <Mdot>, coloured by time

Command line:
    kplot-sphere-mad --sph-dir DIR [--sph-dir DIR ...] --output-dir DIR \
        --radius 25 --radius 50 --radius 100 \
        [--plot --horizon-file F --lnu-file F --lnu-radius 300 --t-merger T]
    kplot-sphere-mad --output-dir DIR --radius 25 --plot-only --horizon-file F ...
"""

import argparse
import os
from multiprocessing import Pool, cpu_count

import numpy as np

from ._files import locate

DEFAULT_RADII = (25.0, 50.0, 100.0)
DEFAULT_JOBNAME = "bns"
DEFAULT_T_AVG_MS = 5.0

# The sph output variables a surface needs, in the order the worker unpacks them.
VARIABLES = ('mhd_w_bcc', 'adm', 'z4c_alpha', 'z4c_betax', 'z4c_betay', 'z4c_betaz')

# Gaussian / Heaviside-Lorentz field units: B_G = sqrt(4π) B_HL.
HL_TO_GAUSS = np.sqrt(4.0 * np.pi)

# MAD saturation of the dimensionless flux in Gaussian units (Tchekhovskoy et al.
# 2011); ≈ 15 in Heaviside-Lorentz units.
PHI_MAD_GAUSS = 50.0

# NDAF reference efficiency for Mdot ~ 0.01 - 1 M_sun/s.
ETA_NDAF = 0.06
NDAF_MDOT_RANGE = (0.01, 1.0)   # [M_sun/s]

# Unit conversions; the same constants as kplot.sphere.plots.
G     = 6.67430e-11
M_SUN = 1.98892e30
C     = 2.99792458e8
MSUN_TO_S = G * M_SUN / C**3
# Code power (M_sun per code time, times c²) in erg/s, i.e. c⁵/G.
CODE_POWER_CGS = M_SUN * 1e3 * (C * 1e2)**2 / MSUN_TO_S

COLUMNS = ('Mdot', 'Mdot_in', 'Mdot_out', 'Phi', 'Phi_N', 'Phi_S')


def _tag(radius):
  return f'{radius:g}'


# ===================================================================
# Main analysis
# ===================================================================

def _process_snapshot_mad(args):
  """Per-snapshot worker: fluxes on every requested surface of one snapshot.

    `args` is a list of ``(radius, files)``, `files` being the (path, surface
    index) of every entry of VARIABLES.  Each dump is parsed once, however many of
    the surfaces it holds.
    """
  import numpy as _np

  import os as _os

  from athplot.load_sph_vtk import SphericalData as _SphericalData

  from ._files import Shell as _Shell, CENTER_TOL as _CENTER_TOL
  from ._integrate import (calc_ur as _calc_ur,
                           sqrt_det_metric_adm as _sqrt_det,
                           get_riemann_weights as _get_riemann_weights)

  parsed = {}

  def shell(entry):
    path, iradius = entry
    if path not in parsed:
      parsed[path] = _SphericalData(path)
    return _Shell(path, iradius, data=parsed[path])

  out = {}
  for radius, files in args:
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

    z4c_alpha_raw = alpha.shell('z4c_alpha')
    phi_zero    = (phi == phi.min())
    theta_north = (theta == theta.min())
    dup_corr = _np.ones(z4c_alpha_raw.shape)
    if z4c_alpha_raw[phi_zero & ~theta_north].max() > 1.0:
      dup_corr[phi_zero] = 0.5
    if z4c_alpha_raw[theta_north].max() > 1.0:
      dup_corr[theta_north] = 0.25

    dens = mhd.shell('dens') * dup_corr
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

    W = _np.sqrt(1.0 + gxx*velx**2 + gyy*vely**2 + gzz*velz**2
                  + 2*gxy*velx*vely + 2*gxz*velx*velz + 2*gyz*vely*velz)
    ut_contrav = W / z4c_alpha
    ux_u = velx - z4c_betax * ut_contrav
    uy_u = vely - z4c_betay * ut_contrav
    uz_u = velz - z4c_betaz * ut_contrav
    ur = _calc_ur(ux_u, uy_u, uz_u, theta, phi)

    # Coordinate area element r² sinθ dθ dφ of the extraction sphere.
    dA = _get_riemann_weights(dens, theta[0, :], phi[:, 0]) * _np.sin(theta) * r**2

    # ------- MASS FLUX ------- sqrt(-g) ρ u^r, outward positive
    mass_flux = _sqrt_det(gxx, gyy, gzz, gxy, gxz, gyz, z4c_alpha) * dens * ur * dA

    # ------- MAGNETIC FLUX ------- bcc is already √γ B^i
    flux_B = _calc_ur(bcc1, bcc2, bcc3, theta, phi) * dA
    north = theta < 0.5 * _np.pi

    out[radius] = {
      'time':     time,
      'center':   centers[0],
      'Mdot':     -mass_flux.sum(),
      'Mdot_in':  -mass_flux[mass_flux < 0].sum(),
      'Mdot_out': mass_flux[mass_flux > 0].sum(),
      'Phi':      0.5 * _np.abs(flux_B).sum(),
      'Phi_N':    flux_B[north].sum(),
      'Phi_S':    flux_B[~north].sum(),
    }
  return out


def analyze(sph_dirs, output_dir, radii=DEFAULT_RADII, jobname=DEFAULT_JOBNAME,
            n_workers=None):
  """Run the accretion-rate / magnetic-flux analysis over every snapshot found in
  `sph_dirs`.

  Parameters
  ----------
  sph_dirs : list of str
    AthenaK ``output-XXXX/sph`` directories to scan (all treated as one run).
  output_dir : str
    Directory the .txt outputs are written to.
  radii : sequence of float
    SPH extraction radii [M_sun]; each must have mhd_w_bcc/adm/z4c output.  A
    snapshot is used at every radius it holds, so the radii may cover different
    time spans.
  jobname : str
    AthenaK job name prefixing the VTK files.
  n_workers : int, optional
    Worker processes for the snapshot loop.  Default: min(8, cpu_count()).
  """
  if n_workers is None:
    n_workers = min(8, cpu_count())
  radii = sorted(set(float(r) for r in radii))

  # ------------------------------------------------------------------
  # Locate the dumps holding each radius, and the snapshots complete at it.
  # ------------------------------------------------------------------
  located = {}
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
  if n_workers > 1:
    with Pool(n_workers) as pool:
      results = list(pool.imap_unordered(
          _process_snapshot_mad, worker_args,
          chunksize=max(1, len(worker_args) // (n_workers * 4))))
  else:
    results = [_process_snapshot_mad(a) for a in worker_args]

  # ------------------------------------------------------------------
  # Unpack and save results, one file per radius
  # ------------------------------------------------------------------
  os.makedirs(output_dir, exist_ok=True)

  # The center is the same for every radius of a snapshot; keep one row each.
  centers = sorted([next(iter(res.values()))['time'], *next(iter(res.values()))['center']]
                   for res in results)
  np.savetxt(os.path.join(output_dir, 'sphere_center_mad.txt'), np.array(centers),
             header='time(Msun)    xc(Msun)    yc(Msun)    zc(Msun)',
             fmt='%.6e')

  for radius in radii:
    rows = sorted([res[radius]['time']] + [res[radius][k] for k in COLUMNS]
                  for res in results if radius in res)
    rows = np.array(rows)
    path = os.path.join(output_dir, f'mad_r{_tag(radius)}.txt')
    np.savetxt(path, rows, fmt='%.8e',
               header=f'r = {radius:g} M_sun, code units (G = c = M_sun = 1, '
                      'Heaviside-Lorentz B)\n'
                      'time  Mdot(net, >0 inflow)  Mdot_in  Mdot_out  '
                      'Phi(=1/2 int|sqrt(g)B^r|dA)  Phi_N  Phi_S')
    print(f'r = {radius:g}: {len(rows)} snapshots, '
          f'peak Mdot = {rows[:, 1].max():.4e}, peak Phi = {rows[:, 4].max():.4e} '
          f'(code units) -> {path}')

  print(f'All outputs saved to {output_dir}')


# ===================================================================
# Derived quantities and plots
# ===================================================================

def load_bh_mass(horizon_file):
  """(time, M_BH) [M_sun] from an AthenaK horizon summary, valid rows only."""
  data = np.loadtxt(horizon_file, comments='#')
  t, m = data[:, 1], data[:, 2]
  keep = np.isfinite(m) & (m > 0)
  t, m = t[keep], m[keep]
  t, first = np.unique(t, return_index=True)
  return t, m[first]


def running_mean(t, y, width):
  """Centred running mean of y(t) over a window of `width` in t.

    The samples need not be equally spaced; NaNs are ignored.
    """
  if width <= 0:
    return y.copy()
  lo = np.searchsorted(t, t - 0.5 * width, side='left')
  hi = np.searchsorted(t, t + 0.5 * width, side='right')
  ok = np.isfinite(y)
  csum = np.concatenate([[0.0], np.cumsum(np.where(ok, y, 0.0))])
  cnt  = np.concatenate([[0],   np.cumsum(ok)])
  n = cnt[hi] - cnt[lo]
  with np.errstate(invalid='ignore', divide='ignore'):
    return np.where(n > 0, (csum[hi] - csum[lo]) / n, np.nan)


def _load_mad(output_dir, radius):
  path = os.path.join(output_dir, f'mad_r{_tag(radius)}.txt')
  if not os.path.exists(path):
    return None
  data = np.atleast_2d(np.loadtxt(path))
  t, first = np.unique(data[:, 0], return_index=True)
  data = data[first]
  return {'time': t, **{k: data[:, i + 1] for i, k in enumerate(COLUMNS)}}


def derive(mad, radius, t_bh=None, m_bh=None, m_bh_const=None, lnu=None,
           lnu_radius=None, t_avg=0.0):
  """φ and η on the snapshot times of `mad` (see the module docstring).

    Parameters
    ----------
    mad : dict
        One radius of :func:`analyze`'s output, as loaded by ``_load_mad``.
    t_bh, m_bh : ndarray, optional
        Black hole mass history [M_sun]; or `m_bh_const` for a fixed mass.
    lnu : (ndarray, ndarray), optional
        Time [M_sun] and total neutrino luminosity [erg/s] at `lnu_radius`.
    t_avg : float
        Averaging window [M_sun].
    """
  t = mad['time']
  if m_bh_const is not None:
    M = np.full_like(t, m_bh_const)
  else:
    M = np.interp(t, t_bh, m_bh, left=np.nan, right=m_bh[-1])

  mdot    = running_mean(t, mad['Mdot'], t_avg)
  mdot_in = running_mean(t, mad['Mdot_in'], t_avg)
  Phi     = running_mean(t, mad['Phi'], t_avg)

  with np.errstate(invalid='ignore', divide='ignore'):
    # φ is undefined where the sphere has net outflow.
    phi_hl = np.where(mdot > 0, Phi / np.sqrt(mdot * M**2), np.nan)
    phi_in = Phi / np.sqrt(mdot_in * M**2) * HL_TO_GAUSS

  eta = np.full_like(t, np.nan)
  lnu_ret = np.full_like(t, np.nan)
  if lnu is not None:
    t_nu, L_nu = lnu
    L_avg = running_mean(t_nu, L_nu, t_avg)
    # L_ν leaves R at t and reaches R_ν at t + (R_ν - R) (c = 1).
    lnu_ret = np.interp(t + (lnu_radius - radius), t_nu, L_avg,
                        left=np.nan, right=np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
      eta = np.where(mdot > 0, lnu_ret / (mdot * CODE_POWER_CGS), np.nan)

  return {
    'time':       t,
    'M_BH':       M,
    'Mdot_Msun_s': mdot / MSUN_TO_S,
    'phi_HL':     phi_hl,
    'phi_G':      phi_hl * HL_TO_GAUSS,
    'phi_G_in':   phi_in,
    'Lnu':        lnu_ret,
    'eta':        eta,
  }


def plot_mad(output_dir, radii=DEFAULT_RADII, horizon_file=None, m_bh=None,
             lnu_file=None, lnu_radius=None, t_merger=None, t_min=None,
             t_avg_ms=DEFAULT_T_AVG_MS):
  """Write mad_derived_r{R}.txt and plot <Mdot>, φ and η (fig_mad.png) and η vs
    <Mdot> (fig_eta_mdot.png).

    The time axis is t - t_merger [ms] if `t_merger` [M_sun] is given, else the
    absolute time [ms].  Only times >= `t_min` [M_sun] are used, default
    `t_merger`.
    """
  from .plots import MSUN_TO_MS, plt
  from matplotlib.colors import Normalize

  if horizon_file is None and m_bh is None:
    raise ValueError('Give a horizon file or a fixed black hole mass.')
  t_bh = m_bh_hist = None
  if m_bh is None:
    t_bh, m_bh_hist = load_bh_mass(horizon_file)
    print(f'  M_BH from {horizon_file}: {m_bh_hist[0]:.4f} -> {m_bh_hist[-1]:.4f} M_sun')

  lnu = None
  if lnu_file is not None:
    if lnu_radius is None:
      raise ValueError('lnu_radius is needed to shift the luminosity in time.')
    if os.path.exists(lnu_file):
      d = np.loadtxt(lnu_file)
      t_nu, first = np.unique(d[:, 0], return_index=True)
      lnu = (t_nu, d[first, 1])
    else:
      print(f'  No neutrino luminosity at {lnu_file}; skipping η.')

  if t_min is None:
    t_min = t_merger
  t_avg = t_avg_ms / MSUN_TO_MS

  derived = {}
  for radius in sorted(set(float(r) for r in radii)):
    mad = _load_mad(output_dir, radius)
    if mad is None:
      print(f'  No mad_r{_tag(radius)}.txt in {output_dir}; skipping r = {radius:g}.')
      continue
    if t_min is not None:
      keep = mad['time'] >= t_min
      mad = {k: v[keep] for k, v in mad.items()}
    if len(mad['time']) == 0:
      print(f'  r = {radius:g}: no snapshot after t_min; skipping.')
      continue
    d = derive(mad, radius, t_bh, m_bh_hist, m_bh, lnu, lnu_radius, t_avg)
    derived[radius] = d

    path = os.path.join(output_dir, f'mad_derived_r{_tag(radius)}.txt')
    np.savetxt(path, np.column_stack([d[k] for k in
               ('time', 'M_BH', 'Mdot_Msun_s', 'phi_HL', 'phi_G', 'phi_G_in',
                'Lnu', 'eta')]),
               fmt='%.6e',
               header=f'r = {radius:g} M_sun, window averages over {t_avg_ms:g} ms; '
                      f'L_nu at r = {lnu_radius if lnu is not None else "-"} '
                      'shifted by the light travel time\n'
                      'time[Msun]  M_BH[Msun]  <Mdot>[Msun/s]  phi_HL  phi_G  '
                      'phi_G(<Mdot_in>)  <L_nu>[erg/s]  eta=L_nu/(Mdot c^2)')
    with np.errstate(invalid='ignore'):
      print(f'  r = {radius:g}: median <Mdot> = {np.nanmedian(d["Mdot_Msun_s"]):.3g} '
            f'M_sun/s, median φ_G = {np.nanmedian(d["phi_G"]):.3g}, max φ_G = '
            f'{np.nanmax(d["phi_G"]) if np.isfinite(d["phi_G"]).any() else np.nan:.3g}, '
            f'median η = {np.nanmedian(d["eta"]) if np.isfinite(d["eta"]).any() else np.nan:.3g}'
            f' -> {path}')

  if not derived:
    print('  Nothing to plot.')
    return

  if t_merger is None:
    def t_ms(t): return t * MSUN_TO_MS
    xlabel = r'$t$ [ms]'
  else:
    def t_ms(t): return (t - t_merger) * MSUN_TO_MS
    xlabel = r'$t - t_\mathrm{merger}$ [ms]'

  # Fixed colour per radius, by its position in the sorted radius list.
  colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
  color = {radius: colors[i % len(colors)] for i, radius in enumerate(derived)}
  has_eta = any(np.isfinite(d['eta']).any() for d in derived.values())

  # ------- TIME EVOLUTION -------
  npanel = 3 if has_eta else 2
  fig, axes = plt.subplots(npanel, 1, figsize=(7.5, 2.6 * npanel + 0.6), sharex=True)

  ax = axes[0]
  for radius, d in derived.items():
    ax.plot(t_ms(d['time']), d['Mdot_Msun_s'], color=color[radius], lw=1.2,
            label=f'$r = {radius:g}\\,M_\\odot$')
  ax.axhline(0.0, color='black', lw=0.6)
  ax.set_yscale('symlog', linthresh=1e-2)
  ax.set_ylabel(r'$\langle\dot M\rangle$  [$M_\odot\,\mathrm{s}^{-1}$]')
  ax.legend(fontsize=9, frameon=False, ncol=len(derived))

  ax = axes[1]
  for radius, d in derived.items():
    ax.semilogy(t_ms(d['time']), d['phi_G'], color=color[radius], lw=1.2)
    ax.semilogy(t_ms(d['time']), d['phi_G_in'], color=color[radius], lw=0.8, ls=':')
  ax.axhline(PHI_MAD_GAUSS, color='gray', ls='--', lw=0.8)
  ax.text(0.01, PHI_MAD_GAUSS, r' MAD, $\phi \approx 50$', color='gray', fontsize=8,
          va='bottom', transform=ax.get_yaxis_transform())
  ax.set_ylabel(r'$\phi = \Phi_B / \sqrt{\langle\dot M\rangle r_g^2 c}$  (Gaussian)')

  if has_eta:
    ax = axes[2]
    for radius, d in derived.items():
      ax.semilogy(t_ms(d['time']), d['eta'], color=color[radius], lw=1.2)
    ax.axhline(ETA_NDAF, color='gray', ls='--', lw=0.8)
    ax.text(0.01, ETA_NDAF, rf' NDAF, $\eta \approx {ETA_NDAF:g}$', color='gray',
            fontsize=8, va='bottom', transform=ax.get_yaxis_transform())
    ax.set_ylabel(r'$\eta = L_\nu / \langle\dot M\rangle c^2$')

  axes[-1].set_xlabel(xlabel)
  axes[0].set_title(rf'window average {t_avg_ms:g} ms; dotted: $\phi$ from inflow only',
                    fontsize=9)
  fig.tight_layout()
  out = os.path.join(output_dir, 'fig_mad.png')
  fig.savefig(out, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved {out}')

  # ------- EFFICIENCY VS ACCRETION RATE -------
  if not has_eta:
    return
  ncol = len(derived)
  fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol + 0.8, 3.6), sharey=True,
                           squeeze=False)
  axes = axes[0]
  tmin = min(t_ms(d['time']).min() for d in derived.values())
  tmax = max(t_ms(d['time']).max() for d in derived.values())
  norm = Normalize(vmin=tmin, vmax=tmax)
  for ax, (radius, d) in zip(axes, derived.items()):
    ok = np.isfinite(d['eta']) & (d['Mdot_Msun_s'] > 0)
    sc = ax.scatter(d['Mdot_Msun_s'][ok], d['eta'][ok], c=t_ms(d['time'][ok]),
                    cmap='viridis', norm=norm, s=8, lw=0)
    ax.axvspan(*NDAF_MDOT_RANGE, color='gray', alpha=0.12, lw=0)
    ax.axhline(ETA_NDAF, color='gray', ls='--', lw=0.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\langle\dot M\rangle$  [$M_\odot\,\mathrm{s}^{-1}$]')
    ax.set_title(f'$r = {radius:g}\\,M_\\odot$', fontsize=10)
  axes[0].set_ylabel(r'$\eta = L_\nu / \langle\dot M\rangle c^2$')
  cbar = fig.colorbar(sc, ax=list(axes), pad=0.02)
  cbar.set_label(xlabel)
  out = os.path.join(output_dir, 'fig_eta_mdot.png')
  fig.savefig(out, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved {out}')


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--sph-dir", action="append", default=[], dest="sph_dirs",
                  metavar="DIR",
                  help="AthenaK output-XXXX/sph directory (repeat for each segment); "
                       "required unless --plot-only.")
  p.add_argument("--output-dir", required=True,
                  help="Directory for the .txt outputs and figures.")
  p.add_argument("--radius", type=float, action="append", default=[], dest="radii",
                  help="SPH extraction radius [M_sun] (repeat for several). "
                       f"Default: {', '.join(f'{r:g}' for r in DEFAULT_RADII)}.")
  p.add_argument("--jobname", default=DEFAULT_JOBNAME,
                  help=f"AthenaK job name prefixing the VTK files. "
                      f"Default: {DEFAULT_JOBNAME}.")
  p.add_argument("--n-workers", type=int, default=None,
                  help="Worker processes for the snapshot loop. "
                      "Default: min(8, cpu_count()).")
  p.add_argument("--plot", action="store_true",
                  help="Derive φ, η and plot them after the analysis.")
  p.add_argument("--plot-only", action="store_true",
                  help="Skip the analysis and plot the existing outputs in --output-dir.")
  p.add_argument("--horizon-file", default=None,
                  help="AthenaK horizon summary (<job>.horizon_summary_0.txt) giving "
                       "M_BH(t) for r_g.")
  p.add_argument("--m-bh", type=float, default=None,
                  help="Fixed black hole mass [M_sun] instead of --horizon-file.")
  p.add_argument("--lnu-file", default=None,
                  help="Lnu_E_total.txt of kplot-sphere-neutrinos; without it η is "
                       "skipped.")
  p.add_argument("--lnu-radius", type=float, default=None,
                  help="Extraction radius [M_sun] of --lnu-file.")
  p.add_argument("--t-merger", type=float, default=None,
                  help="Merger time [M_sun]; the plot then uses t - t_merger. "
                       "Default: absolute time.")
  p.add_argument("--t-min", type=float, default=None,
                  help="First time [M_sun] used in the plot step. Default: --t-merger.")
  p.add_argument("--t-avg", type=float, default=DEFAULT_T_AVG_MS,
                  help=f"Averaging window [ms] for Mdot, Phi and L_nu. "
                       f"Default: {DEFAULT_T_AVG_MS:g}.")
  args = p.parse_args(argv)
  radii = args.radii or list(DEFAULT_RADII)

  if not args.plot_only:
    if not args.sph_dirs:
      p.error("--sph-dir is required unless --plot-only is given")
    analyze(args.sph_dirs, args.output_dir, radii=radii,
            jobname=args.jobname, n_workers=args.n_workers)
  if args.plot or args.plot_only:
    if args.horizon_file is None and args.m_bh is None:
      p.error("plotting needs --horizon-file or --m-bh")
    plot_mad(args.output_dir, radii=radii, horizon_file=args.horizon_file,
             m_bh=args.m_bh, lnu_file=args.lnu_file, lnu_radius=args.lnu_radius,
             t_merger=args.t_merger, t_min=args.t_min, t_avg_ms=args.t_avg)


if __name__ == '__main__':
  main()

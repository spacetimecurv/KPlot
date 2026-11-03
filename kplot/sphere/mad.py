"""
SANE/MAD state and neutrino cooling efficiency of the post-merger disk, from
spherical extraction surfaces centred on the black hole.

Per snapshot and extraction radius R the analysis integrates over the sphere

    Mdot         = - oint alpha sqrt(gamma) rho u^r r^2 dOmega  net accretion rate (> 0: inflow)
    Mdot_in, Mdot_out                                        inflowing / outflowing parts alone
    Phi          = 1/2 oint |sqrt(gamma) B^r| r^2 dOmega     magnetic flux threading one hemisphere
    Phi_N, Phi_S = oint_{N,S} sqrt(gamma) B^r r^2 dOmega     signed flux through each hemisphere

and the plot step forms, with window averages <.> over --t-avg,

    phi = <Phi> / sqrt(<Mdot> r_g^2)           dimensionless flux, r_g = M_BH
    eta = <L_nu>(t + R_nu - R) / (<Mdot> c^2)  neutrino cooling efficiency

L_nu is the total (all species) luminosity of kplot.sphere.neutrinos, extracted at
R_nu, and is shifted back by the light travel time from R to R_nu.

The sphere is centred on the tracker given by the sph block's ``center_tracker``;
before that tracker is enabled the sphere sits at the origin, so the plot step
only uses times from --t-min on (default: the merger).

Required parfile output blocks (one block per variable, any number of radii):
    variable = mhd_w_bcc, adm, z4c_alpha, z4c_betax, z4c_betay, z4c_betaz
    center_tracker = <index of the black hole tracker>

Outputs (written to --output-dir):
    mad_r{R}.txt          time, Mdot, Mdot_in, Mdot_out, Phi, Phi_N, Phi_S  [code units]
    sphere_center_mad.txt t, center of the extraction surface                 [M_sun]
    mad_derived_r{R}.txt  (plot step) time, M_BH, <Mdot> [M_sun/s], phi_HL, phi_G,
                          phi_G from <Mdot_in>, <L_nu> [erg/s], eta
    fig_mad.png           (plot step) <Mdot>, phi and eta vs time, one line per radius
    fig_eta_mdot.png      (plot step) eta vs <Mdot>, coloured by time

Command line:
    kplot-sphere-mad --sph-dir DIR [--sph-dir DIR ...] --output-dir DIR \
        --radius 25 --radius 50 --radius 100 \
        [--plot --horizon-file F --lnu-file F --lnu-radius 300 --t-merger T]
    kplot-sphere-mad --output-dir DIR --radius 25 --plot-only --horizon-file F ...
"""

import os

import numpy as np

from ..utils.cli import require_sph, sphere_parser
from ..utils.constants import CODE_POWER_CGS, MSUN_TO_MS, MSUN_TO_S
from ..utils.grmhd import fluid, sqrt_det
from ..utils.plotting import savefig, time_axis
from ..utils.surface import Surface, run, save_centers

DEFAULT_RADII = (25.0, 50.0, 100.0)
DEFAULT_JOBNAME = "bns"
DEFAULT_T_AVG_MS = 5.0

# Gaussian / Heaviside-Lorentz field units: B_G = sqrt(4pi) B_HL.
HL_TO_GAUSS = np.sqrt(4.0 * np.pi)

# MAD saturation of the dimensionless flux in Gaussian units (Tchekhovskoy et al.
# 2011); ~ 15 in Heaviside-Lorentz units.
PHI_MAD_GAUSS = 50.0

# NDAF reference efficiency for Mdot ~ 0.01 - 1 M_sun/s.
ETA_NDAF = 0.06
NDAF_MDOT_RANGE = (0.01, 1.0)   # [M_sun/s]

COLUMNS = ('Mdot', 'Mdot_in', 'Mdot_out', 'Phi', 'Phi_N', 'Phi_S')


def _tag(radius: float) -> str:
  """
  Radius as used in the output file names.

  Parameters:
  radius (float): Extraction radius [M_sun].

  Returns:
  The radius formatted with :g.
  """
  return f'{radius:g}'


# ===================================================================
# Main analysis
# ===================================================================

def _compute(s: Surface, ctx: None) -> dict:
  """
  Accretion rates and magnetic fluxes through the surface.

  Parameters:
  s (Surface): Extraction surface of one snapshot.
  ctx (None): Unused.

  Returns:
  {'Mdot', 'Mdot_in', 'Mdot_out', 'Phi', 'Phi_N', 'Phi_S'}.
  """
  f = fluid(s.metric, s.alpha, s.beta, s.vel)
  dA = s.dA
  # sqrt(-g) rho u^r, outward positive; bcc is already sqrt(gamma) B^i.
  mass_flux = sqrt_det(s.metric) * s.alpha * s['dens'] * s.radial(f['u_up']) * dA
  flux_B = s.radial(s.bcc) * dA
  north = s.theta < 0.5 * np.pi
  return {
    'Mdot':     -mass_flux.sum(),
    'Mdot_in':  -mass_flux[mass_flux < 0].sum(),
    'Mdot_out': mass_flux[mass_flux > 0].sum(),
    'Phi':      0.5 * np.abs(flux_B).sum(),
    'Phi_N':    flux_B[north].sum(),
    'Phi_S':    flux_B[~north].sum(),
  }


def analyze(sph_dirs: list[str], output_dir: str, radii: list[float] = DEFAULT_RADII,
            jobname: str = DEFAULT_JOBNAME, n_workers: int | None = None) -> None:
  """
  Run the accretion-rate / magnetic-flux analysis over every snapshot found in sph_dirs.

  Parameters:
  sph_dirs (list[str]): AthenaK output-XXXX/sph directories (all treated as one run).
  output_dir (str): Directory the outputs are written to.
  radii (list[float]): SPH extraction radii [M_sun]; each must have mhd_w_bcc/adm/z4c
                       output. A snapshot is used at every radius it holds, so the
                       radii may cover different time spans.
  jobname (str): AthenaK job name prefixing the VTK files.
  n_workers (int): Worker processes for the snapshot loop (default: min(8, cpu_count())).

  Returns:
  None.
  """
  results = run(sph_dirs, jobname, radii, _compute, n_workers=n_workers)
  os.makedirs(output_dir, exist_ok=True)

  # The center is the same for every radius of a snapshot; keep one row each.
  centers = {r['time']: r for res in results.values() for r in res}
  save_centers(os.path.join(output_dir, 'sphere_center_mad.txt'),
               [centers[t] for t in sorted(centers)])

  for radius, res in results.items():
    if not res:
      continue
    rows = np.array([[r['time']] + [r[k] for k in COLUMNS] for r in res])
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

def load_bh_mass(horizon_file: str) -> tuple[np.ndarray, np.ndarray]:
  """
  Black hole mass history from an AthenaK horizon summary, valid rows only.

  Parameters:
  horizon_file (str): Path to the horizon summary file.

  Returns:
  (time, M_BH) [M_sun].
  """
  data = np.loadtxt(horizon_file, comments='#')
  t, m = data[:, 1], data[:, 2]
  keep = np.isfinite(m) & (m > 0)
  t, m = t[keep], m[keep]
  t, first = np.unique(t, return_index=True)
  return t, m[first]


def running_mean(t: np.ndarray, y: np.ndarray, width: float) -> np.ndarray:
  """
  Centred running mean of y(t) over a window of width in t. The samples need
  not be equally spaced; NaNs are ignored.

  Parameters:
  t (np.ndarray): Sorted times.
  y (np.ndarray): Values.
  width (float): Window width (<= 0 returns y).

  Returns:
  The running mean.
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


def _load_mad(output_dir: str, radius: float) -> dict | None:
  """
  Load mad_r{radius}.txt.

  Parameters:
  output_dir (str): Analysis directory.
  radius (float): Extraction radius [M_sun].

  Returns:
  {'time', *COLUMNS}, or None if the file does not exist.
  """
  path = os.path.join(output_dir, f'mad_r{_tag(radius)}.txt')
  if not os.path.exists(path):
    return None
  data = np.atleast_2d(np.loadtxt(path))
  t, first = np.unique(data[:, 0], return_index=True)
  data = data[first]
  return {'time': t, **{k: data[:, i + 1] for i, k in enumerate(COLUMNS)}}


def derive(mad: dict, radius: float, t_bh: np.ndarray | None = None,
           m_bh: np.ndarray | None = None, m_bh_const: float | None = None,
           lnu: tuple[np.ndarray, np.ndarray] | None = None,
           lnu_radius: float | None = None, t_avg: float = 0.0) -> dict:
  """
  phi and eta on the snapshot times of mad (see the module docstring).

  Parameters:
  mad (dict): One radius of the analyze output, as loaded by _load_mad.
  radius (float): Extraction radius [M_sun].
  t_bh (np.ndarray): Times of the black hole mass history [M_sun].
  m_bh (np.ndarray): Black hole mass history [M_sun].
  m_bh_const (float): Fixed black hole mass [M_sun], used without t_bh/m_bh.
  lnu (tuple): Time [M_sun] and total neutrino luminosity [erg/s] at lnu_radius.
  lnu_radius (float): Extraction radius of lnu [M_sun].
  t_avg (float): Averaging window [M_sun].

  Returns:
  The derived time series (running means, phi, eta, ...).
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
    # phi is undefined where the sphere has net outflow.
    phi_hl = np.where(mdot > 0, Phi / np.sqrt(mdot * M**2), np.nan)
    phi_in = Phi / np.sqrt(mdot_in * M**2) * HL_TO_GAUSS

  eta = np.full_like(t, np.nan)
  lnu_ret = np.full_like(t, np.nan)
  if lnu is not None:
    t_nu, L_nu = lnu
    L_avg = running_mean(t_nu, L_nu, t_avg)
    # L_nu leaves R at t and reaches R_nu at t + (R_nu - R) (c = 1).
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


def plot_mad(output_dir: str, radii: list[float] = DEFAULT_RADII,
             horizon_file: str | None = None, m_bh: float | None = None,
             lnu_file: str | None = None, lnu_radius: float | None = None,
             t_merger: float | None = None, t_min: float | None = None,
             t_avg_ms: float = DEFAULT_T_AVG_MS) -> None:
  """
  Write mad_derived_r{R}.txt and plot <Mdot>, phi and eta (fig_mad.png) and eta vs
  <Mdot> (fig_eta_mdot.png).

  Parameters:
  output_dir (str): Analysis directory; the outputs go there too.
  radii (list[float]): Extraction radii [M_sun].
  horizon_file (str): AthenaK horizon summary for the black hole mass.
  m_bh (float): Fixed black hole mass [M_sun] if no horizon file is given.
  lnu_file (str): Total neutrino luminosity file for eta.
  lnu_radius (float): Extraction radius of lnu_file [M_sun].
  t_merger (float): Merger time [M_sun]; the time axis is then t - t_merger [ms].
  t_min (float): Only use times >= t_min [M_sun] (default: t_merger).
  t_avg_ms (float): Averaging window [ms].

  Returns:
  None.
  """
  from .plots import plt
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
      print(f'  No neutrino luminosity at {lnu_file}; skipping eta.')

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
            f'M_sun/s, median phi_G = {np.nanmedian(d["phi_G"]):.3g}, max phi_G = '
            f'{np.nanmax(d["phi_G"]) if np.isfinite(d["phi_G"]).any() else np.nan:.3g}, '
            f'median eta = {np.nanmedian(d["eta"]) if np.isfinite(d["eta"]).any() else np.nan:.3g}'
            f' -> {path}')

  if not derived:
    print('  Nothing to plot.')
    return

  t_ms, xlabel = time_axis(t_merger)

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
  savefig(fig, os.path.join(output_dir, 'fig_mad.png'))

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
  savefig(fig, os.path.join(output_dir, 'fig_eta_mdot.png'), tight=False)


def main(argv: list[str] | None = None) -> None:
  """
  Command-line entry point of the MAD analysis.

  Parameters:
  argv (list[str]): Command-line arguments (default: sys.argv).

  Returns:
  None.
  """
  p = sphere_parser(__doc__, radii=DEFAULT_RADII, plot=True)
  p.add_argument("--horizon-file", default=None,
                  help="AthenaK horizon summary (<job>.horizon_summary_0.txt) giving "
                       "M_BH(t) for r_g.")
  p.add_argument("--m-bh", type=float, default=None,
                  help="Fixed black hole mass [M_sun] instead of --horizon-file.")
  p.add_argument("--lnu-file", default=None,
                  help="Lnu_E_total.txt of kplot-sphere-neutrinos; without it eta is "
                       "skipped.")
  p.add_argument("--lnu-radius", type=float, default=None,
                  help="Extraction radius [M_sun] of --lnu-file.")
  p.add_argument("--t-min", type=float, default=None,
                  help="First time [M_sun] used in the plot step. Default: --t-merger.")
  p.add_argument("--t-avg", type=float, default=DEFAULT_T_AVG_MS,
                  help=f"Averaging window [ms] for Mdot, Phi and L_nu. "
                       f"Default: {DEFAULT_T_AVG_MS:g}.")
  args = p.parse_args(argv)
  require_sph(p, args)
  radii = args.radii or list(DEFAULT_RADII)

  if not args.plot_only:
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

"""
Ejecta analysis on an AthenaK spherical extraction surface: Mej_rate + 2D
ejecta histogram (vinf x theta).

Per snapshot, loads 6 VTK files:
  <job>.r=R.mhd_w_bcc.NNNNN.vtk - primitives dens, s_00, T (primitive variables)
  <job>.r=R.adm.NNNNN.vtk       - adm_gxx..gyz             (ADM 3-metric)
  <job>.r=R.z4c_alpha.NNNNN.vtk - z4c_alpha                (lapse)
  <job>.r=R.z4c_beta*.NNNNN.vtk - z4c_beta*                (shift)

The 2D histogram dMej[i_v, i_theta] unifies the former separate vinf histogram
and rhoej_theta outputs: its marginals reproduce both.

Outputs (written to --output-dir):
  Mej_rate.txt, Mej_rate_Ber.txt, Mej_rate_geo.txt
  sphere_center.txt                            t, center of the extraction surface
  Mej_vinf_geo.txt, Mej_vinf_Ber.txt          (vinf marginal, dM per v-bin)
  rhoej_theta_Ber.txt, rhoej_theta_geo.txt     (theta marginal, dM/dtheta)
  Mej_vinf_theta_geo.npy, Mej_vinf_theta_Ber.npy   (full 2D arrays, shape Nv x Ntheta)
  vinf_centers.txt, theta_centers_2d.txt             (axis labels for 2D arrays)
  Ye_avg_geo.txt, Ye_avg_Ber.txt, Mej_Ye_geo.txt, Mej_Ye_Ber.txt, Ye_centers.txt
  Mej_Ye_theta_geo.npy, Mej_Ye_theta_Ber.npy   (Ye x theta 2D arrays, shape NYe x Ntheta;
                                                 axes: Ye_centers.txt, theta_centers_2d.txt)

Command line:
  kplot-sphere-ejecta --sph-dir DIR [--sph-dir DIR ...] --eos-table TABLE.h5 \
    --output-dir DIR --radius 300 [--t-merger T --t-post-ms 25] \
    [--per_iteration_out True]
"""

import os

import numpy as np
from scipy.integrate import simpson

from ..utils.cli import bool_arg, sphere_parser
from ..utils.constants import MSUN_TO_S
from ..utils.eos import enthalpy, load_eos
from ..utils.grmhd import fluid, sqrt_det
from ..utils.quadrature import sum_over_time
from ..utils.surface import Surface, run, save_centers, stack

DEFAULT_RADIUS = 300.0
DEFAULT_DFLOOR = 3e-15
DEFAULT_JOBNAME = "bns"
DEFAULT_DV = 0.01
DEFAULT_DYE = 0.01


def velocity_bins(dv: float = DEFAULT_DV) -> np.ndarray:
  """
  Bin edges for vinf [c] on [0, 1].

  Parameters:
  dv (float): Bin width.

  Returns:
  The bin edges.
  """
  return np.arange(0.0, 1.0 + dv, dv)


def ye_bins(dye: float = DEFAULT_DYE) -> np.ndarray:
  """
  Bin edges for the electron fraction Y_e on [0, 0.6].

  Parameters:
  dye (float): Bin width.

  Returns:
  The bin edges.
  """
  return np.arange(0.0, 0.6 + dye, dye)


def bin_centers(edges: np.ndarray) -> np.ndarray:
  """
  Centers of the bins.

  Parameters:
  edges (np.ndarray): Bin edges.

  Returns:
  The bin centers.
  """
  return 0.5 * (edges[:-1] + edges[1:])


# ===================================================================
# Main analysis
# ===================================================================

def _compute(s: Surface, ctx: dict) -> dict:
  """
  Ejecta rates, mean properties and histograms on the surface, for the
  geodesic and Bernoulli criteria.

  Parameters:
  s (Surface): Extraction surface of one snapshot.
  ctx (dict): 'eos', 'dfloor', 'v_bins', 'ye_bins' and 't_stop'.

  Returns:
  The per-snapshot results, keyed by quantity and criterion.
  """
  eos = ctx['eos']
  dens, ye = s['dens'], s['s_00']
  f = fluid(s.metric, s.alpha, s.beta, s.vel)
  ur, ut_cov = s.radial(f['u_up']), f['u_t']
  hut_cov = enthalpy(eos, dens, ye, s['temperature']) * ut_cov

  mask_out = (ur > 0) & (dens > ctx['dfloor'])
  mask_Ber = (hut_cov < -eos['h_min']) & mask_out
  mask_geo = (ut_cov < -1) & mask_out

  cell_rate = dens * ur * sqrt_det(s.metric) * s.alpha * s.dA
  flux_rate_geo = np.where(mask_geo, cell_rate, 0.0)
  flux_rate_Ber = np.where(mask_Ber, cell_rate, 0.0)
  total_geo = flux_rate_geo.sum()
  total_Ber = flux_rate_Ber.sum()

  vinf_geo = np.sqrt(np.where(ut_cov < -1, 1.0 - ut_cov**-2, 0.0))
  vinf_Ber = np.sqrt(np.where(hut_cov < -eos['h_min'],
                              1.0 - (eos['h_min'] / hut_cov)**2, 0.0))

  def avg(x: np.ndarray, w: np.ndarray, tot: float) -> float:
    """
    Weighted mean of x (0 if the total weight is 0).

    Parameters:
    x (np.ndarray): Values.
    w (np.ndarray): Weights.
    tot (float): Sum of the weights.

    Returns:
    The weighted mean.
    """
    return np.sum(x * w) / tot if tot > 0 else 0.0

  result = {
    'theta':        s.theta_1d,
    'Mej_rate':     np.where(mask_out, cell_rate, 0.0).sum(),
    'Mej_rate_geo': total_geo,
    'Mej_rate_Ber': total_Ber,
    'vinf_avg_geo': avg(vinf_geo, flux_rate_geo, total_geo),
    'vinf_avg_Ber': avg(vinf_Ber, flux_rate_Ber, total_Ber),
    'ye_avg_geo':   avg(ye, flux_rate_geo, total_geo),
    'ye_avg_Ber':   avg(ye, flux_rate_Ber, total_Ber),
    'in_2d':        s.time <= ctx['t_stop'],
  }
  if not result['in_2d']:
    return result

  theta = s.theta.ravel()
  theta_edges = s.theta_edges
  for crit, sel, vinf, w in (('geo', ut_cov < -1, vinf_geo, flux_rate_geo),
                             ('Ber', hut_cov < -eos['h_min'], vinf_Ber, flux_rate_Ber)):
    sel, w, y = sel.ravel(), w.ravel()[sel.ravel()], ye.ravel()[sel.ravel()]
    result[f'hist2d_{crit}'] = np.histogram2d(vinf.ravel()[sel], theta[sel],
                                              bins=[ctx['v_bins'], theta_edges], weights=w)[0]
    result[f'hist_ye_{crit}'] = np.histogram(y, bins=ctx['ye_bins'], weights=w)[0]
    result[f'hist2d_ye_theta_{crit}'] = np.histogram2d(y, theta[sel],
                                                       bins=[ctx['ye_bins'], theta_edges],
                                                       weights=w)[0]
  return result


def analyze(sph_dirs: list[str], eos_table: str, output_dir: str,
            radius: float = DEFAULT_RADIUS, dfloor: float = DEFAULT_DFLOOR,
            t_stop: float = np.inf, jobname: str = DEFAULT_JOBNAME,
            n_workers: int | None = None, dv: float = DEFAULT_DV, dye: float = DEFAULT_DYE,
            per_iteration_out: bool = False) -> None:
  """
  Run the ejecta analysis over every snapshot found in sph_dirs.

  Parameters:
  sph_dirs (list[str]): AthenaK output-XXXX/sph directories (all treated as one run).
  eos_table (str): Path to the PyCompOSE HDF5 EOS table (must match the simulation EOS).
  output_dir (str): Directory the outputs are written to.
  radius (float): SPH extraction radius [M_sun]; must have mhd_w_bcc/adm/z4c output.
  dfloor (float): Density floor used in the simulation [code units].
  t_stop (float): Include snapshots up to this time [M_sun] in the 2D histogram; the
                  Mej_rate time series always accumulates every snapshot.
  jobname (str): AthenaK job name prefixing the VTK files.
  n_workers (int): Worker processes for the snapshot loop (default: min(8, cpu_count())).
  dv (float): Bin width of vinf [c].
  dye (float): Bin width of Y_e.
  per_iteration_out (bool): Also write the histograms of every snapshot.

  Returns:
  None.
  """
  v_bins     = velocity_bins(dv)
  v_centers  = bin_centers(v_bins)
  ye_bins_   = ye_bins(dye)
  ye_centers = bin_centers(ye_bins_)

  ctx = dict(eos=load_eos(eos_table), v_bins=v_bins, ye_bins=ye_bins_,
             dfloor=dfloor, t_stop=t_stop)
  results = run(sph_dirs, jobname, radius, _compute, ctx=ctx,
                n_workers=n_workers)[float(radius)]
  theta_centers = np.sort(results[0]['theta'])
  theta_edges = np.concatenate([[0.0], 0.5 * (theta_centers[:-1] + theta_centers[1:]),
                                [np.pi]])
  os.makedirs(output_dir, exist_ok=True)
  save_centers(os.path.join(output_dir, 'sphere_center.txt'), results)

  time_arr         = stack(results, 'time')
  Mej_rate_arr     = stack(results, 'Mej_rate')
  Mej_rate_geo_arr = stack(results, 'Mej_rate_geo')
  Mej_rate_Ber_arr = stack(results, 'Mej_rate_Ber')
  vinf_avg_geo_arr = stack(results, 'vinf_avg_geo')
  vinf_avg_Ber_arr = stack(results, 'vinf_avg_Ber')
  ye_avg_geo_arr   = stack(results, 'ye_avg_geo')
  ye_avg_Ber_arr   = stack(results, 'ye_avg_Ber')

  res_2d           = [r for r in results if r['in_2d']]
  if len(res_2d) < 2:
    raise RuntimeError(
      f"Only {len(res_2d)} of {len(results)} snapshots fall at or below "
      f"t_stop = {t_stop:g} M_sun, but the 2D histogram needs at least 2 "
      f"to integrate over time. Snapshots span "
      f"t = {time_arr.min():g} .. {time_arr.max():g} M_sun. "
      f"Check the merger time and the post-merger window (--t-post-ms).")
  time_2d_arr             = stack(res_2d, 'time')
  hist2d_geo_arr          = stack(res_2d, 'hist2d_geo')
  hist2d_Ber_arr          = stack(res_2d, 'hist2d_Ber')
  hist_ye_geo_arr         = stack(res_2d, 'hist_ye_geo')
  hist_ye_Ber_arr         = stack(res_2d, 'hist_ye_Ber')
  hist2d_ye_theta_geo_arr = stack(res_2d, 'hist2d_ye_theta_geo')
  hist2d_ye_theta_Ber_arr = stack(res_2d, 'hist2d_ye_theta_Ber')

  # ------------------------------------------------------------------
  # Post-processing
  # ------------------------------------------------------------------
  dMej_2d_geo = sum_over_time(hist2d_geo_arr, time_2d_arr)   # (Nv, Ntheta)
  dMej_2d_Ber = sum_over_time(hist2d_Ber_arr, time_2d_arr)

  dMej_Ye_geo = sum_over_time(hist_ye_geo_arr, time_2d_arr)  # (NYe,)
  dMej_Ye_Ber = sum_over_time(hist_ye_Ber_arr, time_2d_arr)

  dMej_Ye_theta_geo = sum_over_time(hist2d_ye_theta_geo_arr, time_2d_arr)  # (NYe, Ntheta)
  dMej_Ye_theta_Ber = sum_over_time(hist2d_ye_theta_Ber_arr, time_2d_arr)

  # Marginals
  dMej_geo = dMej_2d_geo.sum(axis=1)          # (Nv,)  - vinf histogram
  dMej_Ber = dMej_2d_Ber.sum(axis=1)

  dtheta = np.diff(theta_edges)                # (Ntheta,)
  rhoej_theta_geo = dMej_2d_geo.sum(axis=0) / dtheta   # dM/dtheta [Msun/rad]
  rhoej_theta_Ber = dMej_2d_Ber.sum(axis=0) / dtheta

  # Total ejecta mass via time-integrated Mej_rate
  Mej     = simpson(y=Mej_rate_arr,     x=time_arr)
  Mej_Ber = simpson(y=Mej_rate_Ber_arr, x=time_arr)
  Mej_geo = simpson(y=Mej_rate_geo_arr, x=time_arr)

  # Per iteration output (currently only Ye) if needed
  if per_iteration_out:
    output_dir_iter = os.path.join(output_dir, 'iteration')
    os.makedirs(os.path.join(output_dir_iter, 'Mej_Ye'), exist_ok=True)
    os.makedirs(os.path.join(output_dir_iter, 'Mej_Ye_theta'), exist_ok=True)

    for i, time in enumerate(time_2d_arr):
      np.savetxt(os.path.join(output_dir_iter, 'Mej_Ye', f'Mej_Ye_geo_{int(time):05d}.txt'),
        np.column_stack((ye_centers, hist_ye_geo_arr[i])),
        header=f'Ye    Mej_per_bin[Msun]  (geodesic: u_t < -1) - T={time}',
        fmt='%.6e')
      np.savetxt(os.path.join(output_dir_iter, 'Mej_Ye', f'Mej_Ye_Ber_{int(time):05d}.txt'),
        np.column_stack((ye_centers, hist_ye_Ber_arr[i])),
        header=f'Ye    Mej_per_bin[Msun]  (Bernoulli: h*u_t < -1) - T={time}',
        fmt='%.6e')
      np.save(os.path.join(output_dir_iter, 'Mej_Ye_theta', f'Mej_Ye_theta_geo_{int(time)}.npy'),
                            hist2d_ye_theta_geo_arr[i])
      np.save(os.path.join(output_dir_iter, 'Mej_Ye_theta', f'Mej_Ye_theta_Ber_{int(time)}.npy'),
                            hist2d_ye_theta_Ber_arr[i])

  print(f'Total Mej:     {Mej:.6e} Msun')
  print(f'Total Mej_Ber: {Mej_Ber:.6e} Msun')
  print(f'Total Mej_geo: {Mej_geo:.6e} Msun')
  print(f'Total Mej (geodesic,  vinf): {np.sum(dMej_geo):.6e} Msun')
  print(f'Total Mej (Bernoulli, vinf): {np.sum(dMej_Ber):.6e} Msun')

  # ------------------------------------------------------------------
  # Save outputs
  # ------------------------------------------------------------------
  # -- Mej_rate --
  with open(os.path.join(output_dir, 'Mej_rate.txt'), 'w') as f:
    f.write('# time(Msun)    Mej_rate(Msun/Msun)\n')
    for t, rate in zip(time_arr, Mej_rate_arr):
      f.write(f'{t:.6e}    {rate:.6e}\n')

  with open(os.path.join(output_dir, 'Mej_rate_Ber.txt'), 'w') as f:
    f.write('# time(Msun)    Mej_rate_Ber(Msun/Msun)\n')
    for t, rate in zip(time_arr, Mej_rate_Ber_arr):
      f.write(f'{t:.6e}    {rate:.6e}\n')

  with open(os.path.join(output_dir, 'Mej_rate_geo.txt'), 'w') as f:
    f.write('# time(Msun)    Mej_rate_geo(Msun/Msun)\n')
    for t, rate in zip(time_arr, Mej_rate_geo_arr):
      f.write(f'{t:.6e}    {rate:.6e}\n')

  # -- Mass-averaged vinf --
  np.savetxt(os.path.join(output_dir, 'vinf_avg_geo.txt'),
              np.column_stack((time_arr, vinf_avg_geo_arr)),
              header='time(Msun)    vinf_avg_geo[c]  (geodesic: u_t < -1)',
              fmt='%.6e')
  np.savetxt(os.path.join(output_dir, 'vinf_avg_Ber.txt'),
              np.column_stack((time_arr, vinf_avg_Ber_arr)),
              header='time(Msun)    vinf_avg_Ber[c]  (Bernoulli: h*u_t < -1)',
              fmt='%.6e')

  # -- vinf (marginal over theta) --
  np.savetxt(os.path.join(output_dir, 'Mej_vinf_geo.txt'),
              np.column_stack((v_centers, dMej_geo)),
              header='vinf[c]    Mej_per_bin[Msun]  (geodesic: u_t < -1)',
              fmt='%.6e')
  np.savetxt(os.path.join(output_dir, 'Mej_vinf_Ber.txt'),
              np.column_stack((v_centers, dMej_Ber)),
              header='vinf[c]    Mej_per_bin[Msun]  (Bernoulli: h*u_t < -1)',
              fmt='%.6e')

  # -- rhoej_theta (marginal over vinf, divided by dtheta) --
  np.savetxt(os.path.join(output_dir, 'rhoej_theta_geo.txt'),
              np.column_stack((theta_centers, rhoej_theta_geo)),
              header='theta(rad)    rho_ej_geo(dM/dtheta)[Msun/rad]', fmt='%.6e')
  np.savetxt(os.path.join(output_dir, 'rhoej_theta_Ber.txt'),
              np.column_stack((theta_centers, rhoej_theta_Ber)),
              header='theta(rad)    rho_ej_Ber(dM/dtheta)[Msun/rad]', fmt='%.6e')

  # -- 2D arrays (Nv x Ntheta) --
  np.save(os.path.join(output_dir, 'Mej_vinf_theta_geo.npy'), dMej_2d_geo)
  np.save(os.path.join(output_dir, 'Mej_vinf_theta_Ber.npy'), dMej_2d_Ber)
  np.savetxt(os.path.join(output_dir, 'vinf_centers.txt'), v_centers,
              header='vinf bin centers [c]', fmt='%.6e')
  np.savetxt(os.path.join(output_dir, 'theta_centers_2d.txt'), theta_centers,
              header='theta bin centers [rad]', fmt='%.6e')

  # -- Y_e outputs --
  np.savetxt(os.path.join(output_dir, 'Ye_avg_geo.txt'),
              np.column_stack((time_arr, ye_avg_geo_arr)),
              header='time(Msun)    Ye_avg_geo  (mass-weighted, geodesic: u_t < -1)',
              fmt='%.6e')
  np.savetxt(os.path.join(output_dir, 'Ye_avg_Ber.txt'),
              np.column_stack((time_arr, ye_avg_Ber_arr)),
              header='time(Msun)    Ye_avg_Ber  (mass-weighted, Bernoulli: h*u_t < -1)',
              fmt='%.6e')
  np.savetxt(os.path.join(output_dir, 'Mej_Ye_geo.txt'),
              np.column_stack((ye_centers, dMej_Ye_geo)),
              header='Ye    Mej_per_bin[Msun]  (geodesic: u_t < -1)',
              fmt='%.6e')
  np.savetxt(os.path.join(output_dir, 'Mej_Ye_Ber.txt'),
              np.column_stack((ye_centers, dMej_Ye_Ber)),
              header='Ye    Mej_per_bin[Msun]  (Bernoulli: h*u_t < -1)',
              fmt='%.6e')
  np.savetxt(os.path.join(output_dir, 'Ye_centers.txt'), ye_centers,
              header='Ye bin centers', fmt='%.6e')

  # -- 2D arrays (NYe x Ntheta) -- axes: Ye_centers.txt, theta_centers_2d.txt
  np.save(os.path.join(output_dir, 'Mej_Ye_theta_geo.npy'), dMej_Ye_theta_geo)
  np.save(os.path.join(output_dir, 'Mej_Ye_theta_Ber.npy'), dMej_Ye_theta_Ber)

  print(f'All outputs saved to {output_dir}')


def main(argv: list[str] | None = None) -> None:
  """
  Command-line entry point of the ejecta analysis.

  Parameters:
  argv (list[str]): Command-line arguments (default: sys.argv).

  Returns:
  None.
  """
  p = sphere_parser(__doc__, radius=DEFAULT_RADIUS)
  p.add_argument("--eos-table", required=True,
                  help="PyCompOSE HDF5 EOS table matching the simulation EOS.")
  p.add_argument("--dfloor", type=float, default=DEFAULT_DFLOOR,
                  help=f"Simulation density floor [code units]. "
                      f"Default: {DEFAULT_DFLOOR:g}.")
  p.add_argument("--t-merger", type=float, default=None,
                  help="Merger time [M_sun]; with --t-post-ms it truncates the 2D "
                      "histogram. Default: no truncation.")
  p.add_argument("--t-post-ms", type=float, default=25.0,
                  help="Post-merger window [ms] included in the 2D histogram "
                      "(needs --t-merger). Default: 25.")
  p.add_argument("--dv", type=float, default=DEFAULT_DV,
                  help=f"vinf bin width [c]. Default: {DEFAULT_DV:g}.")
  p.add_argument("--dye", type=float, default=DEFAULT_DYE,
                  help=f"Y_e bin width. Default: {DEFAULT_DYE:g}.")
  p.add_argument("--per_iteration_out", type=bool_arg, nargs='?', const=True,
                  default=False,
                  help="Y_e hist per iteration output. Default: False.")
  args = p.parse_args(argv)

  if args.t_merger is None:
    t_stop = np.inf
  else:
    t_stop = args.t_merger + args.t_post_ms * 1e-3 / MSUN_TO_S

  analyze(args.sph_dirs, args.eos_table, args.output_dir,
          radius=args.radius, dfloor=args.dfloor, t_stop=t_stop,
          jobname=args.jobname, n_workers=args.n_workers,
          dv=args.dv, dye=args.dye, per_iteration_out=args.per_iteration_out)


if __name__ == '__main__':
  main()

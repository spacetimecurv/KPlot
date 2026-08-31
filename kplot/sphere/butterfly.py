"""
Butterfly diagram analysis on an AthenaK spherical extraction surface.

Each snapshot's VTK files are loaded once and all shared intermediate
quantities (W, u_t, u^r, magnetic field in the fluid frame) are computed once.

Density-weighted, azimuthally-averaged toroidal (comoving b^phi) magnetic
field as a function of theta and time, revealing dynamo-driven polarity
reversals (cf. arXiv:2211.07158).

Outputs (written to --output-dir):
  butterfly_bphi.npy               (density-weighted <b^phi>, shape Nt x Ntheta; axes:
                                     time_sph_butterfly.txt, theta_centers_sph_butterfly.txt)
  sphere_center_butterfly.txt      t, center of the extraction surface

Command line:
    kplot-sphere-butterfly --sph-dir DIR [--sph-dir DIR ...] \
        --output-dir DIR --radius 300
"""

import argparse
import os
from multiprocessing import Pool, cpu_count

import numpy as np

from ._files import Shell, locate

DEFAULT_RADIUS = 300.0
DEFAULT_JOBNAME = "bns"

# The sph output variables a snapshot needs, in the order the worker unpacks them.
VARIABLES = ('mhd_w_bcc', 'adm', 'z4c_alpha', 'z4c_betax', 'z4c_betay', 'z4c_betaz')


# ===================================================================
# Main analysis
# ===================================================================

def _process_snapshot_butterfly(args):
  """Per-snapshot worker for the butterfly diagram analysis."""
  (mhd_file, adm_file, alpha_file, betax_file, betay_file, betaz_file,
    theta_1d, phi_1d) = args

  import numpy as _np

  import os as _os

  from ._files import Shell as _Shell, CENTER_TOL as _CENTER_TOL
  from ._integrate import (calc_uphi as _calc_uphi,
                           sqrt_det_metric_adm as _sqrt_det,
                           integrate_over_phi_riemann as _integrate_over_phi_riemann)

  # Every entry is (path, surface index): a dump may hold more than one radius.
  mhd   = _Shell(*mhd_file)
  adm   = _Shell(*adm_file)
  alpha = _Shell(*alpha_file)
  betax = _Shell(*betax_file)
  betay = _Shell(*betay_file)
  betaz = _Shell(*betaz_file)

  time  = mhd.time
  theta = mhd.theta
  phi   = mhd.phi

  shells  = (mhd, adm, alpha, betax, betay, betaz)
  centers = _np.array([sh.center for sh in shells])
  if not _np.allclose(centers, centers[0], rtol=0.0, atol=_CENTER_TOL):
    detail = '\n'.join(f'  {_os.path.basename(sh.path)}: '
                       f'({c[0]:.6g}, {c[1]:.6g}, {c[2]:.6g})'
                       for sh, c in zip(shells, centers))
    raise RuntimeError(
      f'Sphere centers disagree at t = {time:g} M_sun:\n{detail}\n'
      'Every sph output block feeding the butterfly diagram analysis must use the '
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

  W = _np.sqrt(1.0 + gxx*velx**2 + gyy*vely**2 + gzz*velz**2
                + 2*gxy*velx*vely + 2*gxz*velx*velz + 2*gyz*vely*velz)
  ut_contrav = W / z4c_alpha
  ux_u = velx - z4c_betax * ut_contrav
  uy_u = vely - z4c_betay * ut_contrav

  # ------- TOROIDAL FIELD (fluid frame) ------- (adapted from plot-tools)
  sqrt_det = _sqrt_det(gxx, gyy, gzz, gxy, gxz, gyz, 1.0)
  Bx = bcc1 / sqrt_det
  By = bcc2 / sqrt_det
  Bz = bcc3 / sqrt_det

  BWv = gxx * Bx * velx + gyy * By * vely + gzz * Bz * velz + \
        gxy * (Bx * vely + velx * By) + \
        gxz * (Bx * velz + velx * Bz) + \
        gyz * (By * velz + vely * Bz)

  bx_u = (Bx + BWv * ux_u) / W
  by_u = (By + BWv * uy_u) / W

  bphi_u = _calc_uphi(bx_u, by_u, phi)

  # Density-weighted, azimuthally-averaged toroidal field: <b^phi>(theta)
  num = _integrate_over_phi_riemann(bphi_u * dens, theta_1d, phi_1d)
  den = _integrate_over_phi_riemann(dens, theta_1d, phi_1d)
  bphi_theta = num / den

  return {
    'time':        time,
    'center':      centers[0],
    'bphi_theta':  bphi_theta,
  }


def analyze(sph_dirs, output_dir, radius=DEFAULT_RADIUS, jobname=DEFAULT_JOBNAME,
            n_workers=None):
  """Run the butterfly diagram analysis over every snapshot found in `sph_dirs`.

  Parameters
  ----------
  sph_dirs : list of str
    AthenaK ``output-XXXX/sph`` directories to scan (all treated as one run).
  output_dir : str
    Directory the .txt/.npy outputs are written to.
  radius : float
    SPH extraction radius [M_sun]; must have mhd_w_bcc/adm/z4c output.  A dump
    bundling several radii is accepted, and the surface at `radius` is used.
  jobname : str
    AthenaK job name prefixing the VTK files (``<jobname>.r=R....vtk``, or
    ``<jobname>.r=RMIN-RMAX....vtk`` when one dump holds several radii).
  n_workers : int, optional
    Worker processes for the snapshot loop.  Default: min(8, cpu_count()).
  """
  if n_workers is None:
    n_workers = min(8, cpu_count())

  # ------------------------------------------------------------------
  # Locate the dumps holding `radius`.  An output block may write one surface
  # per file or bundle several radii into one file, and the two forms can be
  # mixed between variables within a run, so every variable is looked up on
  # its own.
  # ------------------------------------------------------------------
  located = {v: locate(sph_dirs, jobname, v, radius) for v in VARIABLES}
  for v in VARIABLES:
    print(f"  {located[v].describe()}")

  index_sets = [set(located[v].paths) for v in VARIABLES]
  indices = sorted(set.intersection(*index_sets))
  if not indices:
    raise RuntimeError(
      f"No snapshot in {list(sph_dirs)} has all of {', '.join(VARIABLES)} "
      f"at r = {radius:g}.\n"
      "Check the sph directories, the job name and the extraction radius.")
  incomplete = len(set.union(*index_sets)) - len(indices)
  if incomplete:
    print(f"  [skip] {incomplete} snapshots missing one of {', '.join(VARIABLES)}")
  print(f"Found {len(indices)} snapshots.")

  # Angular grid from first snapshot's geometry
  _first_mhd = Shell(*located['mhd_w_bcc'].shell(indices[0]))
  theta_1d = _first_mhd.theta[0, :]
  phi_1d   = _first_mhd.phi[:, 0]
  del _first_mhd

  # ------------------------------------------------------------------
  # Build worker argument list
  # ------------------------------------------------------------------
  def _snapshot_files(idx):
    """(path, surface index) of every variable of snapshot `idx`."""
    return tuple(located[v].shell(idx) for v in VARIABLES)

  worker_args = [
    _snapshot_files(i) + (theta_1d, phi_1d)
    for i in indices
  ]
  print(f"Processing {len(worker_args)} snapshots with n_workers={n_workers}...")

  # ------------------------------------------------------------------
  # Parallel snapshot loop
  # ------------------------------------------------------------------
  if n_workers > 1:
    with Pool(n_workers) as pool:
      results = list(pool.imap_unordered(
          _process_snapshot_butterfly, worker_args,
          chunksize=max(1, len(worker_args) // (n_workers * 4))))
  else:
    results = [_process_snapshot_butterfly(a) for a in worker_args]

  results.sort(key=lambda r: r['time'])

  # ------------------------------------------------------------------
  # Unpack results
  # ------------------------------------------------------------------
  time_arr        = np.array([r['time']       for r in results])
  bphi_theta_arr  = np.array([r['bphi_theta'] for r in results])  # (Nt, Ntheta)
  center_arr      = np.array([r['center']     for r in results])  # (Nt, 3)

  center_shift = np.linalg.norm(center_arr - center_arr[0], axis=1).max()
  print(f"Sphere center: ({center_arr[0][0]:.6g}, {center_arr[0][1]:.6g}, "
        f"{center_arr[0][2]:.6g}) -> ({center_arr[-1][0]:.6g}, "
        f"{center_arr[-1][1]:.6g}, {center_arr[-1][2]:.6g}) M_sun, "
        f"max excursion {center_shift:.6g}")

  print(f'Peak |<b^phi>|: {np.abs(bphi_theta_arr).max():.6e} (code units)')

  # ------------------------------------------------------------------
  # Save outputs
  # ------------------------------------------------------------------
  os.makedirs(output_dir, exist_ok=True)

  np.savetxt(os.path.join(output_dir, 'sphere_center_butterfly.txt'),
             np.column_stack((time_arr, center_arr)),
             header='time(Msun)    xc(Msun)    yc(Msun)    zc(Msun)',
             fmt='%.6e')

  # -- Butterfly diagram (Nt x Ntheta) -- axes: time_sph_butterfly.txt,
  #    theta_centers_sph_butterfly.txt.
  np.save(os.path.join(output_dir, 'butterfly_bphi.npy'), bphi_theta_arr)
  np.savetxt(os.path.join(output_dir, 'time_sph_butterfly.txt'), time_arr,
             header='snapshot times [Msun] (axis 0 of butterfly_bphi.npy)', fmt='%.6e')
  np.savetxt(os.path.join(output_dir, 'theta_centers_sph_butterfly.txt'), theta_1d,
             header='theta [rad] (axis 1 of butterfly_bphi.npy)', fmt='%.6e')

  print(f'All outputs saved to {output_dir}')


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--sph-dir", action="append", required=True, dest="sph_dirs",
                  metavar="DIR",
                  help="AthenaK output-XXXX/sph directory (repeat for each segment).")
  p.add_argument("--output-dir", required=True,
                  help="Directory for the .txt/.npy outputs.")
  p.add_argument("--radius", type=float, default=DEFAULT_RADIUS,
                  help=f"SPH extraction radius [M_sun]; picks the surface out of "
                      f"dumps holding several radii. Default: {DEFAULT_RADIUS:g}.")
  p.add_argument("--jobname", default=DEFAULT_JOBNAME,
                  help=f"AthenaK job name prefixing the VTK files. "
                      f"Default: {DEFAULT_JOBNAME}.")
  p.add_argument("--n-workers", type=int, default=None,
                  help="Worker processes for the snapshot loop. "
                      "Default: min(8, cpu_count()).")
  args = p.parse_args(argv)

  analyze(args.sph_dirs, args.output_dir, radius=args.radius,
          jobname=args.jobname, n_workers=args.n_workers)


if __name__ == '__main__':
  main()

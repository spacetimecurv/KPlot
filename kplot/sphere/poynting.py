"""
Poynting-flux analysis on an AthenaK spherical extraction surface.

Each snapshot's VTK files are loaded once and all shared intermediate
quantities (W, u_t, u^r, magnetic field in the fluid frame) are computed once.

Outputs (written to --output-dir):
  poynt_flux.txt                               (sphere-integrated Poynting luminosity vs time)
  poynt_flux_map.npy                           (dL/dOmega, shape Nt x Nphi x Ntheta; axes:
                                                 time_sph.txt, phi_centers_sph.txt,
                                                 theta_centers_sph.txt)
  sphere_center_poynting.txt                   t, center of the extraction surface

Command line:
    kplot-sphere-poynting --sph-dir DIR [--sph-dir DIR ...] \
        --output-dir DIR --radius 300
"""

import os

import numpy as np

from ..utils.cli import sphere_parser
from ..utils.grmhd import fluid, sqrt_det
from ..utils.surface import Surface, run, save_centers, stack

DEFAULT_RADIUS = 300.0
DEFAULT_JOBNAME = "bns"


def _compute(s: Surface, ctx: None) -> dict:
  """
  Poynting luminosity dL/dOmega on the surface and its integral.

  Parameters:
  s (Surface): Extraction surface of one snapshot.
  ctx (None): Unused.

  Returns:
  {'poynt_flux', 'poynt_map', 'theta', 'phi'}.
  """
  f = fluid(s.metric, s.alpha, s.beta, s.vel, s.bcc)
  # dL/dOmega (adapted from plot-tools).
  poynt_map = ((f['b2'] * s.radial(f['u_up']) * f['u_t'] - s.radial(f['b_up']) * f['b_t'])
               * s.alpha * sqrt_det(s.metric) * s.radius**2)
  return {'poynt_flux': (poynt_map * s.dA).sum() / s.radius**2, 'poynt_map': poynt_map,
          'theta': s.theta_1d, 'phi': s.phi_1d}


def analyze(sph_dirs: list[str], output_dir: str, radius: float = DEFAULT_RADIUS,
            jobname: str = DEFAULT_JOBNAME, n_workers: int | None = None) -> None:
  """
  Run the Poynting-flux analysis over every snapshot found in sph_dirs.

  Parameters:
  sph_dirs (list[str]): AthenaK output-XXXX/sph directories (all treated as one run).
  output_dir (str): Directory the outputs are written to.
  radius (float): SPH extraction radius [M_sun]; must have mhd_w_bcc/adm/z4c output.
  jobname (str): AthenaK job name prefixing the VTK files.
  n_workers (int): Worker processes for the snapshot loop (default: min(8, cpu_count())).

  Returns:
  None.
  """
  res = run(sph_dirs, jobname, radius, _compute, n_workers=n_workers)[float(radius)]
  time_arr       = stack(res, 'time')
  poynt_flux_arr = stack(res, 'poynt_flux')

  os.makedirs(output_dir, exist_ok=True)
  save_centers(os.path.join(output_dir, 'sphere_center_poynting.txt'), res)
  print(f'Peak Poynting luminosity: {poynt_flux_arr.max():.6e} (code units)')

  np.savetxt(os.path.join(output_dir, 'poynt_flux.txt'),
             np.column_stack((time_arr, poynt_flux_arr)),
             header='time(Msun)    L_Poynting(code units)', fmt='%.6e')

  # dL/dOmega (Nt x Nphi x Ntheta): integrating with sin(theta) dtheta dphi
  # reproduces poynt_flux.txt.
  np.save(os.path.join(output_dir, 'poynt_flux_map.npy'), stack(res, 'poynt_map'))
  np.savetxt(os.path.join(output_dir, 'time_sph.txt'), time_arr,
             header='snapshot times [Msun] (axis 0 of poynt_flux_map.npy)', fmt='%.6e')
  np.savetxt(os.path.join(output_dir, 'phi_centers_sph.txt'), res[0]['phi'],
             header='phi [rad] (axis 1 of poynt_flux_map.npy)', fmt='%.6e')
  np.savetxt(os.path.join(output_dir, 'theta_centers_sph.txt'), res[0]['theta'],
             header='theta [rad] (axis 2 of poynt_flux_map.npy)', fmt='%.6e')

  print(f'All outputs saved to {output_dir}')


def main(argv: list[str] | None = None) -> None:
  """
  Command-line entry point of the Poynting-flux analysis.

  Parameters:
  argv (list[str]): Command-line arguments (default: sys.argv).

  Returns:
  None.
  """
  args = sphere_parser(__doc__, radius=DEFAULT_RADIUS).parse_args(argv)
  analyze(args.sph_dirs, args.output_dir, radius=args.radius,
          jobname=args.jobname, n_workers=args.n_workers)


if __name__ == '__main__':
  main()

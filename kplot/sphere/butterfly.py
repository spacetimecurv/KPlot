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
  fig_butterfly.png                (with --plot / --plot-only) butterfly diagram vs.
                                     t - t_merger if --t-merger is given, else vs. t

Command line:
    kplot-sphere-butterfly --sph-dir DIR [--sph-dir DIR ...] \
        --output-dir DIR --radius 50 [--plot] [--t-merger T]
    kplot-sphere-butterfly --output-dir DIR --radius 50 --plot-only [--t-merger T]
"""

import os

import numpy as np

from ..utils.cli import require_sph, sphere_parser
from ..utils.grmhd import fluid
from ..utils.plotting import savefig, time_axis
from ..utils.quadrature import integrate_over_phi_riemann
from ..utils.surface import Surface, run, save_centers, stack

DEFAULT_RADIUS = 300.0
DEFAULT_JOBNAME = "bns"


def _compute(s: Surface, ctx: None) -> dict:
  """
  Density-weighted, azimuthally averaged toroidal field <b^phi>(theta).

  Parameters:
  s (Surface): Extraction surface of one snapshot.
  ctx (None): Unused.

  Returns:
  {'bphi_theta', 'theta'}.
  """
  f = fluid(s.metric, s.alpha, s.beta, s.vel, s.bcc)
  dens = s['dens']
  # Density-weighted, azimuthally-averaged toroidal field: <b^phi>(theta).
  num = integrate_over_phi_riemann(s.azimuthal(f['b_up']) * dens, s.theta_1d, s.phi_1d)
  den = integrate_over_phi_riemann(dens, s.theta_1d, s.phi_1d)
  return {'bphi_theta': num / den, 'theta': s.theta_1d}


def analyze(sph_dirs: list[str], output_dir: str, radius: float = DEFAULT_RADIUS,
            jobname: str = DEFAULT_JOBNAME, n_workers: int | None = None) -> None:
  """
  Run the butterfly diagram analysis over every snapshot found in sph_dirs.

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
  bphi_theta_arr = stack(res, 'bphi_theta')  # (Nt, Ntheta)

  os.makedirs(output_dir, exist_ok=True)
  save_centers(os.path.join(output_dir, 'sphere_center_butterfly.txt'), res)
  print(f'Peak |<b^phi>|: {np.abs(bphi_theta_arr).max():.6e} (code units)')

  np.save(os.path.join(output_dir, 'butterfly_bphi.npy'), bphi_theta_arr)
  np.savetxt(os.path.join(output_dir, 'time_sph_butterfly.txt'), time_arr,
             header='snapshot times [Msun] (axis 0 of butterfly_bphi.npy)', fmt='%.6e')
  np.savetxt(os.path.join(output_dir, 'theta_centers_sph_butterfly.txt'), res[0]['theta'],
             header='theta [rad] (axis 1 of butterfly_bphi.npy)', fmt='%.6e')

  print(f'All outputs saved to {output_dir}')


def plot_butterfly(output_dir: str, radius: float = DEFAULT_RADIUS,
                   t_merger: float | None = None) -> None:
  """
  Butterfly diagram: density-weighted <b^phi>(theta, t), cf. Hayashi et al.
  (arXiv:2211.07158). Diverging colormap centered at zero on a symlog scale,
  since the toroidal field reverses sign (dynamo polarity flips) across
  orders of magnitude.

  Parameters:
  output_dir (str): Directory holding the analysis outputs; the figure goes there too.
  radius (float): SPH extraction radius [M_sun].
  t_merger (float): Merger time [M_sun]; the time axis is then t - t_merger [ms].

  Returns:
  None.
  """
  from .plots import edges_from_centers, plt
  from matplotlib.colors import SymLogNorm

  path = os.path.join(output_dir, 'butterfly_bphi.npy')
  if not os.path.exists(path):
    print("  No butterfly diagram output found; skipping.")
    return

  bphi  = np.load(path) # (Nt, Ntheta)
  time  = np.loadtxt(os.path.join(output_dir, 'time_sph_butterfly.txt'))
  theta = np.loadtxt(os.path.join(output_dir, 'theta_centers_sph_butterfly.txt'))

  vmax = np.abs(bphi).max()
  if vmax <= 0:
    print("  Butterfly diagram is all zero; skipping.")
    return
  norm = SymLogNorm(linthresh=vmax / 1e3, vmin=-vmax, vmax=vmax, base=10)
  t_ms, xlabel = time_axis(t_merger)

  fig, ax = plt.subplots(figsize=(8, 4))
  mesh = ax.pcolormesh(edges_from_centers(t_ms(time)), edges_from_centers(theta),
                       bphi.T, cmap='RdBu_r', norm=norm, shading='flat')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(r'$\theta$')
  ax.set_ylim(0.0, np.pi)
  ax.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
  ax.set_yticklabels([r'$0$', r'$\frac{\pi}{4}$', r'$\pi/2$', r'$\frac{3\pi}{4}$', r'$\pi$'])

  cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
  cbar.set_label(r'$\langle b^\phi \rangle$  [code units]')
  savefig(fig, os.path.join(output_dir, 'fig_butterfly.png'))


def main(argv: list[str] | None = None) -> None:
  """
  Command-line entry point of the butterfly analysis.

  Parameters:
  argv (list[str]): Command-line arguments (default: sys.argv).

  Returns:
  None.
  """
  p = sphere_parser(__doc__, radius=DEFAULT_RADIUS, plot=True)
  args = p.parse_args(argv)
  require_sph(p, args)
  if not args.plot_only:
    analyze(args.sph_dirs, args.output_dir, radius=args.radius,
            jobname=args.jobname, n_workers=args.n_workers)
  if args.plot or args.plot_only:
    plot_butterfly(args.output_dir, radius=args.radius, t_merger=args.t_merger)


if __name__ == '__main__':
  main()

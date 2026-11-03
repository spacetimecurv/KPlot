"""
Neutrino luminosity analysis from rad_m1_F / rad_m1_E / rad_m1_N spherical-surface
VTK outputs of AthenaK BNS simulations.

Per snapshot, loads 8 VTK files:
  <job>.r=R.rad_m1_F.NNNNN.vtk  - Fx:s, Fy:s, Fz:s  (Eulerian energy-flux per species)
  <job>.r=R.rad_m1_E.NNNNN.vtk  - E:s                (Eulerian energy density per species)
  <job>.r=R.rad_m1_N.NNNNN.vtk  - N:s                (number density per species)
  <job>.r=R.adm.NNNNN.vtk       - adm_gxx..gyz       (ADM 3-metric)
  <job>.r=R.z4c_alpha.NNNNN.vtk - z4c_alpha           (lapse)
  <job>.r=R.z4c_beta*.NNNNN.vtk - z4c_beta*           (shift)

An output block writing several radii at once bundles them into one dump named
``<job>.r=RMIN-RMAX.<var>.NNNNN.vtk`` instead; --radius then picks the surface to
analyse out of the bundle (see :mod:`kplot.utils.sph`).  Each variable is looked
up on its own, so the radii may be bundled differently from one variable to the next.

Physics
-------
Energy luminosity per species s:
    L_E,s(t) = int (alpha F^r_s - beta^r E_s) * r^2 dOmega
             [code units -> erg/s]
where F^i_s = gamma^ij F_j,s, using the stored M1 flux covector components.
The radial projection uses the metric-normalized coordinate radial covector
used by THC: rhat_i = r_i / sqrt(g^ij_4 r_i r_j), where g^ij_4 is the
spatial block of the inverse 4-metric.
E_s and F_i,s are already densitized M1 evolved variables, so no extra sqrt(gamma)
factor is applied in the surface integral.
This matches the THC/AthenaK M1 calc_E_flux expression.
The angular area integral uses AthenaK's native SPH `weights` field:
    weights = r^2 dOmega  (Gauss-Legendre in mu = cos theta, uniform in phi)

Flux-weighted mean energy per species:
    <eps_s>(t) = int F^r_E,s * r^2 dOmega / int F^r_N,s * r^2 dOmega  -> MeV
THC constructs the denominator from the M1 number flux on the detector sphere.
The AthenaK SPH files used here store E_s, F_i,s, and N_s but not the reconstructed
number-current direction fnu^a, so the post-process uses the same radial transport
velocity as the energy luminosity:
    F^r_N,s ~ F^r_E,s * N_s / E_s.
This is exact when the energy and number currents are advected with the same
radial velocity, and is the appropriate free-streaming detector-sphere limit.

Species ordering (M1_TOTAL_NUM_SPECIES = 4):
    s = 0  nu_e      (electron neutrino)
    s = 1  nubar_e   (electron antineutrino)
    s = 2  nu_x      (heavy-flavour neutrino)
    s = 3  nubar_x   (heavy-flavour antineutrino)
Note: in bns_nurates, species 2 and 3 are symmetric (nux = anux).  The code
stores each separately; both are output here.

Required parfile output blocks (``radius = R`` for one surface, or ``radii = R1, R2,
...`` / ``nradii`` + ``r_min`` + ``r_max`` for several surfaces in one dump):
    <outputXX>
    file_type  = sph
    variable   = rad_m1_F
    dt         = 5.0
    radius     = 400.0
    ntheta     = 256

    <outputXX+1>
    file_type  = sph
    variable   = rad_m1_E
    dt         = 5.0
    radius     = 400.0
    ntheta     = 256

    <outputXX+2>
    file_type  = sph
    variable   = rad_m1_N
    dt         = 5.0
    radius     = 400.0
    ntheta     = 256

Output files (written to --output-dir):
    Lnu_E_{sp}.txt        time, energy luminosity per species [M_sun], [erg/s]
    Lnu_E_total.txt       time, sum over all 4 species         [M_sun], [erg/s]
    Eav_{sp}.txt          time, flux-weighted mean energy       [M_sun], [MeV]
    dEnu_dtheta_{sp}.txt  theta, time-integrated dE_nu/dtheta  [rad],   [erg/rad]
    sphere_center_neutrino.txt  t, center of the extraction surface  [M_sun]

Command line:
    kplot-sphere-neutrinos --sph-dir DIR [--sph-dir DIR ...] --output-dir DIR \
        --radius 300
"""

import os

import numpy as np

from athplot.utils.units import conv_energy, conv_luminosity, cactus, cgs

from ..utils.cli import sphere_parser
from ..utils.constants import LENGTH_CM
from ..utils.grmhd import inverse
from ..utils.quadrature import sum_over_time
from ..utils.surface import Surface, run, save_centers, stack

DEFAULT_RADIUS = 300.0
DEFAULT_JOBNAME = "bns"

# The M1 dumps carrying the neutrino fields, and the geometry dumps they are
# combined with.  Geometry may use a different snapshot counter (a different restart
# offset), so it is matched by time; the M1 variables share one counter.
M1_VARIABLES   = ('rad_m1_F', 'rad_m1_E', 'rad_m1_N')
GEOM_VARIABLES = ('adm', 'z4c_alpha', 'z4c_betax', 'z4c_betay', 'z4c_betaz')

_M1_HINT = ("\nAdd 'variable = rad_m1_F / rad_m1_E / rad_m1_N' sph output blocks "
            "to the parfile.")

# Species labels matching AthenaK's M1 ordering
SPECIES  = ['nue', 'nua', 'nux', 'anux']
NSPECIES = len(SPECIES)

# AthenaK stores N:s (neutrino number density) in EOS units (fm^-3) while
# E:s is in code energy-density units (M_sun^-2).  To form a consistent ratio,
# convert N from fm^-3 to code units (M_sun^-3):
#   1 code_length = G M_sun / c^2 = 1.477e18 fm
#   so 1 fm^-3    = 3.222e54 M_sun^-3
N_CODE_PER_FM3 = (LENGTH_CM * 1e13)**3  # ~ 3.222e54


# ===================================================================
# Main analysis
# ===================================================================

def _radial_projection(V: list[np.ndarray], gu: tuple, alpha: np.ndarray, beta: list[np.ndarray],
                       pos: tuple) -> np.ndarray:
  """
  Project V^i along r_i / sqrt(g^ij_4 r_i r_j), matching THC; the spatial block
  of the inverse 4-metric is g^ij_4 = gamma^ij - beta^i beta^j / alpha^2.

  Parameters:
  V (list[np.ndarray]): Vector components V^i.
  gu (tuple): Inverse spatial metric (xx, xy, xz, yy, yz, zz).
  alpha (np.ndarray): Lapse.
  beta (list[np.ndarray]): Shift.
  pos (tuple): Position relative to the sphere center.

  Returns:
  Radial component of V.
  """
  (guxx, guxy, guxz, guyy, guyz, guzz), (bx, by, bz), (rx, ry, rz) = gu, beta, pos
  a2 = alpha**2
  rr_metric = np.sqrt((guxx - bx * bx / a2) * rx**2 + (guyy - by * by / a2) * ry**2 +
                      (guzz - bz * bz / a2) * rz**2 +
                      2.0 * (guxy - bx * by / a2) * rx * ry +
                      2.0 * (guxz - bx * bz / a2) * rx * rz +
                      2.0 * (guyz - by * bz / a2) * ry * rz)
  return (rx * V[0] + ry * V[1] + rz * V[2]) / rr_metric


def _compute(s: Surface, ctx: None) -> dict:
  """
  Neutrino luminosities and mean energies per species on the surface.

  Parameters:
  s (Surface): Extraction surface of one snapshot.
  ctx (None): Unused.

  Returns:
  {'Lnu_E', 'Eav', 'dLdtheta', 'theta'}.
  """
  gu = inverse(s.metric)
  guxx, guxy, guxz, guyy, guyz, guzz = gu
  alpha, beta, pos = s.alpha, s.beta, s.pos
  weights = s.raw('weights')
  theta_edges = s.theta_edges
  beta_r = _radial_projection(beta, gu, alpha, beta, pos)

  Lnu_E, Eav, dLdtheta = [], [], []
  for sp in range(NSPECIES):
    Fx, Fy, Fz = (s[f'F{k}:{sp}'] for k in 'xyz')
    E, N = s[f'E:{sp}'], s[f'N:{sp}']
    Fu = (guxx * Fx + guxy * Fy + guxz * Fz,
          guxy * Fx + guyy * Fy + guyz * Fz,
          guxz * Fx + guyz * Fy + guzz * Fz)
    radial_flux = alpha * _radial_projection(Fu, gu, alpha, beta, pos) - beta_r * E
    L_code = np.sum(radial_flux * weights)
    Lnu_E.append(conv_luminosity(cactus, cgs, L_code))

    # THC's detector mean energy is energy luminosity divided by number
    # luminosity.  The exact M1 number flux uses fnu^a, which is not stored
    # in the current SPH output, so use the same radial transport velocity
    # as the energy flux: F_N^r ~= F_E^r * N/E.
    number_flux = np.zeros_like(radial_flux)
    np.divide(radial_flux * N * N_CODE_PER_FM3, E, out=number_flux, where=(E > 0.0))
    Ndot_code = np.sum(number_flux * weights)
    Eav.append(conv_energy(cactus, cgs, L_code / Ndot_code) / (cgs.eV * 1e6)
               if Ndot_code > 0.0 else 0.0)

    outgoing_L = np.where(radial_flux > 0.0, radial_flux, 0.0)
    dL_theta, _ = np.histogram(s.theta.ravel(), bins=theta_edges,
                               weights=(outgoing_L * weights).ravel())
    dLdtheta.append(dL_theta / np.diff(theta_edges))

  return {'Lnu_E': Lnu_E, 'Eav': Eav, 'dLdtheta': dLdtheta, 'theta': s.theta_1d}


def analyze(sph_dirs: list[str], output_dir: str, radius: float = DEFAULT_RADIUS,
            jobname: str = DEFAULT_JOBNAME, n_workers: int | None = None) -> None:
  """
  Run the neutrino analysis over every snapshot found in sph_dirs.

  Parameters:
  sph_dirs (list[str]): AthenaK output-XXXX/sph directories (all treated as one run).
  output_dir (str): Directory the outputs are written to.
  radius (float): SPH extraction radius [M_sun]; must have rad_m1_E/F/N and adm/z4c output.
  jobname (str): AthenaK job name prefixing the VTK files.
  n_workers (int): Worker processes for the snapshot loop (default: min(8, cpu_count())).

  Returns:
  None.
  """
  results = run(sph_dirs, jobname, radius, _compute, variables=M1_VARIABLES,
                by_time=GEOM_VARIABLES, n_workers=n_workers, hint=_M1_HINT)[float(radius)]
  os.makedirs(output_dir, exist_ok=True)
  save_centers(os.path.join(output_dir, 'sphere_center_neutrino.txt'), results)

  time_arr      = stack(results, 'time')
  theta_centers = np.sort(results[0]['theta'])
  Lnu_E_snap    = list(stack(results, 'Lnu_E').T)
  Eav_snap      = list(stack(results, 'Eav').T)
  dLdtheta_snap = list(stack(results, 'dLdtheta').transpose(1, 0, 2))

  # Total luminosity across all species (already time-sorted)
  Lnu_total = sum(Lnu_E_snap)

  # Time-integrated dE_nu/dtheta per species [erg/rad]: code_lum/rad times code time,
  # converted to erg/rad.
  dEnu_dtheta = [conv_energy(cactus, cgs, 1.0) * sum_over_time(dLdtheta_snap[s], time_arr)
                 for s in range(NSPECIES)]

  # ------------------------------------------------------------------
  # Print summary
  # ------------------------------------------------------------------
  print("\nPeak energy luminosities:")
  for s, sp in enumerate(SPECIES):
    print(f"  {sp:4s}: {Lnu_E_snap[s].max():.3e} erg/s")
  print(f"  total: {Lnu_total.max():.3e} erg/s")

  print("\nTime-averaged mean energies (last snapshot):")
  for s, sp in enumerate(SPECIES):
    print(f"  {sp:4s}: {Eav_snap[s][-1]:.2f} MeV")

  # ------------------------------------------------------------------
  # Save outputs
  # ------------------------------------------------------------------
  for s, sp in enumerate(SPECIES):
    # Energy luminosity time series
    np.savetxt(
        os.path.join(output_dir, f'Lnu_E_{sp}.txt'),
        np.column_stack((time_arr, Lnu_E_snap[s])),
        header=f'time[M_sun]    Lnu_E_{sp}[erg/s]',
        fmt='%.6e',
    )
    # Mean energy time series
    np.savetxt(
        os.path.join(output_dir, f'Eav_{sp}.txt'),
        np.column_stack((time_arr, Eav_snap[s])),
        header=(f'time[M_sun]    Eav_{sp}[MeV]  '
                f'(flux-weighted mean energy at r={radius})'),
        fmt='%.6e',
    )
    # Time-integrated angular distribution
    np.savetxt(
        os.path.join(output_dir, f'dEnu_dtheta_{sp}.txt'),
        np.column_stack((theta_centers, dEnu_dtheta[s])),
        header=f'theta[rad]    dEnu_dtheta_{sp}[erg/rad]  (time-integrated, outgoing only)',
        fmt='%.6e',
    )

  np.savetxt(
      os.path.join(output_dir, 'Lnu_E_total.txt'),
      np.column_stack((time_arr, Lnu_total)),
      header='time[M_sun]    Lnu_E_total[erg/s]  (sum over all 4 species)',
      fmt='%.6e',
  )

  print(f'\nAll outputs saved to {output_dir}')


def main(argv: list[str] | None = None) -> None:
  """
  Command-line entry point of the neutrino analysis.

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

"""
Neutrino luminosity analysis from rad_m1_F / rad_m1_E / rad_m1_N spherical-surface
VTK outputs of AthenaK BNS simulations.

Per snapshot, loads 8 VTK files:
  <job>.r=R.rad_m1_F.NNNNN.vtk  — Fx:s, Fy:s, Fz:s  (Eulerian energy-flux per species)
  <job>.r=R.rad_m1_E.NNNNN.vtk  — E:s                (Eulerian energy density per species)
  <job>.r=R.rad_m1_N.NNNNN.vtk  — N:s                (number density per species)
  <job>.r=R.adm.NNNNN.vtk       — adm_gxx..gyz       (ADM 3-metric)
  <job>.r=R.z4c_alpha.NNNNN.vtk — z4c_alpha           (lapse)
  <job>.r=R.z4c_beta*.NNNNN.vtk — z4c_beta*           (shift)

An output block writing several radii at once bundles them into one dump named
``<job>.r=RMIN-RMAX.<var>.NNNNN.vtk`` instead; --radius then picks the surface to
analyse out of the bundle (see :mod:`kplot.sphere._files`).  Each variable is looked
up on its own, so the radii may be bundled differently from one variable to the next.

Physics
-------
Energy luminosity per species s:
    L_E,s(t) = ∫ (α F^r_s - β^r E_s) · r² dΩ
             [code units → erg/s]
where F^i_s = γ^ij F_j,s, using the stored M1 flux covector components.
The radial projection uses the metric-normalized coordinate radial covector
used by THC: rhat_i = r_i / sqrt(g^ij_4 r_i r_j), where g^ij_4 is the
spatial block of the inverse 4-metric.
E_s and F_i,s are already densitized M1 evolved variables, so no extra √γ
factor is applied in the surface integral.
This matches the THC/AthenaK M1 calc_E_flux expression.
The angular area integral uses AthenaK's native SPH `weights` field:
    weights = r² dΩ  (Gauss-Legendre in μ = cos θ, uniform in φ)

Flux-weighted mean energy per species:
    <ε_s>(t) = ∫ F^r_E,s · r² dΩ / ∫ F^r_N,s · r² dΩ  → MeV
THC constructs the denominator from the M1 number flux on the detector sphere.
The AthenaK SPH files used here store E_s, F_i,s, and N_s but not the reconstructed
number-current direction fnu^a, so the post-process uses the same radial transport
velocity as the energy luminosity:
    F^r_N,s ≈ F^r_E,s · N_s / E_s.
This is exact when the energy and number currents are advected with the same
radial velocity, and is the appropriate free-streaming detector-sphere limit.

Species ordering (M1_TOTAL_NUM_SPECIES = 4):
    s = 0  νe   (electron neutrino)
    s = 1  ν̄e   (electron antineutrino)
    s = 2  νx   (heavy-flavour neutrino)
    s = 3  ν̄x   (heavy-flavour antineutrino)
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
    dEnu_dtheta_{sp}.txt  θ, time-integrated dE_ν/dθ           [rad],   [erg/rad]
    sphere_center_neutrino.txt  t, center of the extraction surface  [M_sun]

Command line:
    kplot-sphere-neutrinos --sph-dir DIR [--sph-dir DIR ...] --output-dir DIR \
        --radius 300
"""

import argparse
import os
from multiprocessing import Pool, cpu_count

import numpy as np

from athplot.utils.units import conv_energy, cactus, cgs

from ._files import Shell, locate, read_time
from ._integrate import sum_over_time

DEFAULT_RADIUS = 300.0
DEFAULT_JOBNAME = "bns"

# ===================================================================
# Constants
# ===================================================================

# The M1 dumps carrying the neutrino fields, and the geometry dumps they are
# combined with.  Geometry may use a different snapshot counter (a different restart
# offset), so it is matched by time; the M1 variables share one counter.
M1_VARIABLES   = ('rad_m1_F', 'rad_m1_E', 'rad_m1_N')
# Order fixed: the worker unpacks its geometry arguments in it.
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
#   1 M_sun^3     = (1.477e18)^3 fm^3 = 3.222e54 fm^3
#   ∴ 1 fm^-3     = 3.222e54 M_sun^-3
_G_SI        = 6.67430e-11
_M_SUN_KG    = 1.98892e30
_C_SI        = 2.99792458e8
_M_PER_FM    = 1e-15
N_CODE_PER_FM3 = (_G_SI * _M_SUN_KG / _C_SI**2 / _M_PER_FM)**3  # ≈ 3.222e54


# ===================================================================
# Main analysis
# ===================================================================

def _build_time_list(snapshots):
    """Return sorted list of (time_float, (path, surface index)) for a SnapshotSet.

    Only the metadata line of each dump is read, which avoids the full VTK parse.
    """
    result = [(read_time(path), (path, iradius))
              for path, iradius in snapshots.entries.values()]
    result.sort(key=lambda x: x[0])
    return result


def _nearest_path(time_list, t, tol=4.0):
    """Return (path, surface index) of the snapshot nearest in time to t.

    None if the nearest one is further away than `tol`.
    """
    if not time_list:
        return None
    times = np.array([x[0] for x in time_list])
    idx   = np.searchsorted(times, t)
    # check both neighbours
    best_i, best_dt = None, np.inf
    for i in (idx - 1, idx):
        if 0 <= i < len(times):
            dt = abs(times[i] - t)
            if dt < best_dt:
                best_dt, best_i = dt, i
    if best_dt <= tol:
        return time_list[best_i][1]
    return None


def _inverse_spatial_metric(gxx, gyy, gzz, gxy, gxz, gyz):
    gamma_det = (gxx * gyy * gzz +
                 2 * gxy * gxz * gyz -
                 gxx * gyz**2 -
                 gyy * gxz**2 -
                 gzz * gxy**2)

    guxx = (gyy * gzz - gyz**2) / gamma_det
    guyy = (gxx * gzz - gxz**2) / gamma_det
    guzz = (gxx * gyy - gxy**2) / gamma_det
    guxy = (gxz * gyz - gxy * gzz) / gamma_det
    guxz = (gxy * gyz - gxz * gyy) / gamma_det
    guyz = (gxy * gxz - gxx * gyz) / gamma_det
    return guxx, guyy, guzz, guxy, guxz, guyz


def _raise_spatial_covector(Fx, Fy, Fz, gxx, gyy, gzz, gxy, gxz, gyz):
    guxx, guyy, guzz, guxy, guxz, guyz = _inverse_spatial_metric(
        gxx, gyy, gzz, gxy, gxz, gyz)

    Fux = guxx * Fx + guxy * Fy + guxz * Fz
    Fuy = guxy * Fx + guyy * Fy + guyz * Fz
    Fuz = guxz * Fx + guyz * Fy + guzz * Fz
    return Fux, Fuy, Fuz


def _metric_normalized_radial_projection(Vx, Vy, Vz,
                                         gxx, gyy, gzz, gxy, gxz, gyz,
                                         alpha, betax, betay, betaz,
                                         theta, phi, radius):
    """Project V^i along r_i / sqrt(g^ij_4 r_i r_j), matching THC."""
    guxx, guyy, guzz, guxy, guxz, guyz = _inverse_spatial_metric(
        gxx, gyy, gzz, gxy, gxz, gyz)
    oo_alpha2 = 1.0 / alpha**2

    # THC normalizes r_i with the inverse 4-metric.  Its spatial block is
    # g^ij_4 = gamma^ij - beta^i beta^j / alpha^2.
    g4uxx = guxx - betax * betax * oo_alpha2
    g4uyy = guyy - betay * betay * oo_alpha2
    g4uzz = guzz - betaz * betaz * oo_alpha2
    g4uxy = guxy - betax * betay * oo_alpha2
    g4uxz = guxz - betax * betaz * oo_alpha2
    g4uyz = guyz - betay * betaz * oo_alpha2

    rx = radius * np.sin(theta) * np.cos(phi)
    ry = radius * np.sin(theta) * np.sin(phi)
    rz = radius * np.cos(theta)

    rr_metric = np.sqrt(g4uxx * rx**2 + g4uyy * ry**2 + g4uzz * rz**2 +
                        2.0 * g4uxy * rx * ry +
                        2.0 * g4uxz * rx * rz +
                        2.0 * g4uyz * ry * rz)
    return (rx * Vx + ry * Vy + rz * Vz) / rr_metric


def _process_snapshot(args):
    """Per-snapshot worker — runs in a separate process.

    Imports are done inside the function so the worker is self-contained and
    can be pickled by multiprocessing without carrying module-level state.
    """
    (flux_file, ener_file, nden_file, adm_file, alpha_file,
     betax_file, betay_file, betaz_file,
     theta_edges) = args

    import numpy as _np
    from athplot.utils.units import conv_luminosity as _conv_lum, \
                                    conv_energy as _conv_e, \
                                    cactus as _cactus, cgs as _cgs

    import os as _os

    from ._files import Shell as _Shell, CENTER_TOL as _CENTER_TOL

    # Every entry is (path, surface index): a dump may hold more than one radius.
    flux_vtk  = _Shell(*flux_file)
    ener_vtk  = _Shell(*ener_file)
    nden_vtk  = _Shell(*nden_file)
    adm_vtk   = _Shell(*adm_file)
    alpha_vtk = _Shell(*alpha_file)
    betax_vtk = _Shell(*betax_file)
    betay_vtk = _Shell(*betay_file)
    betaz_vtk = _Shell(*betaz_file)

    time  = flux_vtk.time
    radius = flux_vtk.radius
    theta = flux_vtk.theta
    phi   = flux_vtk.phi
    surface_weights = flux_vtk.shell('weights')

    shells  = (flux_vtk, ener_vtk, nden_vtk, adm_vtk, alpha_vtk,
               betax_vtk, betay_vtk, betaz_vtk)
    centers = _np.array([sh.center for sh in shells])
    if not _np.allclose(centers, centers[0], rtol=0.0, atol=_CENTER_TOL):
        detail = '\n'.join(f'  {_os.path.basename(sh.path)}: '
                           f'({c[0]:.6g}, {c[1]:.6g}, {c[2]:.6g})'
                           for sh, c in zip(shells, centers))
        raise RuntimeError(
            f'Sphere centers disagree at t = {time:g} M_sun:\n{detail}\n'
            'Every sph output block feeding the neutrino analysis must use the same '
            'center_tracker.')

    # -- Duplication correction --
    z4c_alpha_raw = alpha_vtk.shell('z4c_alpha')
    phi_zero    = (phi == phi.min())
    theta_north = (theta == theta.min())
    dup_corr = _np.ones(z4c_alpha_raw.shape)
    if z4c_alpha_raw[phi_zero & ~theta_north].max() > 1.0:
        dup_corr[phi_zero] = 0.5
    if z4c_alpha_raw[theta_north].max() > 1.0:
        dup_corr[theta_north] = 0.25

    gxx = adm_vtk.shell('adm_gxx') * dup_corr
    gyy = adm_vtk.shell('adm_gyy') * dup_corr
    gzz = adm_vtk.shell('adm_gzz') * dup_corr
    gxy = adm_vtk.shell('adm_gxy') * dup_corr
    gxz = adm_vtk.shell('adm_gxz') * dup_corr
    gyz = adm_vtk.shell('adm_gyz') * dup_corr
    z4c_alpha  = z4c_alpha_raw * dup_corr
    z4c_betax  = betax_vtk.shell('z4c_betax') * dup_corr
    z4c_betay  = betay_vtk.shell('z4c_betay') * dup_corr
    z4c_betaz  = betaz_vtk.shell('z4c_betaz') * dup_corr

    nspecies = NSPECIES
    n_conv = N_CODE_PER_FM3

    Lnu_E_list   = []
    Eav_list     = []
    dLdtheta_list = []

    dtheta = _np.diff(theta_edges)

    for s in range(nspecies):
        Fx = flux_vtk.shell(f'Fx:{s}') * dup_corr
        Fy = flux_vtk.shell(f'Fy:{s}') * dup_corr
        Fz = flux_vtk.shell(f'Fz:{s}') * dup_corr
        E  = ener_vtk.shell(f'E:{s}')  * dup_corr
        N  = nden_vtk.shell(f'N:{s}')  * dup_corr

        Fux, Fuy, Fuz = _raise_spatial_covector(
            Fx, Fy, Fz, gxx, gyy, gzz, gxy, gxz, gyz)
        Fr            = _metric_normalized_radial_projection(
            Fux, Fuy, Fuz, gxx, gyy, gzz, gxy, gxz, gyz,
            z4c_alpha, z4c_betax, z4c_betay, z4c_betaz,
            theta, phi, radius)
        beta_r        = _metric_normalized_radial_projection(
            z4c_betax, z4c_betay, z4c_betaz,
            gxx, gyy, gzz, gxy, gxz, gyz,
            z4c_alpha, z4c_betax, z4c_betay, z4c_betaz,
            theta, phi, radius)
        radial_flux   = z4c_alpha * Fr - beta_r * E
        L_code        = _np.sum(radial_flux * surface_weights)
        L_erg         = _conv_lum(_cactus, _cgs, L_code)
        Lnu_E_list.append(L_erg)

        # THC's detector mean energy is energy luminosity divided by number
        # luminosity.  The exact M1 number flux uses fnu^a, which is not stored
        # in the current SPH output, so use the same radial transport velocity
        # as the energy flux: F_N^r ~= F_E^r * N/E.
        N_code = N * n_conv
        number_flux = _np.zeros_like(radial_flux)
        _np.divide(radial_flux * N_code, E, out=number_flux, where=(E > 0.0))
        Ndot_code = _np.sum(number_flux * surface_weights)
        eps_MeV = (_conv_e(_cactus, _cgs, L_code / Ndot_code) / (_cgs.eV * 1e6)
                   if Ndot_code > 0.0 else 0.0)
        Eav_list.append(eps_MeV)

        outgoing_L  = _np.where(radial_flux > 0.0, radial_flux, 0.0)
        dL_theta, _ = _np.histogram(
            theta.flatten(), bins=theta_edges,
            weights=outgoing_L.flatten() * surface_weights.flatten(),
        )
        dLdtheta_list.append(dL_theta / dtheta)

    return {'time': time, 'center': centers[0], 'Lnu_E': Lnu_E_list,
            'Eav': Eav_list, 'dLdtheta': dLdtheta_list}


def analyze(sph_dirs, output_dir, radius=DEFAULT_RADIUS, jobname=DEFAULT_JOBNAME,
            n_workers=None):
    """Run the neutrino analysis over every rad_m1_F snapshot in `sph_dirs`.

    Parameters
    ----------
    sph_dirs : list of str
        AthenaK ``output-XXXX/sph`` directories to scan (all treated as one run).
    output_dir : str
        Directory the .txt outputs are written to.
    radius : float
        SPH extraction radius [M_sun]; must have rad_m1_E/F/N output.  A dump
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
    # Locate the dumps holding `radius`, per variable: an output block may write
    # one surface per file or bundle several radii into one file, and the two
    # forms can be mixed between variables within a run.
    # ------------------------------------------------------------------
    m1_files = {v: locate(sph_dirs, jobname, v, radius, hint=_M1_HINT)
                for v in M1_VARIABLES}
    for v in M1_VARIABLES:
        print(f"  {m1_files[v].describe()}")

    # Build the snapshot index from the counters all three M1 variables share.
    m1_indices = [set(m1_files[v].paths) for v in M1_VARIABLES]
    indices = sorted(set.intersection(*m1_indices))
    print(f"Found {len(indices)} rad_m1 snapshots at r={radius} M_sun")
    if len(indices) == 0:
        raise RuntimeError(
            f"No snapshot in {list(sph_dirs)} has all of "
            f"{', '.join(M1_VARIABLES)} at r = {radius:g}.{_M1_HINT}")
    incomplete = len(set.union(*m1_indices)) - len(indices)
    if incomplete:
        print(f"  [skip] {incomplete} snapshots missing one of {', '.join(M1_VARIABLES)}")

    # Build sorted time lists for adm and z4c gauge outputs: they may use a
    # different counter series (different restart offset) but cover the same
    # simulation times, so they are matched by time rather than by counter.
    print("Building time index for adm / z4c gauge files...")
    geom_files = {v: locate(sph_dirs, jobname, v, radius) for v in GEOM_VARIABLES}
    geom_times = {v: _build_time_list(geom_files[v]) for v in GEOM_VARIABLES}
    for v in GEOM_VARIABLES:
        print(f"  {geom_files[v].describe()}")

    # ------------------------------------------------------------------
    # Build theta bins for dL/dtheta diagnostics.  Surface integrals use the
    # native AthenaK SPH weights stored in each VTK file.
    # ------------------------------------------------------------------
    flux_files = m1_files['rad_m1_F']
    first_flux_path, flux_iradius = flux_files.shell(indices[0])
    _fv = Shell(first_flux_path, flux_iradius)
    theta_1d        = _fv.theta[0, :]
    theta_1d_sorted = np.sort(theta_1d)
    theta_edges     = np.concatenate([[0.0],
                                      0.5 * (theta_1d_sorted[:-1] + theta_1d_sorted[1:]),
                                      [np.pi]])
    theta_centers   = theta_1d_sorted
    del _fv

    # dup check on first snapshot with alpha
    _t0 = read_time(first_flux_path)
    _ap = _nearest_path(geom_times['z4c_alpha'], _t0)
    if _ap:
        _av = Shell(*_ap)
        _a  = _av.shell('z4c_alpha')
        print(f"  [dup check] alpha range: min={_a.min():.4f}  max={_a.max():.4f}")

    # ------------------------------------------------------------------
    # Build worker argument list (pre-resolve adm/alpha paths by time).  Every
    # entry is (path, surface index), so a bundled dump is read at the right radius.
    # ------------------------------------------------------------------
    worker_args = []
    skipped = 0
    for index in indices:
        flux_file = flux_files.shell(index)
        t_flux = read_time(flux_file[0])   # metadata line only, no VTK parse
        geom = [_nearest_path(geom_times[v], t_flux) for v in GEOM_VARIABLES]
        if any(entry is None for entry in geom):
            skipped += 1
            continue
        worker_args.append((
            flux_file,
            m1_files['rad_m1_E'].shell(index),
            m1_files['rad_m1_N'].shell(index),
            *geom,
            theta_edges,
        ))
    if skipped:
        print(f"  [skip] {skipped} snapshots with no adm/alpha/beta match")
    print(f"Processing {len(worker_args)} snapshots with n_workers={n_workers}...")

    # ------------------------------------------------------------------
    # Parallel snapshot loop
    # ------------------------------------------------------------------
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = list(pool.imap_unordered(_process_snapshot, worker_args,
                                               chunksize=max(1, len(worker_args) // (n_workers * 4))))
    else:
        results = [_process_snapshot(a) for a in worker_args]

    # Progress summary
    results.sort(key=lambda x: x['time'])
    for i, res in enumerate(results):
        if i % 50 == 0:
            print(f"  snapshot {i:4d}/{len(results)}  t={res['time']:.2f} M_sun  "
                  f"L_nue={res['Lnu_E'][0]:.3e} erg/s")

    # ------------------------------------------------------------------
    # Unpack results into time-sorted accumulators
    # ------------------------------------------------------------------
    time_arr     = np.array([r['time']    for r in results])
    center_arr   = np.array([r['center']  for r in results])

    center_shift = np.linalg.norm(center_arr - center_arr[0], axis=1).max()
    print(f"Sphere center: ({center_arr[0][0]:.6g}, {center_arr[0][1]:.6g}, "
          f"{center_arr[0][2]:.6g}) -> ({center_arr[-1][0]:.6g}, "
          f"{center_arr[-1][1]:.6g}, {center_arr[-1][2]:.6g}) M_sun, "
          f"max excursion {center_shift:.6g}")
    Lnu_E_snap   = [np.array([r['Lnu_E'][s]   for r in results]) for s in range(NSPECIES)]
    Eav_snap     = [np.array([r['Eav'][s]     for r in results]) for s in range(NSPECIES)]
    dLdtheta_snap = [np.array([r['dLdtheta'][s] for r in results]) for s in range(NSPECIES)]

    # Total luminosity across all species (already time-sorted)
    Lnu_total = sum(Lnu_E_snap[s] for s in range(NSPECIES))

    # Time-integrated dE_ν/dθ per species [erg/rad] via midpoint Riemann sum over time.
    # dLdtheta_snap is in code_lum/rad; time_arr is in M_sun (= code time);
    # their integral is in code_energy/rad → convert to erg/rad with conv_energy.
    _conv_e = conv_energy(cactus, cgs, 1.0)  # erg per code_energy ≈ 1.79e54
    dEnu_dtheta = [_conv_e * sum_over_time(dLdtheta_snap[s], time_arr)
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
    os.makedirs(output_dir, exist_ok=True)

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
        os.path.join(output_dir, 'sphere_center_neutrino.txt'),
        np.column_stack((time_arr, center_arr)),
        header='time[M_sun]    xc[M_sun]    yc[M_sun]    zc[M_sun]',
        fmt='%.6e',
    )

    np.savetxt(
        os.path.join(output_dir, 'Lnu_E_total.txt'),
        np.column_stack((time_arr, Lnu_total)),
        header='time[M_sun]    Lnu_E_total[erg/s]  (sum over all 4 species)',
        fmt='%.6e',
    )

    print(f'\nAll outputs saved to {output_dir}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sph-dir", action="append", required=True, dest="sph_dirs",
                   metavar="DIR",
                   help="AthenaK output-XXXX/sph directory (repeat for each segment).")
    p.add_argument("--output-dir", required=True,
                   help="Directory for the .txt outputs.")
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

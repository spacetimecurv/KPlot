#!/usr/bin/env python3
"""
Global properties of the post-merger disk from an AthenaK BHNS 3D snapshot.

The following data is required for this:
<output1>
file_type = bin
variable  = mhd_w_bcc
dt        = 500.0
id        = mhd_w_bcc_3D

<output2>
file_type = bin
variable  = adm
dt        = 500.0
id        = adm_3D

<output3>
file_type = bin
variable  = z4c_alpha
dt        = 500.0
id        = alpha_3D

<output4>
file_type = bin
variable  = z4c_betax
dt        = 500.0
id        = betax_3D

<output5>
file_type = bin
variable  = z4c_betay
dt        = 500.0
id        = betay_3D

<output6>
file_type = bin
variable  = z4c_betaz
dt        = 500.0
id        = betaz_3D

Using the .h5 version of the evolution table used for the simulations,
a bunch of diagnostics involving the post-merger disk are calculated and
stored per snapshot.

Outputs (written to --outdir):
  scalars/disk_scalars_<snapshot>.json: holding scalars such as the disk mass,
                                        angular momentum etc.
  histograms/disk_histograms_<snapshot>.csv: histograms of M_disk vs. Ye, etc.
  profiles/disk_profiles_<profiles>.csv: radial profiles of important quantities, i.e.
                                         temperature etc.
"""

# Built-in libraries.
import argparse
import glob
import json
import os
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Third-party libraries.
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

# AthenaK utility.
from kplot.system.bin_convert import read_binary

# The disk analysis script needs the following 3D output.
VARIABLES = ("mhd_w_bcc", "adm", "alpha", "betax", "betay", "betaz")

# ======================================================================
# MULTIPROCESS WORKER
# ======================================================================
# Per-worker state.
_W_EOS     = None
_W_TRACKER = None
_W_HORIZON = None


def _init_disk_worker(eos, tracker, horizon):
  """Pool initializer: load the EOS and the tracker/horizon files."""
  global _W_EOS, _W_TRACKER, _W_HORIZON
  _W_EOS     = eos
  _W_TRACKER = tracker
  _W_HORIZON = horizon


def _process_snapshot_disk(args):
  """Per-snapshot worker for disk analysis."""
  # Unpack.
  (mhd_file, adm_file, alpha_file, betax_file, betay_file, betaz_file,
  rho_cut, rho_keep, rho_remnant, bound_criterion, r_exclude, center,
  r_disk_max, nbins, rmax, outdir) = args

  snapshot = mhd_file.split("/")[-1].split(".")[-2].strip()

  # ======================================================================
  # MHD
  # ======================================================================
  # Read the header and data of the MHD files first.
  fd            = read_binary(mhd_file)
  pars          = parse_par_dump(fd["header"]) # Parfile parameters.
  time          = fd["time"]                   # Time of the snapshot.
  nx1, nx2, nx3 = fd["nx1_out_mb"], fd["nx2_out_mb"], fd["nx3_out_mb"]
  ncell_block   = nx1 * nx2 * nx3              # Points per meshblock.
  levels        = fd["mb_logical"][:, 3]       # Logical level of meshblock.

  dfloor = par(pars, "mhd", "dfloor", float, 1.0e-14)   # Density floor from parfile.
  xv, yv, zv, dvol, _, _, dzv = block_cell_geometry(fd) # CC-coordinates and volume for each MB.

  # Discard cells at (or barely above) the atmosphere floor.
  if rho_keep is None:
    rho_keep = 1.5 * dfloor * RHO_UNIT
  rho_keep = rho_keep / RHO_UNIT
  if rho_keep <= dfloor:
    raise SystemExit(f"--rho-keep must exceed dfloor = {dfloor * RHO_UNIT:.4g} g/cm^3")
  rho_full     = np.asarray(fd["mb_data"]["dens"]).reshape(-1)
  sel          = np.flatnonzero(rho_full > rho_keep)
  ncells_total = rho_full.size
  del rho_full # Free up memory.

  # Write fields in single precision based on the selection.
  def take(fdict, var):
    return np.asarray(fdict["mb_data"][var]).reshape(-1)[sel]

  rho         = take(fd, "dens")
  press       = take(fd, "press")
  velx        = take(fd, "velx")
  vely        = take(fd, "vely")
  velz        = take(fd, "velz")
  ye          = take(fd, "s_00")
  Bx          = take(fd, "bcc1")
  By          = take(fd, "bcc2")
  Bz          = take(fd, "bcc3")
  temp        = take(fd, "temperature")
  mb_geom_ref = fd["mb_geometry"]
  del fd # Free up memory.

  # Cell coordinates and volumes for the retained cells.
  i   = sel % nx1
  j   = (sel // nx1) % nx2
  k   = (sel // (nx1 * nx2)) % nx3
  m   = sel // ncell_block
  x   = xv[m, i]
  y   = yv[m, j]
  z   = zv[m, k]
  dV  = dvol[m].astype(np.float32)
  dz  = dzv[m].astype(np.float32)
  del i, j, k, m # Free up memory.

  # ======================================================================
  # GAUGE
  # ======================================================================
  def read_scalar_field(file, var):
    f = read_binary(file)
    if not np.array_equal(f["mb_geometry"], mb_geom_ref):
      raise SystemExit(f"{file} has a different mesh than the MHD file!")
    full = np.asarray(f["mb_data"][var]).reshape(-1)
    out = full[sel]
    del f, full # Free up memory.
    return out

  # Fetch the lapse and shift.
  alpha = read_scalar_field(alpha_file, "z4c_alpha")
  betax = read_scalar_field(betax_file, "z4c_betax")
  betay = read_scalar_field(betay_file, "z4c_betay")
  betaz = read_scalar_field(betaz_file, "z4c_betaz")

  # Get the center of the remnant and the radius of the black hole.
  if _W_TRACKER is not None:
    tracker     = _W_TRACKER
    tracker_idx = np.argmin(np.abs(tracker["time"] - time))
    center      = np.array([tracker["x"][tracker_idx],
                            tracker["y"][tracker_idx],
                            tracker["z"][tracker_idx]])
  elif _W_TRACKER is None and center:
    center = np.array(center)
  else:
    raise SystemExit("Either tracker file or manual center has to be specified!")

  r_ah = np.nan
  if _W_HORIZON is not None:
    horizon     = _W_HORIZON
    horizon_idx = np.argmin(np.abs(horizon["time"] - time))
    r_ah  = horizon["radius"][horizon_idx]

  r_exclude = r_exclude if r_exclude is not None else (
              r_ah if np.isfinite(r_ah) else 0.0)

  # ======================================================================
  # METRIC
  # ======================================================================
  fadm = read_binary(adm_file)
  if not np.array_equal(fadm["mb_geometry"], mb_geom_ref):
    raise SystemExit("ADM file has a different mesh than the MHD file!")
  gxx = take(fadm, "adm_gxx")
  gxy = take(fadm, "adm_gxy")
  gxz = take(fadm, "adm_gxz")
  gyy = take(fadm, "adm_gyy")
  gyz = take(fadm, "adm_gyz")
  gzz = take(fadm, "adm_gzz")
  del fadm # Free up memory.

  # Compute the volume factor.
  sqrtg = np.sqrt(gxx * gyy * gzz +
              2 * gxy * gxz * gyz -
                  gxx * gyz * gyz -
                  gyy * gxz * gxz -
                  gzz * gxy * gxy)

  # ======================================================================
  # KINEMATICS
  # ======================================================================
  # W and ut_d for bound vs. unbound.
  W = np.sqrt(1.0 + (gxx * velx * velx + 2.0 * gxy * velx * vely +
                      2.0 * gxz * velx * velz + gyy * vely * vely +
                      2.0 * gyz * vely * velz + gzz * velz * velz).astype(np.float64))
  ut_d = -alpha * W \
        + gxx * betax * velx + gyy * betay * vely \
        + gzz * betaz * velz \
        + gxy * (betax * vely + betay * velx) \
        + gxz * (betax * velz + betaz * velx) \
        + gyz * (betay * velz + betaz * vely)

  # Spatial components of u_\mu.
  ux_d = gxx * velx + gxy * vely + gxz * velz
  uy_d = gxy * velx + gyy * vely + gyz * velz
  uz_d = gxz * velx + gyz * vely + gzz * velz

  # u^i.
  ut_u = W / alpha
  ux_u = velx - ut_u * betax
  uy_u = vely - ut_u * betay
  uz_u = velz - ut_u * betaz

  # Coordinate 3-velocity dx^i/dt = alpha utilde^i / W - beta^i
  vcx = (alpha * velx / W - betax).astype(np.float32)
  vcy = (alpha * vely / W - betay).astype(np.float32)

  # ======================================================================
  # MAGNETIC FIELD
  # ======================================================================
  # bcc is the densitized Eulerian field: B^i = bcc^i / sqrt(gamma).
  Bx /= sqrtg
  By /= sqrtg
  Bz /= sqrtg

  # Magnetic field amplitude in the Eulerian frame.
  Bsq = gxx * Bx * Bx + 2.0 * gxy * Bx * By + 2.0 * gxz * Bx * Bz + \
        gyy * By * By + 2.0 * gyz * By * Bz + gzz * Bz * Bz

  BWv = (gxx * Bx * velx + gyy * By * vely + gzz * Bz * velz + \
        gxy * (Bx * vely + velx * By) + \
        gxz * (Bx * velz + velx * Bz) + \
        gyz * (By * velz + vely * Bz)).astype(np.float32)
  del velx, vely, velz # Free up memory.

  # Magnetic field amplitude in the fluid frame.
  bsq = ((Bsq + BWv * BWv) / (W * W)).astype(np.float32)

  # Magnetic field in fluid frame with upper index.
  bt_u = BWv / alpha
  bx_u = (Bx + alpha * bt_u * ux_u) / W
  by_u = (By + alpha * bt_u * uy_u) / W
  bz_u = (Bz + alpha * bt_u * uz_u) / W
  del ux_u, uy_u, uz_u, Bx, By, Bz # Free up memory.

  # Magnetic field in fluid frame with lower index.
  bx_d = ((gxx * (bx_u + betax * bt_u) +
           gxy * (by_u + betay * bt_u) +
           gxz * (bz_u + betaz * bt_u))).astype(np.float32)
  by_d = ((gxy * (bx_u + betax * bt_u) +
           gyy * (by_u + betay * bt_u) +
           gyz * (bz_u + betaz * bt_u))).astype(np.float32)
  bz_d = ((gxz * (bx_u + betax * bt_u) +
           gyz * (by_u + betay * bt_u) +
           gzz * (bz_u + betaz * bt_u))).astype(np.float32)

  del alpha, betax, betay, betaz, gxx, gxy, gxz, \
      gyy, gyz, gzz, bt_u, bx_u, by_u, bz_u # Free up memory.

  # ======================================================================
  # EOS
  # ======================================================================
  tab      = _W_EOS
  rho_cgs  = rho * RHO_UNIT
  mb_g     = tab.mb * MEV / CLIGHT_CGS**2
  nb       = rho_cgs / mb_g * 1.0e-39
  log_nb   = np.clip(np.log(nb), tab.nb_min, tab.nb_max)
  ye_clip  = np.clip(ye, tab.ye_min, tab.ye_max)
  log_temp = np.clip(np.log(temp), tab.t_min, tab.t_max)

  # Call the interpolators.
  pts = np.column_stack((log_nb, ye_clip, log_temp))
  eosvals = {}
  eosvals["Q1"] = tab.interpolator_Q1(pts)
  eosvals["Q2"] = tab.interpolator_Q2(pts)
  eosvals["Q7"] = tab.interpolator_Q7(pts)

  # Compute the enthalpy and store the entropy.
  h = 1.0 + eosvals["Q7"] + eosvals["Q1"] / tab.mb
  entropy = eosvals["Q2"].astype(np.float32)
  hut_d = h * ut_d

  del eosvals, nb, rho_cgs, log_nb, ye_clip, log_temp, pts # Free up memory.

  # ======================================================================
  # GEOMETRY RELATIVE TO BH/NS
  # ======================================================================
  xg = x - center[0]
  yg = y - center[1]
  zg = z - center[2]
  del x, y, z # Free up memory.
  Rcyl = np.hypot(xg, yg)
  rsph = np.sqrt(xg * xg + yg * yg + zg * zg)

  # ======================================================================
  # MASS, ANGULAR MOMENTUM
  # ======================================================================
  # Compute the conserved rest mass.
  dM = rho * W * sqrtg * dV

  # Exact GR specific angular momentum: j = h u_phi, u_phi = -y u_x + x u_y
  u_phi = -yg * ux_d + xg * uy_d
  j_spec = h * u_phi
  dJ = dM * j_spec

  # Full MHD angular momentum: T^t_phi = (rho h + b^2) u^t u_phi - b^t b_phi,
  # with alpha u^t = W and alpha b^t = BWv.
  b_phi = -yg * bx_d + xg * by_d
  dJ_mhd = sqrtg * ((rho * h + bsq) * W * u_phi - BWv * b_phi) * dV

  # Angular velocity Omega = dphi/dt.
  Rsafe = np.maximum(Rcyl, 1e-30)
  with np.errstate(divide="ignore", invalid="ignore"):
      omega = ((xg * vcy - yg * vcx) / Rsafe**2).astype(np.float32)
      v_rad = ((xg * vcx + yg * vcy) / Rsafe).astype(np.float32)
  del vcx, vcy, ux_d, uy_d, bx_d, by_d, u_phi, Rsafe # Free up memory.

  # ======================================================================
  # MASKS
  # ======================================================================
  # Criteria:
  # - if BH is present, be outside
  # - has to be bound material, i.e. not fulfilling geodesic/Bernoulli crit.
  # - density has to be larger than some cut
  # - density has to be lower than some upper threshold (only for BNS remnant)
  # - radius has to smaller than some value (optional; to remove tidal tails)
  outside_bh = rsph > r_exclude
  unbound_geo = ut_d < -1.0
  unbound_bern = hut_d < -tab.hmin
  unbound = unbound_bern if bound_criterion == "bernoulli" else unbound_geo

  rho_cut_code = rho_cut / RHO_UNIT
  rho_rem_code = rho_remnant / RHO_UNIT

  dense = rho > rho_cut_code
  not_remnant = rho < rho_rem_code
  disk = dense & not_remnant & outside_bh & ~unbound
  if r_disk_max is not None:
      disk &= Rcyl < r_disk_max

  # ======================================================================
  # MRI ANALYSIS
  # ======================================================================
  with np.errstate(divide="ignore", invalid="ignore"):
    lambda_z = np.abs(2.0 * np.pi * bz_d / (omega * np.sqrt(rho * h + bsq)))
  Q_z  = (lambda_z / dz).astype(np.float32)
  # Q_phi = np.pi * np.abs(b_phi) / (Rcyl * omega * np.sqrt((rho * h + bsq) * dz))
  Q_ok = disk & np.isfinite(Q_z)

  wq       = dM[Q_ok]
  Q_z_mean = wmean(Q_z[Q_ok], wq)
  Q_z_pct  = wpercentile(Q_z[Q_ok], wq, [5, 25, 50, 75, 95]).tolist()
  Q_z_f10  = float(wq[Q_z[Q_ok] > 10.0].sum() / wq.sum()) if wq.sum() > 0.0 else np.nan

  del b_phi # Free up memory.

  # ======================================================================
  # INTEGRALS
  # ======================================================================
  def integrate(mask):
    w   = dM[mask]
    M   = float(w.sum())
    out = {
      "n_cells": int(mask.sum()),
      "M_MSUN_CGS": M,
      "M_g": M * MSUN_CGS,
      "J": float(dJ[mask].sum()),
      "J_mhd": float(dJ_mhd[mask].sum()),
    }
    if M > 0:
      out.update({
        "j_mean"          : out["J"] / M,
        "Ye_mean"         : wmean(ye[mask], w),
        "s_mean"          : wmean(entropy[mask], w),
        "T_mean_MeV"      : wmean(temp[mask], w),
        "rho_mean_cgs"    : wmean(rho[mask], w) * RHO_UNIT,
        "rho_max_cgs"     : float(rho[mask].max()) * RHO_UNIT,
        "T_max_MeV"       : float(temp[mask].max()),
        "Rcyl_mean"       : wmean(Rcyl[mask], w),
        "rsph_mean"       : wmean(rsph[mask], w),
        "z_rms"           : float(np.sqrt(wmean(zg[mask]**2, w))),
        "z_half"          : float(wpercentile(np.abs(zg[mask]), w, [50])[0]),
        "Omega_mean"      : wmean(omega[mask], w),
        "bsq_mean"        : wmean(bsq[mask], w),
        "beta_plasma_mean": wmean(2.0 * press[mask] / np.maximum(bsq[mask], 1e-30), w),
        "sigma_mean"      : wmean(bsq[mask] / rho[mask], w),
        "Q_z_mean"        : Q_z_mean,
        "Q_z_pct"         : Q_z_pct,
        "Q_z_f10"         : Q_z_f10,
        "B_rms_G"         : float(np.sqrt(wmean(Bsq[mask], w))) * B_UNIT,
        "B_max_G"         : float(np.sqrt(Bsq[mask].max())) * B_UNIT,
        "E_mag"           : float((0.5 * bsq[mask] * W[mask] * sqrtg[mask] * dV[mask]).sum()),
        "Ye_pct"          : wpercentile(ye[mask], w, [5, 25, 50, 75, 95]).tolist(),
        "s_pct"           : wpercentile(entropy[mask], w, [5, 25, 50, 75, 95]).tolist(),
        "T_pct"           : wpercentile(temp[mask], w, [5, 25, 50, 75, 95]).tolist(),
        "R_pct"           : wpercentile(Rcyl[mask], w, [5, 25, 50, 75, 95]).tolist(),
      })
      out["v_mean"] = float(np.sqrt(np.maximum(1.0 - 1.0 / wmean(W[mask], w)**2, 0.0)))
    return out

  res_disk        = integrate(disk)
  # res_ejecta_bern = integrate(dense & outside_bh & unbound_bern)
  # res_ejecta_geo  = integrate(dense & outside_bh & unbound_geo)
  # res_all_out     = integrate(dense & outside_bh)

  # Sensitivity of the disk mass to the density floor.
  sens = []
  for cut_cgs in (1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12):
    base = (rho > cut_cgs / RHO_UNIT) & outside_bh
    mk = base & not_remnant & ~unbound
    if r_disk_max is not None:
      mk &= Rcyl < r_disk_max
    sens.append((cut_cgs, float(dM[mk].sum()), int(mk.sum()),
                  float(dM[base & unbound].sum())))

  # Enclosed disk mass as a function of cylindrical radius.
  encl = []
  for rr in (10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0):
    mk = disk & (Rcyl < rr)
    encl.append((rr, float(dM[mk].sum())))

  # Mass bookkeeping.
  m_total_kept = float(dM.sum())
  m_inside_bh  = float(dM[~outside_bh].sum())
  m_remnant    = float(dM[dense & outside_bh & ~not_remnant].sum())

  # ======================================================================
  # RADIAL PROFILES
  # ======================================================================
  rmin_prof = max(r_exclude, 1.0)
  edges     = np.geomspace(rmin_prof, rmax, nbins + 1)
  idx       = np.digitize(Rcyl[disk], edges) - 1
  valid     = (idx >= 0) & (idx < nbins)
  idx       = idx[valid]

  wd    = dM[disk][valid]
  prof  = {"R_lo": edges[:-1], "R_hi": edges[1:],
            "R_mid": np.sqrt(edges[:-1] * edges[1:])}
  counts = np.bincount(idx, minlength=nbins)
  mass   = np.bincount(idx, weights=wd, minlength=nbins)

  def wprof(q):
    s = np.bincount(idx, weights=wd * q[disk][valid], minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
      return np.where(mass > 0, s / mass, np.nan)

  zmean = wprof(zg)
  zsq   = wprof(zg**2)
  with np.errstate(invalid="ignore"):
    H = np.sqrt(np.maximum(zsq - zmean**2, 0.0))

  dz_mid = np.abs(zg[disk][valid] - np.nan_to_num(zmean)[idx])
  z_half = binned_wquantile(idx, dz_mid, wd, nbins, 0.5)
  z_90   = binned_wquantile(idx, dz_mid, wd, nbins, 0.9)
  del dz_mid # Free up memory.
  area = np.pi * (edges[1:]**2 - edges[:-1]**2)

  prof.update({
    "n_cells": counts,
    "M_shell_MSUN_CGS": mass,
    "M_cum_MSUN_CGS"  : np.cumsum(mass),
    "Sigma_code"      : np.where(area > 0, mass / area, np.nan),
    "Sigma_g_cm2"     : np.where(area > 0, mass / area, np.nan) * MSUN_CGS / L_UNIT**2,
    "rho_mean_g_cm3"  : wprof(rho) * RHO_UNIT,
    "Ye_mean"         : wprof(ye),
    "s_mean_kB"       : wprof(entropy),
    "T_mean_MeV"      : wprof(temp),
    "Omega_code"      : wprof(omega),
    "Omega_rad_s"     : wprof(omega) / T_UNIT,
    "j_spec"          : wprof(j_spec),
    "z_mean"          : zmean,
    "H"               : H,
    "H_over_R"        : H / prof["R_mid"],
    "z_half"          : z_half,
    "z_half_over_R"   : z_half / prof["R_mid"],
    "z_90"            : z_90,
    "beta_plasma"     : wprof(2.0 * press / np.maximum(bsq, 1e-30)),
    "B_rms_G"         : np.sqrt(np.maximum(wprof(Bsq), 0.0)) * B_UNIT,
    "v_R_code"        : wprof(v_rad),
  })

  # ======================================================================
  # HISTOGRAMS
  # ======================================================================
  wd_all = dM[disk]
  hists = {}
  for name, q, bins in (("Ye", ye[disk], np.linspace(0.0, 0.6, 61)),
                        ("entropy_kB", entropy[disk], np.linspace(0.0, 40.0, 81)),
                        ("T_MeV", temp[disk], np.geomspace(0.4, 20.0, 61))):
    hh, ee = np.histogram(q, bins=bins, weights=wd_all)
    hists[name] = (ee, hh)

  # ======================================================================
  # OUTPUT
  # ======================================================================
  scalars = {
    "time_code": time, "time_ms": time * T_UNIT * 1e3,
    "center": center.tolist(), "r_ah": r_ah, "r_exclude": r_exclude,
    "rho_cut_g_cm3": rho_cut, "rho_remnant_g_cm3": rho_remnant,
    "bound_criterion": bound_criterion, "disk": res_disk,
    "m_total_above_floor": m_total_kept, "m_inside_r_exclude": m_inside_bh,
    "m_inside_remnant": m_remnant, "r_disk_max": r_disk_max,
    "rho_cut_sensitivity": [{"rho_cut_g_cm3": c, "M_disk_MSUN_CGS": mv, "cells": n,
                              "M_ejecta_MSUN_CGS": me} for c, mv, n, me in sens],
    "enclosed_disk_mass": [{"R": r, "M_MSUN_CGS": mv} for r, mv in encl],
    "units": {"L_cm": L_UNIT, "T_s": T_UNIT, "rho_g_cm3": RHO_UNIT,
              "P_erg_cm3": P_UNIT, "B_G": B_UNIT, "J_cgs": J_UNIT},
  }

  with open(os.path.join(outdir, "scalars", f"disk_scalars_{snapshot}.json"), "w") as fp:
    json.dump(scalars, fp, indent=2)

  keys = list(prof.keys())
  with open(os.path.join(outdir, "profiles", f"disk_profiles_{snapshot}.csv"), "w") as fp:
    fp.write(",".join(keys) + "\n")
    for b in range(nbins):
      fp.write(",".join(f"{np.asarray(prof[k])[b]:.8e}" for k in keys) + "\n")

  with open(os.path.join(outdir, "histograms", f"disk_histograms_{snapshot}.csv"), "w") as fp:
    fp.write("quantity,bin_lo,bin_hi,disk_mass_MSUN_CGS\n")
    for name, (ee, hh) in hists.items():
      for b in range(len(hh)):
        fp.write(f"{name},{ee[b]:.8e},{ee[b + 1]:.8e},{hh[b]:.8e}\n")

# ======================================================================
# CONSTANTS AND CONVERSIONS
# ======================================================================
CLIGHT_CGS = 2.99792458e10   # cm/s
GNEWT_CGS  = 6.67408e-8      # cm^3 g^-1 s^-2
MSUN_CGS   = 1.98848e33      # g
MEV        = 1.6021766208e-6 # erg
KBOLTZ     = 1.38064852e-16  # erg/K

L_UNIT   = GNEWT_CGS * MSUN_CGS / CLIGHT_CGS**2 # cm  (1 code length)
T_UNIT   = L_UNIT / CLIGHT_CGS                  # s   (1 code time)
RHO_UNIT = MSUN_CGS / L_UNIT**3                 # g/cm^3
P_UNIT   = RHO_UNIT * CLIGHT_CGS**2             # erg/cm^3
B_UNIT   = np.sqrt(4.0 * np.pi * P_UNIT)        # Gauss
J_UNIT   = GNEWT_CGS * MSUN_CGS**2 / CLIGHT_CGS # g cm^2 / s
KM       = 1.0e5

MN_MEV = 939.56535                # neutron mass, must match the table's `mn` scalar
MEV_PER_FM3_TO_CGS = MEV * 1.0e39 # erg/cm^3 per MeV/fm^3


# ======================================================================
# ATHENAK BINARY READER
# ======================================================================
def parse_par_dump(header):
  """Turn the <block>/key = value par dump carried in the bin header into a dict."""
  pars, block = {}, None
  for line in header:
    line = line.strip()
    if line.startswith("<") and line.endswith(">"):
      block = line[1:-1]
      pars.setdefault(block, {})
    elif "=" in line and block is not None:
      key, val = line.split("=", 1)
      pars[block][key.strip()] = val.strip()
  return pars


def par(pars, block, key, cast=float, default=None):
  try:
    return cast(pars[block][key])
  except (KeyError, ValueError):
    return default


def block_cell_geometry(fd):
  """Cell-centre coordinates and cell volume for every meshblock."""
  geom = fd["mb_geometry"]
  nx1, nx2, nx3 = fd["nx1_out_mb"], fd["nx2_out_mb"], fd["nx3_out_mb"]
  dx1 = (geom[:, 1] - geom[:, 0]) / nx1
  dx2 = (geom[:, 3] - geom[:, 2]) / nx2
  dx3 = (geom[:, 5] - geom[:, 4]) / nx3
  xv = geom[:, 0, None] + dx1[:, None] * (np.arange(nx1) + 0.5)
  yv = geom[:, 2, None] + dx2[:, None] * (np.arange(nx2) + 0.5)
  zv = geom[:, 4, None] + dx3[:, None] * (np.arange(nx3) + 0.5)
  return xv, yv, zv, dx1 * dx2 * dx3, dx1, dx2, dx3


def find_snapshots(tag: str, bindir: str):
  """Locates 3D snapshots or all files with the 3D signature."""
  pat = f"*{tag}_3D.*.bin"
  hits = sorted(glob.glob(os.path.join(bindir, pat)))
  if not hits:
      raise SystemExit(f"No file matching {pat} in {bindir}!")

  return hits

# ======================================================================
# TRACKER AND HORIZON DATA
# ======================================================================
def load_tracker(path: str) -> dict:
  """Loads the tracker data."""
  data = np.loadtxt(path, comments="#")
  out = {
    "time": data[:,1],
    "x"   : data[:,2],
    "y"   : data[:,3],
    "z"   : data[:,4]
  }

  return out


def load_horizon(path: str) -> dict:
  """Load the horizon data."""
  data = np.loadtxt(path, comments="#")
  out = {
    "time"  : data[:,1],
    "mass"  : data[:,2],
    "radius": data[:,-2]
  }

  return out

# ======================================================================
# COMPOSE EOS TABLE
# ======================================================================
class EOSTable:
  """Load a PyCompOSE HDF5 table and build interpolators.

  Sets the following fields:
    interpolator_Q1, interpolator_Q2, interpolator_Q7 - interpolators for Q1, Q2, Q7
    mn_mev - neutron mass [MeV]
    nb_min, nb_max, ye_min, ye_max, t_min, t_max - table bounds
  """
  def __init__(self, path):
    # Read the table.
    with h5py.File(path, 'r') as f:
      nb = f['nb'][:]         # [fm^-3]
      yq = f['yq'][:]         # [dim.less]
      t  = f['t'][:]          # [MeV]
      mn = float(f['mn'][()]) # neutron mass [MeV]
      Q1 = f['Q1'][:]         # p / nb [MeV]
      Q2 = f['Q2'][:]         # s [kB / baryon]
      Q7 = f['Q7'][:]         # e / (nb * m_n) - 1

    # AthenaK convention.
    self.log_nb = np.log(nb)
    self.log_t  = np.log(t)
    self.ye     = yq

    self.nb_min, self.nb_max = self.log_nb.min(), self.log_nb.max()
    self.t_min, self.t_max   = self.log_t.min(), self.log_t.max()
    self.ye_min, self.ye_max = self.ye.min(), self.ye.max()

    if not np.isclose(mn, MN_MEV):
      raise SystemExit(f"Table's mn with {mn}MeV does not match {MN_MEV}MeV!")
    else:
      self.mb = mn

    self.hmin = np.min(1.0 + Q7 + Q1 / self.mb)

    self.interpolator_Q1 = RegularGridInterpolator(
      (self.log_nb, self.ye, self.log_t), Q1,
      method='linear', bounds_error=False, fill_value=None,
    )

    self.interpolator_Q2 = RegularGridInterpolator(
      (self.log_nb, self.ye, self.log_t), Q2,
      method='linear', bounds_error=False, fill_value=None,
    )

    self.interpolator_Q7 = RegularGridInterpolator(
      (self.log_nb, self.ye, self.log_t), Q7,
      method='linear', bounds_error=False, fill_value=None,
    )

# ======================================================================
# WEIGHTED STATISTICS HELPER
# ======================================================================
def wmean(x, w):
    tot = w.sum()
    return float((x * w).sum() / tot) if tot > 0 else float("nan")


def binned_wquantile(idx, x, w, nbins, q):
    """Weighted quantile of x within each bin, in one pass.

    Used for the half-mass scale height: rms(z) is badly inflated by the
    low-density polar material that shares a cylindrical radius with the disk,
    so the |z| containing half the shell mass is the more robust thickness.
    """
    order = np.lexsort((x, idx))
    xs, ws, bs = x[order], w[order], idx[order]
    cum = np.cumsum(ws)
    start = np.searchsorted(bs, np.arange(nbins), side="left")
    stop = np.searchsorted(bs, np.arange(nbins), side="right")
    out = np.full(nbins, np.nan)
    for b in range(nbins):
        lo, hi = start[b], stop[b]
        if hi <= lo:
            continue
        base = cum[lo - 1] if lo > 0 else 0.0
        target = base + q * (cum[hi - 1] - base)
        out[b] = xs[min(np.searchsorted(cum[lo:hi], target) + lo, hi - 1)]
    return out


def wpercentile(x, w, q):
    """Weighted percentiles; q in percent."""
    if x.size == 0 or w.sum() <= 0:
        return np.full(np.shape(q), np.nan)
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cdf = np.cumsum(ws) - 0.5 * ws
    cdf /= ws.sum()
    return np.interp(np.asarray(q, dtype=float) / 100.0, cdf, xs)


def analyze(args):
  """Run the disk analysis over every snapshot in `bindir`.

  Parameters (from args; in detail)
  ---------------------------------
  args.bindir (str): path to the 3D binary files. IMPORTANT: files need
                     to hold the signature *._3D.*.bin. The output needed
                     at each output time is `*.mhd_w_bcc_3D.*.bin`,
                     `*.adm_3D.*.bin`, `*.alpha_3D.*.bin`, `*.betax_3D.*.bin`,
                     `*.betay_3D.*.bin`, `*.betaz_3D.*.bin`.
  args.drop_first_bins (int): how many bins to drop at the start (during inspiral).
  args.eos_table (str): path to the .h5 CompOSE table used for the evolution.
  args.tracker (str): path to one of the tracker files (merged into one segment
                      with BatchMerger; currently only supported this way).
  args.horizon (str): path to the horizon summary file if existing (merged into
                      one segment with BatchMerger; currently only supported this way).
  args.outdir (str): path to the output directory, where files are written.
  args.rho_cut (float): lower density threshold to identify the disk (default: 1e6 g/cm^3).
                        Sensitivity information about that threshold is also in
                        the output.
  args.rho_keep (float): cells below this threshold are dropped from the start (in g/cm^3).
                         Default is 1.5 times the simulation's density floor.
  args.rho_remnant (float): upper density threshold to identify a possible remnant NS (in g/cm^3).
                            Default is 1e13 g/cm^3.
  args.bound_criterion (str): either use the Bernoulli or geodesic criterion to separate the
                              ejecta from the disk.
  args.r_exclude (float): exclude the region r < this (in code units) around a remnant
                          BH. Default is the AH radius if given.
  args.center (list): center of the compact object. If not given, the tracker file has to
                      be given.
  args.r_disk_max (float): outer cylindrical radius for the disk in code units (without the
                           bound tidal tail might count as part of the disk).
  args.nbins (int): number of cylindrical radius bins for the radial profiles. Default: 64.
  args.rmax (float): outer edge of the radial profiles (in code units).
  args.n_workers (int): number of worker processes per snapshot loop.
  """
  # Find the files with the signature.
  files = {tag: find_snapshots(tag, args.bindir) for tag in VARIABLES}

  # Drop snapshots at the beginning.
  if args.drop_first_bins is not None:
    for tag in VARIABLES:
      # Make sure that if one snapshot is missing for one tag,
      # that this one is skipped.
      files[tag] = [
        f for f in sorted(files[tag])
        if int(re.search(r"\.(\d+)\.bin$", f).group(1)) >= args.drop_first_bins
      ]

  lengths = [len(files[tag]) for tag in VARIABLES]
  equal_len = len(set(lengths)) == 1
  if not equal_len:
    raise SystemExit("There are not equally many snapshot for each variable!")
  else:
    print(f"$ Found {len(files["mhd_w_bcc"])} disk snapshots...")

  # Load the EOS and tracker/horizon files.
  eos       = EOSTable(args.eos_table)
  if args.tracker is not None:
    tracker   = load_tracker(args.tracker)
  else:
    tracker   = None

  if args.horizon is not None:
    horizon   = load_horizon(args.horizon)
  else:
    horizon   = None

  init_args = (eos, tracker, horizon)

  # Prepare the worker arguments.
  n_workers = args.n_workers if args.n_workers is not None else min(8, cpu_count())

  def _snapshot_files(idx):
    """Path to every tagged file for a given snapshot index."""
    return tuple(files[v][idx] for v in VARIABLES)
  worker_args = [_snapshot_files(i) +
                 (args.rho_cut, args.rho_keep, args.rho_remnant,
                  args.bound_criterion, args.r_exclude, args.center,
                  args.r_disk_max, args.nbins, args.rmax, args.outdir)
                 for i in range(lengths[0])]
  print(f"$ Processing {len(worker_args)} snapshots with n_worker={n_workers}...")

  # Make directories.
  os.makedirs(os.path.join(args.outdir, "scalars"), exist_ok=True)
  os.makedirs(os.path.join(args.outdir, "profiles"), exist_ok=True)
  os.makedirs(os.path.join(args.outdir, "histograms"), exist_ok=True)

  # Parallel snapshot loop.
  if n_workers > 1:
    with Pool(n_workers,
              initializer=_init_disk_worker,
              initargs=init_args) as pool:
      for _ in tqdm(
        pool.imap_unordered(
          _process_snapshot_disk, worker_args,
          chunksize=max(1, len(worker_args) // (n_workers * 4))),
          total=len(worker_args),
          desc="Progress"):
        pass
  else:
    _init_disk_worker(*init_args)
    for a in tqdm(worker_args, desc="Progress"):
      _process_snapshot_disk(a)

# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main(argv=None):
  ap = argparse.ArgumentParser(
      description="Post-merger disk diagnostics for AthenaK data.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  ap.add_argument("--bindir", required=True,
                  help="directory holding the 3D bin files")
  ap.add_argument("--drop-first-bins", default=None, type=int,
                  help="Drop that many .bin files at the start (inspiral).")
  ap.add_argument("--eos-table",
                  help="CompOSE .h5 used by the run (for h and entropy)")
  ap.add_argument("--tracker", default=None,
                    help="path to the tracker file")
  ap.add_argument("--horizon", default=None,
                    help="path to the horizon file")
  ap.add_argument("--outdir", required=True, help="where to write the results")

  ap.add_argument("--rho-cut", type=float, default=1.0e6,
                  help="disk lower density threshold [g/cm^3]")
  ap.add_argument("--rho-keep", type=float, default=None,
                  help="cells below this are dropped up front [g/cm^3] "
                       "[default: 1.5 x the run's dfloor]")
  ap.add_argument("--rho-remnant", type=float, default=1.0e13,
                  help="upper density threshold excluding a remnant NS [g/cm^3]")
  ap.add_argument("--bound-criterion", choices=("bernoulli", "geodesic"),
                  default="bernoulli",
                  help="which criterion separates the disk from the ejecta")
  ap.add_argument("--r-exclude", type=float, default=None,
                  help="exclude r < this (code units) around the BH"
                       "[default: the AH radius from the horizon-finder]")
  ap.add_argument("--center", type=float, nargs=3, default=None,
                  metavar=("X", "Y", "Z"),
                  help="Centre of compact object")
  ap.add_argument("--r-disk-max", type=float, default=None,
                  help="optional outer cylindrical radius for the disk [code units]; "
                       "without it the bound tidal tail counts as disk")
  ap.add_argument("--nbins", type=int, default=64,
                  help="number of cylindrical-radius bins")
  ap.add_argument("--rmax", type=float, default=1000.0,
                  help="outer edge of the radial profiles [code units]")

  ap.add_argument("--n-workers", type=int, default=None,
                  help="Worker processes for the snapshot loop. "
                        "Default: min(8, cpu_count()).")
  args = ap.parse_args()

  analyze(args)


if __name__ == "__main__":
  main()

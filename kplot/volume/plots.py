"""
Lightweight plotter for kplot.volume.disk output.

Reads the per-snapshot histogram and profile CSVs written by disk.analyze()
and renders one frame per snapshot for each, so scripts/system/make_movies.sh
can turn them into an evolution movie.

    outdir/histograms/disk_histograms_<snap>.csv -> figdir/histograms/disk_histograms_<snap>.png
    outdir/profiles/disk_profiles_<snap>.csv     -> figdir/profiles/disk_profiles_<snap>.png
    outdir/slices/disk_slices_<snap>.npz         -> figdir/Q_slices/disk_Q_slices_<snap>.png
                                                    (upper-half xz over xy),
                                                    figdir/parker_xz/disk_parker_xz_<snap>.png
    outdir/parker/disk_parker_<snap>.csv         -> figdir/scalars/disk_parker_spacetime.png
    outdir/scalars/disk_scalars_<snap>.json      -> figdir/scalars/disk_field_geometry.png
                                                    (Q_z/Q_phi, toroidal vs. poloidal energy)

Command line:
    kplot-volume-plot --outdir DIR [--figdir DIR]
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from kplot.volume.disk import (JRHO_XBINS, JRHO_YBINS_TOP, JRHO_YBINS_BOT, JRHO_R_INT_KM,
                                JRHO_THRESHOLD, JSPEC_UNIT)

HIST_QUANTITIES = [
    ("Ye", r"$Y_e$", False),
    ("entropy_kB", r"$s$  [$k_B$/baryon]", False),
    ("T_MeV", r"$T$  [MeV]", True),
]

PROFILE_PANELS = [
    ("Sigma_g_cm2", r"$\Sigma$  [g/cm$^2$]", True),
    ("rho_mean_g_cm3", r"$\rho$  [g/cm$^3$]", True),
    ("Ye_mean", r"$Y_e$", False),
    ("T_mean_MeV", r"$T$  [MeV]", True),
    ("H_over_R", r"$H/R$", False),
    ("Omega_rad_s", r"$\Omega$  [rad/s]", True),
]

SPECTRUM_CURVES = [
    ("P_v", r"$P_v(k)$", "tab:blue"),
    ("P_B", r"$P_B(k)$", "tab:red"),
    ("P_KE", r"$P_{KE}(k)$", "tab:green"),
]

SPECTRUM_REFS = [
    ("P_v", -5.0 / 3.0, "Kolmogorov"),
    ("P_B", 3.0 / 2.0, "Kazantsev"),
]

SCALAR_PANELS = [
    ("mass_msun", r"$M_{\mathrm{disk}}\ [M_\odot]$", False),
    ("Ye_mean", r"$\langle Y_e\rangle$", False),
    ("T_mean_MeV", r"$\langle T\rangle$ [MeV]", False),
    ("Omega_mean", r"$\langle \Omega \rangle$", False),
    ("Q_z", r"$Q_z$", False),
    ("Rcyl_mean", r"$\langle R_{\mathrm{cycl}}\rangle$ [code units]", False),
    ("z_rms", r"$z_{\mathrm{rms}}$ [code units]", False),
    ("B_max_G", r"$\mathrm{max}(B)$ [G]", True),
]

Q_SLICE_RANGE = (1.0e-1, 1.0e2)
Q_SLICE_CMAP  = "viridis"
Q_MASK_COLOR  = "0.6"

Q_SPACETIME_KEY   = "Q_z_mean"
Q_SPACETIME_RANGE = (1.0e-2, 1.0e1)
Q_PHI_SPACETIME_RANGE = (1.0e-1, 1.0e2)

# Field geometry scalars and the well-resolved MRI references of Hawley, Guan &
# Krolik (2011): Q_z >~ 10 with Q_phi >~ 20 (Sec. 3.5), <B_R^2>/<B_phi^2> ~ 0.2
# (Sec. 3.4) and alpha_mag ~ 0.3-0.4 (Sec. 3.3).
FIELD_GEOMETRY_KEYS = ("Q_z_pct", "Q_phi_pct", "E_tor", "E_pol", "E_R", "E_z", "E_B",
                       "BR2_Bphi2", "Bpol2_Btor2", "alpha_mag")
HGK_Q_REFS     = {"Q_z": 10.0, "Q_phi": 20.0}
HGK_RATIO_REFS = {"BR2_Bphi2": 0.2, "alpha_mag": 0.4}

RHO_CONTOUR_LEVELS = [1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e12]

# Parker panels (Fig. 8 of Jiang et al. 2025, plus the undular Newcomb criterion):
# (npz key, label, cmap, norm class, norm range, overlaid streamlines).
PARKER_SLICE_PANELS = [
    ("P_xz", r"$\mathcal{P}$", "RdBu_r", Normalize, (-0.5, 0.5), "B"),
    ("N_xz", r"$\mathcal{N}_u$", "RdBu_r", Normalize, (-1.0, 1.0), "B"),
    ("betainv_xz", r"$\beta^{-1}$", "viridis", LogNorm, (1.0e-6, 1.0e1), None),
    ("s_xz", r"$s$  [$k_B$/baryon]", "magma", Normalize, (0.0, 40.0), "v"),
    ("W_xz", r"$W$", "plasma", Normalize, (1.0, 1.1), None),
]
PARKER_STREAMS = {"B": ("Bx_xz", "Bz_xz"), "v": ("vx_xz", "vz_xz")}

# Spacetime panels: (csv key, label, cmap, norm class, norm range, zero contour).
PARKER_SPACETIME_PANELS = [
    ("P_p50", r"$\mathcal{P}$  (median)", "RdBu_r", Normalize, (-0.5, 0.5), True),
    ("N_p50", r"$\mathcal{N}_u$  (median)", "RdBu_r", Normalize, (-1.0, 1.0), True),
    ("betainv_mean", r"$\langle\beta^{-1}\rangle_V$", "viridis", LogNorm, (1.0e-6, 1.0e1), False),
]
PARKER_RHO_STYLES = ["dotted", "dashdot", "dashed", "solid"]
PARKER_RHO_LEVELS = [1.0e9, 1.0e10, 1.0e11, 1.0e12]


def find_snapshots(outdir, kind, ending):
  """Snapshot ids for which a disk_<kind>_<snap>.<ending> file exists."""
  pat = os.path.join(outdir, kind, f"disk_{kind}_*.{ending}")
  ids = [re.search(rf"disk_{kind}_(.+)\.{ending}$", f).group(1) for f in glob.glob(pat)]
  return sorted(ids)


def load_time_ms(outdir, snap):
  """Snapshot time in ms from the matching scalars json, or NaN if missing."""
  path = os.path.join(outdir, "scalars", f"disk_scalars_{snap}.json")
  if not os.path.exists(path):
    return float("nan")
  with open(path) as fp:
    return json.load(fp)["time_ms"]


def load_jrho(path):
  return dict(np.load(path))


def load_j_mean(outdir, snap):
  path = os.path.join(outdir, "scalars", f"disk_scalars_{snap}.json")
  if not os.path.exists(path):
    return float("nan")
  with open(path) as fp:
    return json.load(fp)["disk"].get("j_mean", float("nan"))


def load_histograms(path):
  """quantity -> (bin_centers, disk_mass) from a disk_histograms_<snap>.csv file."""
  data = {}
  with open(path) as fp:
    for row in csv.DictReader(fp):
      q = row["quantity"]
      data.setdefault(q, {"lo": [], "hi": [], "mass": []})
      data[q]["lo"].append(float(row["bin_lo"]))
      data[q]["hi"].append(float(row["bin_hi"]))
      data[q]["mass"].append(float(row["disk_mass_MSUN_CGS"]))

  out = {}
  for q, v in data.items():
    lo, hi = np.array(v["lo"]), np.array(v["hi"])
    out[q] = (0.5 * (lo + hi), np.array(v["mass"]))
  return out


def load_profile(path):
  """Structured array of a disk_profiles_<snap>.csv file, keyed by column name."""
  return np.genfromtxt(path, delimiter=",", names=True)


def find_spectrum_snapshots(outdir):
  """Snapshot ids for which a spectrum_<snap>.txt file exists."""
  pat = os.path.join(outdir, "spectra", "spectrum_*.txt")
  ids = [re.search(r"spectrum_(.+)\.txt$", f).group(1) for f in glob.glob(pat)]
  return sorted(ids)


def load_spectrum(path):
  """(time, {P_v, P_B, P_KE}) from a spectrum_<snap>.txt file."""
  with open(path) as fp:
    time = float(fp.readline().split("=")[1])
  k, Pv, PB, PKE = np.loadtxt(path, unpack=True)
  return time, k, {"P_v": Pv, "P_B": PB, "P_KE": PKE}


def hist_ylim(outdir, snaps):
  """Fix the y-limits from the min/max over all snapshots."""
  gmax = {}
  for snap in snaps:
    hist = load_histograms(os.path.join(outdir, "histograms", f"disk_histograms_{snap}.csv"))
    for q, (_, mass) in hist.items():
      gmax[q] = max(gmax.get(q, 0.0), mass.max(initial=0.0))
  return {q: (m * 1e-6, m * 1.5) for q, m in gmax.items() if m > 0}


def common_r_grid(outdir, snaps):
  """Shared R_mid grid (adapts to moving excision regions)."""
  lo, hi, n = [], [], 0
  for snap in snaps:
    prof = load_profile(os.path.join(outdir, "profiles", f"disk_profiles_{snap}.csv"))
    r = np.atleast_1d(prof["R_mid"])
    lo.append(r[0]); hi.append(r[-1]); n = max(n, r.size)
  return np.geomspace(min(lo), max(hi), n)


def resample_profile(prof, R):
  """Profile dict resampled onto the common R grid."""
  r = np.atleast_1d(prof["R_mid"])
  logR, logr = np.log(R), np.log(r)
  out = {}
  for key, _, _ in PROFILE_PANELS:
    vals = np.atleast_1d(prof[key])
    out[key] = np.interp(logR, logr, vals, left=np.nan, right=np.nan)
  return out


def profile_ylim(outdir, snaps):
  """Fix the y-limits from the min/max over all snapshots."""
  gmax = {}
  for snap in snaps:
    prof = load_profile(os.path.join(outdir, "profiles", f"disk_profiles_{snap}.csv"))
    for key, _, _ in PROFILE_PANELS:
      vals = np.atleast_1d(prof[key])
      vals = vals[np.isfinite(vals)]
      gmax[key] = max(gmax.get(key, 0.0), vals.max(initial=0.0))

  ylim = {}
  for key, _, logy in PROFILE_PANELS:
    m = gmax.get(key, 0.0)
    if m <= 0:
      continue
    ylim[key] = (m * 1e-6, m * 1.5) if logy else (0.0, m * 1.1)
  return ylim


def plot_histograms(hist, snap, time_ms, ylim, outfile):
  """Plot mass-histograms for the current snapshot."""
  fig, axes = plt.subplots(1, len(HIST_QUANTITIES), figsize=(12, 3.5))
  for ax, (q, label, logx) in zip(axes, HIST_QUANTITIES):
    if q not in hist:
      continue
    centers, mass = hist[q]
    ax.step(centers, mass, where="mid", color="tab:blue")
    ax.set_xlabel(label)
    ax.set_yscale("log")
    if q in ylim:
      ax.set_ylim(*ylim[q])
    if logx:
      ax.set_xscale("log")
  axes[0].set_ylabel(r"$M_\mathrm{disk}$  [$M_\odot$]")

  fig.suptitle(f"snapshot {snap}, t = {time_ms:.2f} ms")
  fig.tight_layout()
  fig.savefig(outfile, dpi=150)
  plt.close(fig)


def _plot_jrho_panel(ax, H_ext, H_int, ybins):
  cmap = plt.get_cmap("magma").copy()
  cmap.set_under("0.75")
  im = ax.pcolormesh(JRHO_XBINS, ybins, np.where(H_ext > 0, H_ext, np.nan).T,
                      norm=LogNorm(vmin=JRHO_THRESHOLD), cmap=cmap, shading="auto")
  ax.pcolormesh(JRHO_XBINS, ybins, np.where(H_int > 0, 1.0, np.nan).T,
                cmap=ListedColormap(["lightskyblue"]), alpha=0.6, shading="auto")
  ax.set_xscale("log")
  ax.set_xlim(JRHO_XBINS[0], JRHO_XBINS[-1])
  ax.set_ylim(ybins[0], ybins[-1])
  ax.tick_params(which="both", direction="in", top=True, right=True)
  return im


def plot_jrho(hists, snap, time_ms, j_mean_code, outfile):
  a = j_mean_code * JSPEC_UNIT
  fig, axs = plt.subplots(2, sharex=True, figsize=(6, 8),
                           gridspec_kw={"height_ratios": [2, 1]})
  fig.subplots_adjust(hspace=0.05)

  im = _plot_jrho_panel(axs[0], hists["top_ext"], hists["top_int"], JRHO_YBINS_TOP)
  _plot_jrho_panel(axs[1], hists["bot_ext"], hists["bot_int"], JRHO_YBINS_BOT)

  axs[0].set_yscale("log")
  axs[0].set_ylabel(r"$j$  [g cm$^{-1}$ s$^{-1}$]")
  axs[1].set_xlabel(r"$\rho$  [g cm$^{-3}$]")
  axs[1].set_ylabel(r"$j/\rho$  [$10^{16}$ cm$^2$ s$^{-1}$]")

  if np.isfinite(a):
    axs[0].plot(JRHO_XBINS, a * JRHO_XBINS, "g--")
    axs[1].axhline(a * 1.0e-16, color="g", ls="--")

  legend_handles = [
    Line2D([0], [0], color="g", ls="--", label=r"$j=a\rho$"),
    Patch(facecolor="lightskyblue", label=rf"$r<{JRHO_R_INT_KM:.1f}$ km"),
  ]
  axs[0].legend(handles=legend_handles, loc="upper left", fontsize=8, frameon=False)

  cbar = fig.colorbar(im, ax=axs, location="top", orientation="horizontal",
                       pad=0.02, aspect=40, extend="min")
  cbar.set_label("mass fraction")

  fig.text(0.5, 0.005, f"snapshot {snap}, t = {time_ms:.2f} ms", ha="center", fontsize=9)
  fig.savefig(outfile, dpi=150)
  plt.close(fig)


def plot_profiles(R, prof, snap, time_ms, ylim, outfile):
  """Plot the radial profiles at the current snapshot."""
  fig, axes = plt.subplots(2, 3, figsize=(13, 7))
  for ax, (key, label, logy) in zip(axes.flat, PROFILE_PANELS):
    ax.plot(R, prof[key], color="tab:blue")
    ax.set_xlabel(r"$R$  [code units]")
    ax.set_ylabel(label)
    ax.set_xscale("log")
    ax.set_xlim(R[0], R[-1])
    if logy:
      ax.set_yscale("log")
    if key in ylim:
      ax.set_ylim(*ylim[key])

  fig.suptitle(f"snapshot {snap}, t = {time_ms:.2f} ms")
  fig.tight_layout()
  fig.savefig(outfile, dpi=150)
  plt.close(fig)


def _plot_Q_panel(ax, u, v, Q, rho, cmap):
  im = ax.pcolormesh(u, v, np.ma.masked_invalid(Q), norm=LogNorm(*Q_SLICE_RANGE),
                     cmap=cmap, shading="auto")
  if rho is not None and np.isfinite(rho).any():
    with np.errstate(divide="ignore", invalid="ignore"):
      cs = ax.contour(u, v, np.log10(rho), levels=np.log10(RHO_CONTOUR_LEVELS),
                      colors="red", linewidths=0.8)
    ax.clabel(cs, fmt=lambda val: rf"$10^{{{val:.0f}}}$", fontsize=7)
  ax.set_aspect("equal")
  return im


def plot_Q_slice(sl, snap, time_ms, outfile):
  """Upper half of the xz plane stacked on the xy plane (as in kplot.system.plotter)."""
  u = sl["u"]
  top = u >= 0.0
  cmap = plt.get_cmap(Q_SLICE_CMAP).copy()
  cmap.set_bad(Q_MASK_COLOR)
  rho = lambda key: sl[key] if key in sl.files else None

  fig = plt.figure(figsize=(4.5, 6))
  gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.03, right=0.83)
  ax_xz = fig.add_subplot(gs[0])
  ax_xy = fig.add_subplot(gs[1], sharex=ax_xz)

  rho_xz = rho("rho_xz")
  im = _plot_Q_panel(ax_xz, u, u[top], sl["Q_xz"][top],
                     None if rho_xz is None else rho_xz[top], cmap)
  _plot_Q_panel(ax_xy, u, u, sl["Q_xy"], rho("rho_xy"), cmap)

  ax_xz.set_xlim(u[0], u[-1]); ax_xz.set_ylim(0.0, u[-1])
  ax_xy.set_ylim(u[0], u[-1])
  plt.setp(ax_xz.get_xticklabels(), visible=False)
  ax_xz.set_ylabel(r"$z - z_c$")
  ax_xy.set_xlabel(r"$x - x_c$  [code units]")
  ax_xy.set_ylabel(r"$y - y_c$")
  ax_xz.set_title(f"snapshot {snap}, t = {time_ms:.2f} ms")
  ax_xy.yaxis.set_major_locator(MaxNLocator(prune="upper"))
  ax_xz.legend(handles=[Patch(facecolor=Q_MASK_COLOR, label="masked")],
               loc="upper right", fontsize=7)
  fig.align_ylabels([ax_xz, ax_xy])

  p_top = ax_xz.get_position(); p_bot = ax_xy.get_position()
  cax = fig.add_axes([p_top.x1 + 0.015, p_bot.y0, 0.04, p_top.y1 - p_bot.y0])
  cbar = fig.colorbar(im, cax=cax, extend="both")
  cbar.set_label(r"$Q_z$  ($\partial_R \Omega < 0$)")
  fig.savefig(outfile, dpi=150, bbox_inches="tight")
  plt.close(fig)


def plot_Q_spacetime(outdir, snaps, R, outfile, key=Q_SPACETIME_KEY, qrange=Q_SPACETIME_RANGE,
                     label=r"$\langle Q_z \rangle_M$  ($\partial_R \Omega < 0$)"):
  times, rows = [], []
  for snap in snaps:
    prof = load_profile(os.path.join(outdir, "profiles", f"disk_profiles_{snap}.csv"))
    if key not in prof.dtype.names:
      continue
    r = np.atleast_1d(prof["R_mid"])
    q = np.atleast_1d(prof[key])
    rows.append(np.interp(np.log(R), np.log(r), q, left=np.nan, right=np.nan))
    times.append(load_time_ms(outdir, snap))
  if not rows:
    print(f"  No {key} column in the profiles; skipping its spacetime diagram.")
    return

  order = np.argsort(times)
  t = np.asarray(times)[order]
  Q = np.ma.masked_invalid(np.asarray(rows)[order])

  cmap = plt.get_cmap(Q_SLICE_CMAP).copy()
  cmap.set_bad(Q_MASK_COLOR)
  fig, ax = plt.subplots(figsize=(8, 4.5))
  im = ax.pcolormesh(t, R, Q.T, norm=LogNorm(*qrange), cmap=cmap,
                     shading="nearest")
  if t.size > 1:
    ax.contour(t, R, Q.T.filled(np.nan), levels=[1.0], colors="w", linewidths=1.0)
  ax.set_yscale("log")
  ax.set_xlabel(r"$t$ [ms]")
  ax.set_ylabel(r"$R$  [code units]")
  cbar = fig.colorbar(im, ax=ax, extend="both")
  cbar.set_label(label)
  fig.tight_layout()
  fig.savefig(outfile, dpi=150)
  plt.close(fig)


def plot_field_geometry(g, outfile):
  """Q_z and Q_phi, toroidal vs. poloidal energy, and MRI saturation ratios vs. time."""
  t     = np.asarray(g["time_ms"])
  order = np.argsort(t)
  t     = t[order]
  arr   = lambda key: np.asarray(g[key], dtype=float)[order]
  fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

  # Mass-weighted median and 25-75% band of the quality factors (MRI-unstable cells).
  ax = axes[0]
  for name, label, color in (("Q_z", r"$Q_z$", "tab:blue"), ("Q_phi", r"$Q_\phi$", "tab:red")):
    pct = np.asarray(g[f"{name}_pct"], dtype=float)[order]
    ax.fill_between(t, pct[:, 1], pct[:, 3], color=color, alpha=0.25, lw=0)
    ax.plot(t, pct[:, 2], color=color, marker="o", ms=2, label=label)
    ax.axhline(HGK_Q_REFS[name], color=color, ls="--", lw=0.8)
  ax.set_yscale("log")
  ax.set_ylabel(r"$Q$  (median, 25–75%)")
  ax.legend(fontsize=8)

  ax = axes[1]
  for key, label, color in (("E_B", r"$E_B$", "k"),
                            ("E_tor", r"$E_\phi$", "tab:red"),
                            ("E_pol", r"$E_R + E_z$", "tab:blue"),
                            ("E_R", r"$E_R$", "tab:cyan"),
                            ("E_z", r"$E_z$", "tab:purple")):
    ax.plot(t, arr(key), color=color, marker="o", ms=2, label=label,
            ls=":" if key in ("E_R", "E_z") else "-")
  ax.set_yscale("log")
  ax.set_ylabel(r"Eulerian $E$  [code units]")
  ax.legend(fontsize=8, ncol=2)

  ax = axes[2]
  for key, label, color in (("BR2_Bphi2", r"$\langle B_R^2\rangle/\langle B_\phi^2\rangle$", "tab:green"),
                            ("Bpol2_Btor2", r"$\langle B_\mathrm{pol}^2\rangle/\langle B_\phi^2\rangle$", "tab:blue"),
                            ("alpha_mag", r"$\alpha_\mathrm{mag}$", "tab:orange")):
    ax.plot(t, arr(key), color=color, marker="o", ms=2, label=label)
    if key in HGK_RATIO_REFS:
      ax.axhline(HGK_RATIO_REFS[key], color=color, ls="--", lw=0.8)
  vals = np.concatenate([arr(k) for k in ("BR2_Bphi2", "Bpol2_Btor2", "alpha_mag")])
  ax.set_ylim(min(0.0, np.nanmin(vals)) if np.isfinite(vals).any() else 0.0,
              max(0.5, 1.1 * np.nanmax(vals)) if np.isfinite(vals).any() else 0.5)
  ax.set_ylabel("ratio  (dashed: HGK 2011 saturation)")
  ax.legend(fontsize=8)

  for ax in axes:
    ax.set_xlabel(r"$t$ [ms]")
    if t.size > 1:
      ax.set_xlim(t[0], t[-1])
  fig.tight_layout()
  fig.savefig(outfile, dpi=150)
  plt.close(fig)


def _rho_contours(ax, x, y, rho):
  """Cyan iso-density lines at PARKER_RHO_LEVELS (dotted -> solid with rho)."""
  if rho is None or np.ndim(rho) != 2 or min(np.shape(rho)) < 2 or not np.isfinite(rho).any():
    return
  with np.errstate(divide="ignore", invalid="ignore"):
    ax.contour(x, y, np.log10(rho), levels=np.log10(PARKER_RHO_LEVELS), colors="cyan",
               linestyles=PARKER_RHO_STYLES, linewidths=0.8)


def plot_parker_slice(sl, snap, time_ms, outfile):
  """Fig. 8-like xz plane: P, N_u, 1/beta, s, W with field and fluid lines."""
  u = sl["u"]
  fig, axes = plt.subplots(1, len(PARKER_SLICE_PANELS), sharey=True,
                           figsize=(4.2 * len(PARKER_SLICE_PANELS), 4.9))
  for ax, (key, label, cmap_name, norm_cls, vrange, stream) in zip(axes, PARKER_SLICE_PANELS):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(Q_MASK_COLOR)
    im = ax.pcolormesh(u, u, np.ma.masked_invalid(sl[key]), norm=norm_cls(*vrange),
                       cmap=cmap, shading="auto", rasterized=True)
    if stream is not None:
      kx, kz = PARKER_STREAMS[stream]
      ax.streamplot(u, u, np.nan_to_num(sl[kx]), np.nan_to_num(sl[kz]), color="k",
                    density=1.0, linewidth=0.4, arrowsize=0.5)
    _rho_contours(ax, u, u, sl["rho_xz"])
    ax.set_xlim(u[0], u[-1])
    ax.set_ylim(u[0], u[-1])
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x - x_c$  [code units]")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", location="top",
                        extend="both", pad=0.02)
    cbar.set_label(label)
  axes[0].set_ylabel(r"$z - z_c$  [code units]")
  rho_lines = [Line2D([], [], color="cyan", linestyle=ls, label=rf"$10^{{{np.log10(lv):.0f}}}$")
               for ls, lv in zip(PARKER_RHO_STYLES, PARKER_RHO_LEVELS)]
  axes[-1].legend(handles=rho_lines, title=r"$\rho$ [g/cm$^3$]", loc="lower right",
                  fontsize=7, title_fontsize=7)
  fig.suptitle(f"snapshot {snap}, t = {time_ms:.2f} ms", y=0.02, va="bottom", fontsize=10)
  fig.savefig(outfile, dpi=150, bbox_inches="tight")
  plt.close(fig)


def plot_parker_spacetime(outdir, snaps, outfile):
  """Fig. 7-like r-t diagrams of P, N_u and 1/beta over all polar angles."""
  profs, times = [], []
  for snap in snaps:
    profs.append(load_profile(os.path.join(outdir, "parker", f"disk_parker_{snap}.csv")))
    times.append(load_time_ms(outdir, snap))
  if len(profs) < 2:
    print("  Fewer than two Parker profiles; skipping Parker spacetime diagram.")
    return

  # Common r grid (the inner edge follows the moving excision radius).
  lo = min(np.atleast_1d(p["r_mid"])[0] for p in profs)
  hi = max(np.atleast_1d(p["r_mid"])[-1] for p in profs)
  r  = np.geomspace(lo, hi, max(np.atleast_1d(p["r_mid"]).size for p in profs))
  def grid(key):
    return np.asarray([np.interp(np.log(r), np.log(np.atleast_1d(p["r_mid"])),
                                 np.atleast_1d(p[key]), left=np.nan, right=np.nan)
                       for p in profs])
  order = np.argsort(times)
  t     = np.asarray(times)[order]
  rho   = grid("rho_mean_g_cm3")[order]

  fig, axes = plt.subplots(1, len(PARKER_SPACETIME_PANELS), sharey=True,
                           figsize=(5.0 * len(PARKER_SPACETIME_PANELS), 5.0))
  for ax, (key, label, cmap_name, norm_cls, vrange, zero) in zip(axes, PARKER_SPACETIME_PANELS):
    Z    = np.ma.masked_invalid(grid(key)[order])
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(Q_MASK_COLOR)
    im = ax.pcolormesh(r, t, Z, norm=norm_cls(*vrange), cmap=cmap, shading="nearest")
    if zero:
      ax.contour(r, t, Z.filled(np.nan), levels=[0.0], colors="w", linewidths=1.0)
    _rho_contours(ax, r, t, rho)
    ax.set_xscale("log")
    ax.set_xlabel(r"$r$  [code units]")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", location="top",
                        extend="both", pad=0.02)
    cbar.set_label(label)
  axes[0].set_ylabel(r"$t$ [ms]")
  fig.savefig(outfile, dpi=150, bbox_inches="tight")
  plt.close(fig)


def plot_scalars(data, outfile):
  """Plot the scalar evolution of some important disk measures."""
  fig, axes = plt.subplots(2, 4, figsize=(18,6))
  time = data["time_ms"]
  for ax, (key, label, logy) in zip(axes.flat, SCALAR_PANELS):
    ax.plot(time, data[key], marker="o", markerfacecolor="black",
            markeredgecolor="black", markersize=2)
    ax.set_xlabel(r"$t$ [ms]")
    ax.set_ylabel(label)
    ax.set_xlim(np.min(time), np.max(time))
    if logy:
      ax.set_yscale("log")

  fig.tight_layout()
  fig.savefig(outfile, dpi=150)
  plt.close(fig)


def plot_spectrum(k, spectra, snap, time, outfile):
  """Plot the isotropic power spectra with Kolmogorov/Kazantsev reference slopes."""
  fig, ax = plt.subplots(figsize=(6, 5))
  for key, label, color in SPECTRUM_CURVES:
    ax.plot(k, spectra[key], color=color, label=label)

  # Reference slopes, anchored to the spectrum a third of the way into the
  # resolved k-range.
  for key, exponent, name in SPECTRUM_REFS:
    finite = (k > 0) & np.isfinite(spectra[key]) & (spectra[key] > 0)
    if finite.sum() < 4:
      continue
    kk, ss = k[finite], spectra[key][finite]
    i_ref = len(kk) // 3
    amp = ss[i_ref] / kk[i_ref]**exponent
    k_line = kk[[0, -1]]
    ax.plot(k_line, amp * k_line**exponent, ls="--", lw=1, color="gray",
            label=rf"{name} ($k^{{{exponent:+.2f}}}$)".replace("+", ""))

  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel(r"$k$  [code units]")
  ax.set_ylabel(r"$P(k)$")
  ax.legend(fontsize=8, loc="upper right")
  ax.set_title(f"snapshot {snap}, t = {time:.2f} code units")

  fig.tight_layout()
  fig.savefig(outfile, dpi=150)
  plt.close(fig)


def plot_all(outdir, figdir, no_histograms=False, no_profiles=False, no_scalars=False,
             no_spectra=False, no_jrho=False, no_slices=False, no_parker=False):
  os.makedirs(os.path.join(figdir, "histograms"), exist_ok=True)
  os.makedirs(os.path.join(figdir, "parker_xz"), exist_ok=True)
  os.makedirs(os.path.join(figdir, "profiles"), exist_ok=True)
  os.makedirs(os.path.join(figdir, "scalars"), exist_ok=True)
  os.makedirs(os.path.join(figdir, "spectra"), exist_ok=True)
  os.makedirs(os.path.join(figdir, "jrho"), exist_ok=True)
  os.makedirs(os.path.join(figdir, "Q_slices"), exist_ok=True)

  if not no_histograms:
    snaps = find_snapshots(outdir, "histograms", "csv")
    print(f"$ Plotting {len(snaps)} histogram frames...")
    ylim = hist_ylim(outdir, snaps)
    for snap in snaps:
      hist = load_histograms(os.path.join(outdir, "histograms", f"disk_histograms_{snap}.csv"))
      outfile = os.path.join(figdir, "histograms", f"disk_histograms_{snap}.png")
      plot_histograms(hist, snap, load_time_ms(outdir, snap), ylim, outfile)

  if not no_jrho:
    snaps = find_snapshots(outdir, "jrho", "npz")
    print(f"$ Plotting {len(snaps)} j-rho frames...")
    for snap in snaps:
      hists = load_jrho(os.path.join(outdir, "jrho", f"disk_jrho_{snap}.npz"))
      j_mean = load_j_mean(outdir, snap)
      outfile = os.path.join(figdir, "jrho", f"disk_jrho_{snap}.png")
      plot_jrho(hists, snap, load_time_ms(outdir, snap), j_mean, outfile)

  if not no_slices:
    snaps = find_snapshots(outdir, "slices", "npz")
    print(f"$ Plotting {len(snaps)} Q slice frames...")
    for snap in snaps:
      sl = np.load(os.path.join(outdir, "slices", f"disk_slices_{snap}.npz"))
      outfile = os.path.join(figdir, "Q_slices", f"disk_Q_slices_{snap}.png")
      plot_Q_slice(sl, snap, load_time_ms(outdir, snap), outfile)

  if not no_parker:
    # Slices from before the Parker analysis lack the P_xz etc. keys.
    snaps = [s for s in find_snapshots(outdir, "slices", "npz")
             if "P_xz" in np.load(os.path.join(outdir, "slices", f"disk_slices_{s}.npz")).files]
    print(f"$ Plotting {len(snaps)} Parker slice frames...")
    for snap in snaps:
      sl = np.load(os.path.join(outdir, "slices", f"disk_slices_{snap}.npz"))
      outfile = os.path.join(figdir, "parker_xz", f"disk_parker_xz_{snap}.png")
      plot_parker_slice(sl, snap, load_time_ms(outdir, snap), outfile)

    print(f"$ Plotting Parker spacetime diagram...")
    plot_parker_spacetime(outdir, find_snapshots(outdir, "parker", "csv"),
                          os.path.join(figdir, "scalars", "disk_parker_spacetime.png"))

  if not no_profiles:
    snaps = find_snapshots(outdir, "profiles", "csv")
    print(f"$ Plotting {len(snaps)} profile frames...")
    ylim = profile_ylim(outdir, snaps)
    R = common_r_grid(outdir, snaps)
    for snap in snaps:
      prof = load_profile(os.path.join(outdir, "profiles", f"disk_profiles_{snap}.csv"))
      prof = resample_profile(prof, R)
      outfile = os.path.join(figdir, "profiles", f"disk_profiles_{snap}.png")
      plot_profiles(R, prof, snap, load_time_ms(outdir, snap), ylim, outfile)

    print(f"$ Plotting Q spacetime diagrams...")
    plot_Q_spacetime(outdir, snaps, R,
                     os.path.join(figdir, "scalars", "disk_Q_spacetime.png"))
    plot_Q_spacetime(outdir, snaps, R,
                     os.path.join(figdir, "scalars", "disk_Qphi_spacetime.png"),
                     key="Q_phi_mean", qrange=Q_PHI_SPACETIME_RANGE,
                     label=r"$\langle Q_\phi \rangle_M$  ($\partial_R \Omega < 0$)")

  if not no_scalars:
    snaps = find_snapshots(outdir, "scalars", "json")
    print(f"$ Plotting scalar evolution...")
    d = defaultdict(list)
    for snap in snaps:
      scal = Path(outdir) / "scalars" / f"disk_scalars_{snap}.json"
      data = json.loads(scal.read_text())

      # Fill the dictionary.
      d["time_ms"].append(data["time_ms"])
      d["mass_msun"].append(data["disk"]["M_MSUN_CGS"])
      d["Ye_mean"].append(data["disk"]["Ye_mean"])
      d["T_mean_MeV"].append(data["disk"]["T_mean_MeV"])
      d["Omega_mean"].append(data["disk"]["Omega_mean"])
      d["Q_z"].append(data["disk"]["Q_z_mean"])
      d["Rcyl_mean"].append(data["disk"]["Rcyl_mean"])
      d["z_rms"].append(data["disk"]["z_rms"])
      d["B_max_G"].append(data["disk"]["B_max_G"])

    outfile = os.path.join(figdir, "scalars", f"disk_scalars.png")
    plot_scalars(d, outfile)

    # Field geometry; scalars from before the Q_phi/field split lack these keys.
    g = defaultdict(list)
    for snap in snaps:
      data = json.loads((Path(outdir) / "scalars" / f"disk_scalars_{snap}.json").read_text())
      if "E_tor" not in data["disk"]:
        continue
      g["time_ms"].append(data["time_ms"])
      for key in FIELD_GEOMETRY_KEYS:
        g[key].append(data["disk"].get(key, np.nan))
    if g:
      print(f"$ Plotting field geometry...")
      plot_field_geometry(g, os.path.join(figdir, "scalars", "disk_field_geometry.png"))

  if not no_spectra:
    snaps = find_spectrum_snapshots(outdir)
    print(f"$ Plotting {len(snaps)} spectrum frames...")
    for snap in snaps:
      time, k, spectra = load_spectrum(os.path.join(outdir, "spectra", f"spectrum_{snap}.txt"))
      outfile = os.path.join(figdir, "spectra", f"spectrum_{snap}.png")
      plot_spectrum(k, spectra, snap, time, outfile)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--outdir", required=True,
                   help="disk analysis output directory (disk.py's --outdir)")
  ap.add_argument("--figdir", default=None,
                   help="where to write the frames [default: <outdir>/frames]")
  ap.add_argument("--no-histograms", action="store_true", help="skip histogram frames")
  ap.add_argument("--no-profiles", action="store_true", help="skip profile frames")
  ap.add_argument("--no-scalars", action="store_true", help="skip scalar evolution")
  ap.add_argument("--no-spectra", action="store_true", help="skip spectrum frames")
  ap.add_argument("--no-jrho", action="store_true", help="skip j-rho frames")
  ap.add_argument("--no-slices", action="store_true", help="skip Q slice frames")
  ap.add_argument("--no-parker", action="store_true",
                   help="skip Parker slice frames and spacetime diagram")
  args = ap.parse_args(argv)

  figdir = args.figdir or os.path.join(args.outdir, "frames")
  plot_all(args.outdir, figdir, args.no_histograms, args.no_profiles, args.no_scalars,
           args.no_spectra, args.no_jrho, args.no_slices, args.no_parker)


if __name__ == "__main__":
  main()

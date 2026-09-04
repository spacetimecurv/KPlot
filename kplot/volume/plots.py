"""
Lightweight plotter for kplot.volume.disk output.

Reads the per-snapshot histogram and profile CSVs written by disk.analyze()
and renders one frame per snapshot for each, so scripts/system/make_movies.sh
can turn them into an evolution movie.

    outdir/histograms/disk_histograms_<snap>.csv -> figdir/histograms/disk_histograms_<snap>.png
    outdir/profiles/disk_profiles_<snap>.csv     -> figdir/profiles/disk_profiles_<snap>.png

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
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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
             no_spectra=False, no_jrho=False):
  os.makedirs(os.path.join(figdir, "histograms"), exist_ok=True)
  os.makedirs(os.path.join(figdir, "profiles"), exist_ok=True)
  os.makedirs(os.path.join(figdir, "scalars"), exist_ok=True)
  os.makedirs(os.path.join(figdir, "spectra"), exist_ok=True)
  os.makedirs(os.path.join(figdir, "jrho"), exist_ok=True)

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
  args = ap.parse_args(argv)

  figdir = args.figdir or os.path.join(args.outdir, "frames")
  plot_all(args.outdir, figdir, args.no_histograms, args.no_profiles, args.no_scalars,
           args.no_spectra, args.no_jrho)


if __name__ == "__main__":
  main()

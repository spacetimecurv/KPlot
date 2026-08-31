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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def find_snapshots(outdir, kind):
    """Snapshot ids for which a disk_<kind>_<snap>.csv file exists."""
    pat = os.path.join(outdir, kind, f"disk_{kind}_*.csv")
    ids = [re.search(rf"disk_{kind}_(.+)\.csv$", f).group(1) for f in glob.glob(pat)]
    return sorted(ids)


def load_time_ms(outdir, snap):
    """Snapshot time in ms from the matching scalars json, or NaN if missing."""
    path = os.path.join(outdir, "scalars", f"disk_scalars_{snap}.json")
    if not os.path.exists(path):
        return float("nan")
    with open(path) as fp:
        return json.load(fp)["time_ms"]


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


def hist_ylim(outdir, snaps):
    """quantity -> (lo, hi) log-scale ylim, fixed from the max mass over all snapshots."""
    gmax = {}
    for snap in snaps:
        hist = load_histograms(os.path.join(outdir, "histograms", f"disk_histograms_{snap}.csv"))
        for q, (_, mass) in hist.items():
            gmax[q] = max(gmax.get(q, 0.0), mass.max(initial=0.0))
    return {q: (m * 1e-6, m * 1.5) for q, m in gmax.items() if m > 0}


def common_r_grid(outdir, snaps):
    """Shared R_mid grid spanning all snapshots, so a shrinking/growing excised
    region around the BH doesn't shift the radial axis frame to frame."""
    lo, hi, n = [], [], 0
    for snap in snaps:
        prof = load_profile(os.path.join(outdir, "profiles", f"disk_profiles_{snap}.csv"))
        r = np.atleast_1d(prof["R_mid"])
        lo.append(r[0]); hi.append(r[-1]); n = max(n, r.size)
    return np.geomspace(min(lo), max(hi), n)


def resample_profile(prof, R):
    """Profile dict resampled (log-log) onto the common R grid; NaN outside its own range."""
    r = np.atleast_1d(prof["R_mid"])
    logR, logr = np.log(R), np.log(r)
    out = {}
    for key, _, _ in PROFILE_PANELS:
        vals = np.atleast_1d(prof[key])
        out[key] = np.interp(logR, logr, vals, left=np.nan, right=np.nan)
    return out


def profile_ylim(outdir, snaps):
    """key -> (lo, hi) ylim, fixed from the max value over all snapshots."""
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


def plot_profiles(R, prof, snap, time_ms, ylim, outfile):
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


def plot_all(outdir, figdir, no_histograms=False, no_profiles=False):
    os.makedirs(os.path.join(figdir, "histograms"), exist_ok=True)
    os.makedirs(os.path.join(figdir, "profiles"), exist_ok=True)

    if not no_histograms:
        snaps = find_snapshots(outdir, "histograms")
        print(f"$ Plotting {len(snaps)} histogram frames...")
        ylim = hist_ylim(outdir, snaps)
        for snap in snaps:
            hist = load_histograms(os.path.join(outdir, "histograms", f"disk_histograms_{snap}.csv"))
            outfile = os.path.join(figdir, "histograms", f"disk_histograms_{snap}.png")
            plot_histograms(hist, snap, load_time_ms(outdir, snap), ylim, outfile)

    if not no_profiles:
        snaps = find_snapshots(outdir, "profiles")
        print(f"$ Plotting {len(snaps)} profile frames...")
        ylim = profile_ylim(outdir, snaps)
        R = common_r_grid(outdir, snaps)
        for snap in snaps:
            prof = load_profile(os.path.join(outdir, "profiles", f"disk_profiles_{snap}.csv"))
            prof = resample_profile(prof, R)
            outfile = os.path.join(figdir, "profiles", f"disk_profiles_{snap}.png")
            plot_profiles(R, prof, snap, load_time_ms(outdir, snap), ylim, outfile)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True,
                     help="disk analysis output directory (disk.py's --outdir)")
    ap.add_argument("--figdir", default=None,
                     help="where to write the frames [default: <outdir>/frames]")
    ap.add_argument("--no-histograms", action="store_true", help="skip histogram frames")
    ap.add_argument("--no-profiles", action="store_true", help="skip profile frames")
    args = ap.parse_args(argv)

    figdir = args.figdir or os.path.join(args.outdir, "frames")
    plot_all(args.outdir, figdir, args.no_histograms, args.no_profiles)


if __name__ == "__main__":
    main()

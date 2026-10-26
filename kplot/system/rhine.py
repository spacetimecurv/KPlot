#########################################################################
# File: rhine.py                                                        #
# Description: Script to analyze transition EOS + RHINE results.        #
#########################################################################

# Built-in libraries.
from pathlib import Path
import argparse
import re
from tqdm import tqdm

# Third-party libraries.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.patches import Circle
from scipy.integrate import cumulative_trapezoid

# KPlot utility.
from kplot.system.units import CGS, GEOMETRIC_SOLAR
from kplot.system.horizon import HorizonFinder
from kplot.system.tracker import Tracker
from kplot.system.checkpoints import find_segments
import kplot.system.bin_convert as bc

# Animation keys.
PANELS = [
  # (key, source, title, norm-spec, cmap, cbar label)
  ("dens", "w", r"$\rho$", ("log", 1e3, 1e15), "magma", r"$\rho$ [g cm$^{-3}$]"),
  ("s_00", "w", r"$Y_e$", ("lin", 0.0, 0.5), "coolwarm", r"$Y_e$"),
  ("temperature", "w", r"$T$", ("log", 1e-5, 3e1), "inferno", r"$T$ [MeV]"),
  ("rhine_fnu", "a", r"$f_\nu$", ("lin", 0.0, 0.5), "viridis", r"$f_\nu$"),
  ("rhine_dye", "a", r"$\dot Y_e$", ("sym", 1e-4, 1e1), "RdBu_r", r"[ms$^{-1}$]"),
  ("rhine_dyn", "a", r"$\dot Y_n$", ("sym", 1e-3, 1e3), "RdBu_r", r"[ms$^{-1}$]"),
  ("rhine_dya", "a", r"$\dot Y_\alpha$", ("sym", 1e-3, 1e3), "RdBu_r", r"[ms$^{-1}$]"),
  ("rhine_dyh", "a", r"$\dot Y_h$", ("sym", 1e-4, 1e1), "RdBu_r", r"[ms$^{-1}$]"),
  ("rhine_dah", "a", r"$\langle \dot A_h\rangle$", ("sym", 1e-1, 1e5), "RdBu_r",
    r"[ms$^{-1}$]"),
  ("rhine_dma", "a", r"$\dot m_a$", ("sym", 1e-1, 1e4), "RdBu_r",
    r"[MeV baryon$^{-1}$ ms$^{-1}$]"),
  ("rhine_heating_rate", "a", r"$\dot q$", ("sym", 1e25, 1e32), "RdBu_r",
    r"[erg cm$^{-3}$ s$^{-1}$]"),
  ("rhine_transition_w", "a", r"$w$ (transition weight)", ("lin", 0.0, 1.0),
    "cividis", r"$w$"),
]

# Per-second -> per-millisecond for the rate channels.
RATE_KEYS = {"rhine_dye", "rhine_dyn", "rhine_dya", "rhine_dyh", "rhine_dah",
             "rhine_dma"}

# Density contour levels drawn on every panel [g/cm^3].
RHO_LEVELS = [1e6, 1e8, 1e10, 1e12]

# Animation helper functions.
def make_norm(spec):
  kind, lo, hi = spec
  if kind == "log":
    return LogNorm(vmin=lo, vmax=hi)
  if kind == "lin":
    return Normalize(vmin=lo, vmax=hi)
  return SymLogNorm(linthresh=lo, vmin=-hi, vmax=hi, base=10)


def composite(blocks, key, npix, half_width):
  """Paint SMR blocks onto one uniform npix x npix grid, coarse levels first."""
  out = np.full((npix, npix), np.nan)
  edge = np.linspace(-half_width, half_width, npix + 1)
  ctr = 0.5 * (edge[1:] + edge[:-1])
  for ext, lvl, vals in sorted(blocks, key=lambda b: b[1]):
    z = np.nan_to_num(np.array(vals[key], dtype=float), nan=0.0,
                      posinf=1e300, neginf=-1e300)
    ny, nx = z.shape
    ix = np.floor((ctr - ext[0]) / (ext[1] - ext[0]) * nx).astype(int)
    iy = np.floor((ctr - ext[2]) / (ext[3] - ext[2]) * ny).astype(int)
    mx = (ix >= 0) & (ix < nx)
    my = (iy >= 0) & (iy < ny)
    if not (mx.any() and my.any()):
      continue
    out[np.ix_(my, mx)] = z[np.ix_(iy[my], ix[mx])]
  return out

# Slice (.bin) handling.
def load_slice(path, plane, len_conv):
  """
  Read one slice dump.  Returns (time, blocks, names).
  """
  d = bc.read_binary(str(path))
  geom = np.array(d["mb_geometry"]) * len_conv
  lvl = np.array(d["mb_logical"])[:, 3]
  g0 = 0 if plane == "xy" else 2
  blocks = []
  for b in range(d["n_mbs"]):
    ext = tuple(geom[b, g0:g0 + 4])
    if plane == "xy":
      vals = {v: np.array(d["mb_data"][v])[b, 0] for v in d["var_names"]}
    else:
      vals = {v: np.array(d["mb_data"][v])[b, :, :, 0] for v in d["var_names"]}
    blocks.append((ext, lvl[b], vals))
  return d["time"], blocks, d["var_names"]

# Batchtools segment handling.
def merge_segments(chunks, time_key="time", tol=1e-8):
  """
  Concatenate per-segment data dicts (in segment order) along time.
  """
  kept, cutoff = [], np.inf
  for chunk in reversed(chunks):
    t = chunk[time_key]
    mask = t < cutoff - tol
    kept.append({k: v[mask] if isinstance(v, np.ndarray) else v for k, v in chunk.items()})
    if t.size:
      cutoff = min(cutoff, np.nanmin(t))
  kept.reverse()

  keys = [k for k in kept[0] if all(k in c for c in kept)]
  return {k: np.concatenate([c[k] for c in kept]) if isinstance(kept[0][k], np.ndarray)
          else kept[0][k] for k in keys}

# Define the RHINE class.
class RHINE:
  """
  A class to analyze AthenaK data from a run with transition EOS + RHINE.
  """

  def __init__(self, simpath, jobname, batchtools=True,
               hst_path=None, figpath=None):
    """
    Initialize the RHINE class.

    Parameters:
    simpath      (str): Simulation folder (holding the output-XXXX segments).
    jobname      (str): AthenaK job basename, i.e. <job> in <job>.user.hst.
    batchtools  (bool): Whether simpath holds batchtools output-XXXX segments.
    hst_path     (str): Explicit <job>.user.hst file (overrides the segments).
    figpath      (str): Default output folder (defaults to <simpath>/Figs).
    """
    self.simpath = Path(simpath).expanduser()
    self.jobname = jobname
    self.batchtools = batchtools
    self.hst_path = Path(hst_path) if hst_path is not None else None
    self.figpath = Path(figpath) if figpath is not None else self.simpath / "Figs"
    self.segments = [Path(s) for s in find_segments(str(self.simpath), batchtools)]
    print(f"$ Found {len(self.segments)} segment(s) under {self.simpath}...")

    # Load the data.
    self.load_hst_data()


  def segment_files(self, name, explicit=None):
    """
    Returns the paths of 'name' over all segments, or [explicit] if given.
    """
    if explicit is not None:
      if not Path(explicit).is_file():
        raise FileNotFoundError(f"File not found at: {explicit}")
      return [Path(explicit)]

    return [s / name for s in self.segments if (s / name).is_file()]


  def output_path(self, output_dir, name):
    """
    Returns <output_dir>/<name>, defaulting to the figpath, and creates the folder.
    """
    out_dir = Path(output_dir) if output_dir is not None else self.figpath
    out_dir.mkdir(exist_ok=True, parents=True)

    return out_dir / name


  def load_hst_data(self):
    """
    Locate the <job>.user.hst file(s) and load the data into a dict,
    merged over all segments.

    Returns:
    Dictionary holding the hst data.
    """
    paths = self.segment_files(f"{self.jobname}.user.hst", self.hst_path)
    if not paths:
      raise FileNotFoundError(f"No {self.jobname}.user.hst found under {self.simpath}!")

    chunks = []
    for path in paths:
      # Find the headers.
      with open(path) as f:
        f.readline()
        header_line = f.readline().strip()
      headers = re.findall(r'\[\d+\]=(\S+)', header_line)

      # Read the data with numpy (skip segments without data).
      data = np.loadtxt(path, comments='#', ndmin=2)
      if data.shape[0] == 0:
        continue

      # Create a dictionary.
      chunks.append({name: data[:, i] for i, name in enumerate(headers)})

    if not chunks:
      raise SystemExit(f"No data in {self.jobname}.user.hst!")
    self.hst_user_data = merge_segments(chunks)

    print(f"$ Successfully loaded the data from {len(paths)} <job>.user.hst file(s)...")
    return self.hst_user_data


  def load_horizon_data(self, horizon_path=None, index=0):
    """
    Load the horizon summary <job>.horizon_summary_<index>.txt with the
    HorizonFinder class, merged over all segments.

    Parameters:
    horizon_path (str): Explicit horizon file (overrides the segments).
    index        (int): Horizon index.

    Returns:
    Dictionary holding the horizon data, or None if no file exists.
    """
    paths = self.segment_files(f"{self.jobname}.horizon_summary_{index}.txt", horizon_path)
    chunks = [HorizonFinder(horizon_path=str(p)).horizon_data for p in paths]
    chunks = [c for c in chunks if c["time"].size]

    return merge_segments(chunks) if chunks else None


  def load_tracker_data(self, tracker_path=None, index=0):
    """
    Load the tracker <job>.co_<index>.txt with the Tracker class, merged
    over all segments.

    Parameters:
    tracker_path (str): Explicit tracker file (overrides the segments).
    index        (int): Tracker index.

    Returns:
    Dictionary holding the tracker data, or None if no file exists.
    """
    paths = self.segment_files(f"{self.jobname}.co_{index}.txt", tracker_path)
    chunks = []
    for p in paths:
      tracker = Tracker(str(p))
      tracker.load_tracker_data()
      chunks.append(tracker.tracker_data[0])
    chunks = [c for c in chunks if c["time"].size]

    return merge_segments(chunks) if chunks else None


  def find_slices(self, var, plane):
    """
    Locates the <job>.<var>_<plane>.<n>.bin slices over all segments.

    Returns:
    Dictionary {dump index: path}, sorted by index.
    """
    regex = re.compile(rf"^{re.escape(self.jobname)}\.{var}_{plane}\.(\d+)\.bin$")
    found = []
    for s in self.segments:
      d = s / "bin" if (s / "bin").is_dir() else s
      found.append({int(m[1]): f for f in d.iterdir() if (m := regex.match(f.name))})

    hits, cutoff = {}, np.inf
    for seg in reversed(found):
      hits.update({n: f for n, f in seg.items() if n < cutoff})
      if seg:
        cutoff = min(cutoff, min(seg))

    return dict(sorted(hits.items()))


  def plot_rhine_diagnostics(self, output_dir = None):
    """
    Plot the mass fraction conservation and heating rate.

    Parameters:
    output_dir (str): Directory to store 'rhine_diagnostics.png' in.
    """
    # Conversion factors.
    geo = GEOMETRIC_SOLAR
    cgs = CGS
    lum_geo_to_cgs = geo.energy_conversion(cgs) / geo.time_conversion(cgs)
    time_conv = geo.time_conversion(cgs) * 1e3
    dens_conv = geo.mass_density_conversion(cgs)

    keys   = ["time", "rho-max", "T-max", "Xsum-err", "rh-qphys", "rhine-Lfnu"]
    colors = ["#009E73", "#D55E00", "#F0A860", "#A7C7E7"]
    labels = [r"$\rho_\mathrm{max}\ [\mathrm{g}/\mathrm{cm}^3]$", r"$T_\mathrm{max}$ [MeV]",
              r"$\max|\sum_i X_i - 1|$", r"$|\dot{q}|,\ L_{f_\nu}$ [ergs/s]"]
    for key in keys:
      if key not in self.hst_user_data.keys():
        raise SystemExit(f".hst data is missing key '{key}!'")

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    data = self.hst_user_data
    for i, a in enumerate(ax.flat):
      t_ms = data["time"] * time_conv
      if keys[i+1] != "rh-qphys":
        if keys[i+1] == "rho-max":
          a.plot(t_ms, data[keys[i+1]] * dens_conv, color=colors[i])
          a.set_yscale("log")
        else:
          a.plot(t_ms, data[keys[i+1]], color=colors[i])
      else:
        a.plot(t_ms, data[keys[i+1]] * lum_geo_to_cgs, color=colors[i], ls="solid", label=r"$|\dot{q}|$")
        a.plot(t_ms, data[keys[i+2]] * lum_geo_to_cgs, color=colors[i], ls="dashed", label=r"$L_{f_\nu}$")
        a.set_yscale("log")
        a.legend(frameon=False, fontsize=14)
      a.set_xlabel(r"$t$ [ms]", fontsize=14); a.set_ylabel(labels[i], fontsize=14)
      a.set_xlim(np.min(t_ms), np.max(t_ms))

    fig.tight_layout()

    out_path = self.output_path(output_dir, "rhine_diagnostics.png")
    plt.savefig(out_path, dpi=150)
    print(f"$ Saved diagnostics plot to {out_path}...")
    plt.close(fig)


  def plot_heating_split(self, output_dir = None):
    """
    Plot the heating rate split between physical contribution,
    matter entering in-and-put of NSE etc..

    Parameters:
    output_dir (str): Directory to store 'heating_split.png' in.
    """
    # Conversion factors.
    geo = GEOMETRIC_SOLAR
    cgs = CGS
    energy_geo_to_cgs = geo.energy_conversion(cgs)
    lum_geo_to_cgs = energy_geo_to_cgs / geo.time_conversion(cgs)
    time_conv = geo.time_conversion(cgs) * 1e3

    keys   = ["time", "rhine-qdot", "rh-qphys", "rh-qexit", "rh-qreent", "rh-qphys"]
    colors = ["#009E73", "#D55E00", "#F0A860", "black"]
    labels = [r"$\dot{Q}\ [\mathrm{erg}\ \mathrm{s}^{-1}]$",
              r"$\int \dot{Q}dt\ [\mathrm{erg}]$"]
    for key in keys:
      if key not in self.hst_user_data.keys():
        raise SystemExit(f".hst data is missing key '{key}!'")

    fig, ax = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    data = self.hst_user_data

    # Compute the splits.
    t     = data["time"]
    qphys = data["rh-qphys"]
    qnse  = data["rh-qexit"] + data["rh-qreent"]
    qmix  = data["rhine-qdot"] - data["rh-qphys"] - data["rh-qexit"]
    qnet  = data["rhine-qdot"] + data["rh-qreent"]
    splits = [(qphys, r"Physical $\dot{Q}_\mathrm{phys}$", "solid"),
              (qnse, r"NSE boundary $\dot{Q}_\mathrm{exit}+\dot{Q}_\mathrm{reent}$", "solid"),
              (qmix, r"Mixing $\dot{Q}-\dot{Q}_\mathrm{phys}-\dot{Q}_\mathrm{exit}$", "solid"),
              (qnet, r"Net $\dot{Q}+\dot{Q}_\mathrm{reent}$", "dashed")]

    # Compute cumulative integral of the splits (geometric time -> geometric energy).
    int_qphys = cumulative_trapezoid(qphys, t, initial=0)
    int_qnse  = cumulative_trapezoid(qnse, t, initial=0)
    int_qmix  = cumulative_trapezoid(qmix, t, initial=0)
    int_qnet  = cumulative_trapezoid(qnet, t, initial=0)
    int_splits = [(int_qphys, "solid"),
                  (int_qnse, "solid"),
                  (int_qmix, "solid"),
                  (int_qnet, "dashed")]

    t_ms = data["time"] * time_conv
    for i, (quant, lab, linest) in enumerate(splits):
      ax[0].plot(t_ms, quant * lum_geo_to_cgs, label=lab, ls=linest, color=colors[i])

    for i, (quant, linest) in enumerate(int_splits):
      ax[1].plot(t_ms, quant * energy_geo_to_cgs, ls=linest, color=colors[i])

    for a, label in zip(ax.flat, labels):
      a.axhline(0.0, color="gray", lw=0.8)
      a.set_ylabel(label, fontsize=14)
      a.set_xlim(np.min(t_ms), np.max(t_ms))
      a.tick_params(which="both", direction="in", top=True, right=True)
      a.minorticks_on()
    ax[1].set_xlabel(r"$t$ [ms]", fontsize=14)

    all_q = np.concatenate([q for q, _, _ in splits]) * lum_geo_to_cgs
    lo, hi = np.percentile(all_q, [0.2, 99.8])
    pad = 0.1 * (hi - lo)
    ax[0].set_ylim(lo - pad, hi + pad)

    ax[0].legend(loc="upper left", fontsize=12)

    fig.tight_layout()

    out_path = self.output_path(output_dir, "heating_split.png")
    plt.savefig(out_path, dpi=150)
    print(f"$ Saved diagnostics plot to {out_path}...")
    plt.close(fig)


  def plot_rhine_slices(self, plane="xy", output_dir=None, half_width=150.0, npix=600,
                        stride=1, fps=20, nmax=None, horizon_path=None, tracker_path=None,
                        bh_index=0):
    """
    Plot animation of 2D slices of RHINE rates and other diagnostics, with
    density contours and the BH horizon (mean coordinate radius around the
    tracker position) once the horizon finder has locked on.

    Parameters:
    plane        (str): Slice plane, 'xy' or 'yz'.
    output_dir   (str): Directory to store 'rhine_<plane>.mp4' in.
    half_width (float): Half-width of the field of view [km].
    npix         (int): Pixels per axis of the composited image.
    stride       (int): Use every stride-th dump.
    fps          (int): Frames per second of the movie.
    nmax         (int): Use only the first nmax dumps.
    horizon_path (str): Explicit horizon file (overrides the segments).
    tracker_path (str): Explicit BH tracker file (overrides the segments).
    bh_index     (int): Index of the BH horizon and tracker, i.e. co_<bh_index>.txt.
    """
    # Conversion factors.
    geo = GEOMETRIC_SOLAR
    cgs = CGS
    dens_conv = geo.mass_density_conversion(cgs)
    length_conv = geo.length_conversion(cgs) * 1e-5
    time_conv = geo.time_conversion(cgs) * 1e3

    # Locate the slices in all segments.
    aux = self.find_slices("rhine_aux", plane)
    prim = self.find_slices("mhd_w_bcc", plane)
    frames = sorted(set(aux) & set(prim))[:nmax][::stride]
    if not frames:
      raise SystemExit(f"No rhine_aux/mhd_w_bcc {plane} slices found under {self.simpath}!")
    print(f"$ Rendering {len(frames)} {plane} slices ({frames[0]:05d}-{frames[-1]:05d})...")

    # Load the horizon and the BH tracker.
    horizon = self.load_horizon_data(horizon_path, bh_index)
    tracker = self.load_tracker_data(tracker_path, bh_index)
    have_horizon = horizon is not None and tracker is not None
    if have_horizon:
      hz = np.isfinite(horizon["meanradius"])
      hz_t, hz_r = horizon["time"][hz], horizon["meanradius"][hz] * length_conv
      have_horizon = hz_t.size > 0
    if not have_horizon:
      print("$ No horizon and tracker data (or horizon never found), skipping the BH...")
    i1, i2 = ("x", "y") if plane == "xy" else ("y", "z")

    # Set up the figure.
    ext = (-half_width, half_width, -half_width, half_width)
    fig, axes = plt.subplots(3, 4, figsize=(19.0, 11.5))
    axes = axes.ravel()
    ims, bhs = [], []
    blank = np.full((npix, npix), np.nan)
    for a, (key, src, title, spec, cmap, clab) in zip(axes, PANELS):
      a.set_title(title)
      a.set_xlabel(f"{i1} [km]")
      a.set_ylabel(f"{i2} [km]")
      im = a.imshow(blank, origin="lower", extent=ext, norm=make_norm(spec),
                    cmap=cmap, interpolation="nearest")
      cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
      cb.set_label(clab, fontsize=8)
      cb.ax.tick_params(labelsize=7)
      ims.append(im)
      bh = Circle((0, 0), 0, fc="k", ec="w", lw=0.5, visible=False)
      a.add_patch(bh)
      bhs.append(bh)
    suptitle = fig.suptitle("")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    state = {"cs": []}
    g = np.linspace(-half_width, half_width, npix)

    def draw(n):
      for cs in state["cs"]:
        cs.remove()
      state["cs"] = []
      t, ba, _ = load_slice(aux[n], plane, length_conv)
      _, bw, _ = load_slice(prim[n], plane, length_conv)
      rho = composite(bw, "dens", npix, half_width) * dens_conv

      # BH horizon (mean coordinate radius) once the finder has locked on.
      have_bh = have_horizon and hz_t[0] <= t <= hz_t[-1]
      if have_bh:
        ctr = (np.interp(t, tracker["time"], tracker[i1]) * length_conv,
               np.interp(t, tracker["time"], tracker[i2]) * length_conv)
        r = np.interp(t, hz_t, hz_r)

      for im, a, bh, (key, src, _t, _s, _c, _l) in zip(ims, axes, bhs, PANELS):
        z = composite(ba if src == "a" else bw, key, npix, half_width)
        if key == "dens":
          z = z * dens_conv
        elif key in RATE_KEYS:
          z = z * 1e-3
        im.set_data(z)
        cs = a.contour(g, g, rho, levels=RHO_LEVELS, colors="k", linewidths=0.5)
        state["cs"].append(cs)
        bh.set_visible(have_bh)
        if have_bh:
          bh.set_center(ctr)
          bh.set_radius(r)
      suptitle.set_text(
        f"{plane} slice, "
        f"t = {t:.1f} $M_\\odot$ = {t*time_conv:.2f} ms   "
        r"($\rho$ contours at $10^{6},10^{8},10^{10},10^{12}$ g cm$^{-3}$)")
      return ims

    # Render the movie.
    out_path = self.output_path(output_dir, f"rhine_{plane}.mp4")
    ani = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=6000)
    pbar = tqdm(total=len(frames), desc="Rendering movie")
    ani.save(out_path, writer=writer, progress_callback=lambda i, n: pbar.update(1))
    pbar.close()
    plt.close(fig)
    print(f"$ Saved slice movie to {out_path}...")


if __name__ == "__main__":
  p = argparse.ArgumentParser(description="Analyze AthenaK transition EOS + RHINE runs.")
  p.add_argument("simpath", help="simulation folder (holding the output-XXXX segments)")
  p.add_argument("--jobname", default="bhns_rhine")
  p.add_argument("--batchtools", action=argparse.BooleanOptionalAction, default=True,
                 help="read the output-XXXX segments under simpath (default); "
                      "--no-batchtools reads simpath directly")
  p.add_argument("--plots", nargs="+", choices=["diagnostics", "heating", "slices"],
                 default=["diagnostics", "heating", "slices"])
  p.add_argument("-o", "--output-dir", default=None, help="defaults to <simpath>/Figs")
  p.add_argument("--hst", default=None, help="explicit <job>.user.hst file")
  p.add_argument("--horizon", default=None, help="explicit horizon summary file")
  p.add_argument("--tracker", default=None, help="explicit BH tracker file")
  p.add_argument("--plane", choices=["xy", "yz"], default="xy")
  p.add_argument("--width", type=float, default=150.0,
                 help="half-width of the field of view [km]")
  p.add_argument("--npix", type=int, default=600)
  p.add_argument("--stride", type=int, default=1)
  p.add_argument("--fps", type=int, default=20)
  p.add_argument("--nmax", type=int, default=None)
  args = p.parse_args()

  rhine = RHINE(args.simpath, jobname=args.jobname, batchtools=args.batchtools,
                hst_path=args.hst, figpath=args.output_dir)
  if "diagnostics" in args.plots:
    rhine.plot_rhine_diagnostics()
  if "heating" in args.plots:
    rhine.plot_heating_split()
  if "slices" in args.plots:
    rhine.plot_rhine_slices(plane=args.plane, half_width=args.width, npix=args.npix,
                            stride=args.stride, fps=args.fps, nmax=args.nmax,
                            horizon_path=args.horizon, tracker_path=args.tracker)

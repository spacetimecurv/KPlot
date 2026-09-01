"""
Approximate post-merger accretion rate.

Baryon conservation for the post-merger disk
    (disk mass loss) = (mass accreted onto BH) + (ejected mass),
so the accretion rate is:

    Mdot_accr(t) = -dM_b/dt  +  Mdot_ej(t - r_ej/<v_ej>)

    M_b(t)   domain baryon (rest) mass       -> <job>.mhd.hst
    Mdot_ej  ejecta mass flux                -> Mej_rate_<crit>.txt
    <v_ej>   characteristic ejecta speed [c] -> flux-weighted vinf_avg_<crit>.txt
    r_ej     matter extraction radius        -> --r-ej

Command line:
    kplot-sphere-accretion --analysis-dir <dir> --r-ej 400 [--crit Ber]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

MSUN_TO_MS = 6.67430e-11 * 1.98892e30 / 2.99792458e8**3 * 1e3  # 4.9268e-3


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--analysis-dir", required=True,
                 help="Directory with the Mej_rate_*.txt, vinf_avg_*.txt")
  p.add_argument("--hst-file", default="bhns.mhd.hst",
                 help="AthenaK history file holding the domain baryon mass, "
                      "relative to --analysis-dir. Default: bhns.mhd.hst.")
  p.add_argument("--r-ej", type=float, default=400.0,
                 help="Matter extraction radius [M_sun], must match the sph radius.")
  p.add_argument("--crit", choices=["geo", "Ber"], default="Ber",
                 help="Ejecta criterion. Default: Ber.")
  p.add_argument("--smooth", type=int, default=9,
                 help="Smoothing width for dM_b/dt. Default: 9.")
  p.add_argument("--t-settle", type=float, default=3.0,
                 help="Skip this many ms after merger to exclude the plunge. Default: 3.")
  args = p.parse_args(argv)

  adir = args.analysis_dir

  # Domain baryon mass M_b(t) (drop duplicate/restart times, then resample
  # onto a uniform grid).
  hst = np.loadtxt(os.path.join(adir, args.hst_file))
  t_hst, mb = hst[:, 0], hst[:, 2]
  keep = np.concatenate([[True], np.diff(t_hst) > 0])
  t_hst, mb = t_hst[keep], mb[keep]
  dt = np.median(np.diff(t_hst))
  t_uni = np.arange(t_hst[0], t_hst[-1], dt)
  mb = np.interp(t_uni, t_hst, mb)
  t_hst = t_uni

  # Ejecta flux and characteristic speed.
  rate = np.loadtxt(os.path.join(adir, f"Mej_rate_{args.crit}.txt"))
  t_ej, r_ej = rate[:, 0], rate[:, 1]
  vv = np.loadtxt(os.path.join(adir, f"vinf_avg_{args.crit}.txt"))
  w = np.interp(vv[:, 0], t_ej, r_ej)
  m = (vv[:, 1] > 0) & (w > 0)
  vej = np.sum(vv[m, 1] * w[m]) / np.sum(w[m])
  delay = args.r_ej / vej

  # Accretion rate [Msun/ms].
  mbdot = np.gradient(uniform_filter1d(mb, size=max(1, args.smooth), mode="nearest"), t_hst)
  rate_shift = np.interp(t_hst - delay, t_ej, r_ej, left=0.0)
  mdot_accr = (-mbdot + rate_shift) / MSUN_TO_MS

  # Merger.
  t_merger = t_hst[np.argmin(mbdot)]
  t0 = t_merger + args.t_settle / MSUN_TO_MS
  post = t_hst >= t0

  print(f"crit={args.crit}  <v_ej>={vej:.4f} c  r_ej={args.r_ej:g}  "
        f"delay={delay*MSUN_TO_MS:.2f} ms  t_merger={t_merger:.1f} Msun")

  # Plot.
  fig, ax = plt.subplots(figsize=(7, 4.3))
  ax.plot((t_hst[post] - t_merger) * MSUN_TO_MS, mdot_accr[post],
          color="tab:red", lw=1.4)
  ax.axhline(0.0, color="black", lw=0.6)
  ax.set_xlim(left=(t0 - t_merger) * MSUN_TO_MS)
  ax.set_xlabel(r"$t - t_\mathrm{merger}$ [ms]")
  ax.set_ylabel(r"$\dot M_\mathrm{accr}$  [$M_\odot\,\mathrm{ms}^{-1}$]")
  fig.tight_layout()
  out = os.path.join(adir, "fig_accretion.png")
  fig.savefig(out, dpi=150, bbox_inches="tight")
  print(f"Saved {out}")


if __name__ == "__main__":
  main()

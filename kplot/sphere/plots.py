"""
Plot ejecta and neutrino diagnostics produced by kplot.sphere.ejecta and
kplot.sphere.neutrinos.

Reads the .txt outputs from --output-dir and writes:
    fig_ejecta.pdf    (2x3 panel summary of the ejecta properties)
    fig_neutrino.pdf  (2 panels: neutrino luminosity + mean energy)

Command line:
    kplot-sphere-plot --output-dir DIR --t-merger T [--radius 300] [--from-merger]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Set explicitly rather than inherited, so the figures do not depend on which
# other kplot module happened to be imported first.  Every label below renders
# under mathtext, so a LaTeX install is not required.
matplotlib.rcParams["text.usetex"] = False

# Unit conversions
G       = 6.67430e-11
M_SUN   = 1.98892e30
C       = 2.99792458e8
MSUN_TO_MS = G * M_SUN / C**3 * 1e3


def cumulative(arr):
    t = arr[:, 0]; r = arr[:, 1]; dt = np.diff(t)
    return t[1:], np.cumsum(0.5 * (r[:-1] + r[1:]) * dt)


def plot_ejecta(adir, t_ms, radius):
    def load(f): return np.loadtxt(os.path.join(adir, f))

    Mej_rate_geo = load('Mej_rate_geo.txt')
    Mej_rate_Ber = load('Mej_rate_Ber.txt')
    Mej_vinf_geo = load('Mej_vinf_geo.txt')
    Mej_vinf_Ber = load('Mej_vinf_Ber.txt')
    rhoej_geo    = load('rhoej_theta_geo.txt')
    rhoej_Ber    = load('rhoej_theta_Ber.txt')
    Ye_avg_geo   = load('Ye_avg_geo.txt')
    Ye_avg_Ber   = load('Ye_avg_Ber.txt')
    Mej_Ye_geo   = load('Mej_Ye_geo.txt')
    Mej_Ye_Ber   = load('Mej_Ye_Ber.txt')

    t_geo_cum, Mej_geo_cum = cumulative(Mej_rate_geo)
    t_Ber_cum, Mej_Ber_cum = cumulative(Mej_rate_Ber)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    ax = axes[0, 0]
    ax.plot(t_ms(Mej_rate_geo[:, 0]), Mej_rate_geo[:, 1] / MSUN_TO_MS * 1e3, color='tab:blue', label='geodesic', lw=1.2)
    ax.plot(t_ms(Mej_rate_Ber[:, 0]), Mej_rate_Ber[:, 1] / MSUN_TO_MS * 1e3, color='tab:orange', ls='--', label='Bernoulli', lw=1.2)
    ax.set_xlabel(r'$t - t_\mathrm{merger}$ [ms]'); ax.set_ylabel(r'$\dot{M}_\mathrm{ej}$  [$10^{-3}M_\odot\,\mathrm{ms}^{-1}$]')
    ax.legend(fontsize=9, frameon=False); ax.set_xlim(left=0, right=np.max(t_ms(Mej_rate_geo[:, 0]))) 
    ax.set_title('Mass ejection rate')

    ax = axes[0, 1]
    ax.plot(t_ms(t_geo_cum), Mej_geo_cum * 1e3, color='tab:blue', label='geodesic', lw=1.2)
    ax.plot(t_ms(t_Ber_cum), Mej_Ber_cum * 1e3, color='tab:orange', ls='--', label='Bernoulli', lw=1.2)
    ax.set_xlabel(r'$t - t_\mathrm{merger}$ [ms]'); ax.set_ylabel(r'$M_\mathrm{ej}$  [$10^{-3}M_\odot$]')
    ax.set_xlim(left=0, right=np.max(t_ms(t_geo_cum))); ax.set_title('Cumulative ejecta mass')

    ax = axes[0, 2]
    ax.plot(t_ms(Ye_avg_geo[:, 0]), Ye_avg_geo[:, 1], color='tab:blue', label='geodesic', lw=1.2)
    ax.plot(t_ms(Ye_avg_Ber[:, 0]), Ye_avg_Ber[:, 1], color='tab:orange', ls='--', label='Bernoulli', lw=1.2)
    ax.axhline(0.5, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$t - t_\mathrm{merger}$ [ms]'); ax.set_ylabel(r'$\langle Y_e \rangle_\mathrm{ej}$')
    ax.set_xlim(left=0, right=np.max(t_ms(Ye_avg_geo[:, 0]))); ax.set_ylim(0, 0.6); ax.set_title(r'Mass-weighted $\langle Y_e \rangle$')

    ax = axes[1, 0]
    dv = np.diff(Mej_vinf_geo[:, 0]); dv = np.append(dv, dv[-1])
    ax.step(Mej_vinf_geo[:, 0], Mej_vinf_geo[:, 1] / dv * 1e3, color='tab:blue', label='geodesic', lw=1.2, where='mid')
    ax.step(Mej_vinf_Ber[:, 0], Mej_vinf_Ber[:, 1] / dv * 1e3, color='tab:orange', ls='--', label='Bernoulli', lw=1.2, where='mid')
    ax.set_xlabel(r'$v_\infty$ [$c$]'); ax.set_ylabel(r'$dM_\mathrm{ej}/dv_\infty$  [$10^{-3}M_\odot\,c^{-1}$]')
    ax.set_xlim(0, 0.8); ax.set_title('Velocity distribution')

    ax = axes[1, 1]
    ax.plot(rhoej_geo[:, 0] * 180 / np.pi, rhoej_geo[:, 1] * 1e3, color='tab:blue', label='geodesic', lw=1.2)
    ax.plot(rhoej_Ber[:, 0] * 180 / np.pi, rhoej_Ber[:, 1] * 1e3, color='tab:orange', ls='--', label='Bernoulli', lw=1.2)
    ax.set_xlabel(r'$\theta$ [deg]'); ax.set_ylabel(r'$dM_\mathrm{ej}/d\theta$  [$10^{-3}M_\odot\,\mathrm{rad}^{-1}$]')
    ax.set_xlim(0, 180); ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.set_title('Angular distribution')

    ax = axes[1, 2]
    dye = np.diff(Mej_Ye_geo[:, 0]); dye = np.append(dye, dye[-1])
    ax.step(Mej_Ye_geo[:, 0], Mej_Ye_geo[:, 1] / dye * 1e3, color='tab:blue', label='geodesic', lw=1.2, where='mid')
    ax.step(Mej_Ye_Ber[:, 0], Mej_Ye_Ber[:, 1] / dye * 1e3, color='tab:orange', ls='--', label='Bernoulli', lw=1.2, where='mid')
    ax.axvline(0.5, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$Y_e$'); ax.set_ylabel(r'$dM_\mathrm{ej}/dY_e$  [$10^{-3}M_\odot$]')
    ax.set_xlim(0, 0.6); ax.set_title(r'$Y_e$ distribution')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.01, 0.99, f'R = {radius:g} $M_\\odot$', fontsize=12,
            ha='left', verticalalignment='top', bbox=props)

    fig.tight_layout()
    out = os.path.join(adir, 'fig_ejecta.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')

    # Check whether per-iteration output present.
    if os.path.exists(os.path.join(adir, 'iteration')):
        print("Per-iteration output found. Making plots...")
        plot_ye_evolution(adir, t_ms)
    else:
        print("Skipping per-iteration output; not found.")


def edges_from_centers(centers):
    """Cell edges bracketing 1D bin centers (midpoints, extrapolated ends)."""
    centers = np.asarray(centers, dtype=float)
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def plot_ye_evolution(adir, t_ms):
    """Time-vs-Y_e maps of the per-bin ejected mass, one panel per criterion.

    Reads the per-iteration histograms written by kplot.sphere.ejecta under
    ``<adir>/iteration`` (``Mej_Ye_<crit>_<time>.txt``, columns Ye | M[Msun])
    and renders each criterion as a pcolormesh with time on x, Y_e on y and the
    per-bin mass on a shared log color scale.  Empty (zero-mass) cells are left
    blank, as in the reference figure.
    """
    idir = os.path.join(adir, 'iteration')

    # Group the per-iteration files by ejecta criterion, sorted in time.
    # Filenames look like ``Mej_Ye_geo_00500.txt`` -> criterion 'geo', time 500.
    per_crit = {}
    for fname in sorted(os.listdir(idir)):
        if not fname.endswith('.txt'):
            continue
        stem = fname.split('.')[0].split('_')
        criterion, time = stem[-2], int(stem[-1])
        per_crit.setdefault(criterion, []).append((time, fname))

    if not per_crit:
        print("  No per-iteration histograms to plot.")
        return

    labels = {'geo': 'Geodesic', 'Ber': 'Bernoulli'}
    criteria = sorted(per_crit) 

    # Load every criterion into (Ntime, Nye) mass arrays sharing the Ye grid.
    data = {}
    ye_centers = None
    vmax = 0.0
    for crit in criteria:
        entries = sorted(per_crit[crit])
        times = np.array([t for t, _ in entries], dtype=float)
        stack = np.array([np.loadtxt(os.path.join(idir, f)) for _, f in entries])
        if ye_centers is None:
            ye_centers = stack[0, :, 0]
        mass = stack[:, :, 1]                        # (Ntime, Nye)
        data[crit] = (times, mass)
        vmax = max(vmax, mass.max())

    if vmax <= 0:
        print("  All per-iteration histograms are empty; skipping.")
        return

    from matplotlib.colors import LogNorm

    norm = LogNorm(vmin=vmax / 1e5, vmax=vmax)
    y_edges = edges_from_centers(ye_centers)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]

    mesh = None
    for ax, crit in zip(axes, criteria):
        times, mass = data[crit]
        x_edges = edges_from_centers(t_ms(times))

        # Blank out empty bins so they render white, as in the reference figure.
        masked = np.ma.masked_less_equal(mass.T, 0.0)   # (Nye, Ntime)
        mesh = ax.pcolormesh(x_edges, y_edges, masked,
                             norm=norm, cmap='viridis', shading='flat')
        ax.set_xlabel(r'$t - t_\mathrm{merger}$ [ms]')
        ax.set_ylim(0, 0.5)
        ax.set_title(labels.get(crit, crit))
    axes[0].set_ylabel(r'$Y_e$')

    cbar = fig.colorbar(mesh, ax=axes, label=r'$M$  [$M_\odot$]')
    cbar.ax.set_ylabel(r'$M$  [$M_\odot$]')

    out = os.path.join(adir, 'fig_ejecta_ye_evolution.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')


def plot_neutrino(adir, t_ms, t_merger_ms, nu_xlabel, nu_xlim, radius):
    def load(f): return np.loadtxt(os.path.join(adir, f))

    SPECIES  = ['nue', 'nua', 'nux', 'anux']
    SP_LABEL = [r'$\nu_e$', r'$\bar\nu_e$', r'$\nu_x$', r'$\bar\nu_x$']
    SP_COLOR = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    Lnu  = {sp: load(f'Lnu_E_{sp}.txt') for sp in SPECIES}
    Ltot = load('Lnu_E_total.txt')
    Eav  = {sp: load(f'Eav_{sp}.txt')   for sp in SPECIES}

    # Two panels: (1) luminosity, log scale 1e51–3e54;  (2) mean energy, 0–60.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.semilogy(t_ms(Ltot[:, 0]), Ltot[:, 1], color='black', lw=1.5, label='total')
    for sp, lab, col in zip(SPECIES, SP_LABEL, SP_COLOR):
        d = Lnu[sp]; ax.semilogy(t_ms(d[:, 0]), d[:, 1], color=col, label=lab, lw=1.0)
    ax.axvline(t_merger_ms, color='gray', ls='--', lw=0.8)
    ax.set_xlabel(nu_xlabel); ax.set_ylabel(r'$L_{\nu,E}$  [erg s$^{-1}$]')
    ax.set_ylim(1e51, 3e54)
    ax.set_xlim(0, np.max(t_ms(d[:, 0])))
    ax.legend(fontsize=9, ncol=2, frameon=False); ax.set_xlim(**nu_xlim)
    ax.set_title('Energy luminosity')

    ax = axes[1]
    for sp, lab, col in zip(SPECIES, SP_LABEL, SP_COLOR):
        d = Eav[sp]; ax.plot(t_ms(d[:, 0]), d[:, 1], color=col, label=lab, lw=1.2)
    ax.axvline(t_merger_ms, color='gray', ls='--', lw=0.8)
    ax.set_xlabel(nu_xlabel); ax.set_ylabel(r'$\langle\varepsilon_\nu\rangle$  [MeV]')
    ax.set_ylim(0, 60)
    ax.set_xlim(0, np.max(t_ms(d[:, 0])))
    ax.legend(fontsize=9, ncol=2, frameon=False); ax.set_xlim(**nu_xlim)
    ax.set_title('Mean neutrino energy')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.01, 0.99, f'R = {radius:g} $M_\\odot$', fontsize=12,
            ha='left', verticalalignment='top', bbox=props)

    fig.tight_layout()
    out = os.path.join(adir, 'fig_neutrino.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True,
                        help="Directory holding the analysis .txt outputs.")
    parser.add_argument("--t-merger", type=float, required=True,
                        help="Merger time [M_sun] (from merger_time.txt).")
    parser.add_argument("--radius", type=float, default=300.0,
                        help="SPH extraction radius (for plot titles). Default: 300.")
    parser.add_argument("--from-merger", action="store_true",
                        help="Use t - t_merger [ms] as the neutrino x-axis (xlim left=0) "
                             "instead of absolute time.")
    parser.add_argument("--no-ejecta", action="store_true", help="Skip the ejecta figure.")
    parser.add_argument("--no-neutrino", action="store_true", help="Skip the neutrino figure.")
    args = parser.parse_args(argv)

    adir = args.output_dir
    t_merger = args.t_merger

    if args.from_merger:
        def t_ms(t): return (t - t_merger) * MSUN_TO_MS
        nu_xlabel = r'$t - t_\mathrm{merger}$ [ms]'
        nu_xlim = {'left': 0}
        t_merger_ms = 0.0
    else:
        def t_ms(t): return t * MSUN_TO_MS
        nu_xlabel = r'$t$ [ms]'
        nu_xlim = {}
        t_merger_ms = t_merger * MSUN_TO_MS

    if not args.no_ejecta:
        plot_ejecta(adir, t_ms, args.radius)
    if not args.no_neutrino:
        plot_neutrino(adir, t_ms, t_merger_ms, nu_xlabel, nu_xlim, args.radius)


if __name__ == "__main__":
    main()

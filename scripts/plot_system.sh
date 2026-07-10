#!/usr/bin/env bash
# plot_system.sh — KPlot full-run visualization driver
# ----------------------------------------------------
# Thin wrapper around the KPlot full-run visualizer (kplot.system /
# SystemPlotter).  All plotting logic lives in the installed KPlot package, so
# this shell script is the only file you edit: set SIMPATH (and optionally
# FIGPATH), pick JOBNAME/PLANE/UNITS, then comment/uncomment entries in
# SECTIONS to choose what to plot.
#
# Requires KPlot to be installed in the active Python environment:
#   pip install -e external/plot-tools   # (from the KPlot checkout)
#   pip install -e .
#
# SIMPATH should be the parent directory that contains the output-XXXX
# restart-segment subdirectories, e.g.:
#   DD2_m1_VVLR/
#     output-0000/
#     output-0001/
#     ...

# SIMPATH="/lus/flare/projects/CompactBinaryMerger/yiqiu/Large_sim/bundle_run/VVLR"
SIMPATH="/leonardo_scratch/large/userexternal/osteppoh/basem1/temp"
FIGPATH="${SIMPATH}/Figs"    # default: <SIMPATH>/Figs

# AthenaK job name prefixing all input/output files (input_block <job> name).
JOBNAME=bhns

# Slice plane to render: 'xy' (default; auto-adds an xz companion panel when
# those files exist), 'xz', or 'yz'.
PLANE=xy
# PLANE=xz
# PLANE=yz

# Set to "false" (uncomment the second line) to hide tracker markers on 2-D plots
# SHOW_TRACKERS=true
SHOW_TRACKERS=false

# Set to "true" to overlay black-hole apparent-horizon circles on 2-D plots
# (needs <JOBNAME>.horizon_summary_k.txt and <JOBNAME>.co_k.txt tracker files)
SHOW_HORIZON=false
# SHOW_HORIZON=true

# Set to "true" to skip frames whose PNG already exists in the Figs directory
SKIP_EXISTING=true
# SKIP_EXISTING=true

# Set to "true" to show the full simulation domain instead of the ±60 Msun window
# FULL_DOMAIN=true
FULL_DOMAIN=false

# Unit system for physical quantities: 'code', 'cgs', or 'ngs'  (default: cgs)
UNITS=cgs
# UNITS=code
# UNITS=ngs

# Time display units for plot titles / time-series axes: 'Msun' or 'ms'
TIME_UNITS=Msun
# TIME_UNITS=ms

# Set to "true" to delete the rendered *_frame_* PNGs after the movies are made
# (keeps the movies in <FIGPATH>/all_movies/, reclaims the frame disk space).
DELETE_FRAMES=false
# DELETE_FRAMES=true

# ---------------------------------------------------------------------------
# SECTIONS — comment out any line to disable that diagnostic
# ---------------------------------------------------------------------------
SECTIONS=(
    history       # baryonic mass & Hamiltonian constraint norm vs time
    rho_max       # central density & min lapse vs time
    # trackers      # compact-object trajectory plot (both objects, smoothed)
    run_speed     # AthenaK cycles/sec from log file

    density       # density (dens) slice frames  [+ trackers + AMR grid]
    temperature   # temperature slice frames      [+ trackers + AMR grid]
    s00           # electron fraction (s_00) frames
    # s00_avg       # area-weighted average Y_e vs time (1-D timeseries)
    #z4c
    rad_E         # radiation energy E:0 frames          [<job>.rad_m1_E]
    rad_N         # radiation number density N:0 frames  [<job>.rad_m1_N]
    # rad_N2        # radiation number density N:2 (heavy lepton) frames
    # rad_abs       # absorption opacity abs_1:0 frames    [<job>.rad_m1_abs_1]
    # rad_abs0    # absorption opacity abs_0:0 frames    [<job>.rad_m1_abs_0]
    # rad_eta0    # emissivity eta_0:0 frames            [<job>.rad_m1_eta_0]
    # rad_eta1    # emissivity eta_1:0 frames            [<job>.rad_m1_eta_1]
    # avg_energy  # derived average neutrino energy frames
    nu_energy   # neutrino energy E/N species 0 (ν_e)    [MeV in cgs]
    nu_energy1  # neutrino energy E/N species 1 (ν̄_e)    [MeV in cgs]
    nu_energy2  # neutrino energy E/N species 2 (ν_x)    [MeV in cgs]
)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#module load python py-matplotlib py-numpy py-scipy
#module load ffmpeg          # provides ffmpeg for make_movies.sh

ARGS=(--simpath "${SIMPATH}")
[[ -n "${FIGPATH:-}"  ]] && ARGS+=(--figpath  "${FIGPATH}")
[[ -n "${JOBNAME:-}"  ]] && ARGS+=(--jobname  "${JOBNAME}")
[[ -n "${PLANE:-}"      ]] && ARGS+=(--plane      "${PLANE}")
[[ -n "${TIME_UNITS:-}" ]] && ARGS+=(--time-units "${TIME_UNITS}")
[[ "${SHOW_TRACKERS}" == "false" ]] && ARGS+=(--no-trackers)
[[ "${SHOW_HORIZON}"  == "true"  ]] && ARGS+=(--show-horizon)
[[ "${SKIP_EXISTING}" == "true"  ]] && ARGS+=(--skip-existing)
[[ "${FULL_DOMAIN}"   == "true"  ]] && ARGS+=(--full-domain)
ARGS+=(--units "${UNITS}")
ARGS+=(--sections "${SECTIONS[@]}")

# Run the KPlot full-run visualizer.  Prefer the installed console script; fall
# back to the module form if KPlot has not been reinstalled since it was added.
if command -v kplot-system >/dev/null 2>&1; then
    kplot-system "${ARGS[@]}"
else
    python3 -m kplot.system "${ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Movie generation — automatically create MP4s from rendered frames
# ---------------------------------------------------------------------------
MOVIE_FPS=24       # frames per second
MOVIE_STRIDE=2     # use every Nth frame (1 = all)

echo ""
echo "=== Making movies from ${FIGPATH} ==="
bash "${SCRIPT_DIR}/make_movies.sh" "${MOVIE_FPS}" "${MOVIE_STRIDE}" "${FIGPATH}"

# ---------------------------------------------------------------------------
# Optional cleanup — delete the rendered frame PNGs (movies are kept)
# ---------------------------------------------------------------------------
if [[ "${DELETE_FRAMES}" == "true" ]]; then
    echo ""
    echo "=== Deleting *_frame_* PNGs under ${FIGPATH} (keeping all_movies/) ==="
    find "${FIGPATH}" -type f -name '*_frame_*_?????.png' -delete
    # Remove any now-empty per-series frame folders (all_movies is left intact).
    find "${FIGPATH}" -mindepth 1 -type d -empty -delete
fi

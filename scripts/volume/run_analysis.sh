#!/usr/bin/env bash
# =============================================================================
# run_analysis.sh  —  KPlot post-merger disk analysis script
#
# Usage:
#   bash run_analysis.sh [--disk] [--plot] [--all]
#
# With no flags both steps (disk, plot) are run.
#
# Thin wrapper around the kplot.volume analysis + plotting modules.  All
# analysis logic lives in the installed KPlot package, so this shell script
# is the only file you edit: set the paths in config.ini, then the run
# settings below.
#
# Requires KPlot to be installed in the active Python environment:
#   pip install -e external/plot-tools   # (from the KPlot checkout)
#   pip install -e .
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.ini"

# ─────────────────────────────────────────────────────────────────────────────
# Read paths from config.ini  (plain INI: parsed, never executed)
# ─────────────────────────────────────────────────────────────────────────────
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: ${CONFIG_FILE} not found."
    echo "       Copy config.example.ini to config.ini and set 'bindir', 'eos_table' + 'athenak_root'."
    exit 1
fi

# Scalar `key = value`.  A trailing inline comment (whitespace followed by # or
# ;) is stripped, so a '#' inside a path is still safe.
_cfg_get() {
    awk -v key="$1" '
        /^[[:space:]]*[#;]/ { next }
        /^[[:space:]]*\[/   { next }
        {
            k = $0; sub(/=.*/, "", k)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", k)
            if (k == key && index($0, "=")) {
                v = substr($0, index($0, "=") + 1)
                sub(/[[:space:]]+[#;].*/, "", v)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
                print v; exit
            }
        }
    ' "${CONFIG_FILE}"
}

BINDIR="$(_cfg_get bindir)"
EOS_TABLE="$(_cfg_get eos_table)"
PYTHONPATH_EXTRA="$(_cfg_get pythonpath_extra)"   # optional (e.g. a vtk install)

if [[ -z "${BINDIR}" ]]; then
    echo "ERROR: 'bindir' is not set in ${CONFIG_FILE}."; exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# ARG PARSING
# ─────────────────────────────────────────────────────────────────────────────
RUN_DISK=0; RUN_PLOT=0
if [[ $# -eq 0 ]]; then
    RUN_DISK=1; RUN_PLOT=1
fi
for arg in "$@"; do
    case "$arg" in
        --disk) RUN_DISK=1 ;;
        --plot) RUN_PLOT=1 ;;
        --all)  RUN_DISK=1; RUN_PLOT=1 ;;
        *) echo "Unknown flag: $arg"; echo "Usage: $0 [--disk] [--plot] [--all]"; exit 1 ;;
    esac
done

if [[ $RUN_DISK -eq 1 && -z "${EOS_TABLE}" ]]; then
    echo "ERROR: 'eos_table' is not set in ${CONFIG_FILE} (required for --disk)."; exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# RUN SETTINGS  ← edit per run
# ─────────────────────────────────────────────────────────────────────────────

# Directory where all output json/csv files will be written.
OUTPUT_DIR="${BINDIR}/disk"

# Extract the job-specific parameters from config.ini
DROP_FIRST_BINS="$(_cfg_get drop_first_bins)"
TRACKER="$(_cfg_get tracker)"
CENTER="$(_cfg_get center)"
HORIZON="$(_cfg_get horizon)"
R_EXCLUDE="$(_cfg_get r_exclude)"
RHO_CUT="$(_cfg_get rho_cut)"
RHO_KEEP="$(_cfg_get rho_keep)"
RHO_REMNANT="$(_cfg_get rho_remnant)"
BOUND_CRITERION="$(_cfg_get bound_criterion)"
R_DISK_MAX="$(_cfg_get r_disk_max)"
NBINS="$(_cfg_get nbins)"
RMAX="$(_cfg_get rmax)"
N_WORKERS="$(_cfg_get n_workers)"

if [[ $RUN_DISK -eq 1 && -z "${TRACKER}" && -z "${CENTER}" ]]; then
    echo "ERROR: either 'tracker' or 'center' must be set in ${CONFIG_FILE} (required for --disk)."; exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
# Aurora uses environment modules. Lmod's `module` is usually only a shell
# function for interactive shells, so source its init first if needed.
#if ! command -v module &>/dev/null; then
#    for _init in "${MODULESHOME:-/usr/share/lmod/lmod}/init/bash" \
#                 /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh; do
#        [[ -r "${_init}" ]] && source "${_init}" && break
#    done
#fi
#if command -v module &>/dev/null; then
#    module load python py-numpy py-scipy py-matplotlib py-h5py
#fi
# The module Python may have no pip/user-site; provide extra packages (e.g. vtk
# needed by athplot.load_sph_vtk) via pythonpath_extra in config.ini.
#if [[ -n "${PYTHONPATH_EXTRA}" ]]; then
#    export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
#fi

# Prefer the installed console script; fall back to the module form if KPlot
# has not been reinstalled since the kplot-volume-disk entry point was added.
_kplot() {
    local script="$1" module="$2"; shift 2
    if command -v "${script}" >/dev/null 2>&1; then
        "${script}" "$@"
    else
        python3 -m "${module}" "$@"
    fi
}

mkdir -p "${OUTPUT_DIR}"

# Optional settings: a key left blank in config.ini is omitted here, so the
# analysis module applies its own default rather than receiving an empty string.
DISK_ARGS=()
[[ -n "${DROP_FIRST_BINS}" ]] && DISK_ARGS+=(--drop-first-bins "${DROP_FIRST_BINS}")
[[ -n "${TRACKER}"         ]] && DISK_ARGS+=(--tracker         "${TRACKER}")
[[ -n "${HORIZON}"         ]] && DISK_ARGS+=(--horizon         "${HORIZON}")
[[ -n "${R_EXCLUDE}"       ]] && DISK_ARGS+=(--r-exclude       "${R_EXCLUDE}")
[[ -n "${RHO_CUT}"         ]] && DISK_ARGS+=(--rho-cut         "${RHO_CUT}")
[[ -n "${RHO_KEEP}"        ]] && DISK_ARGS+=(--rho-keep        "${RHO_KEEP}")
[[ -n "${RHO_REMNANT}"     ]] && DISK_ARGS+=(--rho-remnant     "${RHO_REMNANT}")
[[ -n "${BOUND_CRITERION}" ]] && DISK_ARGS+=(--bound-criterion "${BOUND_CRITERION}")
[[ -n "${R_DISK_MAX}"      ]] && DISK_ARGS+=(--r-disk-max      "${R_DISK_MAX}")
[[ -n "${NBINS}"           ]] && DISK_ARGS+=(--nbins           "${NBINS}")
[[ -n "${RMAX}"            ]] && DISK_ARGS+=(--rmax            "${RMAX}")
[[ -n "${N_WORKERS}"       ]] && DISK_ARGS+=(--n-workers       "${N_WORKERS}")
if [[ -n "${CENTER}" ]]; then
    read -ra CENTER_ARR <<< "${CENTER}"
    DISK_ARGS+=(--center "${CENTER_ARR[@]}")
fi

echo "============================================================"
echo "  Post-merger disk analysis"
echo "  3D dir     : ${BINDIR}"
echo "  Output     : ${OUTPUT_DIR}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: DISK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if [[ $RUN_DISK -eq 1 ]]; then
    echo ""
    echo "── Disk analysis ─────────────────────────────────────────────"
    _kplot kplot-volume-disk kplot.volume.disk \
        --bindir       "${BINDIR}" \
        --eos-table    "${EOS_TABLE}" \
        --outdir       "${OUTPUT_DIR}" \
        "${DISK_ARGS[@]+"${DISK_ARGS[@]}"}"
    echo "── Disk analysis done ───────────────────────────────────────"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: PLOTS — one histogram/profile frame per snapshot, stitched into movies.
# ─────────────────────────────────────────────────────────────────────────────
if [[ $RUN_PLOT -eq 1 ]]; then
    FIGDIR="${OUTPUT_DIR}/frames"

    echo ""
    echo "── Plotting histograms and profiles ─────────────────────────"
    _kplot kplot-volume-plot kplot.volume.plots \
        --outdir "${OUTPUT_DIR}" \
        --figdir "${FIGDIR}"

    MOVIE_FPS=10       # frames per second
    MOVIE_STRIDE=1     # use every Nth frame (1 = all)

    echo ""
    echo "=== Making movies from ${FIGDIR} ==="
    bash "${SCRIPT_DIR}/../system/make_movies.sh" "${MOVIE_FPS}" "${MOVIE_STRIDE}" "${FIGDIR}"
    echo "── Plotting done ────────────────────────────────────────────"
fi

echo ""
echo "All done. Results in: ${OUTPUT_DIR}"

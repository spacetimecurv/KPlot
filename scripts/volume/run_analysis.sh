#!/usr/bin/env bash
# =============================================================================
# run_analysis.sh  —  KPlot post-merger disk analysis script
#
# Usage:
#   bash run_analysis.sh [--disk] [--spectra] [--plot] [--all]
#
# With no flags all steps (disk, spectra, plot) are run.
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
    echo "       Copy config.example.ini to config.ini and set 'simpath' + 'eos_table'."
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

SIMPATH="$(_cfg_get simpath)"
BATCHTOOLS="$(_cfg_get batchtools)"
case "${BATCHTOOLS,,}" in
    false|0|no|off) BATCHTOOLS=false ;;
    *)              BATCHTOOLS=true  ;;
esac
EOS_TABLE="$(_cfg_get eos_table)"
PYTHONPATH_EXTRA="$(_cfg_get pythonpath_extra)"   # optional (e.g. a vtk install)

if [[ -z "${SIMPATH}" ]]; then
    echo "ERROR: 'simpath' is not set in ${CONFIG_FILE}."; exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# ARG PARSING
# ─────────────────────────────────────────────────────────────────────────────
RUN_DISK=0; RUN_SPECTRA=0; RUN_PLOT=0
if [[ $# -eq 0 ]]; then
    RUN_DISK=1; RUN_SPECTRA=1; RUN_PLOT=1
fi
for arg in "$@"; do
    case "$arg" in
        --disk) RUN_DISK=1 ;;
        --spectra) RUN_SPECTRA=1 ;;
        --plot) RUN_PLOT=1 ;;
        --all)  RUN_DISK=1; RUN_SPECTRA=1; RUN_PLOT=1 ;;
        *) echo "Unknown flag: $arg"; echo "Usage: $0 [--disk] [--spectra] [--plot] [--all]"; exit 1 ;;
    esac
done

if [[ $RUN_DISK -eq 1 && -z "${EOS_TABLE}" ]]; then
    echo "ERROR: 'eos_table' is not set in ${CONFIG_FILE} (required for --disk)."; exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# RUN SETTINGS  ← edit per run
# ─────────────────────────────────────────────────────────────────────────────

# Directory where all output json/csv files will be written.
OUTPUT_DIR="${SIMPATH}/disk"

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
SLICE_EXTENT="$(_cfg_get slice_extent)"
SLICE_NPTS="$(_cfg_get slice_npts)"
PARKER_DIR="$(_cfg_get parker_dir)"
PARKER_EPS="$(_cfg_get parker_eps)"
PARKER_RMAX="$(_cfg_get parker_rmax)"
PARKER_SMOOTH="$(_cfg_get parker_smooth)"
WIN_RAD="$(_cfg_get window_radius)"
TAR_DX="$(_cfg_get target_dx)"
N_WORKERS="$(_cfg_get n_workers)"
MOV_FPS="$(_cfg_get movie_fps)"
MOV_STR="$(_cfg_get movie_stride)"

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
[[ "${BATCHTOOLS}" == "false" ]] && DISK_ARGS+=(--no-batchtools)
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
[[ -n "${SLICE_EXTENT}"    ]] && DISK_ARGS+=(--slice-extent    "${SLICE_EXTENT}")
[[ -n "${SLICE_NPTS}"      ]] && DISK_ARGS+=(--slice-npts      "${SLICE_NPTS}")
[[ -n "${PARKER_DIR}"      ]] && DISK_ARGS+=(--parker-dir      "${PARKER_DIR}")
[[ -n "${PARKER_EPS}"      ]] && DISK_ARGS+=(--parker-eps      "${PARKER_EPS}")
[[ -n "${PARKER_RMAX}"     ]] && DISK_ARGS+=(--parker-rmax     "${PARKER_RMAX}")
[[ -n "${PARKER_SMOOTH}"   ]] && DISK_ARGS+=(--parker-smooth   "${PARKER_SMOOTH}")
[[ -n "${N_WORKERS}"       ]] && DISK_ARGS+=(--n-workers       "${N_WORKERS}")
if [[ -n "${CENTER}" ]]; then
  read -ra CENTER_ARR <<< "${CENTER}"
  DISK_ARGS+=(--center "${CENTER_ARR[@]}")
fi

SPECTRA_ARGS=()
[[ "${BATCHTOOLS}" == "false" ]] && SPECTRA_ARGS+=(--no-batchtools)
[[ -n "${DROP_FIRST_BINS}" ]] && SPECTRA_ARGS+=(--drop-first-bins "${DROP_FIRST_BINS}")
[[ -n "${WIN_RAD}"   ]] && SPECTRA_ARGS+=(--win-radius "${WIN_RAD}")
[[ -n "${TAR_DX}"    ]] && SPECTRA_ARGS+=(--target-dx   "${TAR_DX}")
[[ -n "${TRACKER}"   ]] && SPECTRA_ARGS+=(--tracker     "${TRACKER}")
[[ -n "${N_WORKERS}" ]] && SPECTRA_ARGS+=(--n-workers   "${N_WORKERS}")
if [[ -n "${CENTER}" ]]; then
  read -ra CENTER_ARR <<< "${CENTER}"
  SPECTRA_ARGS+=(--center "${CENTER_ARR[@]}")
fi

echo "============================================================"
echo "  Post-merger disk analysis"
echo "  Simulation : ${SIMPATH} (batchtools: ${BATCHTOOLS})"
echo "  Output     : ${OUTPUT_DIR}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: DISK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if [[ $RUN_DISK -eq 1 ]]; then
    echo ""
    echo "── Disk analysis ─────────────────────────────────────────────"
    _kplot kplot-volume-disk kplot.volume.disk \
        --simpath      "${SIMPATH}" \
        --eos-table    "${EOS_TABLE}" \
        --outdir       "${OUTPUT_DIR}" \
        "${DISK_ARGS[@]+"${DISK_ARGS[@]}"}"
    echo "── Disk analysis done ───────────────────────────────────────"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: SPECTRA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if [[ $RUN_SPECTRA -eq 1 ]]; then
    echo ""
    echo "── Spectra analysis ─────────────────────────────────────────────"
    _kplot kplot-volume-spectra kplot.volume.spectra \
        --simpath      "${SIMPATH}" \
        --outdir       "${OUTPUT_DIR}" \
        "${SPECTRA_ARGS[@]+"${SPECTRA_ARGS[@]}"}"
    echo "── Spectra analysis done ───────────────────────────────────────"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: PLOTS — one histogram/profile frame per snapshot, stitched into movies.
# ─────────────────────────────────────────────────────────────────────────────
if [[ $RUN_PLOT -eq 1 ]]; then
    FIGDIR="${OUTPUT_DIR}/frames"

    echo ""
    echo "── Plotting histograms and profiles ─────────────────────────"
    _kplot kplot-volume-plot kplot.volume.plots \
        --outdir "${OUTPUT_DIR}" \
        --figdir "${FIGDIR}"

    MOVIE_FPS="${MOV_FPS}"       # frames per second
    MOVIE_STRIDE="${MOV_STR}"    # use every Nth frame (1 = all)

    echo ""
    echo "=== Making movies from ${FIGDIR} ==="
    bash "${SCRIPT_DIR}/../system/make_movies.sh" "${MOVIE_FPS}" "${MOVIE_STRIDE}" "${FIGDIR}"
    echo "── Plotting done ────────────────────────────────────────────"
fi

echo ""
echo "All done. Results in: ${OUTPUT_DIR}"

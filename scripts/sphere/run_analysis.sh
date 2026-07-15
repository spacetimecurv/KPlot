#!/usr/bin/env bash
# =============================================================================
# run_analysis.sh  —  KPlot spherical-surface ejecta + neutrino analysis driver
#
# Usage:
#   bash run_analysis.sh [--ejecta] [--neutrino] [--plot] [--all]
#
# With no flags all steps (merger time, ejecta, neutrino, plot) are run.
#
# Thin wrapper around the kplot.sphere analysis modules.  All analysis logic
# lives in the installed KPlot package, so this shell script is the only file
# you edit: set the paths in config.ini, then the run settings below.
#
# Requires KPlot to be installed in the active Python environment:
#   pip install -e external/plot-tools   # (from the KPlot checkout)
#   pip install -e .
#
# Machine-specific PATHS (simpath, eos_table, optional pythonpath_extra) are
# read from config.ini next to this script.  The run settings below stay in
# this script — edit per run.
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
EOS_TABLE="$(_cfg_get eos_table)"
PYTHONPATH_EXTRA="$(_cfg_get pythonpath_extra)"   # optional (e.g. a vtk install)

if [[ -z "${SIMPATH}" ]]; then
    echo "ERROR: 'simpath' is not set in ${CONFIG_FILE}."; exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# RUN SETTINGS  ← edit per run
# ─────────────────────────────────────────────────────────────────────────────

# Directory where all output txt/npy/pdf files will be written.
OUTPUT_DIR="${SIMPATH}/analysis"

# Extract the job-specific parameters from config.ini
JOBNAME="$(_cfg_get jobname)"
RADIUS="$(_cfg_get radius)"
WF_RADIUS="$(_cfg_get wf_radius)"
PLOT_FROM_MERGER="$(_cfg_get plot_from_merger)"
T_POST_MS="$(_cfg_get t_post_ms)"
DFLOOR="$(_cfg_get dfloor)"
N_WORKERS="$(_cfg_get n_workers)"

# ── auto-discover all output-*/sph directories (sorted) ──────────────────────
SPH_DIRS=()
for d in "${SIMPATH}"/output-*/sph; do
    [[ -d "$d" ]] && SPH_DIRS+=("$d")
done
if [[ ${#SPH_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: no output-*/sph directories found under ${SIMPATH}"; exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# ARG PARSING
# ─────────────────────────────────────────────────────────────────────────────
RUN_EJECTA=0; RUN_NEUTRINO=0; RUN_PLOT=0
if [[ $# -eq 0 ]]; then
    RUN_EJECTA=1; RUN_NEUTRINO=1; RUN_PLOT=1
fi
for arg in "$@"; do
    case "$arg" in
        --ejecta)   RUN_EJECTA=1   ;;
        --neutrino) RUN_NEUTRINO=1 ;;
        --plot)     RUN_PLOT=1     ;;
        --all)      RUN_EJECTA=1; RUN_NEUTRINO=1; RUN_PLOT=1 ;;
        *) echo "Unknown flag: $arg"; echo "Usage: $0 [--ejecta] [--neutrino] [--plot] [--all]"; exit 1 ;;
    esac
done

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

# Prefer the installed console scripts; fall back to the module form if KPlot
# has not been reinstalled since the kplot.sphere entry points were added.
_kplot() {
    local script="$1" module="$2"; shift 2
    if command -v "${script}" >/dev/null 2>&1; then
        "${script}" "$@"
    else
        python3 -m "${module}" "$@"
    fi
}

mkdir -p "${OUTPUT_DIR}"

# Repeated --sph-dir DIR flags, one per restart segment.
SPH_ARGS=()
for d in "${SPH_DIRS[@]}"; do SPH_ARGS+=(--sph-dir "$d"); done

# Optional settings: a key left blank in config.ini is omitted here, so the
# analysis module applies its own default rather than receiving an empty string.
COMMON_ARGS=()
[[ -n "${JOBNAME}"   ]] && COMMON_ARGS+=(--jobname   "${JOBNAME}")
[[ -n "${RADIUS}"    ]] && COMMON_ARGS+=(--radius    "${RADIUS}")
[[ -n "${N_WORKERS}" ]] && COMMON_ARGS+=(--n-workers "${N_WORKERS}")

EJECTA_ARGS=()
[[ -n "${DFLOOR}"    ]] && EJECTA_ARGS+=(--dfloor    "${DFLOOR}")
[[ -n "${T_POST_MS}" ]] && EJECTA_ARGS+=(--t-post-ms "${T_POST_MS}")

MERGER_ARGS=()
[[ -n "${WF_RADIUS}" ]] && MERGER_ARGS+=(--radius "${WF_RADIUS}")

echo "============================================================"
echo "  BNS SPH analysis"
echo "  SPH dirs   : ${#SPH_DIRS[@]} segment(s)"
echo "  Radius     : ${RADIUS} M_sun"
echo "  Output     : ${OUTPUT_DIR}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: MERGER TIME  (from max |psi4_22|^2 of the (2,2) GW mode)
# ─────────────────────────────────────────────────────────────────────────────
MERGER_FILE="${OUTPUT_DIR}/merger_time.txt"
echo ""
echo "── Merger time (GW (2,2) mode) ──────────────────────────────"
_kplot kplot-sphere-mergertime kplot.sphere.mergertime \
    --simpath "${SIMPATH}" "${MERGER_ARGS[@]+"${MERGER_ARGS[@]}"}" --out "${MERGER_FILE}"
# Read the merger time (first non-comment numeric line) from the file.
T_MERGER_MSUN="$(awk '!/^#/ && NF {print $1; exit}' "${MERGER_FILE}")"
if [[ -z "${T_MERGER_MSUN}" ]]; then
    echo "ERROR: could not read merger time from ${MERGER_FILE}"; exit 1
fi
echo "  T_merger = ${T_MERGER_MSUN} M_sun"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 + 2: EJECTA and NEUTRINO  (run concurrently — they share no state)
# ─────────────────────────────────────────────────────────────────────────────
EJECTA_PID=""; NEUTRINO_PID=""

if [[ $RUN_EJECTA -eq 1 ]]; then
    echo ""
    echo "── Ejecta analysis (background) ─────────────────────────────"
    if [[ -z "${EOS_TABLE}" ]]; then
        echo "ERROR: 'eos_table' is not set in ${CONFIG_FILE} (required for ejecta)."; exit 1
    fi
    _kplot kplot-sphere-ejecta kplot.sphere.ejecta \
        "${SPH_ARGS[@]}" \
        --eos-table  "${EOS_TABLE}" \
        --output-dir "${OUTPUT_DIR}" \
        --t-merger   "${T_MERGER_MSUN}" \
        "${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"}" \
        "${EJECTA_ARGS[@]+"${EJECTA_ARGS[@]}"}" &
    EJECTA_PID=$!
fi

if [[ $RUN_NEUTRINO -eq 1 ]]; then
    echo "── Neutrino analysis (background) ───────────────────────────"
    _kplot kplot-sphere-neutrinos kplot.sphere.neutrinos \
        "${SPH_ARGS[@]}" \
        --output-dir "${OUTPUT_DIR}" \
        "${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"}" &
    NEUTRINO_PID=$!
fi

if [[ -n "${EJECTA_PID}" ]]; then
    wait "${EJECTA_PID}" && echo "── Ejecta done ──" || { echo "ERROR: ejecta failed"; exit 1; }
fi
if [[ -n "${NEUTRINO_PID}" ]]; then
    wait "${NEUTRINO_PID}" && echo "── Neutrino done ──" || { echo "ERROR: neutrino failed"; exit 1; }
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: PLOT
# ─────────────────────────────────────────────────────────────────────────────
if [[ $RUN_PLOT -eq 1 ]]; then
    echo ""
    echo "── Plotting ─────────────────────────────────────────────────"
    PLOT_ARGS=(--output-dir "${OUTPUT_DIR}" --t-merger "${T_MERGER_MSUN}")
    [[ -n "${RADIUS}" ]] && PLOT_ARGS+=(--radius "${RADIUS}")
    [[ "${PLOT_FROM_MERGER}" == "1" ]] && PLOT_ARGS+=(--from-merger)
    _kplot kplot-sphere-plot kplot.sphere.plots "${PLOT_ARGS[@]}"
    echo "── Plotting done ────────────────────────────────────────────"
fi

echo ""
echo "All done. Results in: ${OUTPUT_DIR}"

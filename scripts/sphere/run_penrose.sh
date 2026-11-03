#!/usr/bin/env bash
# =============================================================================
# run_penrose.sh  -  KPlot generalized Penrose process analysis
#
# Usage:
#   bash run_penrose.sh [--analysis] [--plot] [--all] [--slices]
#
# --slices also plots P^2 on the x-z plane through the BH from the 3D dumps
# (penrose_slice_snapshots; blank = all).
#
# With no flags (or --all) all steps (analysis, plot, slices) are run.
#
# Loads spheres close to the horizon and then evaluates the generalized
# criteria for an active Penrose process by Lasota et al. 2014.
#
# Requires KPlot to be installed in the active Python environment:
#   pip install -e external/plot-tools   # (from the KPlot checkout)
#   pip install -e .
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.ini"

# -----------------------------------------------------------------------------
# Read paths from config.ini  (plain INI: parsed, never executed)
# -----------------------------------------------------------------------------
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: ${CONFIG_FILE} not found."
  echo "       Copy config.example.ini to config.ini and set 'simpath'."
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

if [[ -z "${SIMPATH}" ]]; then
  echo "ERROR: 'simpath' is not set in ${CONFIG_FILE}."; exit 1
fi

# -----------------------------------------------------------------------------
# RUN SETTINGS
# -----------------------------------------------------------------------------
JOBNAME="$(_cfg_get jobname)"
N_WORKERS="$(_cfg_get n_workers)"
PEN_RADII="$(_cfg_get penrose_radii)"
PEN_HORIZON="$(_cfg_get penrose_horizon)"
PEN_T_MERGER="$(_cfg_get penrose_t_merger)"
PEN_SLICE_SNAPS="$(_cfg_get penrose_slice_snapshots)"
EOS_TABLE="$(_cfg_get eos_table)"

# Defaults: the black hole horizon of the merged run, extracted at `radius`.
[[ -z "${PEN_HORIZON}" ]] && PEN_HORIZON="${SIMPATH}/merged/${JOBNAME:-bns}.horizon_summary_0.txt"

# Directory where the Penrose outputs are written.
OUTPUT_DIR="${SIMPATH}/penrose"

# -- auto-discover all output-*/sph directories (sorted) ----------------------
SPH_PATTERN="output-*/sph"
if [[ "${BATCHTOOLS}" == "false" ]]; then
  SPH_PATTERN="sph"
  [[ -d "${SIMPATH}/sph" ]] || SPH_PATTERN="."
fi
SPH_DIRS=()
for d in "${SIMPATH}"/${SPH_PATTERN}; do
  [[ -d "$d" ]] && SPH_DIRS+=("$d")
done

# -----------------------------------------------------------------------------
# ARG PARSING
# -----------------------------------------------------------------------------
RUN_ANALYSIS=0; RUN_PLOT=0; RUN_SLICES=0
if [[ $# -eq 0 ]]; then
  RUN_ANALYSIS=1; RUN_PLOT=1; RUN_SLICES=1
fi
for arg in "$@"; do
  case "$arg" in
    --analysis) RUN_ANALYSIS=1 ;;
    --plot)     RUN_PLOT=1     ;;
    --all)      RUN_ANALYSIS=1; RUN_PLOT=1; RUN_SLICES=1 ;;
    --slices)   RUN_SLICES=1   ;;
    *) echo "Unknown flag: $arg"; echo "Usage: $0 [--analysis] [--plot] [--all] [--slices]"; exit 1 ;;
  esac
done
[[ $RUN_SLICES -eq 1 && $RUN_ANALYSIS -eq 0 ]] && RUN_PLOT=1

if [[ $RUN_ANALYSIS -eq 1 && ${#SPH_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: no ${SPH_PATTERN} directories found under ${SIMPATH}"; exit 1
fi

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
# Prefer the installed console script; fall back to the module form if KPlot
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

# Optional settings: a key left blank in config.ini is omitted here, so the
# analysis module applies its own default rather than receiving an empty string.
PEN_ARGS=(--output-dir "${OUTPUT_DIR}")
[[ -n "${EOS_TABLE}" ]] && PEN_ARGS+=(--eos-table "${EOS_TABLE}")
for r in ${PEN_RADII//,/ }; do PEN_ARGS+=(--radius "$r"); done
[[ -n "${JOBNAME}"   ]] && PEN_ARGS+=(--jobname   "${JOBNAME}")
[[ -n "${N_WORKERS}" ]] && PEN_ARGS+=(--n-workers "${N_WORKERS}")
[[ $RUN_SLICES -eq 1 ]] && PEN_ARGS+=(--slices "${SIMPATH}" --horizon-file "${PEN_HORIZON}")
if [[ $RUN_SLICES -eq 1 ]]; then
  for n in ${PEN_SLICE_SNAPS//,/ }; do PEN_ARGS+=(--slice-snapshot "$n"); done
fi

PLOT_ARGS=()
PLOT_ARGS+=(--horizon-file "${PEN_HORIZON}")
[[ -n "${PEN_T_MERGER}" ]] && PLOT_ARGS+=(--t-merger  "${PEN_T_MERGER}")

SPH_ARGS=()
for d in "${SPH_DIRS[@]+"${SPH_DIRS[@]}"}"; do SPH_ARGS+=(--sph-dir "$d"); done

echo "============================================================"
echo "  Generalized Penrose process analysis"
echo "  SPH dirs   : ${#SPH_DIRS[@]} segment(s)"
echo "  Radii      : ${PEN_RADII:-default} M_sun"
echo "  t_merger   : ${PEN_T_MERGER:-none (absolute time)}"
echo "  Output     : ${OUTPUT_DIR}"
echo "============================================================"

if [[ $RUN_ANALYSIS -eq 1 ]]; then
  echo ""
  echo "-- Penrose analysis ---------------------------------------------"
  PLOT_FLAG=()
  [[ $RUN_PLOT -eq 1 ]] && PLOT_FLAG=(--plot "${PLOT_ARGS[@]}")
  _kplot kplot-sphere-penrose kplot.sphere.penrose \
    "${SPH_ARGS[@]}" "${PEN_ARGS[@]}" "${PLOT_FLAG[@]+"${PLOT_FLAG[@]}"}"
  echo "-- Penrose analysis done ----------------------------------------"
elif [[ $RUN_PLOT -eq 1 ]]; then
  echo ""
  echo "-- Plotting Penrose diagnostics ---------------------------------"
  _kplot kplot-sphere-penrose kplot.sphere.penrose "${PEN_ARGS[@]}" --plot-only "${PLOT_ARGS[@]}"
  echo "-- Plotting done --------------------------------------------"
fi

echo ""
echo "All done. Results in: ${OUTPUT_DIR}"

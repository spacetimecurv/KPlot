#!/usr/bin/env bash
# =============================================================================
# run_butterfly.sh  —  KPlot butterfly diagram (toroidal field) analysis script
#
# Usage:
#   bash run_butterfly.sh [--analysis] [--plot] [--all]
#
# With no flags both steps (analysis, plot) are run.
#
# Independent of run_analysis.sh: the butterfly diagram is extracted on its own
# (typically inner-disk) radius and needs no GW merger time.  Shares config.ini
# with run_analysis.sh; the butterfly settings live in its [butterfly] section.
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
  echo "       Copy config.example.ini to config.ini and set 'simpath' + 'butterfly_radius'."
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

# ─────────────────────────────────────────────────────────────────────────────
# RUN SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
JOBNAME="$(_cfg_get jobname)"
N_WORKERS="$(_cfg_get n_workers)"
BF_RADIUS="$(_cfg_get butterfly_radius)"
BF_T_MERGER="$(_cfg_get butterfly_t_merger)"

if [[ -z "${BF_RADIUS}" ]]; then
  echo "ERROR: 'butterfly_radius' is not set in ${CONFIG_FILE}."; exit 1
fi

# Directory where the butterfly outputs are written, separate from the
# ejecta/neutrino results of run_analysis.sh.
OUTPUT_DIR="${SIMPATH}/butterfly_$(printf '%gM' "${BF_RADIUS}")"

# ── auto-discover all output-*/sph directories (sorted) ──────────────────────
SPH_PATTERN="output-*/sph"
if [[ "${BATCHTOOLS}" == "false" ]]; then
  SPH_PATTERN="sph"
  [[ -d "${SIMPATH}/sph" ]] || SPH_PATTERN="."
fi
SPH_DIRS=()
for d in "${SIMPATH}"/${SPH_PATTERN}; do
  [[ -d "$d" ]] && SPH_DIRS+=("$d")
done

# ─────────────────────────────────────────────────────────────────────────────
# ARG PARSING
# ─────────────────────────────────────────────────────────────────────────────
RUN_ANALYSIS=0; RUN_PLOT=0
if [[ $# -eq 0 ]]; then
  RUN_ANALYSIS=1; RUN_PLOT=1
fi
for arg in "$@"; do
  case "$arg" in
    --analysis) RUN_ANALYSIS=1 ;;
    --plot)     RUN_PLOT=1     ;;
    --all)      RUN_ANALYSIS=1; RUN_PLOT=1 ;;
    *) echo "Unknown flag: $arg"; echo "Usage: $0 [--analysis] [--plot] [--all]"; exit 1 ;;
  esac
done

if [[ $RUN_ANALYSIS -eq 1 && ${#SPH_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: no ${SPH_PATTERN} directories found under ${SIMPATH}"; exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
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
BF_ARGS=(--output-dir "${OUTPUT_DIR}" --radius "${BF_RADIUS}")
[[ -n "${JOBNAME}"     ]] && BF_ARGS+=(--jobname   "${JOBNAME}")
[[ -n "${N_WORKERS}"   ]] && BF_ARGS+=(--n-workers "${N_WORKERS}")
[[ -n "${BF_T_MERGER}" ]] && BF_ARGS+=(--t-merger  "${BF_T_MERGER}")

SPH_ARGS=()
for d in "${SPH_DIRS[@]+"${SPH_DIRS[@]}"}"; do SPH_ARGS+=(--sph-dir "$d"); done

echo "============================================================"
echo "  Butterfly diagram"
echo "  SPH dirs   : ${#SPH_DIRS[@]} segment(s)"
echo "  Radius     : ${BF_RADIUS} M_sun"
echo "  t_merger   : ${BF_T_MERGER:-none (absolute time)}"
echo "  Output     : ${OUTPUT_DIR}"
echo "============================================================"

if [[ $RUN_ANALYSIS -eq 1 ]]; then
  echo ""
  echo "── Butterfly diagram analysis ───────────────────────────────"
  PLOT_FLAG=()
  [[ $RUN_PLOT -eq 1 ]] && PLOT_FLAG=(--plot)
  _kplot kplot-sphere-butterfly kplot.sphere.butterfly \
    "${SPH_ARGS[@]}" "${BF_ARGS[@]}" "${PLOT_FLAG[@]+"${PLOT_FLAG[@]}"}"
  echo "── Butterfly diagram done ───────────────────────────────────"
elif [[ $RUN_PLOT -eq 1 ]]; then
  echo ""
  echo "── Plotting butterfly diagram ───────────────────────────────"
  _kplot kplot-sphere-butterfly kplot.sphere.butterfly "${BF_ARGS[@]}" --plot-only
  echo "── Plotting done ────────────────────────────────────────────"
fi

echo ""
echo "All done. Results in: ${OUTPUT_DIR}"

#!/usr/bin/env bash
# =============================================================================
# run_mad.sh  —  KPlot SANE/MAD state and neutrino cooling efficiency
#
# Usage:
#   bash run_mad.sh [--analysis] [--plot] [--all]
#
# With no flags both steps (analysis, plot) are run.
#
# Integrates the accretion rate and the magnetic flux on the black-hole centred
# sph surfaces (mad_radii) and forms the dimensionless flux
# phi = Phi_B / sqrt(Mdot r_g^2) and the cooling efficiency eta = L_nu / (Mdot c^2).
# eta needs the neutrino luminosity of run_analysis.sh (Lnu_E_total.txt); without
# it only Mdot and phi are plotted.  The merger time is taken from the
# merger_time.txt of run_analysis.sh when present.  Shares config.ini with
# run_analysis.sh; the settings live in its [mad] section.
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

# ─────────────────────────────────────────────────────────────────────────────
# RUN SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
JOBNAME="$(_cfg_get jobname)"
N_WORKERS="$(_cfg_get n_workers)"
RADIUS="$(_cfg_get radius)"
T_POST_MS="$(_cfg_get t_post_ms)"
MAD_RADII="$(_cfg_get mad_radii)"
MAD_HORIZON="$(_cfg_get mad_horizon)"
MAD_M_BH="$(_cfg_get mad_m_bh)"
MAD_LNU_DIR="$(_cfg_get mad_lnu_dir)"
MAD_T_AVG="$(_cfg_get mad_t_avg)"
MAD_T_MIN="$(_cfg_get mad_t_min)"

# Defaults: the black hole horizon of the merged run, and the neutrino results of
# run_analysis.sh, extracted at `radius`.
[[ -z "${MAD_HORIZON}" ]] && MAD_HORIZON="${SIMPATH}/merged/${JOBNAME:-bns}.horizon_summary_0.txt"
[[ -z "${MAD_LNU_DIR}" ]] && \
  MAD_LNU_DIR="${SIMPATH}/$(LC_NUMERIC=C printf '%gM_%gms' "${RADIUS:-300}" "${T_POST_MS:-25}")"
LNU_FILE="${MAD_LNU_DIR}/Lnu_E_total.txt"
MERGER_FILE="${MAD_LNU_DIR}/merger_time.txt"

# Directory where the MAD outputs are written, separate from the
# ejecta/neutrino results of run_analysis.sh.
OUTPUT_DIR="${SIMPATH}/mad"

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
MAD_ARGS=(--output-dir "${OUTPUT_DIR}")
for r in ${MAD_RADII//,/ }; do MAD_ARGS+=(--radius "$r"); done
[[ -n "${JOBNAME}"   ]] && MAD_ARGS+=(--jobname   "${JOBNAME}")
[[ -n "${N_WORKERS}" ]] && MAD_ARGS+=(--n-workers "${N_WORKERS}")

PLOT_ARGS=()
if [[ -n "${MAD_M_BH}" ]]; then
  PLOT_ARGS+=(--m-bh "${MAD_M_BH}")
else
  PLOT_ARGS+=(--horizon-file "${MAD_HORIZON}")
fi
if [[ -f "${LNU_FILE}" ]]; then
  PLOT_ARGS+=(--lnu-file "${LNU_FILE}" --lnu-radius "${RADIUS:-300}")
else
  echo "WARNING: ${LNU_FILE} not found; run run_analysis.sh first for eta."
fi
T_MERGER_MSUN=""
if [[ -f "${MERGER_FILE}" ]]; then
  T_MERGER_MSUN="$(awk '!/^#/ && NF {print $1; exit}' "${MERGER_FILE}")"
fi
[[ -n "${T_MERGER_MSUN}" ]] && PLOT_ARGS+=(--t-merger "${T_MERGER_MSUN}")
[[ -n "${MAD_T_MIN}"     ]] && PLOT_ARGS+=(--t-min    "${MAD_T_MIN}")
[[ -n "${MAD_T_AVG}"     ]] && PLOT_ARGS+=(--t-avg    "${MAD_T_AVG}")

SPH_ARGS=()
for d in "${SPH_DIRS[@]+"${SPH_DIRS[@]}"}"; do SPH_ARGS+=(--sph-dir "$d"); done

echo "============================================================"
echo "  SANE/MAD state and neutrino cooling efficiency"
echo "  SPH dirs   : ${#SPH_DIRS[@]} segment(s)"
echo "  Radii      : ${MAD_RADII:-default} M_sun"
echo "  M_BH       : ${MAD_M_BH:-${MAD_HORIZON}}"
echo "  L_nu       : ${LNU_FILE} (r = ${RADIUS:-300} M_sun)"
echo "  t_merger   : ${T_MERGER_MSUN:-none (absolute time)}"
echo "  Output     : ${OUTPUT_DIR}"
echo "============================================================"

if [[ $RUN_ANALYSIS -eq 1 ]]; then
  echo ""
  echo "── MAD analysis ─────────────────────────────────────────────"
  PLOT_FLAG=()
  [[ $RUN_PLOT -eq 1 ]] && PLOT_FLAG=(--plot "${PLOT_ARGS[@]}")
  _kplot kplot-sphere-mad kplot.sphere.mad \
    "${SPH_ARGS[@]}" "${MAD_ARGS[@]}" "${PLOT_FLAG[@]+"${PLOT_FLAG[@]}"}"
  echo "── MAD analysis done ────────────────────────────────────────"
elif [[ $RUN_PLOT -eq 1 ]]; then
  echo ""
  echo "── Plotting MAD diagnostics ─────────────────────────────────"
  _kplot kplot-sphere-mad kplot.sphere.mad "${MAD_ARGS[@]}" --plot-only "${PLOT_ARGS[@]}"
  echo "── Plotting done ────────────────────────────────────────────"
fi

echo ""
echo "All done. Results in: ${OUTPUT_DIR}"

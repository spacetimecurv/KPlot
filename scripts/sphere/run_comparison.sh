#!/usr/bin/env bash
# =============================================================================
# run_comparison.sh  —  overlay neutrino diagnostics from 2+ models
#
# Wraps the kplot.sphere.comparison module.  Reuses the .txt outputs already
# produced by run_analysis.sh in each model's analysis directory (no data
# regeneration).
#
# Usage:
#   bash run_comparison.sh                 # uses the MODELS list below
#   bash run_comparison.sh DIR1 LABEL1 DIR2 LABEL2 [...]   # override on CLI
#
# Edit the MODELS list below (one "analysis_dir|label" per model), then run.
# First model is drawn solid, second dashed, third dotted, fourth dash-dot.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.ini"

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS  ← edit per run
# ─────────────────────────────────────────────────────────────────────────────

# Models to overlay: one "analysis_dir|legend label" per line (2 to 4 models).
# Each analysis_dir must contain the merger_time.txt / Lnu_E_*.txt / Eav_*.txt
# produced by run_analysis.sh.
MODELS=(
    "/lus/flare/projects/CompactBinaryMerger/yiqiu/Large_sim/bundle_run/VVLR_test_15/analysis|eq (w/ NEPS)"
    "/lus/flare/projects/CompactBinaryMerger/yiqiu/Large_sim/bundle_run/archive_20260625_235523/VVLR_eq_number/analysis|non-eq (w/ NEPS)"
)

# Output figure filename.
OUT="fig_neutrino_comparison_noneps.png"

# Where to write the figure (empty = <first analysis dir>/comparison_plot/).
OUTDIR=""

# Override x-range [ms] about merger, e.g. XLIM=(-2 20).  Empty = model overlap.
XLIM=()

# ─────────────────────────────────────────────────────────────────────────────
# Allow overriding the MODELS list from the command line (DIR LABEL pairs).
# ─────────────────────────────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
    if [[ $(( $# % 2 )) -ne 0 ]]; then
        echo "ERROR: command-line args must be DIR LABEL pairs (got $# args)."
        echo "Usage: $0 [DIR1 LABEL1 DIR2 LABEL2 ...]"
        exit 1
    fi
    MODELS=()
    while [[ $# -gt 0 ]]; do
        MODELS+=("$1|$2")
        shift 2
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# SETUP  (same Lmod handling as run_analysis.sh)
# ─────────────────────────────────────────────────────────────────────────────
if ! command -v module &>/dev/null; then
    for _init in "${MODULESHOME:-/usr/share/lmod/lmod}/init/bash" \
                 /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh; do
        [[ -r "${_init}" ]] && source "${_init}" && break
    done
fi
if command -v module &>/dev/null; then
    module load python py-numpy py-matplotlib
fi

# Optional extra PYTHONPATH from config.ini (the plotting step needs no vtk,
# so config.ini is not required here — only honoured when present).
if [[ -f "${CONFIG_FILE}" ]]; then
    PYTHONPATH_EXTRA="$(awk -v key=pythonpath_extra '
        /^[[:space:]]*[#;]/ { next }
        /^[[:space:]]*\[/   { next }
        {
            k = $0; sub(/=.*/, "", k)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", k)
            if (k == key && index($0, "=")) {
                v = substr($0, index($0, "=") + 1)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
                print v; exit
            }
        }
    ' "${CONFIG_FILE}")"
    if [[ -n "${PYTHONPATH_EXTRA}" ]]; then
        export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Build the comparison args from MODELS
# ─────────────────────────────────────────────────────────────────────────────
ARGS=()
for entry in "${MODELS[@]}"; do
    dir="${entry%%|*}"
    label="${entry#*|}"
    if [[ ! -d "${dir}" ]]; then
        echo "ERROR: analysis directory not found: ${dir}"; exit 1
    fi
    ARGS+=(--dir "${dir}" --label "${label}")
done

ARGS+=(--out "${OUT}")
[[ -n "${OUTDIR}" ]] && ARGS+=(--outdir "${OUTDIR}")
[[ ${#XLIM[@]} -eq 2 ]] && ARGS+=(--xlim-ms "${XLIM[0]}" "${XLIM[1]}")

echo "============================================================"
echo "  BNS neutrino comparison plot"
echo "  Models : ${#MODELS[@]}"
for entry in "${MODELS[@]}"; do echo "    - ${entry%%|*}  (${entry#*|})"; done
echo "============================================================"

# Prefer the installed console script; fall back to the module form if KPlot
# has not been reinstalled since the kplot.sphere entry points were added.
if command -v kplot-sphere-compare >/dev/null 2>&1; then
    kplot-sphere-compare "${ARGS[@]}"
else
    python3 -m kplot.sphere.comparison "${ARGS[@]}"
fi

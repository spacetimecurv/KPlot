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

# Whitespace-separated list for a key whose value continues over the following
# lines (INI has no array syntax), e.g.
#     sections =
#         density        # rendered
#         # temperature  <- disabled, stays commented out
# The list ends at the next `key =` or [section].  Comments are stripped and
# stray parentheses are tolerated, so a bash-style `sections=( ... )` block
# parses too.
_cfg_get_list() {
    awk -v key="$1" '
        function clean(s) {
            sub(/[[:space:]]*[#;].*/, "", s)   # drop comments
            gsub(/[()]/, " ", s)               # tolerate ( ... ) wrappers
            return s
        }
        !found {
            if ($0 ~ /^[[:space:]]*[#;]/ || $0 ~ /^[[:space:]]*\[/) next
            k = $0; sub(/=.*/, "", k)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", k)
            if (k == key && index($0, "=")) {
                found = 1
                print clean(substr($0, index($0, "=") + 1))
            }
            next
        }
        $0 ~ /^[[:space:]]*\[/                                   { exit }
        $0 ~ /^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=/  { exit }
        { print clean($0) }
    ' "${CONFIG_FILE}"
}

SIMPATH="$(_cfg_get simpath)"

if [[ -z "${SIMPATH}" ]]; then
    echo "ERROR: 'simpath' is not set in ${CONFIG_FILE}."; exit 1
fi

FIGPATH="${SIMPATH}/Figs"    # default: <SIMPATH>/Figs

# Job-specifc input parameters parsed from config.ini
JOBNAME="$(_cfg_get jobname)"
PLANE="$(_cfg_get plane)"
SHOW_TRACKERS="$(_cfg_get show_trackers)"
SHOW_HORIZON="$(_cfg_get show_horizon)"
SKIP_EXISTING="$(_cfg_get skip_existing)"
FULL_DOMAIN="$(_cfg_get full_domain)"
UNITS="$(_cfg_get units)"
TIME_UNITS="$(_cfg_get time_units)"
DELETE_FRAMES="$(_cfg_get delete_frames)"

# ---------------------------------------------------------------------------
# SECTIONS — comment out any line in config.ini to disable that diagnostic.
# Left empty, kplot-system falls back to its default ("all").
# ---------------------------------------------------------------------------
SECTIONS=()
while read -r _tok; do
    [[ -n "${_tok}" ]] && SECTIONS+=("${_tok}")
done < <(_cfg_get_list sections | tr -s '[:space:]' '\n')
# ---------------------------------------------------------------------------

#module load python py-matplotlib py-numpy py-scipy
#module load ffmpeg          # provides ffmpeg for make_movies.sh

# Every key is optional: left blank in config.ini, the flag is omitted and
# kplot-system applies its own default.
ARGS=(--simpath "${SIMPATH}")
[[ -n "${FIGPATH:-}"    ]] && ARGS+=(--figpath    "${FIGPATH}")
[[ -n "${JOBNAME:-}"    ]] && ARGS+=(--jobname    "${JOBNAME}")
[[ -n "${PLANE:-}"      ]] && ARGS+=(--plane      "${PLANE}")
[[ -n "${TIME_UNITS:-}" ]] && ARGS+=(--time-units "${TIME_UNITS}")
[[ -n "${UNITS:-}"      ]] && ARGS+=(--units      "${UNITS}")
[[ "${SHOW_TRACKERS}" == "false" ]] && ARGS+=(--no-trackers)
[[ "${SHOW_HORIZON}"  == "true"  ]] && ARGS+=(--show-horizon)
[[ "${SKIP_EXISTING}" == "true"  ]] && ARGS+=(--skip-existing)
[[ "${FULL_DOMAIN}"   == "true"  ]] && ARGS+=(--full-domain)
[[ ${#SECTIONS[@]} -gt 0 ]] && ARGS+=(--sections "${SECTIONS[@]}")

# Run the KPlot full-run visualizer.  Prefer the installed console script; fall
# back to the module form if KPlot has not been reinstalled since it was added.
if command -v kplot-system >/dev/null 2>&1; then
    kplot-system "${ARGS[@]}"
else
    python3 -m kplot.system "${ARGS[@]}"
fi

# Stop here if rendering failed — otherwise the movie and DELETE_FRAMES steps
# below would run against stale frames and could delete a previous good run.
_rc=$?
if [[ ${_rc} -ne 0 ]]; then
    echo ""
    echo "ERROR: the KPlot system plotter failed (exit ${_rc}); skipping movies and cleanup."
    exit "${_rc}"
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

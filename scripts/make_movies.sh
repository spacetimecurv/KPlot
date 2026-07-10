#!/usr/bin/env bash
# make_movies.sh
#
# Creates one MP4 per frame sequence found in FIGPATH.
# STRIDE controls sub-sampling: 1 = every frame, 10 = every 10th frame, etc.
#
# Usage:
#   bash make_movies.sh               # use settings below
#   bash make_movies.sh 25 1          # FPS=25, every frame
#   bash make_movies.sh 25 10         # FPS=25, every 10th frame
#   bash make_movies.sh 25 10 /path   # also override FIGPATH

# Log everything to a file so errors are readable even if the terminal closes.
_LOGFILE="${HOME}/make_movies.log"
exec > >(tee -a "${_LOGFILE}") 2>&1
echo "===== $(date) ====="
echo "Log: ${_LOGFILE}"

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------
FIGPATH="/lus/flare/projects/CompactBinaryMerger/yiqiu/AthenaK_runs/lorene_bns/DD2_m1_VVLR_test/no_inelastic/Figs"



# Frames per second in the output movie.
# 10–15  → slow, good for inspecting individual frames
# 24–30  → cinematic / standard
FPS=10

# Use every STRIDEth frame from the available PNGs.
# 1  → all frames (00000, 00001, 00002, ...)
# 10 → every 10th (00000, 00010, 00020, ...)
STRIDE=10

# H.264 quality: 0 (lossless) – 51 (worst).  18 ≈ visually lossless.
# Only used by encoders that support -crf (libx264, libx265, h264_nvenc).
CRF=18

# Video encoder.  Leave empty for auto-detection (recommended).
# Override if needed, e.g.:  VCODEC="mpeg4"
VCODEC=""
# ---------------------------------------------------------------------------

# Command-line overrides
FPS="${1:-${FPS}}"
STRIDE="${2:-${STRIDE}}"
FIGPATH="${3:-${FIGPATH}}"

if ! command -v ffmpeg &>/dev/null; then
    echo "ERROR: ffmpeg not found in PATH."
    echo "  Install via conda:  conda install -c conda-forge ffmpeg"
    echo "  Then activate the environment and re-run this script."
    return 1 2>/dev/null || exit 1
fi

# Auto-detect the best available encoder (libx264 > h264_nvenc > libx265 > mpeg4).
# Falls back to GIF, which is always supported by ffmpeg.
if [[ -z "${VCODEC}" ]]; then
    _encoders=$(ffmpeg -hide_banner -encoders 2>/dev/null)
    for _c in libx264 h264_nvenc libx265 mpeg4; do
        if echo "${_encoders}" | grep -q " ${_c} "; then
            VCODEC="${_c}"
            break
        fi
    done
    [[ -z "${VCODEC}" ]] && VCODEC="gif"
fi
echo "Encoder : ${VCODEC}"

# Quality flags (not used for GIF — palette filter handles quality instead)
case "${VCODEC}" in
    libx264|libx265|h264_nvenc)  QUALITY_FLAGS=(-crf "${CRF}") ;;
    gif)                         QUALITY_FLAGS=()               ;;
    *)                           QUALITY_FLAGS=(-q:v 5)         ;;
esac

if [[ ! -d "${FIGPATH}" ]]; then
    echo "ERROR: FIGPATH not found: ${FIGPATH}"
    echo "  Edit FIGPATH at the top of make_movies.sh to point to your Figs/ directory."
    return 1 2>/dev/null || exit 1
fi

cd "${FIGPATH}"

# ---------------------------------------------------------------------------
# Discover per-series frame folders
# Each rendering operation writes its frames to its own subfolder, named
# <diagnostic>_<plane>/ (e.g. temperature_xy/, dens_xz/), containing
# <something>_NNNNN.png frames.  One movie is built per such folder.
# ---------------------------------------------------------------------------
MOVIEDIR="${FIGPATH}/all_movies"

shopt -s nullglob
SERIES_DIRS=()
for d in */ ; do
    d="${d%/}"
    [[ "${d}" == "all_movies" ]] && continue
    _f=( "${d}"/*_?????.png )
    [[ ${#_f[@]} -gt 0 ]] && SERIES_DIRS+=("${d}")
done
shopt -u nullglob

if [[ ${#SERIES_DIRS[@]} -eq 0 ]]; then
    echo "No <series>/*_NNNNN.png frame folders found in ${FIGPATH}"
    return 0 2>/dev/null || exit 0
fi

echo "FIGPATH : ${FIGPATH}"
echo "FPS     : ${FPS}"
echo "STRIDE  : ${STRIDE}"
echo "Found ${#SERIES_DIRS[@]} frame folder(s)"
echo ""

mkdir -p "${MOVIEDIR}"

# ---------------------------------------------------------------------------
# Build one movie per frame folder, written into all_movies/ (separate from
# the rendered images).
# ---------------------------------------------------------------------------
for series in "${SERIES_DIRS[@]}"; do
    shopt -s nullglob
    ALL_FRAMES=( $(ls "${series}"/*_?????.png | sort) )
    shopt -u nullglob

    if [[ ${#ALL_FRAMES[@]} -eq 0 ]]; then
        continue
    fi

    # Sub-sample: keep every STRIDEth frame
    FRAMES=()
    for (( j=0; j<${#ALL_FRAMES[@]}; j+=STRIDE )); do
        FRAMES+=("${ALL_FRAMES[j]}")
    done

    # GIF → .gif; everything else → .mp4.  Movie is named after the folder.
    [[ "${VCODEC}" == "gif" ]] && _ext="gif" || _ext="mp4"
    outfile="${MOVIEDIR}/${series}.${_ext}"
    listfile="${series}/.fflist"

    echo "[${series}]  ${#ALL_FRAMES[@]} available, ${#FRAMES[@]} used  ->  $(basename "${outfile}")"

    # Write ffmpeg concat file list
    : > "${listfile}"
    for f in "${FRAMES[@]}"; do
        printf "file '%s'\n" "${PWD}/${f}" >> "${listfile}"
    done

    if [[ "${VCODEC}" == "gif" ]]; then
        # High-quality GIF via palette generation (single-pass filtergraph)
        # palettegen builds a per-stream optimal 256-colour palette;
        # paletteuse+bayer dithering gives clean colour gradients.
        ffmpeg -y \
            -r "${FPS}" \
            -f concat -safe 0 -i "${listfile}" \
            -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse=dither=bayer" \
            -loop 0 \
            "${outfile}" \
            -loglevel warning -stats
    else
        # -r before -i  : input framerate (1 frame = 1/FPS seconds)
        # scale filter  : round width and height to even numbers (required for yuv420p)
        ffmpeg -y \
            -r "${FPS}" \
            -f concat -safe 0 -i "${listfile}" \
            -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
            -c:v "${VCODEC}" -pix_fmt yuv420p \
            "${QUALITY_FLAGS[@]}" \
            "${outfile}" \
            -loglevel warning -stats
    fi

    rm -f "${listfile}"
    echo ""
done

echo "Movies written to ${MOVIEDIR}/"

echo "Done."

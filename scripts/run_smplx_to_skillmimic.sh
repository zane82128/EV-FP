#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PIPELINE_SCRIPT="${REPO_ROOT}/skillmimic/utils/smplx_to_skillmimic_pipeline.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_smplx_to_skillmimic.sh \
    --smpl data/archive/<session_name>/subject-1.smpl \
    --output data/converted/<session_name>_<skill_name> \
    [--results data/archive/<session_name>/results.pkl] \
    [--segments data/annotations/<session_name>/subject-1_<skill_name>_segments.json] \
    [--skill-name serve] \
    [--subject subject-1]

Current scope:
  This wrapper is for the current Phase 1 / human-only workflow.
  It always calls the Python pipeline with:
    --coord-transform y_up_to_z_up
    --phase1-human-only

Recommended use:
  Use this script when you want more control than
  convert_smplx_session_to_skillmimic.sh provides.
  For example:
    - you already have a segments.json
    - you want to pass an explicit .smpl path
    - you want to override model-path / asset-xml

Common examples:
  1) Standard case: .smpl + results.pkl + segments.json
     bash scripts/run_smplx_to_skillmimic.sh \
       --smpl data/archive/match4_001/subject-1.smpl \
       --results data/archive/match4_001/results.pkl \
       --segments data/annotations/match4_001/subject-1_forehand_segments.json \
       --output data/converted/match4_001_forehand \
       --skill-name forehand \
       --subject subject-1

  2) Explicit segments.json
     bash scripts/run_smplx_to_skillmimic.sh \
       --smpl data/archive/serve_01/subject-1.smpl \
       --results data/archive/serve_01/results.pkl \
       --segments data/annotations/serve_01/subject-1_serve_segments.json \
       --output data/converted/serve_01_serve \
       --subject subject-1 \
       --skill-name serve

Main options:
  --smpl PATH                  Path to subject-*.smpl
  --output PATH                Output bundle root
  --results PATH               Optional explicit results.pkl
  --segments PATH              Optional explicit segments JSON
  --skill-name NAME            Auto full-clip label: serve / forehand / backhand
  --skill-id INT               Optional skill id override for auto full-clip mode
  --subject NAME               Optional subject name, default = .smpl stem
  --track-id INT               Optional explicit track id for results.pkl
  --gender NAME                Optional SMPL-X gender override
  --model-path PATH            Optional SMPL-X model root, default = <repo>/models
  --motion-output-dir PATH     Optional final .pt output dir
  --full-clip-skill-id INT     Required with full-clip mode
  --full-clip-skill-name NAME  Required with full-clip mode
  --asset-xml PATH             Optional target asset xml, default = mjcf/mocap_humanoid.xml
  --dummy-obj-pos X Y Z        Dummy object position, default = 2.0 0.0 1.0
  --inclusive-end              Treat end_frame in segments JSON as inclusive
  --skip-validate              Skip final validate_motion_pt.py
  --strict-warn                Treat validator WARN as non-zero exit
  --help                       Show this message

Notes:
  - Recommended default:
      <repo>/models/smplx/SMPLX_NEUTRAL.npz
  - If --results is omitted, this script tries to auto-find results.pkl near the .smpl path.
  - If --segments is omitted, pass --skill-name and the script will auto-use start_frame=0, end_frame=<clip_length>.
  - For advanced options not covered here, call:
      python skillmimic/utils/smplx_to_skillmimic_pipeline.py --help
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

resolve_results_pkl() {
  local smpl_path="$1"
  local smpl_dir parent grandparent
  smpl_dir=$(cd "$(dirname "${smpl_path}")" && pwd)
  parent=$(dirname "${smpl_dir}")
  grandparent=$(dirname "${parent}")

  local candidate
  for candidate in \
    "${smpl_dir}/results.pkl" \
    "${smpl_dir}/world/results.pkl" \
    "${parent}/results.pkl" \
    "${parent}/world/results.pkl" \
    "${grandparent}/results.pkl" \
    "${grandparent}/world/results.pkl"
  do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  local -a found=()
  mapfile -t found < <(
    find "${smpl_dir}" "${parent}" "${grandparent}" \
      -maxdepth 3 -type f -name 'results.pkl' 2>/dev/null | sort -u
  )

  if [[ ${#found[@]} -eq 1 ]]; then
    echo "${found[0]}"
    return 0
  fi

  if [[ ${#found[@]} -gt 1 ]]; then
    echo "error: multiple results.pkl candidates found near ${smpl_path}" >&2
    printf '  %s\n' "${found[@]}" >&2
    echo "Pass --results explicitly." >&2
    exit 1
  fi

  return 1
}

SMPL_PATH=""
RESULTS_PKL=""
SEGMENTS_JSON=""
SKILL_NAME=""
SKILL_ID=""
OUTPUT_ROOT=""
SUBJECT=""
TRACK_ID=""
GENDER=""
MODEL_PATH="${REPO_ROOT}/models"
MOTION_OUTPUT_DIR=""
FULL_CLIP_SKILL_ID=""
FULL_CLIP_SKILL_NAME=""
ASSET_XML="mjcf/mocap_humanoid.xml"
DUMMY_OBJ_POS_X="2.0"
DUMMY_OBJ_POS_Y="0.0"
DUMMY_OBJ_POS_Z="1.0"
INCLUSIVE_END=0
SKIP_VALIDATE=0
STRICT_WARN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smpl)
      SMPL_PATH="${2:-}"; shift 2 ;;
    --results)
      RESULTS_PKL="${2:-}"; shift 2 ;;
    --segments)
      SEGMENTS_JSON="${2:-}"; shift 2 ;;
    --skill-name)
      SKILL_NAME="${2:-}"; shift 2 ;;
    --skill-id)
      SKILL_ID="${2:-}"; shift 2 ;;
    --output)
      OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --subject)
      SUBJECT="${2:-}"; shift 2 ;;
    --track-id)
      TRACK_ID="${2:-}"; shift 2 ;;
    --gender)
      GENDER="${2:-}"; shift 2 ;;
    --model-path)
      MODEL_PATH="${2:-}"; shift 2 ;;
    --motion-output-dir)
      MOTION_OUTPUT_DIR="${2:-}"; shift 2 ;;
    --full-clip-skill-id)
      FULL_CLIP_SKILL_ID="${2:-}"; shift 2 ;;
    --full-clip-skill-name)
      FULL_CLIP_SKILL_NAME="${2:-}"; shift 2 ;;
    --asset-xml)
      ASSET_XML="${2:-}"; shift 2 ;;
    --dummy-obj-pos)
      DUMMY_OBJ_POS_X="${2:-}"
      DUMMY_OBJ_POS_Y="${3:-}"
      DUMMY_OBJ_POS_Z="${4:-}"
      shift 4 ;;
    --inclusive-end)
      INCLUSIVE_END=1; shift ;;
    --skip-validate)
      SKIP_VALIDATE=1; shift ;;
    --strict-warn)
      STRICT_WARN=1; shift ;;
    --help|-h)
      usage
      exit 0 ;;
    *)
      die "unknown argument: $1" ;;
  esac
done

[[ -n "${SMPL_PATH}" ]] || { usage; die "--smpl is required"; }
[[ -n "${OUTPUT_ROOT}" ]] || { usage; die "--output is required"; }
[[ -f "${SMPL_PATH}" ]] || die ".smpl not found: ${SMPL_PATH}"
[[ -f "${PIPELINE_SCRIPT}" ]] || die "pipeline script not found: ${PIPELINE_SCRIPT}"

if [[ -z "${SUBJECT}" ]]; then
  SUBJECT=$(basename "${SMPL_PATH}" .smpl)
fi

if [[ -z "${RESULTS_PKL}" ]]; then
  if ! RESULTS_PKL=$(resolve_results_pkl "${SMPL_PATH}"); then
    die "could not auto-find results.pkl near ${SMPL_PATH}; pass --results explicitly"
  fi
fi

[[ -f "${RESULTS_PKL}" ]] || die "results.pkl not found: ${RESULTS_PKL}"

if [[ -n "${SEGMENTS_JSON}" ]]; then
  [[ -f "${SEGMENTS_JSON}" ]] || die "segments JSON not found: ${SEGMENTS_JSON}"
else
  if [[ -n "${SKILL_NAME}" ]]; then
    FULL_CLIP_SKILL_NAME="${SKILL_NAME}"
    if [[ -z "${SKILL_ID}" ]]; then
      case "${SKILL_NAME}" in
        serve) SKILL_ID="1" ;;
        forehand) SKILL_ID="2" ;;
        backhand) SKILL_ID="3" ;;
        *) die "unknown --skill-name '${SKILL_NAME}'; use serve, forehand, backhand, or pass --skill-id" ;;
      esac
    fi
    FULL_CLIP_SKILL_ID="${SKILL_ID}"
  fi
  [[ -n "${FULL_CLIP_SKILL_ID}" ]] || die "missing --segments or --skill-name"
  [[ -n "${FULL_CLIP_SKILL_NAME}" ]] || die "missing --segments or --skill-name"
fi

[[ -d "${MODEL_PATH}" ]] || die "SMPL-X model path not found: ${MODEL_PATH}"
if [[ ! -f "${MODEL_PATH}/smplx/SMPLX_NEUTRAL.npz" && ! -f "${MODEL_PATH}/smplx/SMPLX_NEUTRAL.pkl" ]]; then
  die "expected ${MODEL_PATH}/smplx/SMPLX_NEUTRAL.npz or .pkl"
fi

CMD=(
  "${PYTHON_BIN}" "${PIPELINE_SCRIPT}"
  --smpl-path "${SMPL_PATH}"
  --results-pkl "${RESULTS_PKL}"
  --output-root "${OUTPUT_ROOT}"
  --subject "${SUBJECT}"
  --model-path "${MODEL_PATH}"
  --coord-transform "y_up_to_z_up"
  --asset-xml "${ASSET_XML}"
  --dummy-obj-pos "${DUMMY_OBJ_POS_X}" "${DUMMY_OBJ_POS_Y}" "${DUMMY_OBJ_POS_Z}"
  --phase1-human-only
)

if [[ -n "${SEGMENTS_JSON}" ]]; then
  CMD+=(--segments-json "${SEGMENTS_JSON}")
else
  CMD+=(--full-clip-skill-id "${FULL_CLIP_SKILL_ID}")
  CMD+=(--full-clip-skill-name "${FULL_CLIP_SKILL_NAME}")
fi

if [[ -n "${TRACK_ID}" ]]; then
  CMD+=(--track-id "${TRACK_ID}")
fi

if [[ -n "${GENDER}" ]]; then
  CMD+=(--gender "${GENDER}")
fi

if [[ -n "${MOTION_OUTPUT_DIR}" ]]; then
  CMD+=(--motion-output-dir "${MOTION_OUTPUT_DIR}")
fi

if [[ ${INCLUSIVE_END} -eq 1 ]]; then
  CMD+=(--inclusive-end)
fi

if [[ ${SKIP_VALIDATE} -eq 1 ]]; then
  CMD+=(--skip-validate)
fi

if [[ ${STRICT_WARN} -eq 1 ]]; then
  CMD+=(--strict-warn)
fi

echo "=== SMPL-X -> SkillMimic wrapper ==="
echo "repo_root: ${REPO_ROOT}"
echo "smpl_path: ${SMPL_PATH}"
echo "results_pkl: ${RESULTS_PKL}"
if [[ -n "${SEGMENTS_JSON}" ]]; then
  echo "segments_json: ${SEGMENTS_JSON}"
else
  echo "segments_json: <full-clip mode>"
  echo "full_clip_skill_id: ${FULL_CLIP_SKILL_ID}"
  echo "full_clip_skill_name: ${FULL_CLIP_SKILL_NAME}"
fi
echo "subject: ${SUBJECT}"
echo "output_root: ${OUTPUT_ROOT}"
echo "model_path: ${MODEL_PATH}"
echo
printf 'running:'
printf ' %q' "${CMD[@]}"
printf '\n\n'

"${CMD[@]}"

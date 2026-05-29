#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
GENERATE_SEGMENTS_SCRIPT="${REPO_ROOT}/skillmimic/utils/generate_full_clip_segments.py"
RUN_PIPELINE_SCRIPT="${REPO_ROOT}/scripts/run_smplx_to_skillmimic.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/convert_smplx_session_to_skillmimic.sh \
    --session-dir data/archive/<session_name> \
    --skill-name serve \
    [--skill-id 1] \
    --output data/converted/<session_name>_serve \
    [--subject subject-1] \
    [--model-path models]

Expected session-dir contents:
  data/archive/<session_name>/
    subject-1.smpl
    results.pkl

Recommended use:
  This is the main entry script for teammates.
  If you only have:
    - subject-1.smpl
    - results.pkl
  then use this script first.

This script does two things automatically:
  1. Generate a full-clip segments.json
  2. Run the full SMPL-X -> SkillMimic pipeline

Supported skill names:
  serve
  forehand
  backhand

Custom labels:
  If --skill-name is not in the default set above, pass --skill-id explicitly.

Example:
  bash scripts/convert_smplx_session_to_skillmimic.sh \
    --session-dir data/archive/match4_001 \
    --skill-name serve \
    --output data/converted/match4_001_serve
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

SESSION_DIR=""
SKILL_NAME=""
SKILL_ID=""
OUTPUT_ROOT=""
SUBJECT="subject-1"
MODEL_PATH="${REPO_ROOT}/models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-dir)
      SESSION_DIR="${2:-}"; shift 2 ;;
    --skill-name)
      SKILL_NAME="${2:-}"; shift 2 ;;
    --skill-id)
      SKILL_ID="${2:-}"; shift 2 ;;
    --output)
      OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --subject)
      SUBJECT="${2:-}"; shift 2 ;;
    --model-path)
      MODEL_PATH="${2:-}"; shift 2 ;;
    --help|-h)
      usage
      exit 0 ;;
    *)
      die "unknown argument: $1" ;;
  esac
done

[[ -n "${SESSION_DIR}" ]] || { usage; die "--session-dir is required"; }
[[ -n "${SKILL_NAME}" ]] || { usage; die "--skill-name is required"; }
[[ -n "${OUTPUT_ROOT}" ]] || { usage; die "--output is required"; }

SESSION_DIR=$(cd "${SESSION_DIR}" && pwd)
OUTPUT_ROOT=$(mkdir -p "${OUTPUT_ROOT}" && cd "${OUTPUT_ROOT}" && pwd)

SMPL_PATH="${SESSION_DIR}/${SUBJECT}.smpl"
RESULTS_PKL="${SESSION_DIR}/results.pkl"
SEGMENTS_DIR="${OUTPUT_ROOT}/generated_segments"
SEGMENTS_PATH="${SEGMENTS_DIR}/${SUBJECT}_${SKILL_NAME}_segments.json"

[[ -f "${SMPL_PATH}" ]] || die ".smpl not found: ${SMPL_PATH}"
[[ -f "${RESULTS_PKL}" ]] || die "results.pkl not found: ${RESULTS_PKL}"
[[ -d "${MODEL_PATH}" ]] || die "SMPL-X model path not found: ${MODEL_PATH}"
if [[ ! -f "${MODEL_PATH}/smplx/SMPLX_NEUTRAL.npz" && ! -f "${MODEL_PATH}/smplx/SMPLX_NEUTRAL.pkl" ]]; then
  die "expected ${MODEL_PATH}/smplx/SMPLX_NEUTRAL.npz or .pkl"
fi

mkdir -p "${SEGMENTS_DIR}"

echo "=== Step 1: generate full-clip segments.json ==="
GENERATE_CMD=(
  "${PYTHON_BIN}" "${GENERATE_SEGMENTS_SCRIPT}"
  "${SMPL_PATH}" \
  --skill-name "${SKILL_NAME}"
  --subject "${SUBJECT}"
  --output "${SEGMENTS_PATH}"
)

if [[ -n "${SKILL_ID}" ]]; then
  GENERATE_CMD+=(--skill-id "${SKILL_ID}")
fi

"${GENERATE_CMD[@]}"

echo
echo "=== Step 2: run SMPL-X -> SkillMimic pipeline ==="
"${RUN_PIPELINE_SCRIPT}" \
  --smpl "${SMPL_PATH}" \
  --results "${RESULTS_PKL}" \
  --segments "${SEGMENTS_PATH}" \
  --output "${OUTPUT_ROOT}" \
  --subject "${SUBJECT}" \
  --model-path "${MODEL_PATH}"

echo
echo "=== done ==="
echo "segments_json: ${SEGMENTS_PATH}"
echo "output_root: ${OUTPUT_ROOT}"
echo "final_motions: ${OUTPUT_ROOT}/motions"

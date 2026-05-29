#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
CONVERT_SCRIPT="${REPO_ROOT}/scripts/convert_smplx_session_to_skillmimic.sh"
DEFAULT_INPUT_ROOT="${REPO_ROOT}/skillmimic/data/motions/TableTennis_splits_ori"
DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/skillmimic/data/motions/TableTennis_splits_converted"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/convert_table_tennis_splits_to_skillmimic.sh \
    [--input-root skillmimic/data/motions/TableTennis_splits_ori] \
    [--output-root skillmimic/data/motions/TableTennis_splits_converted] \
    [--model-path models] \
    [--skill-map non_hit:4] \
    [--only-skill forehand] \
    [--skip-existing] \
    [--dry-run] \
    [--fail-fast]

Expected input layout:
  <input-root>/<skill_name>/<sample_name>/
    subject-*.smpl
    results.pkl

Default skill id mapping:
  serve=1
  forehand=2
  backhand=3
  non_hit=4
  uncertain=5

Notes:
  - The script mirrors the input tree under --output-root.
  - Each sample output folder contains generated_segments/, clips/, source_joints/, canonical/, motions/, and manifest.json.
  - Use --skill-map LABEL:ID to override or add mappings.
  - Use --dry-run to preview the per-sample commands without running conversion.

Examples:
  bash scripts/convert_table_tennis_splits_to_skillmimic.sh

  bash scripts/convert_table_tennis_splits_to_skillmimic.sh \
    --only-skill forehand \
    --output-root skillmimic/data/motions/TableTennis_splits_converted_forehand

  bash scripts/convert_table_tennis_splits_to_skillmimic.sh \
    --skill-map non_hit:40 \
    --skill-map uncertain:41
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

log() {
  echo "[batch] $*"
}

warn() {
  echo "[batch][warn] $*" >&2
}

normalize_skill_name() {
  local value="$1"
  value="${value,,}"
  value="${value// /_}"
  printf '%s\n' "${value}"
}

resolve_subject() {
  local session_dir="$1"
  local -a smpl_files=()
  local smpl_path

  while IFS= read -r smpl_path; do
    smpl_files+=("${smpl_path}")
  done < <(find "${session_dir}" -maxdepth 1 -type f -name '*.smpl' | sort)

  if [[ ${#smpl_files[@]} -eq 0 ]]; then
    echo "no .smpl found in ${session_dir}" >&2
    return 1
  fi

  if [[ ${#smpl_files[@]} -gt 1 ]]; then
    echo "multiple .smpl files found in ${session_dir}" >&2
    printf '  %s\n' "${smpl_files[@]}" >&2
    return 1
  fi

  basename "${smpl_files[0]}" .smpl
}

should_process_skill() {
  local skill_name="$1"
  local only_skill

  if [[ ${#ONLY_SKILLS[@]} -eq 0 ]]; then
    return 0
  fi

  for only_skill in "${ONLY_SKILLS[@]}"; do
    if [[ "${only_skill}" == "${skill_name}" ]]; then
      return 0
    fi
  done

  return 1
}

lookup_skill_id() {
  local skill_name="$1"

  if [[ -n "${SKILL_IDS[${skill_name}]+x}" ]]; then
    printf '%s\n' "${SKILL_IDS[${skill_name}]}"
    return 0
  fi

  return 1
}

print_command() {
  printf 'running:'
  printf ' %q' "$@"
  printf '\n'
}

INPUT_ROOT="${DEFAULT_INPUT_ROOT}"
OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}"
MODEL_PATH="${REPO_ROOT}/models"
SKIP_EXISTING=0
DRY_RUN=0
FAIL_FAST=0

declare -A SKILL_IDS=(
  [serve]=1
  [forehand]=2
  [backhand]=3
  [non_hit]=4
  [uncertain]=5
)
declare -a ONLY_SKILLS=()
declare -a FAILURES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root)
      INPUT_ROOT="${2:-}"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --model-path)
      MODEL_PATH="${2:-}"; shift 2 ;;
    --skill-map)
      MAP_ENTRY="${2:-}"
      [[ "${MAP_ENTRY}" == *:* ]] || die "--skill-map must be LABEL:ID"
      SKILL_LABEL=$(normalize_skill_name "${MAP_ENTRY%%:*}")
      SKILL_ID_OVERRIDE="${MAP_ENTRY##*:}"
      [[ "${SKILL_ID_OVERRIDE}" =~ ^[0-9]+$ ]] || die "skill id must be an integer: ${MAP_ENTRY}"
      SKILL_IDS["${SKILL_LABEL}"]="${SKILL_ID_OVERRIDE}"
      shift 2 ;;
    --only-skill)
      ONLY_SKILLS+=("$(normalize_skill_name "${2:-}")")
      shift 2 ;;
    --skip-existing)
      SKIP_EXISTING=1; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --fail-fast)
      FAIL_FAST=1; shift ;;
    --help|-h)
      usage
      exit 0 ;;
    *)
      die "unknown argument: $1" ;;
  esac
done

[[ -f "${CONVERT_SCRIPT}" ]] || die "convert script not found: ${CONVERT_SCRIPT}"
[[ -d "${INPUT_ROOT}" ]] || die "input root not found: ${INPUT_ROOT}"
[[ -d "${MODEL_PATH}" ]] || die "model path not found: ${MODEL_PATH}"

INPUT_ROOT=$(cd "${INPUT_ROOT}" && pwd)
OUTPUT_ROOT=$(mkdir -p "${OUTPUT_ROOT}" && cd "${OUTPUT_ROOT}" && pwd)

shopt -s nullglob

total_samples=0
converted_samples=0
planned_samples=0
skipped_samples=0
failed_samples=0

for skill_dir in "${INPUT_ROOT}"/*; do
  [[ -d "${skill_dir}" ]] || continue

  skill_name=$(normalize_skill_name "$(basename "${skill_dir}")")
  if ! should_process_skill "${skill_name}"; then
    continue
  fi

  if ! skill_id=$(lookup_skill_id "${skill_name}"); then
    die "missing skill id mapping for '${skill_name}'. Pass --skill-map ${skill_name}:<id>"
  fi

  sample_dirs=("${skill_dir}"/*)
  if [[ ${#sample_dirs[@]} -eq 0 ]]; then
    warn "no sample directories found under ${skill_dir}"
    continue
  fi

  for sample_dir in "${sample_dirs[@]}"; do
    [[ -d "${sample_dir}" ]] || continue
    total_samples=$((total_samples + 1))

    sample_name=$(basename "${sample_dir}")
    output_dir="${OUTPUT_ROOT}/${skill_name}/${sample_name}"

    if [[ ${SKIP_EXISTING} -eq 1 && -f "${output_dir}/manifest.json" ]]; then
      log "skip existing sample: ${skill_name}/${sample_name}"
      skipped_samples=$((skipped_samples + 1))
      continue
    fi

    if ! subject=$(resolve_subject "${sample_dir}"); then
      warn "failed to infer subject for ${skill_name}/${sample_name}"
      FAILURES+=("${skill_name}/${sample_name}: subject detection failed")
      failed_samples=$((failed_samples + 1))
      if [[ ${FAIL_FAST} -eq 1 ]]; then
        break 2
      fi
      continue
    fi

    cmd=(
      bash "${CONVERT_SCRIPT}"
      --session-dir "${sample_dir}"
      --skill-name "${skill_name}"
      --skill-id "${skill_id}"
      --output "${output_dir}"
      --subject "${subject}"
      --model-path "${MODEL_PATH}"
    )

    if [[ ${DRY_RUN} -eq 1 ]]; then
      print_command "${cmd[@]}"
      planned_samples=$((planned_samples + 1))
      continue
    fi

    log "converting ${skill_name}/${sample_name} -> ${output_dir}"
    if "${cmd[@]}"; then
      converted_samples=$((converted_samples + 1))
    else
      warn "conversion failed for ${skill_name}/${sample_name}"
      FAILURES+=("${skill_name}/${sample_name}")
      failed_samples=$((failed_samples + 1))
      if [[ ${FAIL_FAST} -eq 1 ]]; then
        break 2
      fi
    fi
  done
done

echo
echo "=== batch summary ==="
echo "input_root: ${INPUT_ROOT}"
echo "output_root: ${OUTPUT_ROOT}"
echo "total_samples: ${total_samples}"
if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "planned_samples: ${planned_samples}"
else
  echo "converted_samples: ${converted_samples}"
fi
echo "skipped_samples: ${skipped_samples}"
echo "failed_samples: ${failed_samples}"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo "failed_items:" >&2
  printf '  %s\n' "${FAILURES[@]}" >&2
  exit 1
fi

if [[ ${total_samples} -eq 0 ]]; then
  die "no sample directories found under ${INPUT_ROOT}"
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "dry_run_only: true"
fi
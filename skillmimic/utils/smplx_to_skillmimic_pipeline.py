import argparse
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utils_dir() -> Path:
    return repo_root() / "skillmimic" / "utils"


def asset_root() -> Path:
    return repo_root() / "skillmimic" / "data" / "assets"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end SMPL-X -> SkillMimic conversion pipeline for the current "
            "human-only / Phase 1 workflow. "
            "Wraps clip extraction, source_joints decode, canonical retarget, "
            "final .pt packing, and optional validation."
        )
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--archive-zip",
        type=str,
        default="",
        help=(
            "Path to an archive like match4_001_smplx.zip or tabletennis_serve.zip. "
            "The script will extract it and auto-locate subject-*.smpl plus results.pkl."
        ),
    )
    input_group.add_argument(
        "--smpl-path",
        type=str,
        default="",
        help="Path to a .smpl archive. Use this when starting from raw SMPL-X output.",
    )
    input_group.add_argument(
        "--clips-input",
        type=str,
        default="",
        help="Path to an existing clip .npz or a directory of clip .npz files.",
    )

    parser.add_argument(
        "--results-pkl",
        type=str,
        default="",
        help=(
            "Path to results.pkl containing people[track_id].smplx_world. "
            "Required unless --archive-zip is used."
        ),
    )
    parser.add_argument(
        "--segments-json",
        type=str,
        default="",
        help="Segment annotation JSON. Required when --smpl-path is used unless full-clip mode is enabled.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="",
        help="Subject name override, e.g. subject-1. Defaults to the .smpl stem.",
    )
    parser.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="Override track id in results.pkl. Default infers from subject, e.g. subject-1 -> 1.",
    )
    parser.add_argument(
        "--gender",
        type=str,
        choices=["neutral", "male", "female"],
        default=None,
        help="Optional SMPL-X model gender override.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help=(
            "Optional SMPL-X model root. "
            "If omitted, results_pkl_to_source_joints.py searches <repo>/models and <cwd>/models."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Root directory for all generated artifacts.",
    )
    parser.add_argument(
        "--motion-output-dir",
        type=str,
        default="",
        help=(
            "Optional final .pt output directory. "
            "If omitted, final .pt files are written to <output-root>/motions."
        ),
    )
    parser.add_argument(
        "--mapping-json",
        type=str,
        default="",
        help="Optional source->target mapping override for smpl_to_canonical.py.",
    )
    parser.add_argument(
        "--asset-xml",
        type=str,
        default="mjcf/mocap_humanoid.xml",
        help=(
            "Target asset XML for target skeleton parsing. "
            "Can be an absolute path or a path relative to skillmimic/data/assets/."
        ),
    )
    parser.add_argument(
        "--coord-transform",
        type=str,
        choices=["none", "y_up_to_z_up"],
        default="y_up_to_z_up",
        help="Coordinate transform applied before packing canonical clips.",
    )
    parser.add_argument(
        "--root-joint-index",
        type=int,
        default=0,
        help="Fallback root joint index for smpl_to_canonical.py.",
    )
    parser.add_argument(
        "--dummy-obj-pos",
        type=float,
        nargs=3,
        default=[2.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Dummy object position used when packing human-only motion into BallPlay format.",
    )
    parser.add_argument(
        "--phase1-human-only",
        action="store_true",
        help=(
            "Mark this run as Phase 1 human-only. "
            "Enables human-only validation expectations (dummy object + zero contact)."
        ),
    )
    parser.add_argument(
        "--allow-missing-source-joints",
        action="store_true",
        help="Allow smpl_to_canonical.py fallback without source_joints. Only recommended for smoke tests.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip validate_motion_pt.py on final .pt files.",
    )
    parser.add_argument(
        "--strict-warn",
        action="store_true",
        help="Treat validator WARN as non-zero exit.",
    )
    parser.add_argument(
        "--inclusive-end",
        action="store_true",
        help="Treat end_frame in segments JSON as inclusive during clip extraction.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="FPS passed to validate_motion_pt.py.",
    )
    parser.add_argument(
        "--termination-height",
        type=float,
        default=0.25,
        help="terminationHeight passed to validate_motion_pt.py.",
    )
    parser.add_argument(
        "--full-clip-skill-id",
        type=int,
        default=None,
        help="If set, auto-generate a one-segment JSON covering the whole .smpl clip.",
    )
    parser.add_argument(
        "--full-clip-skill-name",
        type=str,
        default="",
        help="Skill name used with --full-clip-skill-id in full-clip mode.",
    )
    parser.add_argument(
        "--manifest-json",
        type=str,
        default="",
        help="Optional path to save a machine-readable JSON manifest of outputs.",
    )

    args = parser.parse_args()

    if not args.archive_zip and not args.results_pkl:
        parser.error("--results-pkl is required unless --archive-zip is used.")

    if (args.smpl_path or args.archive_zip) and not args.segments_json:
        if args.full_clip_skill_id is None or not args.full_clip_skill_name:
            parser.error(
                "--smpl-path/--archive-zip requires either --segments-json or both "
                "--full-clip-skill-id and --full-clip-skill-name."
            )

    return args


def resolve_existing_path(path_str: str, kind: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")
    return path.resolve()


def resolve_optional_existing_path(path_str: str, kind: str) -> Optional[Path]:
    if not path_str:
        return None
    return resolve_existing_path(path_str, kind)


def resolve_asset_xml(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path.resolve()

    alt = asset_root() / path
    if alt.exists():
        return alt.resolve()

    raise FileNotFoundError(
        f"Asset XML not found: {path_str}. "
        f"Tried {path} and {alt}"
    )


def extract_archive_zip(archive_zip: Path, output_root: Path) -> Path:
    extract_root = output_root / "archive_extract" / archive_zip.stem
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_zip, "r") as archive:
        archive.extractall(extract_root)
    return extract_root


def resolve_archive_contents(
    archive_zip: Path,
    output_root: Path,
    subject: str,
) -> Tuple[Path, Path, Path]:
    extract_root = extract_archive_zip(archive_zip, output_root)

    smpl_files = sorted(extract_root.rglob("*.smpl"))
    if not smpl_files:
        raise FileNotFoundError(f"No .smpl files found after extracting {archive_zip}")

    if subject:
        target_name = f"{subject}.smpl"
        matched = [p for p in smpl_files if p.name == target_name]
        if not matched:
            raise FileNotFoundError(
                f"Could not find {target_name} inside {archive_zip}. "
                f"Available .smpl files: {[p.name for p in smpl_files]}"
            )
        smpl_path = matched[0]
    else:
        if len(smpl_files) != 1:
            raise ValueError(
                f"{archive_zip} contains multiple .smpl files. "
                "Pass --subject to choose one explicitly."
            )
        smpl_path = smpl_files[0]

    results_candidates = sorted(extract_root.rglob("results.pkl"))
    if len(results_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one results.pkl in {archive_zip}, found {len(results_candidates)}"
        )
    results_pkl = results_candidates[0]

    return extract_root, smpl_path, results_pkl


def load_smpl_arrays(smpl_path: Path) -> dict:
    import numpy as np

    arrays = {}
    with zipfile.ZipFile(smpl_path, "r") as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".npy"):
                continue
            with archive.open(name, "r") as handle:
                arrays[name] = np.load(io.BytesIO(handle.read()), allow_pickle=True)
    return arrays


def build_full_clip_segments_json(
    smpl_path: Path,
    output_root: Path,
    skill_id: int,
    skill_name: str,
) -> Path:
    arrays = load_smpl_arrays(smpl_path)
    if "bodyPose.npy" not in arrays:
        raise KeyError(f"{smpl_path} does not contain bodyPose.npy")

    total_frames = int(arrays["bodyPose.npy"].shape[0])
    tmp_dir = output_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / "full_clip_segments.json"
    payload = [
        {
            "skill_id": int(skill_id),
            "skill_name": str(skill_name),
            "start_frame": 0,
            "end_frame": total_frames,
        }
    ]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def iter_files(path: Path, suffix: str):
    if path.is_file():
        return [path] if path.suffix == suffix else []
    return sorted(path.rglob(f"*{suffix}"))


def run_step(name: str, cmd: List[str]):
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def validate_pt_files(
    pt_files: List[Path],
    fps: float,
    termination_height: float,
    expect_dummy_object: bool,
    strict_warn: bool,
):
    validator = utils_dir() / "validate_motion_pt.py"
    for pt_path in pt_files:
        cmd = [
            sys.executable,
            str(validator),
            str(pt_path),
            "--fps",
            str(fps),
            "--termination-height",
            str(termination_height),
        ]
        if expect_dummy_object:
            cmd.append("--expect-dummy-object")
        if strict_warn:
            cmd.append("--strict-warn")
        run_step(f"validate {pt_path.name}", cmd)


def write_manifest(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    root = repo_root()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    archive_zip = resolve_optional_existing_path(args.archive_zip, "archive zip")
    smpl_path = resolve_optional_existing_path(args.smpl_path, ".smpl archive")
    clips_input = resolve_optional_existing_path(args.clips_input, "clips input")
    results_pkl = resolve_optional_existing_path(args.results_pkl, "results.pkl")
    segments_json = resolve_optional_existing_path(args.segments_json, "segments JSON")
    model_path = resolve_optional_existing_path(args.model_path, "SMPL-X model path")
    mapping_json = resolve_optional_existing_path(args.mapping_json, "mapping JSON")
    asset_xml = resolve_asset_xml(args.asset_xml)

    archive_extract_root = None
    if archive_zip is not None:
        archive_extract_root, smpl_path, results_pkl = resolve_archive_contents(
            archive_zip=archive_zip,
            output_root=output_root,
            subject=args.subject,
        )
    elif results_pkl is None:
        raise FileNotFoundError("--results-pkl is required when not using --archive-zip")

    if smpl_path is not None and segments_json is None:
        segments_json = build_full_clip_segments_json(
            smpl_path=smpl_path,
            output_root=output_root,
            skill_id=args.full_clip_skill_id,
            skill_name=args.full_clip_skill_name,
        )

    clips_dir = output_root / "clips"
    source_joints_dir = output_root / "source_joints"
    canonical_dir = output_root / "canonical"
    motions_dir = (
        Path(args.motion_output_dir).resolve()
        if args.motion_output_dir
        else output_root / "motions"
    )

    if smpl_path is not None:
        extract_script = utils_dir() / "extract_smpl_clip.py"
        cmd = [
            sys.executable,
            str(extract_script),
            str(smpl_path),
            "--segments-json",
            str(segments_json),
            "--output-dir",
            str(clips_dir),
        ]
        if args.subject:
            cmd.extend(["--subject", args.subject])
        if args.inclusive_end:
            cmd.append("--inclusive-end")
        run_step("extract clips", cmd)
        clip_input_path = clips_dir
    else:
        clip_input_path = clips_input

    joints_script = utils_dir() / "results_pkl_to_source_joints.py"
    cmd = [
        sys.executable,
        str(joints_script),
        str(clip_input_path),
        "--results-pkl",
        str(results_pkl),
        "--output-dir",
        str(source_joints_dir),
    ]
    if model_path is not None:
        cmd.extend(["--model-path", str(model_path)])
    if args.gender is not None:
        cmd.extend(["--gender", args.gender])
    if args.track_id is not None:
        cmd.extend(["--track-id", str(args.track_id)])
    run_step("decode source joints", cmd)

    canonical_script = utils_dir() / "smpl_to_canonical.py"
    cmd = [
        sys.executable,
        str(canonical_script),
        str(clip_input_path),
        "--output-dir",
        str(canonical_dir),
        "--source-joints-dir",
        str(source_joints_dir),
        "--coord-transform",
        args.coord_transform,
        "--asset-xml",
        str(asset_xml),
        "--root-joint-index",
        str(args.root_joint_index),
    ]
    if mapping_json is not None:
        cmd.extend(["--mapping-json", str(mapping_json)])
    if args.allow_missing_source_joints:
        cmd.append("--allow-missing-source-joints")
    run_step("build canonical clips", cmd)

    pack_script = utils_dir() / "canonical_to_skillmimic_pt.py"
    cmd = [
        sys.executable,
        str(pack_script),
        str(canonical_dir),
        "--output-dir",
        str(motions_dir),
        "--dummy-obj-pos",
        str(args.dummy_obj_pos[0]),
        str(args.dummy_obj_pos[1]),
        str(args.dummy_obj_pos[2]),
    ]
    run_step("pack final .pt", cmd)

    pt_files = iter_files(motions_dir, ".pt")
    if not args.skip_validate:
        validate_pt_files(
            pt_files=pt_files,
            fps=args.fps,
            termination_height=args.termination_height,
            expect_dummy_object=args.phase1_human_only,
            strict_warn=args.strict_warn,
        )

    manifest = {
        "repo_root": str(root),
        "inputs": {
            "archive_zip": str(archive_zip) if archive_zip is not None else None,
            "archive_extract_root": str(archive_extract_root) if archive_extract_root is not None else None,
            "smpl_path": str(smpl_path) if smpl_path is not None else None,
            "clips_input": str(clips_input) if clips_input is not None else None,
            "results_pkl": str(results_pkl),
            "segments_json": str(segments_json) if segments_json is not None else None,
            "subject": args.subject or None,
            "track_id": args.track_id,
            "gender": args.gender,
            "model_path": str(model_path) if model_path is not None else None,
        },
        "settings": {
            "coord_transform": args.coord_transform,
            "asset_xml": str(asset_xml),
            "dummy_obj_pos": args.dummy_obj_pos,
            "phase1_human_only": args.phase1_human_only,
            "allow_missing_source_joints": args.allow_missing_source_joints,
            "fps": args.fps,
            "termination_height": args.termination_height,
        },
        "outputs": {
            "output_root": str(output_root),
            "clips_dir": str(clips_dir),
            "source_joints_dir": str(source_joints_dir),
            "canonical_dir": str(canonical_dir),
            "motions_dir": str(motions_dir),
            "clip_files": [str(p) for p in iter_files(clips_dir, ".npz")],
            "source_joint_files": [str(p) for p in iter_files(source_joints_dir, ".npy")],
            "canonical_files": [str(p) for p in iter_files(canonical_dir, ".npz")],
            "motion_files": [str(p) for p in pt_files],
        },
    }

    manifest_path = (
        Path(args.manifest_json).resolve()
        if args.manifest_json
        else output_root / "manifest.json"
    )
    write_manifest(manifest_path, manifest)

    print("\n=== pipeline complete ===")
    print(f"manifest: {manifest_path}")
    print(f"final_motion_files: {len(pt_files)}")
    for path in pt_files:
        print(path)


if __name__ == "__main__":
    main()

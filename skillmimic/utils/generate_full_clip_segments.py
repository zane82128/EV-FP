import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np


DEFAULT_SKILL_IDS = {
    "serve": 1,
    "forehand": 2,
    "backhand": 3,
}


def normalize_skill_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def resolve_skill_id(skill_name: str, override_skill_id):
    if override_skill_id is not None:
        return int(override_skill_id)

    key = normalize_skill_name(skill_name)
    if key in DEFAULT_SKILL_IDS:
        return DEFAULT_SKILL_IDS[key]

    raise ValueError(
        f"Unknown skill_name '{skill_name}'. "
        "Pass --skill-id explicitly or use one of: serve, forehand, backhand."
    )


def scalar_or_default(value, default):
    if value is None:
        return default
    array = np.asarray(value)
    return array.item() if array.shape == () else value


def load_smpl_arrays(path: Path):
    arrays = {}
    with zipfile.ZipFile(path, "r") as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".npy"):
                continue
            with archive.open(name, "r") as handle:
                arrays[name] = np.load(io.BytesIO(handle.read()), allow_pickle=True)
    return arrays


def infer_clip_length(path: Path):
    if path.suffix == ".smpl":
        arrays = load_smpl_arrays(path)
        if "bodyPose.npy" not in arrays:
            raise KeyError(f"{path} does not contain bodyPose.npy")
        return int(arrays["bodyPose.npy"].shape[0])

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "body_pose" in data:
            return int(data["body_pose"].shape[0])
        raise KeyError(f"{path} does not contain body_pose")

    raise ValueError(f"Unsupported input type: {path}")


def infer_subject(path: Path, explicit_subject: str):
    if explicit_subject:
        return explicit_subject

    if path.suffix == ".smpl":
        return path.stem

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "subject" in data:
            return str(scalar_or_default(data["subject"], path.stem))
        return path.stem

    return path.stem


def build_output_path(output_path: str, input_path: Path, skill_name: str):
    if output_path:
        return Path(output_path).resolve()

    skill_name = normalize_skill_name(skill_name)
    return input_path.with_name(f"{input_path.stem}_{skill_name}_segments.json")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a one-segment JSON for a full clip. "
            "The segment always spans start_frame=0 to end_frame=<clip_length>."
        )
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a .smpl file or an extracted clip .npz file.",
    )
    parser.add_argument(
        "--skill-name",
        type=str,
        required=True,
        help="Skill label, usually serve / forehand / backhand.",
    )
    parser.add_argument(
        "--skill-id",
        type=int,
        default=None,
        help="Optional skill id override. Default mapping: serve=1, forehand=2, backhand=3.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="",
        help="Optional subject override. Default uses .smpl stem or clip subject field.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSON path. Default writes next to the input file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    skill_name = normalize_skill_name(args.skill_name)
    skill_id = resolve_skill_id(skill_name, args.skill_id)
    subject = infer_subject(input_path, args.subject)
    clip_length = infer_clip_length(input_path)
    output_path = build_output_path(args.output, input_path, skill_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "subject": subject,
            "skill_id": skill_id,
            "skill_name": skill_name,
            "start_frame": 0,
            "end_frame": clip_length,
        }
    ]

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"saved_segments: {output_path}")
    print(f"subject: {subject}")
    print(f"skill_id: {skill_id}")
    print(f"skill_name: {skill_name}")
    print(f"start_frame: 0")
    print(f"end_frame: {clip_length}")


if __name__ == "__main__":
    main()

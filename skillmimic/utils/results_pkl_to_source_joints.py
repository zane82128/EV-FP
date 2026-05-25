import argparse
from pathlib import Path

import joblib
import numpy as np
import smplx
import torch


SOURCE_JOINT_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
]

GENDER_CODE_TO_NAME = {
    0: "neutral",
    1: "male",
    2: "female",
}


def scalar(value):
    array = np.asarray(value)
    return array.item() if array.shape == () else array


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def iter_clip_paths(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.npz"))


def normalize_subject(subject):
    subject = str(subject)
    if subject.startswith("subject-"):
        return int(subject.split("-", 1)[1])
    if subject.isdigit():
        return int(subject)
    raise ValueError(f"Cannot infer track id from subject '{subject}'")


def infer_gender_name(raw_gender, override_gender):
    if override_gender is not None:
        return override_gender

    if raw_gender is None:
        return "neutral"

    raw_gender = scalar(raw_gender)
    if isinstance(raw_gender, str):
        value = raw_gender.strip().lower()
        if value in {"neutral", "male", "female"}:
            return value
    if isinstance(raw_gender, (int, np.integer)):
        return GENDER_CODE_TO_NAME.get(int(raw_gender), "neutral")
    return "neutral"


def patch_numpy_compat():
    # 4D-Humans joblib payloads may reference numpy._core when pickled with newer NumPy.
    import sys

    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric


def load_results(path):
    patch_numpy_compat()
    return joblib.load(path)


def resolve_model_path(explicit_path):
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"SMPL-X model path does not exist: {path}")
        return path

    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    candidates = [
        repo_root / "models",
        repo_root.parent / "InterAct" / "models",
        Path.cwd() / "models",
    ]
    for candidate in candidates:
        if (candidate / "smplx" / "SMPLX_NEUTRAL.npz").exists():
            return candidate
        if (candidate / "smplx" / "SMPLX_NEUTRAL.pkl").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find SMPL-X model files. Pass --model-path pointing to a folder that contains smplx/SMPLX_NEUTRAL.npz."
    )


def build_smplx_model(model_path, gender_name, batch_size):
    return smplx.create(
        str(model_path),
        model_type="smplx",
        gender=gender_name,
        use_pca=False,
        ext="npz",
        batch_size=batch_size,
    )


def split_full_pose(full_pose):
    if full_pose.ndim != 2 or full_pose.shape[1] != 165:
        raise ValueError(f"Expected full SMPL-X pose shape [T, 165], got {full_pose.shape}")

    return {
        "global_orient": full_pose[:, :3],
        "body_pose": full_pose[:, 3:66],
        "jaw_pose": full_pose[:, 66:69],
        "leye_pose": full_pose[:, 69:72],
        "reye_pose": full_pose[:, 72:75],
        "left_hand_pose": full_pose[:, 75:120],
        "right_hand_pose": full_pose[:, 120:165],
    }


def validate_clip_against_results(clip, person, start, end):
    clip_body_pose = np.asarray(clip["body_pose"], dtype=np.float32)
    clip_body_translation = np.asarray(clip["body_translation"], dtype=np.float32)

    world_pose = np.asarray(person["smplx_world"]["pose"][start:end], dtype=np.float32)
    world_trans = np.asarray(person["smplx_world"]["trans"][start:end], dtype=np.float32)

    if clip_body_pose.shape != (end - start, 22, 3):
        raise ValueError(f"Unexpected clip body_pose shape: {clip_body_pose.shape}")
    if clip_body_translation.shape != (end - start, 3):
        raise ValueError(f"Unexpected clip body_translation shape: {clip_body_translation.shape}")
    if world_pose.shape != (end - start, 165):
        raise ValueError(f"Unexpected results smplx_world pose shape: {world_pose.shape}")
    if world_trans.shape != (end - start, 3):
        raise ValueError(f"Unexpected results smplx_world trans shape: {world_trans.shape}")

    pose_view = world_pose[:, :66].reshape(end - start, 22, 3)
    pose_diff = np.max(np.abs(pose_view - clip_body_pose))
    trans_diff = np.max(np.abs(world_trans - clip_body_translation))

    if pose_diff > 1e-4:
        raise ValueError(f"Clip body_pose does not match results.pkl pose slice, max diff={pose_diff}")
    if trans_diff > 1e-4:
        raise ValueError(
            f"Clip body_translation does not match results.pkl translation slice, max diff={trans_diff}"
        )

    return world_pose, world_trans


def decode_source_joints(model, full_pose, trans, betas, align_to_clip_root):
    pose_parts = split_full_pose(full_pose)

    with torch.no_grad():
        output = model(
            betas=torch.from_numpy(betas),
            transl=torch.from_numpy(trans),
            global_orient=torch.from_numpy(pose_parts["global_orient"]),
            body_pose=torch.from_numpy(pose_parts["body_pose"]),
            jaw_pose=torch.from_numpy(pose_parts["jaw_pose"]),
            leye_pose=torch.from_numpy(pose_parts["leye_pose"]),
            reye_pose=torch.from_numpy(pose_parts["reye_pose"]),
            left_hand_pose=torch.from_numpy(pose_parts["left_hand_pose"]),
            right_hand_pose=torch.from_numpy(pose_parts["right_hand_pose"]),
            return_verts=False,
        )

    joints = output.joints[:, : len(SOURCE_JOINT_NAMES), :].detach().cpu().numpy().astype(np.float32)

    if align_to_clip_root:
        delta = trans - joints[:, 0, :]
        joints = joints + delta[:, None, :]

    return joints


def resolve_output_path(output_dir, clip_path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{Path(clip_path).stem}.npy"


def main():
    parser = argparse.ArgumentParser(
        description="Decode per-clip SMPL-X source_joints [T, 22, 3] from a results.pkl track."
    )
    parser.add_argument("input_path", type=str, help="Path to a clip .npz or a directory of clips")
    parser.add_argument(
        "--results-pkl",
        type=str,
        required=True,
        help="Path to results.pkl that contains people[track_id].smplx_world",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save per-clip source_joints .npy files",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional SMPL-X model root. If omitted, the script searches common local paths.",
    )
    parser.add_argument(
        "--gender",
        type=str,
        choices=["neutral", "male", "female"],
        default=None,
        help="Override model gender. Default infers from the clip gender field when possible.",
    )
    parser.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="Override the results.pkl person track id. Default infers from clip subject, e.g. subject-1 -> 1.",
    )
    parser.add_argument(
        "--no-align-to-root",
        action="store_true",
        help="Do not shift decoded joints so Pelvis matches clip body_translation.",
    )
    args = parser.parse_args()

    results = load_results(args.results_pkl)
    model_path = resolve_model_path(args.model_path)
    saved = []

    model_cache = {}

    for clip_path in iter_clip_paths(args.input_path):
        clip = load_npz(clip_path)
        start = int(scalar(clip["start_frame"]))
        end = int(scalar(clip["end_frame"]))
        subject = scalar(clip["subject"])
        track_id = args.track_id if args.track_id is not None else normalize_subject(subject)

        if "people" not in results or track_id not in results["people"]:
            raise KeyError(f"Track id {track_id} not found in {args.results_pkl}")

        person = results["people"][track_id]
        full_pose, trans = validate_clip_against_results(clip, person, start, end)
        betas_full = np.asarray(person["smplx_world"]["shape"][start:end], dtype=np.float32)
        if betas_full.shape != (end - start, 10):
            raise ValueError(f"Unexpected results smplx_world shape shape: {betas_full.shape}")

        gender_name = infer_gender_name(clip.get("gender"), args.gender)
        cache_key = (str(model_path), gender_name, end - start)
        if cache_key not in model_cache:
            model_cache[cache_key] = build_smplx_model(model_path, gender_name, end - start)
        model = model_cache[cache_key]

        joints = decode_source_joints(
            model=model,
            full_pose=full_pose,
            trans=trans,
            betas=betas_full,
            align_to_clip_root=not args.no_align_to_root,
        )

        output_path = resolve_output_path(args.output_dir, clip_path)
        np.save(output_path, joints.astype(np.float32))
        saved.append(str(output_path))

    print(f"saved_source_joints: {len(saved)}")
    print(f"model_path: {model_path}")
    print(f"joint_order: {SOURCE_JOINT_NAMES}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()

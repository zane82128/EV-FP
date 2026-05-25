import argparse
from pathlib import Path

import numpy as np
import torch


def scalar(value):
    array = np.asarray(value)
    return array.item() if array.shape == () else array


def normalize_skill_name(name):
    return str(name).strip().replace(" ", "_")


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def iter_canonical_paths(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.npz"))


def validate_canonical(clip):
    required = ["root_pos", "root_rot_3d", "dof_pos", "body_pos"]
    missing = [key for key in required if key not in clip]
    if missing:
        raise KeyError(f"Missing required canonical fields: {missing}")

    root_pos = np.asarray(clip["root_pos"], dtype=np.float32)
    root_rot_3d = np.asarray(clip["root_rot_3d"], dtype=np.float32)
    dof_pos = np.asarray(clip["dof_pos"], dtype=np.float32)
    body_pos = np.asarray(clip["body_pos"], dtype=np.float32)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must have shape [T, 3], got {root_pos.shape}")
    if root_rot_3d.shape != root_pos.shape:
        raise ValueError(f"root_rot_3d must match root_pos shape, got {root_rot_3d.shape}")
    if dof_pos.ndim != 3 or dof_pos.shape[1:] != (52, 3):
        raise ValueError(f"dof_pos must have shape [T, 52, 3], got {dof_pos.shape}")
    if body_pos.ndim != 3 or body_pos.shape[1:] != (53, 3):
        raise ValueError(f"body_pos must have shape [T, 53, 3], got {body_pos.shape}")
    if not (root_pos.shape[0] == dof_pos.shape[0] == body_pos.shape[0]):
        raise ValueError("All canonical arrays must share the same frame length")

    return root_pos, root_rot_3d, dof_pos, body_pos


def pack_human_only_pt(root_pos, root_rot_3d, dof_pos, body_pos, dummy_obj_pos):
    T = root_pos.shape[0]
    motion = np.zeros((T, 337), dtype=np.float32)

    obj_pos = np.tile(np.asarray(dummy_obj_pos, dtype=np.float32)[None, :], (T, 1))
    obj_rot_3d = np.zeros((T, 3), dtype=np.float32)
    contact = np.zeros((T, 1), dtype=np.float32)

    motion[:, 0:3] = root_pos
    motion[:, 3:6] = root_rot_3d
    motion[:, 9:165] = dof_pos.reshape(T, 156)
    motion[:, 165:324] = body_pos.reshape(T, 159)
    motion[:, 324:327] = obj_pos
    motion[:, 327:330] = obj_rot_3d
    motion[:, 336:337] = contact

    return torch.from_numpy(motion)


def resolve_output_path(clip, fallback_path, output_dir):
    skill_id = int(scalar(clip.get("skill_id", 0)))
    skill_name = normalize_skill_name(scalar(clip.get("skill_name", fallback_path.parent.name)))
    clip_index = int(scalar(clip.get("clip_index", 1)))

    folder = Path(output_dir) / skill_name
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{skill_id:03d}_{skill_name}_{clip_index:04d}.pt"
    return folder / filename


def main():
    parser = argparse.ArgumentParser(description="Pack canonical human-only clips into SkillMimic BallPlay-style .pt files.")
    parser.add_argument("input_path", type=str, help="Path to a canonical clip .npz or a directory of clips")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save .pt files",
    )
    parser.add_argument(
        "--dummy-obj-pos",
        type=float,
        nargs=3,
        default=[2.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Dummy object position to use for human-only clips",
    )
    args = parser.parse_args()

    saved = []

    for canonical_path in iter_canonical_paths(args.input_path):
        clip = load_npz(canonical_path)
        root_pos, root_rot_3d, dof_pos, body_pos = validate_canonical(clip)
        motion = pack_human_only_pt(
            root_pos=root_pos,
            root_rot_3d=root_rot_3d,
            dof_pos=dof_pos,
            body_pos=body_pos,
            dummy_obj_pos=args.dummy_obj_pos,
        )
        output_path = resolve_output_path(clip, canonical_path, args.output_dir)
        torch.save(motion, output_path)
        saved.append(str(output_path))

    print(f"saved_pt_files: {len(saved)}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()

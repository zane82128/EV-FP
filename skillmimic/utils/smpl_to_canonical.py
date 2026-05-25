import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


TARGET_BODY_NAMES = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Spine2",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]

TARGET_DOF_NAMES = TARGET_BODY_NAMES[1:]

DEFAULT_SOURCE_JOINT_NAMES = [
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

DEFAULT_MAPPING = {
    "Pelvis": ["Pelvis"],
    "L_Hip": ["L_Hip"],
    "R_Hip": ["R_Hip"],
    "Spine1": ["Torso"],
    "L_Knee": ["L_Knee"],
    "R_Knee": ["R_Knee"],
    "Spine2": ["Spine"],
    "L_Ankle": ["L_Ankle"],
    "R_Ankle": ["R_Ankle"],
    "Spine3": ["Spine2", "Chest"],
    "L_Foot": ["L_Toe"],
    "R_Foot": ["R_Toe"],
    "Neck": ["Neck"],
    "L_Collar": ["L_Thorax"],
    "R_Collar": ["R_Thorax"],
    "Head": ["Head"],
    "L_Shoulder": ["L_Shoulder"],
    "R_Shoulder": ["R_Shoulder"],
    "L_Elbow": ["L_Elbow"],
    "R_Elbow": ["R_Elbow"],
    "L_Wrist": ["L_Wrist"],
    "R_Wrist": ["R_Wrist"],
}

BODY_PROXY = {
    "L_Index1": "L_Wrist",
    "L_Index2": "L_Wrist",
    "L_Index3": "L_Wrist",
    "L_Middle1": "L_Wrist",
    "L_Middle2": "L_Wrist",
    "L_Middle3": "L_Wrist",
    "L_Pinky1": "L_Wrist",
    "L_Pinky2": "L_Wrist",
    "L_Pinky3": "L_Wrist",
    "L_Ring1": "L_Wrist",
    "L_Ring2": "L_Wrist",
    "L_Ring3": "L_Wrist",
    "L_Thumb1": "L_Wrist",
    "L_Thumb2": "L_Wrist",
    "L_Thumb3": "L_Wrist",
    "R_Index1": "R_Wrist",
    "R_Index2": "R_Wrist",
    "R_Index3": "R_Wrist",
    "R_Middle1": "R_Wrist",
    "R_Middle2": "R_Wrist",
    "R_Middle3": "R_Wrist",
    "R_Pinky1": "R_Wrist",
    "R_Pinky2": "R_Wrist",
    "R_Pinky3": "R_Wrist",
    "R_Ring1": "R_Wrist",
    "R_Ring2": "R_Wrist",
    "R_Ring3": "R_Wrist",
    "R_Thumb1": "R_Wrist",
    "R_Thumb2": "R_Wrist",
    "R_Thumb3": "R_Wrist",
}

TARGET_SOURCE_ANCHOR = {
    "Pelvis": "Pelvis",
    "L_Hip": "L_Hip",
    "L_Knee": "L_Knee",
    "L_Ankle": "L_Ankle",
    "L_Toe": "L_Foot",
    "R_Hip": "R_Hip",
    "R_Knee": "R_Knee",
    "R_Ankle": "R_Ankle",
    "R_Toe": "R_Foot",
    "Torso": "Spine1",
    "Spine": "Spine2",
    "Spine2": "Spine3",
    "Chest": "Spine3",
    "Neck": "Neck",
    "Head": "Head",
    "L_Thorax": "L_Collar",
    "L_Shoulder": "L_Shoulder",
    "L_Elbow": "L_Elbow",
    "L_Wrist": "L_Wrist",
    "R_Thorax": "R_Collar",
    "R_Shoulder": "R_Shoulder",
    "R_Elbow": "R_Elbow",
    "R_Wrist": "R_Wrist",
}

CHILD_SOURCE_OVERRIDE = {
    ("Spine2", "Chest"): "Neck",
}

COORD_TRANSFORMS = {
    "none": np.eye(3, dtype=np.float32),
    # Source world appears to be Y-up. Isaac Gym expects Z-up.
    "y_up_to_z_up": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    ),
}


def scalar(value):
    array = np.asarray(value)
    return array.item() if array.shape == () else array


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def normalize_skill_name(name):
    return str(name).strip().replace(" ", "_")


def load_mapping(path):
    if path is None:
        return DEFAULT_MAPPING
    with open(path, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    return {str(k): list(v) if isinstance(v, list) else [str(v)] for k, v in mapping.items()}


def lookup_source_joints(clip_path, source_joints_path, source_joints_dir):
    if source_joints_path is not None:
        return source_joints_path

    if source_joints_dir is None:
        return None

    base = Path(clip_path).stem
    candidates = [
        Path(source_joints_dir) / f"{base}.npy",
        Path(source_joints_dir) / f"{base}.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_source_joints(path):
    path = Path(path)
    if path.suffix == ".npy":
        joints = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "source_joints" in data:
            joints = data["source_joints"]
        elif "joints" in data:
            joints = data["joints"]
        else:
            raise KeyError(f"{path} must contain source_joints or joints")
    else:
        raise ValueError(f"Unsupported source joints file: {path}")

    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected source joints shape [T, J, 3], got {joints.shape}")
    return joints.astype(np.float32)


def fill_body_proxies(body_pos):
    for target_name, proxy_name in BODY_PROXY.items():
        target_idx = TARGET_BODY_NAMES.index(target_name)
        proxy_idx = TARGET_BODY_NAMES.index(proxy_name)
        body_pos[:, target_idx] = body_pos[:, proxy_idx]
    return body_pos


def parse_vec3(value):
    if value is None:
        return np.zeros(3, dtype=np.float32)
    return np.fromstring(value, sep=" ", dtype=np.float32)


def resolve_asset_xml(path):
    if path is not None:
        asset_path = Path(path)
        if not asset_path.exists():
            raise FileNotFoundError(f"Asset XML not found: {asset_path}")
        return asset_path

    repo_root = Path(__file__).resolve().parents[2]
    asset_path = repo_root / "skillmimic" / "data" / "assets" / "mjcf" / "mocap_humanoid.xml"
    if not asset_path.exists():
        raise FileNotFoundError(f"Default asset XML not found: {asset_path}")
    return asset_path


def load_target_skeleton(asset_xml_path):
    tree = ET.parse(asset_xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{asset_xml_path} has no <worldbody>")

    pelvis = None
    for body in worldbody.iter("body"):
        if body.attrib.get("name") == "Pelvis":
            pelvis = body
            break

    if pelvis is None:
        raise ValueError(f"{asset_xml_path} has no body named Pelvis")

    nodes = {}

    def visit(elem, parent_name):
        name = elem.attrib.get("name")
        if name not in TARGET_BODY_NAMES:
            return

        nodes[name] = {
            "parent": parent_name,
            "offset": parse_vec3(elem.attrib.get("pos")),
            "children": [],
        }
        if parent_name is not None:
            nodes[parent_name]["children"].append(name)

        for child in elem.findall("body"):
            visit(child, name)

    visit(pelvis, None)

    missing = [name for name in TARGET_BODY_NAMES if name not in nodes]
    if missing:
        raise ValueError(f"Target asset is missing expected bodies: {missing}")

    return nodes


def safe_normalize(vec, eps=1e-8):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < eps:
        return None
    return vec / norm


def rotation_between_vectors(source, target):
    source_unit = safe_normalize(source)
    target_unit = safe_normalize(target)
    if source_unit is None or target_unit is None:
        return np.eye(3, dtype=np.float32)

    dot = np.clip(np.dot(source_unit, target_unit), -1.0, 1.0)
    if dot > 1.0 - 1e-6:
        return np.eye(3, dtype=np.float32)

    if dot < -1.0 + 1e-6:
        trial = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if np.abs(np.dot(source_unit, trial)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = np.cross(source_unit, trial)
        axis = safe_normalize(axis)
        if axis is None:
            return np.eye(3, dtype=np.float32)
        return Rotation.from_rotvec(axis * np.pi).as_matrix().astype(np.float32)

    axis = np.cross(source_unit, target_unit)
    axis = safe_normalize(axis)
    if axis is None:
        return np.eye(3, dtype=np.float32)
    angle = np.arccos(dot)
    return Rotation.from_rotvec(axis * angle).as_matrix().astype(np.float32)


def fit_rotation(rest_vectors, desired_vectors):
    src = []
    dst = []
    for rest, desired in zip(rest_vectors, desired_vectors):
        rest_unit = safe_normalize(rest)
        desired_unit = safe_normalize(desired)
        if rest_unit is None or desired_unit is None:
            continue
        src.append(rest_unit)
        dst.append(desired_unit)

    if not src:
        return None
    if len(src) == 1:
        return rotation_between_vectors(src[0], dst[0])

    src = np.stack(src, axis=0)
    dst = np.stack(dst, axis=0)
    covariance = src.T @ dst
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    return rotation.astype(np.float32)


def target_body_source_name(body_name):
    return TARGET_SOURCE_ANCHOR.get(body_name)


def target_child_source_name(parent_name, child_name):
    return CHILD_SOURCE_OVERRIDE.get((parent_name, child_name), TARGET_SOURCE_ANCHOR.get(child_name))


def retarget_from_source_joints(root_pos, source_joints, target_skeleton, source_local_pose=None):
    source_name_to_index = {name: idx for idx, name in enumerate(DEFAULT_SOURCE_JOINT_NAMES)}
    body_index = {name: idx for idx, name in enumerate(TARGET_BODY_NAMES)}

    T = source_joints.shape[0]
    body_pos = np.zeros((T, len(TARGET_BODY_NAMES), 3), dtype=np.float32)
    global_rot = np.zeros((T, len(TARGET_BODY_NAMES), 3, 3), dtype=np.float32)
    local_rot = np.zeros((T, len(TARGET_DOF_NAMES), 3, 3), dtype=np.float32)

    traversal = TARGET_BODY_NAMES

    for t in range(T):
        frame_source = source_joints[t]
        body_pos[t, body_index["Pelvis"]] = root_pos[t]

        for body_name in traversal:
            body_id = body_index[body_name]
            node = target_skeleton[body_name]
            parent_name = node["parent"]

            if parent_name is None:
                parent_global = np.eye(3, dtype=np.float32)
            else:
                parent_id = body_index[parent_name]
                parent_global = global_rot[t, parent_id]
                body_pos[t, body_id] = (
                    body_pos[t, parent_id] + parent_global @ node["offset"]
                ).astype(np.float32)

            source_current_name = target_body_source_name(body_name)
            rest_vectors = []
            desired_vectors = []
            fallback_local = None

            if (
                source_local_pose is not None
                and source_current_name is not None
                and source_current_name in source_name_to_index
            ):
                source_idx = source_name_to_index[source_current_name]
                if source_idx < source_local_pose.shape[1]:
                    fallback_local = Rotation.from_rotvec(
                        source_local_pose[t, source_idx]
                    ).as_matrix().astype(np.float32)

            if source_current_name is not None and source_current_name in source_name_to_index:
                current_source_pos = frame_source[source_name_to_index[source_current_name]]
                for child_name in node["children"]:
                    source_child_name = target_child_source_name(body_name, child_name)
                    if source_child_name is None or source_child_name not in source_name_to_index:
                        continue
                    desired = frame_source[source_name_to_index[source_child_name]] - current_source_pos
                    rest = target_skeleton[child_name]["offset"]
                    if np.linalg.norm(desired) < 1e-6 or np.linalg.norm(rest) < 1e-6:
                        continue
                    rest_vectors.append(rest)
                    desired_vectors.append(desired)

            if rest_vectors:
                if len(rest_vectors) == 1:
                    rest_world = parent_global @ rest_vectors[0]
                    delta = rotation_between_vectors(rest_world, desired_vectors[0])
                    body_global = (delta @ parent_global).astype(np.float32)
                else:
                    body_global = fit_rotation(rest_vectors, desired_vectors)
                    if body_global is None:
                        body_global = parent_global
            else:
                if parent_name is not None and fallback_local is not None:
                    body_global = (parent_global @ fallback_local).astype(np.float32)
                else:
                    body_global = parent_global

            global_rot[t, body_id] = body_global

            if parent_name is not None:
                local_matrix = parent_global.T @ body_global
                local_rot[t, body_id - 1] = local_matrix.astype(np.float32)

    root_rot_3d = Rotation.from_matrix(global_rot[:, 0]).as_rotvec().astype(np.float32)
    dof_pos = Rotation.from_matrix(local_rot.reshape(-1, 3, 3)).as_rotvec().astype(np.float32)
    dof_pos = dof_pos.reshape(T, len(TARGET_DOF_NAMES), 3)

    return root_rot_3d, dof_pos, body_pos


def get_coord_transform(name):
    if name not in COORD_TRANSFORMS:
        raise ValueError(f"Unsupported coord transform '{name}'")
    return COORD_TRANSFORMS[name]


def transform_positions(array, transform):
    return np.einsum("ij,...j->...i", transform, array).astype(np.float32)


def transform_rotvecs(rotvecs, transform):
    flat = np.asarray(rotvecs, dtype=np.float32).reshape(-1, 3)
    if flat.shape[0] == 0:
        return np.asarray(rotvecs, dtype=np.float32)
    matrices = Rotation.from_rotvec(flat).as_matrix()
    transformed = transform[None, :, :] @ matrices @ transform.T[None, :, :]
    rotvec_out = Rotation.from_matrix(transformed).as_rotvec().astype(np.float32)
    return rotvec_out.reshape(rotvecs.shape)


def build_canonical(
    clip,
    source_joints,
    mapping,
    root_joint_index,
    coord_transform_name,
    target_skeleton,
):
    body_pose = np.asarray(clip["body_pose"], dtype=np.float32)
    body_translation = np.asarray(clip["body_translation"], dtype=np.float32)

    if body_pose.ndim != 3 or body_pose.shape[-1] != 3:
        raise ValueError(f"Expected body_pose shape [T, J, 3], got {body_pose.shape}")
    if body_translation.ndim != 2 or body_translation.shape[-1] != 3:
        raise ValueError(
            f"Expected body_translation shape [T, 3], got {body_translation.shape}"
        )

    T = body_pose.shape[0]
    if source_joints is not None and source_joints.shape[0] != T:
        raise ValueError(
            f"source_joints length {source_joints.shape[0]} does not match clip length {T}"
        )

    coord_transform = get_coord_transform(coord_transform_name)
    body_pose = transform_rotvecs(body_pose, coord_transform)
    body_translation = transform_positions(body_translation, coord_transform)
    if source_joints is not None:
        source_joints = transform_positions(source_joints, coord_transform)

    root_pos = body_translation.astype(np.float32)

    if source_joints is not None:
        root_rot_3d, dof_pos, body_pos = retarget_from_source_joints(
            root_pos=root_pos,
            source_joints=source_joints,
            target_skeleton=target_skeleton,
            source_local_pose=body_pose,
        )
    else:
        root_rot_3d = body_pose[:, root_joint_index].astype(np.float32)
        dof_pos = np.zeros((T, len(TARGET_DOF_NAMES), 3), dtype=np.float32)
        body_pos = np.zeros((T, len(TARGET_BODY_NAMES), 3), dtype=np.float32)
        body_pos[:, TARGET_BODY_NAMES.index("Pelvis")] = root_pos

        source_name_to_index = {name: idx for idx, name in enumerate(DEFAULT_SOURCE_JOINT_NAMES)}
        target_body_to_index = {name: idx for idx, name in enumerate(TARGET_BODY_NAMES)}
        target_dof_to_index = {name: idx for idx, name in enumerate(TARGET_DOF_NAMES)}

        for source_name, target_names in mapping.items():
            if source_name not in source_name_to_index:
                continue
            source_idx = source_name_to_index[source_name]

            for target_name in target_names:
                if target_name in target_dof_to_index and source_idx < body_pose.shape[1]:
                    dof_pos[:, target_dof_to_index[target_name]] = body_pose[:, source_idx]

                if (
                    source_joints is not None
                    and target_name in target_body_to_index
                    and source_idx < source_joints.shape[1]
                ):
                    body_pos[:, target_body_to_index[target_name]] = source_joints[:, source_idx]

        if source_joints is None:
            body_pos[:] = root_pos[:, None, :]

        body_pos = fill_body_proxies(body_pos)

    payload = {
        "skill_id": np.array(int(scalar(clip["skill_id"])), dtype=np.int32),
        "skill_name": np.array(normalize_skill_name(scalar(clip["skill_name"]))),
        "clip_index": np.array(int(scalar(clip["clip_index"])), dtype=np.int32),
        "fps": np.array(float(scalar(clip["fps"])), dtype=np.float32),
        "root_pos": root_pos,
        "root_rot_3d": root_rot_3d,
        "dof_pos": dof_pos,
        "body_pos": body_pos,
    }
    return payload


def iter_clip_paths(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.npz"))


def save_payload(payload, output_dir):
    skill_id = int(scalar(payload["skill_id"]))
    skill_name = normalize_skill_name(scalar(payload["skill_name"]))
    clip_index = int(scalar(payload["clip_index"]))

    folder = Path(output_dir) / skill_name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{skill_id:03d}_{skill_name}_{clip_index:04d}.npz"
    np.savez_compressed(path, **payload)
    return path


def main():
    parser = argparse.ArgumentParser(description="Convert SMPL clip .npz files to canonical human-only clips.")
    parser.add_argument("input_path", type=str, help="Path to a clip .npz or a directory of clips")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save canonical clip .npz files",
    )
    parser.add_argument(
        "--source-joints",
        type=str,
        default=None,
        help="Path to a single .npy/.npz file containing source_joints [T, J, 3]",
    )
    parser.add_argument(
        "--source-joints-dir",
        type=str,
        default=None,
        help="Directory containing per-clip source joints files matching clip basenames",
    )
    parser.add_argument(
        "--mapping-json",
        type=str,
        default=None,
        help="Optional JSON file overriding the default source->target mapping",
    )
    parser.add_argument(
        "--root-joint-index",
        type=int,
        default=0,
        help="Index in body_pose used as the root rotation candidate",
    )
    parser.add_argument(
        "--allow-missing-source-joints",
        action="store_true",
        help="Allow conversion without source_joints by filling body_pos from root_pos",
    )
    parser.add_argument(
        "--coord-transform",
        type=str,
        choices=sorted(COORD_TRANSFORMS.keys()),
        default="none",
        help="Optional source->target coordinate transform to apply before packing canonical clips.",
    )
    parser.add_argument(
        "--asset-xml",
        type=str,
        default=None,
        help="Optional target skeleton MJCF asset. Defaults to skillmimic/data/assets/mjcf/mocap_humanoid.xml.",
    )
    args = parser.parse_args()

    mapping = load_mapping(args.mapping_json)
    target_skeleton = load_target_skeleton(resolve_asset_xml(args.asset_xml))
    saved = []

    for clip_path in iter_clip_paths(args.input_path):
        clip = load_npz(clip_path)
        joints_path = lookup_source_joints(clip_path, args.source_joints, args.source_joints_dir)

        if joints_path is None and not args.allow_missing_source_joints:
            raise FileNotFoundError(
                f"Missing source joints for {clip_path}. "
                "Provide --source-joints/--source-joints-dir or pass --allow-missing-source-joints."
            )

        source_joints = load_source_joints(joints_path) if joints_path is not None else None
        payload = build_canonical(
            clip,
            source_joints,
            mapping,
            args.root_joint_index,
            args.coord_transform,
            target_skeleton,
        )
        saved.append(str(save_payload(payload, args.output_dir)))

    print(f"saved_canonical_clips: {len(saved)}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()

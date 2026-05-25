import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch


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

BALLPLAY_LAYOUT = {
    "root_pos": (0, 3),
    "root_rot_3d": (3, 6),
    "reserved_a": (6, 9),
    "dof_pos": (9, 165),
    "body_pos": (165, 324),
    "obj_pos": (324, 327),
    "obj_rot_3d": (327, 330),
    "reserved_b": (330, 336),
    "contact": (336, 337),
}

BONE_PAIRS = [
    ("Pelvis", "L_Hip"),
    ("L_Hip", "L_Knee"),
    ("L_Knee", "L_Ankle"),
    ("L_Ankle", "L_Toe"),
    ("Pelvis", "R_Hip"),
    ("R_Hip", "R_Knee"),
    ("R_Knee", "R_Ankle"),
    ("R_Ankle", "R_Toe"),
    ("Pelvis", "Torso"),
    ("Torso", "Spine"),
    ("Spine", "Spine2"),
    ("Spine2", "Chest"),
    ("Chest", "Neck"),
    ("Neck", "Head"),
    ("Chest", "L_Thorax"),
    ("L_Thorax", "L_Shoulder"),
    ("L_Shoulder", "L_Elbow"),
    ("L_Elbow", "L_Wrist"),
    ("Chest", "R_Thorax"),
    ("R_Thorax", "R_Shoulder"),
    ("R_Shoulder", "R_Elbow"),
    ("R_Elbow", "R_Wrist"),
]

BODY_INDEX = {name: idx for idx, name in enumerate(TARGET_BODY_NAMES)}


@dataclass
class CheckResult:
    level: str
    name: str
    message: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate BallPlay-style SkillMimic motion .pt files."
    )
    parser.add_argument("path", type=str, help="Path to a motion .pt file")
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="FPS used to estimate velocities from adjacent frames",
    )
    parser.add_argument(
        "--termination-height",
        type=float,
        default=0.25,
        help="Early termination root-height threshold from the task config",
    )
    parser.add_argument(
        "--expect-dummy-object",
        action="store_true",
        help="Expect human-only dummy object and zero contact",
    )
    parser.add_argument(
        "--strict-warn",
        action="store_true",
        help="Exit with non-zero status on WARN in addition to FAIL",
    )
    return parser.parse_args()


def slice_field(tensor: torch.Tensor, name: str) -> torch.Tensor:
    start, end = BALLPLAY_LAYOUT[name]
    return tensor[:, start:end]


def exp_map_to_quat(exp_map: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(exp_map, dim=-1, keepdim=True)
    half_angle = 0.5 * angle

    axis = torch.zeros_like(exp_map)
    nonzero = angle.squeeze(-1) > 1e-8
    axis[nonzero] = exp_map[nonzero] / angle[nonzero]

    xyz = axis * torch.sin(half_angle)
    w = torch.cos(half_angle)
    quat = torch.cat((xyz, w), dim=-1)
    return quat


def angular_speed_from_exp_map(exp_map: torch.Tensor, fps: float) -> torch.Tensor:
    quat = exp_map_to_quat(exp_map)
    quat_prev = quat[:-1]
    quat_next = quat[1:]

    dot = torch.sum(quat_prev * quat_next, dim=-1).abs().clamp(max=1.0)
    angle = 2.0 * torch.arccos(dot)
    return angle * fps


def max_abs(tensor: torch.Tensor) -> float:
    return float(torch.max(torch.abs(tensor)).item())


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def add_result(results: List[CheckResult], level: str, name: str, message: str):
    results.append(CheckResult(level=level, name=name, message=message))


def validate_tensor_shape(tensor: torch.Tensor, results: List[CheckResult]):
    if not isinstance(tensor, torch.Tensor):
        add_result(results, "FAIL", "type", f"Expected torch.Tensor, got {type(tensor)}")
        return

    if tensor.ndim != 2:
        add_result(results, "FAIL", "shape", f"Expected [T, 337], got {tuple(tensor.shape)}")
        return

    frames, width = tensor.shape
    if width != 337:
        add_result(results, "FAIL", "shape", f"Expected width 337, got {width}")
    elif frames < 4:
        add_result(results, "FAIL", "shape", f"Need at least 4 frames for sampling, got {frames}")
    elif frames < 15:
        add_result(results, "WARN", "shape", f"Very short clip: {frames} frames")
    else:
        add_result(results, "PASS", "shape", f"Tensor shape is {tuple(tensor.shape)}")


def validate_finite(tensor: torch.Tensor, results: List[CheckResult]):
    finite_mask = torch.isfinite(tensor)
    if torch.all(finite_mask):
        add_result(results, "PASS", "finite", "All values are finite")
        return

    bad = int((~finite_mask).sum().item())
    add_result(results, "FAIL", "finite", f"Found {bad} non-finite values")


def validate_reserved_fields(tensor: torch.Tensor, results: List[CheckResult]):
    reserved_a = slice_field(tensor, "reserved_a")
    reserved_b = slice_field(tensor, "reserved_b")
    reserved_max = max(max_abs(reserved_a), max_abs(reserved_b))
    if reserved_max <= 1e-6:
        add_result(results, "PASS", "reserved", "Reserved fields are zero")
    elif reserved_max <= 1e-3:
        add_result(results, "WARN", "reserved", f"Reserved fields are near-zero, max abs={reserved_max:.6f}")
    else:
        add_result(results, "WARN", "reserved", f"Reserved fields are not zero, max abs={reserved_max:.6f}")


def validate_contact(tensor: torch.Tensor, results: List[CheckResult], expect_dummy_object: bool):
    contact = slice_field(tensor, "contact")
    rounded = torch.round(contact)
    diff = torch.abs(contact - rounded)
    max_diff = float(diff.max().item())
    contact_non_binary = max_diff > 1e-4

    if contact_non_binary:
        add_result(results, "FAIL", "contact", f"Contact contains non-binary values, max diff from round={max_diff:.6f}")
    elif expect_dummy_object and torch.any(torch.abs(contact) > 1e-6):
        active = int((torch.abs(contact) > 1e-6).sum().item())
        add_result(results, "WARN", "contact", f"Expected zero contact for human-only data, but found {active} active entries")
    else:
        add_result(results, "PASS", "contact", "Contact values look valid")


def validate_root_body_consistency(root_pos: torch.Tensor, body_pos: torch.Tensor, results: List[CheckResult]):
    pelvis = body_pos[:, 0, :]
    error = torch.linalg.norm(pelvis - root_pos, dim=-1)
    mean_error = float(error.mean().item())
    max_error = float(error.max().item())

    if max_error <= 1e-4:
        add_result(results, "PASS", "pelvis-root", "Pelvis body_pos matches root_pos")
    elif max_error <= 5e-2:
        add_result(results, "WARN", "pelvis-root", f"Pelvis-root mismatch mean={mean_error:.5f}, max={max_error:.5f}")
    else:
        add_result(results, "FAIL", "pelvis-root", f"Pelvis-root mismatch mean={mean_error:.5f}, max={max_error:.5f}")


def validate_root_height(root_pos: torch.Tensor, termination_height: float, results: List[CheckResult]):
    root_z = root_pos[:, 2]
    min_z = float(root_z.min().item())
    max_z = float(root_z.max().item())
    below = int((root_z < termination_height).sum().item())

    if below > 0:
        add_result(
            results,
            "FAIL",
            "root-height",
            f"{below} frames fall below termination height {termination_height:.3f}; min root z={min_z:.3f}, max={max_z:.3f}",
        )
    elif min_z < termination_height + 0.1:
        add_result(
            results,
            "WARN",
            "root-height",
            f"Root is close to termination height; min root z={min_z:.3f}, max={max_z:.3f}",
        )
    else:
        add_result(results, "PASS", "root-height", f"Root height looks safe; min z={min_z:.3f}, max z={max_z:.3f}")


def validate_root_motion(root_pos: torch.Tensor, root_rot_3d: torch.Tensor, fps: float, results: List[CheckResult]):
    if root_pos.shape[0] < 2:
        return

    root_speed = torch.linalg.norm(root_pos[1:] - root_pos[:-1], dim=-1) * fps
    max_root_speed = float(root_speed.max().item())
    p99_root_speed = float(torch.quantile(root_speed, 0.99).item()) if root_speed.numel() > 1 else max_root_speed

    if max_root_speed > 30.0:
        add_result(results, "FAIL", "root-speed", f"Root speed spike too large: max={max_root_speed:.3f} m/s, p99={p99_root_speed:.3f}")
    elif max_root_speed > 10.0:
        add_result(results, "WARN", "root-speed", f"Root speed looks high: max={max_root_speed:.3f} m/s, p99={p99_root_speed:.3f}")
    else:
        add_result(results, "PASS", "root-speed", f"Root speed looks reasonable: max={max_root_speed:.3f} m/s, p99={p99_root_speed:.3f}")

    root_ang_speed = angular_speed_from_exp_map(root_rot_3d, fps)
    max_root_ang = float(root_ang_speed.max().item()) if root_ang_speed.numel() > 0 else 0.0
    over_loader_threshold = float((root_ang_speed > 5.0).float().mean().item()) if root_ang_speed.numel() > 0 else 0.0

    if max_root_ang > 8.0:
        add_result(
            results,
            "FAIL",
            "root-ang-speed",
            f"Root angular speed exceeds replay abnormal threshold badly: max={max_root_ang:.3f} rad/s, over-5 ratio={percent(over_loader_threshold)}",
        )
    elif max_root_ang > 5.0:
        add_result(
            results,
            "WARN",
            "root-ang-speed",
            f"Root angular speed exceeds replay abnormal threshold: max={max_root_ang:.3f} rad/s, over-5 ratio={percent(over_loader_threshold)}",
        )
    else:
        add_result(results, "PASS", "root-ang-speed", f"Root angular speed looks reasonable: max={max_root_ang:.3f} rad/s")


def validate_dof_rotations(dof_pos: torch.Tensor, results: List[CheckResult]):
    norms = torch.linalg.norm(dof_pos, dim=-1)
    max_norm = float(norms.max().item())
    p99_norm = float(torch.quantile(norms.reshape(-1), 0.99).item())

    if max_norm > 6.4:
        add_result(results, "FAIL", "dof-rot", f"dof_pos rotvec magnitude is implausibly large: max={max_norm:.3f}, p99={p99_norm:.3f}")
    elif p99_norm > 3.5:
        add_result(results, "WARN", "dof-rot", f"dof_pos rotvec magnitudes look high: max={max_norm:.3f}, p99={p99_norm:.3f}")
    else:
        add_result(results, "PASS", "dof-rot", f"dof_pos magnitudes look reasonable: max={max_norm:.3f}, p99={p99_norm:.3f}")


def validate_body_not_collapsed(root_pos: torch.Tensor, body_pos: torch.Tensor, results: List[CheckResult]):
    offsets = body_pos[:, 1:, :] - root_pos[:, None, :]
    dist = torch.linalg.norm(offsets, dim=-1)
    median_dist = float(torch.median(dist).item())
    p95_dist = float(torch.quantile(dist.reshape(-1), 0.95).item())

    if p95_dist < 1e-3:
        add_result(
            results,
            "FAIL",
            "body-collapse",
            "body_pos is effectively collapsed to root_pos. This usually means you used the fallback path without real source_joints.",
        )
    elif median_dist < 1e-2:
        add_result(
            results,
            "WARN",
            "body-collapse",
            f"body_pos offsets are unusually small: median={median_dist:.5f}, p95={p95_dist:.5f}",
        )
    else:
        add_result(results, "PASS", "body-collapse", f"body_pos offsets look non-degenerate: median={median_dist:.3f}, p95={p95_dist:.3f}")


def validate_bone_lengths(body_pos: torch.Tensor, results: List[CheckResult]):
    unstable = []

    for parent_name, child_name in BONE_PAIRS:
        parent_idx = BODY_INDEX[parent_name]
        child_idx = BODY_INDEX[child_name]
        bone = body_pos[:, child_idx] - body_pos[:, parent_idx]
        lengths = torch.linalg.norm(bone, dim=-1)
        mean_len = float(lengths.mean().item())
        std_len = float(lengths.std().item())
        rel_std = 0.0 if mean_len < 1e-6 else std_len / mean_len
        unstable.append((rel_std, mean_len, std_len, f"{parent_name}->{child_name}"))

    unstable.sort(reverse=True)
    worst_rel_std, worst_mean, worst_std, worst_name = unstable[0]

    if worst_mean < 1e-4:
        add_result(results, "FAIL", "bone-length", f"Bone {worst_name} is nearly zero length across the clip")
    elif worst_rel_std > 0.20:
        add_result(
            results,
            "FAIL",
            "bone-length",
            f"Bone lengths are unstable. Worst bone {worst_name}: mean={worst_mean:.3f}, std={worst_std:.3f}, rel_std={worst_rel_std:.3f}",
        )
    elif worst_rel_std > 0.05:
        add_result(
            results,
            "WARN",
            "bone-length",
            f"Bone lengths vary more than expected. Worst bone {worst_name}: mean={worst_mean:.3f}, std={worst_std:.3f}, rel_std={worst_rel_std:.3f}",
        )
    else:
        add_result(
            results,
            "PASS",
            "bone-length",
            f"Bone lengths are stable. Worst bone {worst_name}: mean={worst_mean:.3f}, std={worst_std:.3f}, rel_std={worst_rel_std:.3f}",
        )


def validate_object_fields(obj_pos: torch.Tensor, obj_rot_3d: torch.Tensor, expect_dummy_object: bool, results: List[CheckResult]):
    obj_pos_std = obj_pos.std(dim=0)
    obj_rot_std = obj_rot_3d.std(dim=0)
    max_obj_pos_std = float(obj_pos_std.max().item())
    max_obj_rot_std = float(obj_rot_std.max().item())

    if expect_dummy_object:
        if max_obj_pos_std > 1e-5 or max_obj_rot_std > 1e-5:
            add_result(
                results,
                "WARN",
                "dummy-object",
                f"Expected constant dummy object, but std(pos)={max_obj_pos_std:.6f}, std(rot)={max_obj_rot_std:.6f}",
            )
        else:
            add_result(results, "PASS", "dummy-object", "Object fields match a constant human-only dummy object")
    else:
        add_result(
            results,
            "PASS",
            "object",
            f"Object stats: std(pos)={max_obj_pos_std:.6f}, std(rot)={max_obj_rot_std:.6f}",
        )


def print_results(path: Path, results: Sequence[CheckResult]):
    print(f"path: {path}")
    for result in results:
        print(f"[{result.level}] {result.name}: {result.message}")


def exit_code(results: Sequence[CheckResult], strict_warn: bool) -> int:
    has_fail = any(result.level == "FAIL" for result in results)
    has_warn = any(result.level == "WARN" for result in results)
    if has_fail:
        return 1
    if strict_warn and has_warn:
        return 2
    return 0


def main():
    args = parse_args()
    path = Path(args.path)
    tensor = torch.load(path, map_location="cpu")
    results: List[CheckResult] = []

    validate_tensor_shape(tensor, results)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[1] != 337:
        print_results(path, results)
        raise SystemExit(exit_code(results, args.strict_warn))

    tensor = tensor.to(dtype=torch.float32)
    validate_finite(tensor, results)
    validate_reserved_fields(tensor, results)
    validate_contact(tensor, results, args.expect_dummy_object)

    root_pos = slice_field(tensor, "root_pos")
    root_rot_3d = slice_field(tensor, "root_rot_3d")
    dof_pos = slice_field(tensor, "dof_pos").view(tensor.shape[0], 52, 3)
    body_pos = slice_field(tensor, "body_pos").view(tensor.shape[0], 53, 3)
    obj_pos = slice_field(tensor, "obj_pos")
    obj_rot_3d = slice_field(tensor, "obj_rot_3d")

    validate_root_body_consistency(root_pos, body_pos, results)
    validate_root_height(root_pos, args.termination_height, results)
    validate_root_motion(root_pos, root_rot_3d, args.fps, results)
    validate_dof_rotations(dof_pos, results)
    validate_body_not_collapsed(root_pos, body_pos, results)
    validate_bone_lengths(body_pos, results)
    validate_object_fields(obj_pos, obj_rot_3d, args.expect_dummy_object, results)

    print_results(path, results)
    raise SystemExit(exit_code(results, args.strict_warn))


if __name__ == "__main__":
    main()

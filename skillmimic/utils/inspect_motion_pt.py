import argparse
import os

import torch


BALLPLAY_LAYOUT = [
    ("root_pos", 0, 3),
    ("root_rot_3d", 3, 6),
    ("reserved_a", 6, 9),
    ("dof_pos", 9, 165),
    ("body_pos", 165, 324),
    ("obj_pos", 324, 327),
    ("obj_rot_3d", 327, 330),
    ("reserved_b", 330, 336),
    ("contact", 336, 337),
]

PARAHOME_LAYOUT = [
    ("root_pos", 0, 3),
    ("root_rot_3d", 3, 6),
    ("reserved_a", 6, 9),
    ("dof_pos", 9, 189),
    ("joint_pos", 189, 402),
    ("obj_pos", 402, 405),
    ("obj_rot_3d", 405, 408),
    ("contact", 408, 409),
]


def infer_format(width):
    if width == 337:
        return "ballplay"
    if width == 409:
        return "parahome"
    return None


def get_layout(format_name):
    if format_name == "ballplay":
        return BALLPLAY_LAYOUT
    if format_name == "parahome":
        return PARAHOME_LAYOUT
    raise ValueError(f"Unsupported format: {format_name}")


def preview_tensor(tensor, max_values=8):
    flat = tensor.reshape(-1)
    values = flat[:max_values].tolist()
    return ", ".join(f"{v:.4f}" for v in values)


def print_header(path, tensor, format_name):
    print(f"path: {path}")
    print(f"filename: {os.path.basename(path)}")
    print(f"type: {type(tensor).__name__}")
    print(f"shape: {tuple(tensor.shape)}")
    print(f"dtype: {tensor.dtype}")
    print(f"format: {format_name}")

    skill_prefix = os.path.basename(path).split("_")[0]
    if skill_prefix.isdigit():
        print(f"skill_id_from_filename: {skill_prefix}")


def print_layout(layout):
    print("\nfield layout:")
    for name, start, end in layout:
        print(f"  {name:12s} [{start:3d}:{end:3d}]  dim={end - start}")


def print_frame_preview(tensor, layout, frames_to_show):
    total_frames = tensor.shape[0]
    frames_to_show = min(frames_to_show, total_frames)

    for frame_idx in range(frames_to_show):
        frame = tensor[frame_idx]
        print(f"\nframe {frame_idx}:")
        for name, start, end in layout:
            field = frame[start:end]
            print(
                f"  {name:12s} shape={tuple(field.shape)!s:>8s} "
                f"preview=[{preview_tensor(field)}]"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Inspect SkillMimic motion .pt files."
    )
    parser.add_argument("path", type=str, help="Path to a motion .pt file")
    parser.add_argument(
        "--format",
        choices=["auto", "ballplay", "parahome"],
        default="auto",
        help="Motion format to use when decoding fields",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1,
        help="Number of frames to preview",
    )
    args = parser.parse_args()

    tensor = torch.load(args.path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D tensor [T, D], got shape {tuple(tensor.shape)}")

    format_name = args.format
    if format_name == "auto":
        format_name = infer_format(tensor.shape[1])
        if format_name is None:
            raise ValueError(
                f"Cannot infer format from width {tensor.shape[1]}. "
                "Use --format ballplay or --format parahome."
            )

    layout = get_layout(format_name)

    print_header(args.path, tensor, format_name)
    print_layout(layout)
    print_frame_preview(tensor, layout, args.frames)


if __name__ == "__main__":
    main()

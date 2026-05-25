import argparse
import io
import os
import zipfile

import numpy as np


def load_smpl_archive(path):
    arrays = {}
    with zipfile.ZipFile(path, "r") as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".npy"):
                continue
            with archive.open(name, "r") as handle:
                arrays[name] = np.load(io.BytesIO(handle.read()), allow_pickle=True)
    return arrays


def preview_array(array, max_values=8):
    if getattr(array, "shape", ()) == ():
        return repr(array.item())

    flat = np.asarray(array).reshape(-1)
    values = flat[:max_values]
    return ", ".join(f"{float(v):.4f}" for v in values)


def print_summary(path, arrays):
    print(f"path: {path}")
    print(f"filename: {os.path.basename(path)}")
    print(f"num_entries: {len(arrays)}")

    print("\nentries:")
    for name, array in arrays.items():
        shape = getattr(array, "shape", ())
        dtype = getattr(array, "dtype", type(array).__name__)
        print(f"  {name:24s} shape={shape!s:16s} dtype={dtype}")


def print_previews(arrays, max_values):
    print("\npreview:")
    for name, array in arrays.items():
        print(f"  {name}: [{preview_array(array, max_values=max_values)}]")


def print_motion_hint(arrays):
    body_translation = arrays.get("bodyTranslation.npy")
    body_pose = arrays.get("bodyPose.npy")
    frame_count = arrays.get("frameCount.npy")
    frame_rate = arrays.get("frameRate.npy")

    print("\ninterpreted fields:")
    if frame_count is not None:
        print(f"  frameCount: {int(np.asarray(frame_count).item())}")
    if frame_rate is not None:
        print(f"  frameRate: {float(np.asarray(frame_rate).item())}")
    if body_translation is not None:
        print(f"  bodyTranslation: {body_translation.shape}  # root/world translation per frame")
    if body_pose is not None:
        print(f"  bodyPose: {body_pose.shape}  # joint pose parameters per frame")


def main():
    parser = argparse.ArgumentParser(description="Inspect a .smpl archive.")
    parser.add_argument("path", type=str, help="Path to a .smpl file")
    parser.add_argument(
        "--max-values",
        type=int,
        default=8,
        help="Maximum number of flattened values to preview for each entry",
    )
    args = parser.parse_args()

    arrays = load_smpl_archive(args.path)
    if not arrays:
        raise ValueError(f"No .npy entries found inside {args.path}")

    print_summary(args.path, arrays)
    print_motion_hint(arrays)
    print_previews(arrays, args.max_values)


if __name__ == "__main__":
    main()

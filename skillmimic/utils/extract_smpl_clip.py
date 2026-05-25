import argparse
import io
import json
import os
import zipfile
from pathlib import Path

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


def scalar_or_none(arrays, key):
    value = arrays.get(key)
    if value is None:
        return None
    value = np.asarray(value)
    return value.item() if value.shape == () else value


def load_segments(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError("Segments JSON must be a list of objects.")
    return data


def normalize_skill_name(name):
    return str(name).strip().replace(" ", "_")


def validate_segment(segment, total_frames):
    required = ["skill_id", "skill_name", "start_frame", "end_frame"]
    missing = [key for key in required if key not in segment]
    if missing:
        raise KeyError(f"Segment missing required keys: {missing}")

    start = int(segment["start_frame"])
    end = int(segment["end_frame"])
    if start < 0 or end <= start or end > total_frames:
        raise ValueError(
            f"Invalid segment range start={start}, end={end}, total_frames={total_frames}"
        )


def build_clip_payload(arrays, segment, subject, clip_index):
    body_pose = arrays["bodyPose.npy"]
    body_translation = arrays["bodyTranslation.npy"]
    frame_rate = float(np.asarray(arrays["frameRate.npy"]).item())
    shape_parameters = arrays.get("shapeParameters.npy")
    gender = scalar_or_none(arrays, "gender.npy")
    codec_version = scalar_or_none(arrays, "codecVersion.npy")
    smpl_version = scalar_or_none(arrays, "smplVersion.npy")

    start = int(segment["start_frame"])
    end = int(segment["end_frame"])
    skill_id = int(segment["skill_id"])
    skill_name = normalize_skill_name(segment["skill_name"])

    payload = {
        "subject": np.array(subject),
        "skill_id": np.array(skill_id, dtype=np.int32),
        "skill_name": np.array(skill_name),
        "clip_index": np.array(clip_index, dtype=np.int32),
        "start_frame": np.array(start, dtype=np.int32),
        "end_frame": np.array(end, dtype=np.int32),
        "fps": np.array(frame_rate, dtype=np.float32),
        "body_pose": body_pose[start:end].astype(np.float32),
        "body_translation": body_translation[start:end].astype(np.float32),
    }

    if shape_parameters is not None:
        payload["shape_parameters"] = np.asarray(shape_parameters).astype(np.float32)
    if gender is not None:
        payload["gender"] = np.array(gender)
    if codec_version is not None:
        payload["codec_version"] = np.array(codec_version)
    if smpl_version is not None:
        payload["smpl_version"] = np.array(smpl_version)

    return payload, skill_id, skill_name


def resolve_output_path(output_dir, skill_id, skill_name, clip_index):
    folder = Path(output_dir) / skill_name
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{skill_id:03d}_{skill_name}_{clip_index:04d}.npz"
    return folder / filename


def main():
    parser = argparse.ArgumentParser(description="Extract skill clips from a .smpl archive.")
    parser.add_argument("smpl_path", type=str, help="Path to the .smpl archive")
    parser.add_argument(
        "--segments-json",
        type=str,
        required=True,
        help="Path to a JSON file containing a list of clip segments",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save extracted clip .npz files",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Override subject name; default uses the .smpl filename stem",
    )
    parser.add_argument(
        "--inclusive-end",
        action="store_true",
        help="Treat end_frame in the segments JSON as inclusive",
    )
    args = parser.parse_args()

    arrays = load_smpl_archive(args.smpl_path)
    if "bodyPose.npy" not in arrays or "bodyTranslation.npy" not in arrays:
        raise KeyError("The .smpl archive must contain bodyPose.npy and bodyTranslation.npy")

    total_frames = arrays["bodyPose.npy"].shape[0]
    subject = args.subject or Path(args.smpl_path).stem
    segments = load_segments(args.segments_json)

    skill_counts = {}
    saved = []

    for raw_segment in segments:
        segment = dict(raw_segment)
        if args.inclusive_end:
            segment["end_frame"] = int(segment["end_frame"]) + 1

        validate_segment(segment, total_frames)
        key = (int(segment["skill_id"]), normalize_skill_name(segment["skill_name"]))
        clip_index = skill_counts.get(key, 0) + 1
        skill_counts[key] = clip_index

        payload, skill_id, skill_name = build_clip_payload(arrays, segment, subject, clip_index)
        output_path = resolve_output_path(args.output_dir, skill_id, skill_name, clip_index)
        np.savez_compressed(output_path, **payload)
        saved.append(str(output_path))

    print(f"saved_clips: {len(saved)}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()

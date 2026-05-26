# Data Layout

## Purpose

This document defines the local `data/` layout used for table-tennis preprocessing.

The goal is to separate:

- raw source material
- manual annotations
- intermediate conversion artifacts
- final training artifacts

The `data/` folder should be treated as a **local workspace**.  
The final SkillMimic training files should still live under:

- `skillmimic/data/motions/`

---

## Top-Level Structure

```text
data/
  raw/
  annotations/
  clips/
  source_joints/
  canonical/
  archive/
```

---

## Folder Meanings

### `data/raw/`

Stores original source material and source-format reconstructions.

Each subfolder represents one source package or capture session.

Current examples:

- `data/raw/serve_01/`
- `data/raw/match4_001_forehand/`
- `data/raw/match4_001_legacy/`

Recommended internal structure:

```text
raw/<session>/
  media/
  world/
  smpl/
```

Example:

```text
data/raw/serve_01/
  media/serve.mov
  world/results.pkl
  world/world4d.glb
  world/world4d.mcs
  world/spec_calib/0000.jpg
  smpl/subject-1.smpl
  smpl/subject-2.smpl
```

### `data/annotations/`

Stores human-made segmentation metadata.

Example:

```text
data/annotations/serve_01/subject-1_serve_segments.json
data/annotations/match4_001_forehand/subject-1_forehand_segments.json
```

### `data/clips/`

Stores skill-specific clips cut from source `.smpl` data.

Example:

```text
data/clips/serve_01/subject-1/serve/001_serve_0001.npz
data/clips/match4_001_forehand/subject-1/forehand/002_forehand_0001.npz
```

### `data/source_joints/`

Stores extracted `source_joints [T, 22, 3]` used by retargeting.

Example:

```text
data/source_joints/serve_01/subject-1/001_serve_0001.npy
data/source_joints/match4_001_forehand/subject-1/002_forehand_0001.npy
```

### `data/canonical/`

Stores canonicalized intermediate motion clips after retarget preprocessing.

Example:

```text
data/canonical/serve_01/subject-1/001_serve_0001.npz
data/canonical/match4_001_forehand/subject-1/002_forehand_0001.npz
```

### `data/archive/`

Stores compressed source packages and large archives that are not part of the active working tree.

Example:

```text
data/archive/tabletennis_serve.zip
data/archive/match4_001_smplx.zip
```

---

## Naming Rules

### Rule 1: Raw data is organized by source session, not by skill

Do this:

- `data/raw/serve_01/`
- `data/raw/match4_001_forehand/`

Avoid this:

- `data/Serve/...`
- `data/Forehand/...`

Reason:

- one source session may contain multiple skills
- `skill` should be introduced later at the clip stage, not at the raw-data stage

### Rule 2: Processing stages get separate folders

Do not mix:

- raw `.smpl`
- segment JSON
- extracted clips
- source joints
- canonical clips

Each stage should have its own directory group.

### Rule 3: Final training motions do not belong in `data/`

Final `.pt` files that SkillMimic loads for replay or training should stay in:

- `skillmimic/data/motions/TableTennis/`

That folder is the runtime dataset location.

---

## Current Session Mapping

### Serve

Source session:

- `serve_01`

Key files:

- raw source: `data/raw/serve_01/`
- annotation: `data/annotations/serve_01/subject-1_serve_segments.json`
- clip: `data/clips/serve_01/subject-1/serve/001_serve_0001.npz`
- source joints: `data/source_joints/serve_01/subject-1/001_serve_0001.npy`
- canonical: `data/canonical/serve_01/subject-1/001_serve_0001.npz`

### Forehand

Active source session:

- `match4_001_forehand`

Legacy copy retained separately:

- `match4_001_legacy`

Key files:

- raw source: `data/raw/match4_001_forehand/`
- annotation: `data/annotations/match4_001_forehand/subject-1_forehand_segments.json`
- clip: `data/clips/match4_001_forehand/subject-1/forehand/002_forehand_0001.npz`
- source joints: `data/source_joints/match4_001_forehand/subject-1/002_forehand_0001.npy`
- canonical: `data/canonical/match4_001_forehand/subject-1/002_forehand_0001.npz`

---

## Recommended Future Pattern

When adding new data, follow this pattern:

```text
data/raw/<session>/
data/annotations/<session>/
data/clips/<session>/<subject>/<skill>/
data/source_joints/<session>/<subject>/
data/canonical/<session>/<subject>/
```

Examples:

- `data/raw/backhand_01/`
- `data/annotations/backhand_01/`
- `data/clips/backhand_01/subject-1/backhand/`

---

## Practical Rule Of Thumb

If a file is:

- original input -> `raw/`
- manual segmentation metadata -> `annotations/`
- cut skill clip -> `clips/`
- retarget helper -> `source_joints/`
- retargeted intermediate -> `canonical/`
- compressed backup -> `archive/`
- final training motion -> `skillmimic/data/motions/`

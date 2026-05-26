# SMPL-X To SkillMimic Pipeline

更新日期: `2026-05-26`

這份文件是直接給你用的。

你手上預期只有：

- `subject-1.smpl`
- `results.pkl`

你要先準備：

- `SkillMimic-V2/models/smplx/SMPLX_NEUTRAL.npz`

---

## 1. 先準備模型檔

你先把 SMPL-X model 放在：

```text
SkillMimic-V2/models/smplx/SMPLX_NEUTRAL.npz
```

---

## 2. 預設輸入格式

你的資料夾應該長這樣：

```text
some_session/
  subject-1.smpl
  results.pkl
```

例如：

```text
SkillMimic-V2/data/archive/match4_001/
  subject-1.smpl
  results.pkl
```

---

## 3. 一鍵指令

你先進環境：

```bash
conda activate skillmimic
cd ./SkillMimic-V2
```

然後直接跑：

```bash
bash scripts/convert_smplx_session_to_skillmimic.sh \
  --session-dir data/archive/match4_001 \
  --skill-name serve \
  --output data/converted/match4_001_serve
```

建議你固定這樣放：

- `--session-dir`
  放原始輸入資料夾，例如：
  `SkillMimic-V2/data/archive/<session_name>/`

- `--output`
  放轉檔輸出資料夾，例如：
  `SkillMimic-V2/data/converted/<session_name>_<skill_name>/`

例如：

- `session-dir`
  `SkillMimic-V2/data/archive/match4_001/`

- `output`
  `SkillMimic-V2/data/converted/match4_001_serve/`

可用的 `skill-name`：

- `serve`
- `forehand`
- `backhand`

如果你的檔案不是 `subject-1`，就補：

```bash
--subject subject-2
```

---

## 4. 這個腳本會自動做什麼

它會自動做兩步：

1. 先產生 `segments.json`
2. 再把 `.smpl + results.pkl` 轉成 SkillMimic final `.pt`

`segments.json` 會自動用整段 clip：

- `start_frame = 0`
- `end_frame = clip_length`

例如 `--skill-name serve` 時，它會自動生成：

```json
[
  {
    "subject": "subject-1",
    "skill_id": 1,
    "skill_name": "serve",
    "start_frame": 0,
    "end_frame": 60
  }
]
```

---

## 5. 預期輸出

如果你指定：

```text
--output data/converted/match4_001_serve
```

輸出會是：

```text
data/converted/match4_001_serve
  generated_segments/
  clips/
  source_joints/
  canonical/
  motions/
  manifest.json
```

重點看這三個地方：

- `generated_segments/`
  自動產生的 `segments.json`

- `motions/`
  最終給 SkillMimic / Isaac Gym replay 用的 `.pt`

- `manifest.json`
  這次輸入輸出記錄

---

## 6. 成功標準

成功至少要有：

1. `data/converted/match4_001_serve/generated_segments/*.json`
2. `data/converted/match4_001_serve/motions/.../*.pt`
3. `data/converted/match4_001_serve/manifest.json`

---

## 7. replay 指令

如果你要在 Isaac Gym 看結果：

```bash
python skillmimic/run.py \
  --play_dataset \
  --test \
  --task SkillMimic1BallPlay \
  --num_envs 1 \
  --episode_length 60 \
  --cfg_env skillmimic/data/cfg/skillmimic.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
  --motion_file data/converted/match4_001_serve/motions \
  --state_init 2
```

---

## 8. 補充

如果你之後真的想手動控制：

- `segments.json` 生成腳本：  
  [generate_full_clip_segments.py](../../skillmimic/utils/generate_full_clip_segments.py)

- 主 pipeline：  
  [smplx_to_skillmimic_pipeline.py](../../skillmimic/utils/smplx_to_skillmimic_pipeline.py)

但一般情況下，你直接用：

- [convert_smplx_session_to_skillmimic.sh](../../scripts/convert_smplx_session_to_skillmimic.sh)

就夠了。

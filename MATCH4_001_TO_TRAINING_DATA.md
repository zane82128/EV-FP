# `match4_001` to SkillMimic-V2 Training Data

這份文件的目的很單純：

**把 [data/match4_001_smplx](/home/zane82128/EV/FP/SkillMimic-V2/data/match4_001_smplx) 或 [data/match4_001](/home/zane82128/EV/FP/SkillMimic-V2/data/match4_001) 轉成 SkillMimic-V2 可以拿來訓練的 `.pt`。**

這裡只考慮目前最實際的版本：

- `Phase 1`: human-only imitation
- 不做 ball tracking
- 不做球拍獨立剛體
- 先學：
  - `serve`
  - `forehand`
- `backhand` 先延後

---

## TL;DR

你現在不要直接把 `.smpl/.smplx` 當成訓練資料。

你要做的是：

```text
match4_001_smplx
  -> 挑一位球員
  -> 切成 skill clips
  -> 解出人體 motion
  -> retarget 到 SkillMimic 骨架
  -> 產生 canonical human-only clip
  -> 打包成 BallPlay-compatible [T, 337] .pt
  -> 用 --play_dataset 檢查
  -> 再開始 train
```

最終要做出來的不是一個大檔，而是一批 skill clips，例如：

```text
skillmimic/data/motions/TableTennis/
  serve/001_serve_0001.pt
  forehand/002_forehand_0001.pt
```

目前先以：

- `serve`
- `forehand`

為主，`backhand` 之後再補。

---

## 1. 最終目標

對目前的 `Phase 1: human-only imitation`，你最後要得到：

- 一批 `BallPlay-style` 的 `.pt`
- 每個檔案都是：

```python
shape = [T, 337]
```

其中：

- 人體欄位要正確：
  - `root_pos`
  - `root_rot_3d`
  - `dof_pos`
  - `body_pos`
- 球欄位先用 dummy 值：
  - `obj_pos`
  - `obj_rot_3d`
  - `contact`

---

## 2. 你現在手上的來源資料

目前最推薦優先用：

- [data/match4_001_smplx/match4_001/subject-1.smpl](/home/zane82128/EV/FP/SkillMimic-V2/data/match4_001_smplx/match4_001/subject-1.smpl)
- [data/match4_001_smplx/match4_001/subject-2.smpl](/home/zane82128/EV/FP/SkillMimic-V2/data/match4_001_smplx/match4_001/subject-2.smpl)

這些檔案目前 inspect 出來的核心欄位是：

- `bodyTranslation.npy: (250, 3)`
- `bodyPose.npy: (250, 22, 3)`
- `shapeParameters.npy: (10,)`
- `frameRate.npy: 30.0`

這代表：

- 你手上有人體 motion 參數
- 但你手上**還沒有** SkillMimic 直接能吃的：
  - `dof_pos (T, 52, 3)`
  - `body_pos (T, 53, 3)`
  - `obj_pos`
  - `contact`

---

## 3. 最終 `.pt` 欄位格式

目前建議先完全相容現有 BallPlay loader。

單幀 layout：

| 欄位 | index | dim | Phase 1 狀態 |
|---|---|---:|---|
| `root_pos` | `0:3` | 3 | 要正確 |
| `root_rot_3d` | `3:6` | 3 | 要正確 |
| `reserved_a` | `6:9` | 3 | 先填 0 |
| `dof_pos` | `9:165` | 156 | 要正確 |
| `body_pos` | `165:324` | 159 | 要正確 |
| `obj_pos` | `324:327` | 3 | dummy |
| `obj_rot_3d` | `327:330` | 3 | dummy |
| `reserved_b` | `330:336` | 6 | 先填 0 |
| `contact` | `336:337` | 1 | dummy |

---

## 4. 整體流程圖

```text
subject-1.smpl / subject-2.smpl
  -> Step 1. 選來源與檢查資料
  -> Step 2. 切出 serve / forehand
  -> Step 3. 解出 SMPL/SMPL-X 人體 motion
  -> Step 4. retarget 到 SkillMimic 骨架
  -> Step 5. 產生 canonical clip
  -> Step 6. 打包成 [T, 337] .pt
  -> Step 7. replay 驗證
  -> Step 8. 再開始訓練
```

---

## 5. 實作 SOP

### Step 1. 選來源與檢查資料

**目的**

- 先固定只用一位球員
- 確認 `.smpl` 檔內容正常

**建議輸入**

- [subject-1.smpl](/home/zane82128/EV/FP/SkillMimic-V2/data/match4_001_smplx/match4_001/subject-1.smpl)

**工具**

- [inspect_smpl.py](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/utils/inspect_smpl.py)

**建議命令**

```bash
cd /home/zane82128/EV/FP/SkillMimic-V2

python skillmimic/utils/inspect_smpl.py \
  data/match4_001_smplx/match4_001/subject-1.smpl
```

**完成條件**

- 看得到 `frameCount`
- 看得到 `frameRate`
- 看得到 `bodyTranslation`
- 看得到 `bodyPose`

**Checklist**

- [x] 已決定先用 `subject-1` 或 `subject-2`
- [x] 已確認這段資料幀數正常
- [x] 已確認 `bodyPose` / `bodyTranslation` 存在

---

### Step 2. 切出 skill clips

**目的**

- 不拿整段 250 幀直接訓練
- 先切出單一動作 skill

**你要做的事**

最小可行版本不一定要一次標 3 段。

如果你手上的來源本質上只是一段單一 skill，那可以先只做 1 段。

例如：

- `Serve/01` 這種已經剪好的發球片段
- 或 `match4_001` 裡你只先挑一段 `forehand`

都可以先各自做成一段 clip。

目前最小版本可以是其中一種：

- `001 serve`

或：

- `002 forehand`

如果之後資料更多，再擴充成：

- `001 serve`
- `002 forehand`
- `003 backhand`

如果你現在用的是已經剪好的 `serve` 片段，而且該 `.smpl` 的 `frameCount = 60`，那 metadata 可以長這樣：

```python
[
    {"subject": "subject-1", "skill_id": 1, "skill_name": "serve", "start_frame": 0, "end_frame": 60}
]
```

如果你現在用的是 `match4_001` 裡的一段 `forehand`，那 metadata 可以長這樣：

```python
[
    {"subject": "subject-1", "skill_id": 2, "skill_name": "forehand", "start_frame": 0, "end_frame": 250}
]
```

多段版本的 metadata 也可以長這樣：

```python
[
    {"subject": "subject-1", "skill_id": 1, "skill_name": "serve", "start_frame": 40, "end_frame": 85},
    {"subject": "subject-1", "skill_id": 2, "skill_name": "forehand", "start_frame": 110, "end_frame": 145},
    {"subject": "subject-1", "skill_id": 3, "skill_name": "backhand", "start_frame": 165, "end_frame": 195},
]
```

**輸出**

- 一份 clip metadata

**完成條件**

- 每段 clip 都只有一個主要動作
- 每段 clip 的 frame range 明確

**Checklist**

- [x] 已建立至少一段主要 skill 的 metadata
- [x] 如果目前是已經剪好的發球片段，已建立一段 `serve` metadata
- [x] 如果目前是已經剪好的正手片段，已建立一段 `forehand` metadata
- [ ] 已人工確認這段 clip 的主要動作確實和 `skill_name` 一致
- [x] 目前 scope 已確認先做 `serve + forehand`
- [x] `backhand` 之後若需要再補
- [x] 每段都有 `start_frame` / `end_frame`

**對應腳本**

- [extract_smpl_clip.py](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/utils/extract_smpl_clip.py)

---

### Step 3. 解出 SMPL/SMPL-X 人體 motion

**目的**

- 把 `bodyPose + bodyTranslation + shapeParameters`
- 轉成可以操作的 3D 人體 joints / pose

**重點**

`bodyPose` 不是 joint xyz。  
這一步不是手動做，而是用現成的 SMPL / SMPL-X body model implementation 去解。

**輸入**

- `bodyPose`
- `bodyTranslation`
- `shapeParameters`

**輸出**

至少要能得到：

- 每幀的人體 joints
- 每幀的人體世界座標姿態

**完成條件**

- 你已經有可以用來做 retarget 的人體骨架結果

**Checklist**

- [x] 已能從 `.smpl` 讀出 `bodyPose`
- [x] 已能從 `.smpl` 讀出 `bodyTranslation`
- [x] 已能從 `.smpl` 讀出 `shapeParameters`
- [x] 已能還原每幀人體 joints / pose
- Step 3 outputs: `data/source_joints_subject1/001_serve_0001.npy`
- Step 3 outputs: `data/source_joints_subject1/002_forehand_0001.npy`

**注意**

這一步仍然需要你自己的 SMPL / SMPL-X body model 解碼流程。  
目前 repo 內沒有直接把 `.smpl` 解成 joints 的現成模型程式。

---

### Step 4. 做 `SMPL/SMPL-X -> SkillMimic` retarget

**目的**

- 把來源人體骨架轉成 SkillMimic humanoid 骨架

**這一步最重要**

因為：

- 來源只有 `22` 個 body joints
- SkillMimic 最終要：
  - `52 joints`
  - `53 bodies`

你要建立：

- `source joints -> SkillMimic joints` mapping

**Phase 1 建議**

- 軀幹、肩膀、手臂、腿：認真對
- 手指：先全部補 0 或固定 pose

**這一步最後要產生**

- `dof_pos (T, 52, 3)`  
  目標骨架的 local rotation rotvec
- `body_pos (T, 53, 3)`  
  目標骨架的 world position

**注意**

- 不要直接把來源 joints reshape 成 `52*3`
- 不要直接把來源 xyz 當成 SkillMimic `body_pos`

**完成條件**

- 已經有 `dof_pos`
- 已經有 `body_pos`

**Checklist**

- [x] 已建立 `source -> SkillMimic` joint mapping
- [x] 已決定手指先怎麼補
- [x] 已產生 `dof_pos (T, 52, 3)`
- [x] 已產生 `body_pos (T, 53, 3)`

**對應腳本**

- [smpl_to_canonical.py](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/utils/smpl_to_canonical.py)

**重要**

這支腳本支援兩種模式：

1. `正式模式`
   - 提供 `source_joints`
   - 這時才能真正把來源骨架 positions 映到 `body_pos`
2. `流程測試模式`
   - 加 `--allow-missing-source-joints`
   - 這時 `body_pos` 只會用非常粗略的 fallback 補值
   - 只能拿來測試 pipeline，不建議直接拿去正式訓練

---

### Step 5. 取 root 資料

**目的**

- 補齊 `.pt` 中的人體根節點資訊

**要產生**

- `root_pos (T, 3)`
- `root_rot_3d (T, 3)`

**建議來源**

- `root_pos`：先用 `bodyTranslation`
- `root_rot_3d`：先用來源 pose 中對應 root 的 rotation 候選

**注意**

- `root_rot_3d` 很容易方向不對
- 如果 replay 後人物朝向怪，就先檢查這欄

**完成條件**

- 已經有 `root_pos`
- 已經有 `root_rot_3d`

**Checklist**

- [x] 已產生 `root_pos (T, 3)`
- [x] 已產生 `root_rot_3d (T, 3)`
- [x] 已確認 root 朝向沒有明顯反掉

---

### Step 6. 先存成 canonical human-only clip

**目的**

- 不要直接一步打包 `.pt`
- 先做一層中間格式，方便 debug

**建議格式**

```python
clip = {
    "skill_id": 1,
    "skill_name": "serve",
    "fps": 30.0,
    "root_pos": ...,      # (T, 3)
    "root_rot_3d": ...,   # (T, 3)
    "dof_pos": ...,       # (T, 52, 3)
    "body_pos": ...,      # (T, 53, 3)
}
```

**完成條件**

- 你已經有一份 human-only canonical clip
- 所有人體欄位都已經準備好

**Checklist**

- [x] 已建立 canonical clip 結構
- [x] `root_pos` 已放入
- [x] `root_rot_3d` 已放入
- [x] `dof_pos` 已放入
- [x] `body_pos` 已放入
- [x] 已輸出 `data/canonical_subject1/serve/001_serve_0001.npz`
- [x] 已輸出 `data/canonical_subject1/forehand/002_forehand_0001.npz`

**對應腳本**

- [smpl_to_canonical.py](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/utils/smpl_to_canonical.py)

---

### Step 7. 補 dummy object

**目的**

- 讓資料先相容現有 BallPlay loader
- 但 Phase 1 不需要真球

**建議**

```python
obj_pos = np.tile(np.array([[2.0, 0.0, 1.0]], dtype=np.float32), (T, 1))
obj_rot_3d = np.zeros((T, 3), dtype=np.float32)
contact = np.zeros((T, 1), dtype=np.float32)
```

**完成條件**

- 即使沒有球，也能先包成 `[T, 337]`

**Checklist**

- [x] `obj_pos` 已補 dummy 值
- [x] `obj_rot_3d` 已補 0
- [x] `contact` 已補 0

---

### Step 8. 打包成 `[T, 337]`

**目的**

- 產生目前 repo 可以直接讀的 training `.pt`

**範例**

```python
import numpy as np
import torch

def pack_human_only_pt(root_pos, root_rot_3d, dof_pos, body_pos):
    T = root_pos.shape[0]

    obj_pos = np.tile(np.array([[2.0, 0.0, 1.0]], dtype=np.float32), (T, 1))
    obj_rot_3d = np.zeros((T, 3), dtype=np.float32)
    contact = np.zeros((T, 1), dtype=np.float32)

    motion = np.zeros((T, 337), dtype=np.float32)
    motion[:, 0:3] = root_pos
    motion[:, 3:6] = root_rot_3d
    motion[:, 9:165] = dof_pos.reshape(T, 156)
    motion[:, 165:324] = body_pos.reshape(T, 159)
    motion[:, 324:327] = obj_pos
    motion[:, 327:330] = obj_rot_3d
    motion[:, 336:337] = contact

    return torch.from_numpy(motion)
```

**輸出檔名建議**

```text
skillmimic/data/motions/TableTennis/
  serve/001_serve_0001.pt
  forehand/002_forehand_0001.pt
```

之後若要擴充，再補：

```text
skillmimic/data/motions/TableTennis/
  backhand/003_backhand_0001.pt
```

**完成條件**

- 已產生至少 2 個 `.pt`
- 檔名符合 `skill_id + skill_name` 規則

**Checklist**

- [x] 已成功打包 `001_serve_0001.pt`
- [x] 已成功打包 `002_forehand_0001.pt`
- [ ] 若之後擴充，再打包 `003_backhand_0001.pt`
- [x] 每個 `.pt` shape 都是 `[T, 337]`
- [x] `validate_motion_pt.py` 已通過 `001_serve_0001.pt`
- [x] `validate_motion_pt.py` 已通過 `002_forehand_0001.pt`
- [x] 已輸出 `skillmimic/data/motions/TableTennis/serve/001_serve_0001.pt`
- [x] 已輸出 `skillmimic/data/motions/TableTennis/forehand/002_forehand_0001.pt`

**對應腳本**

- [canonical_to_skillmimic_pt.py](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/utils/canonical_to_skillmimic_pt.py)

---

### Step 9. 用 `--play_dataset` replay

**目的**

- 先看資料能不能正常播放
- 不要先急著 train

**命令**

```bash
LD_LIBRARY_PATH=/home/zane82128/miniconda3/envs/skillmimic/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH} \
PATH=/home/zane82128/miniconda3/envs/skillmimic/bin:${PATH} \
/home/zane82128/miniconda3/envs/skillmimic/bin/python skillmimic/run.py \
  --play_dataset \
  --test \
  --task SkillMimic1BallPlay \
  --num_envs 1 \
  --episode_length 60 \
  --cfg_env skillmimic/data/cfg/skillmimic.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
  --motion_file skillmimic/data/motions/TableTennis/serve \
  --state_init 2 \
  --save_images

LD_LIBRARY_PATH=/home/zane82128/miniconda3/envs/skillmimic/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH} \
PATH=/home/zane82128/miniconda3/envs/skillmimic/bin:${PATH} \
/home/zane82128/miniconda3/envs/skillmimic/bin/python skillmimic/run.py \
  --play_dataset \
  --test \
  --task SkillMimic1BallPlay \
  --num_envs 1 \
  --episode_length 250 \
  --cfg_env skillmimic/data/cfg/skillmimic.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
  --motion_file skillmimic/data/motions/TableTennis/forehand \
  --state_init 2 \
  --save_images
```

**你要檢查**

- 人沒有炸掉
- root 高度合理
- 朝向合理
- 動作像 `serve / forehand`

**完成條件**

- 這些 `.pt` 可以正常 replay

**Checklist**

- [x] replay 時人物沒有爆炸
- [x] root 高度合理
- [x] 人物朝向合理
- [x] `serve` 看起來像發球動作
- [x] `forehand` 看起來像正手動作
- [ ] 若之後擴充，再確認 `backhand` 看起來像反手動作

**目前狀態（2026-05-21）**

- 已在 GPU 上重新執行 replay，Isaac Gym / GPU PhysX 正常啟動
- `serve` 最新 replay 輸出影像：`skillmimic/data/images/20260521_155403/`
- `forehand` 最新 replay 輸出影像：`skillmimic/data/images/20260521_155437/`
- 最新 replay 預覽：`skillmimic/data/replay_preview/serve_replay.gif`
- 最新 replay 預覽：`skillmimic/data/replay_preview/forehand_replay.gif`
- 已修正 retarget / DOF 對應，不再直接複製 SMPL-X rotvec 到 SkillMimic target skeleton
- 已微調 `Chest / Neck / Head` 的 asset 外觀比例，脖子區域視覺上較合理
- 結論：Step 9 以目前 `serve + forehand` scope 判定完成，可進 Step 10

---

### Step 10. 再開始 training

**目的**

- 只有在 replay 正常後，才進訓練

**Phase 1 訓練建議**

- 先做 human-only imitation
- object / contact 類 reward 權重先關掉或設很低

**完成條件**

- 已經有可播的資料
- 再開始 train

**Checklist**

- [ ] replay 已經正常
- [ ] 已確認先做 human-only imitation
- [ ] 已確認球相關 reward 先不作為主目標

---

## 6. 你需要自己做的部分 vs 現成可用的部分

### 現成可用

- `inspect_smpl.py`
- `inspect_motion_pt.py`
- 現有 BallPlay loader
- `--play_dataset` replay 檢查流程

### 需要你自己寫

- `extract_smpl_clip.py`
- `smpl_to_canonical.py`
- `canonical_to_skillmimic_pt.py`
- `source -> SkillMimic` joint mapping

---

## 7. 建議拆成 3 支腳本

### `extract_smpl_clip.py`

功能：

- 讀 `.smpl`
- 依 `start_frame / end_frame` 切出 clips
- 已完成

### `smpl_to_canonical.py`

功能：

- 還原人體 motion
- retarget 到 SkillMimic 骨架
- 產生：
  - `root_pos`
  - `root_rot_3d`
  - `dof_pos`
  - `body_pos`
- 已完成基礎版本
- 若未提供 `source_joints`，只能做 pipeline smoke test

### `canonical_to_skillmimic_pt.py`

功能：

- 補 dummy object
- 打包成 `[T, 337]`
- 已完成

---

## 8. 目前先不要做什麼

先不要做：

- ball tracking
- 球拍獨立剛體
- contact 標記
- 雙人
- 改 `.pt` shape
- 改 loader

先把 human-only replay 跑通。

---

## 9. 最後總檢查

在你開始 train 之前，請確認：

- [ ] 已選定單一球員
- [ ] 已切出 `serve / forehand`
- [ ] 已解出人體 motion
- [ ] 已完成 `source -> SkillMimic` retarget
- [ ] 已產生 canonical clip
- [ ] 已產生 `[T, 337]` `.pt`
- [ ] 已用 `--play_dataset` replay 成功

---

## 10. 一句話總結

你現在不是在做完整桌球互動資料，而是在做：

**`SMPL/SMPL-X human motion -> SkillMimic human-only training .pt`**

先把人體動作學會，再把球交給 simulator。

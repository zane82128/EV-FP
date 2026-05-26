# Table Tennis Workflow

更新日期: `2026-05-25`

這份文件是目前桌球專案的主入口。

它整合了原本分散的幾份文件：

- current progress
- replay commands
- serve learning plan

如果你現在只是想知道：

- 做到哪裡了
- replay 要怎麼跑
- 接下來怎麼讓 agent 學 serve

先看這一份就夠。

如果你現在要看的重點是：

- `phase 1 / phase 2` 該怎麼切
- reference 要怎麼設計
- reward 要怎麼拆
- contact graph 什麼時候該用

直接看：

- [TRAINING_PHASES.md](./TRAINING_PHASES.md)

---

## 1. Current Scope

目前只做兩個動作：

- `serve`
- `forehand`

`backhand` 先不做。

目前的最終 motion 檔放在：

- [skillmimic/data/motions/TableTennis/serve/001_serve_0001.pt](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/data/motions/TableTennis/serve/001_serve_0001.pt:1)
- [skillmimic/data/motions/TableTennis/forehand/002_forehand_0001.pt](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/data/motions/TableTennis/forehand/002_forehand_0001.pt:1)

---

## 2. Current Data Pipeline Status

### `serve`

- raw source: `data/raw/serve_01/`
- annotation: `data/annotations/serve_01/subject-1_serve_segments.json`
- clip: `data/clips/serve_01/subject-1/serve/001_serve_0001.npz`
- source joints: `data/source_joints/serve_01/subject-1/001_serve_0001.npy`
- canonical: `data/canonical/serve_01/subject-1/001_serve_0001.npz`
- final motion: `skillmimic/data/motions/TableTennis/serve/001_serve_0001.pt`

### `forehand`

- raw source: `data/raw/match4_001_forehand/`
- annotation: `data/annotations/match4_001_forehand/subject-1_forehand_segments.json`
- clip: `data/clips/match4_001_forehand/subject-1/forehand/002_forehand_0001.npz`
- source joints: `data/source_joints/match4_001_forehand/subject-1/002_forehand_0001.npy`
- canonical: `data/canonical/match4_001_forehand/subject-1/002_forehand_0001.npz`
- final motion: `skillmimic/data/motions/TableTennis/forehand/002_forehand_0001.pt`

### Validation Status

已完成：

- final `.pt` shape 驗證
- replay 驗證
- 球拍 asset 顯示驗證

補充：

- 詳細轉檔和驗證方法看
  [../pipeline/SMPLX_TO_SKILLMIMIC_FORMAT.md](../pipeline/SMPLX_TO_SKILLMIMIC_FORMAT.md)

---

## 3. Replay Commands

### Environment Setup

```bash
source /home/zane82128/miniconda3/etc/profile.d/conda.sh
conda activate skillmimic
cd /home/zane82128/EV/FP/SkillMimic-V2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
```

### Optional Viewer Settings

```bash
export SKILLMIMIC_VIEWER_WIDTH=960
export SKILLMIMIC_VIEWER_HEIGHT=640
```

### Optional Racket Grip Tuning

```bash
export SKILLMIMIC_RACKET_GRIP_SCALE=1.0
```

- `0.7`: 握得比較鬆
- `1.2`: 握得比較緊

如果要關掉 replay 裡的握拍補正：

```bash
export SKILLMIMIC_ENABLE_RACKET_GRIP=0
```

### Replay `serve`

```bash
python skillmimic/run.py \
  --play_dataset \
  --test \
  --task SkillMimic1BallPlay \
  --num_envs 1 \
  --episode_length 60 \
  --cfg_env skillmimic/data/cfg/skillmimic.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
  --motion_file skillmimic/data/motions/TableTennis/serve \
  --state_init 2
```

### Replay `forehand`

```bash
python skillmimic/run.py \
  --play_dataset \
  --test \
  --task SkillMimic1BallPlay \
  --num_envs 1 \
  --episode_length 250 \
  --cfg_env skillmimic/data/cfg/skillmimic.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
  --motion_file skillmimic/data/motions/TableTennis/forehand \
  --state_init 2
```

### Replay With No-Hand Asset

```bash
python skillmimic/run.py \
  --play_dataset \
  --test \
  --task SkillMimic1BallPlay \
  --num_envs 1 \
  --episode_length 60 \
  --cfg_env skillmimic/data/cfg/skillmimic.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
  --motion_file skillmimic/data/motions/TableTennis/serve \
  --state_init 2 \
  --asset_file_name mjcf/mocap_humanoid_racket_nohand.xml
```

### Save Image Frames

在任一 replay 指令後面加：

```bash
--save_images
```

---

## 4. Current Racket Asset State

目前球拍是掛在 `R_Wrist` 上的固定 asset，不是獨立動力學 rigid body。

目前有兩個版本：

- 一般版 asset:
  `skillmimic/data/assets/mjcf/mocap_humanoid.xml`
- hand-only 訓練版 asset:
  `skillmimic/data/assets/mjcf/mocap_humanoid_hand_only.xml`
- `nohand` 版 asset:
  `skillmimic/data/assets/mjcf/mocap_humanoid_racket_nohand.xml`

目前球拍已拆成兩個 mesh：

- `skillmimic/data/assets/obj/r_racket_handle.obj`
- `skillmimic/data/assets/obj/r_racket_blade.obj`

目前顏色：

- 握把：木頭棕
- 拍面：亮桃紅

---

## 5. How To Train Serve: Practical Roadmap

### Stage 1: Learn Body Serve Motion First

先不要要求 agent 真正把球發出去。

你現在手上的 `serve` reference 主要適合學：

- 發球站姿
- 揮手 / 揮拍動作
- 身體重心轉移

目前這份 `.pt` 的限制是：

- `obj_pos` 固定
- `contact` 全是 0

所以這個階段只能先學 **body motion imitation**。

#### Stage 1 Training Command

```bash
python skillmimic/run.py \
  --task SkillMimic1BallPlay \
  --num_envs 256 \
  --episode_length 60 \
  --cfg_env skillmimic/data/cfg/skillmimic.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
  --motion_file skillmimic/data/motions/TableTennis/serve \
  --state_init Random \
  --headless
```

如果 GPU 壓力太大，可以把：

- `--num_envs 256` 改成 `128`

#### Stage 1 Success Criteria

- training 沒有 NaN
- checkpoint 正常產生
- agent 不會一開始就倒地
- test 時動作開始接近 serve replay

### Stage 2: Add Ball Reference

這時才開始讓 agent 學：

- 球在手邊
- 拋球
- 擊球
- 球飛走

最小要求：

1. 球一開始在手附近
2. 中間被拋起來
3. 某幾幀接近拍面
4. 後面往前飛
5. `contact` 合理變化

### Stage 3: Add More Serve Variants

當 Stage 1 和 Stage 2 穩了之後，再補：

- 更多 serve clips
- 不同節奏
- 不同拋球高度
- 更強泛化

---

## 6. Recommended Next Step

你現在最合理的下一步是：

1. 先用目前的 `serve` 跑一次最小 training
2. 確認 agent 能學到發球身體動作
3. 再決定要不要補第一版人工 ball trajectory

不要一開始就把目標定成「完整真實發球」。

先學會：

- **serve body motion**

再學：

- **serve with ball**

---

## 7. What This File Replaces

這份文件整合了以下舊文件的主要用途：

- `TABLE_TENNIS_CURRENT_PROGRESS.md`
- `TABLE_TENNIS_REPLAY_COMMANDS.md`
- `SERVE_LEARNING_PLAN.md`

# Table Tennis Replay Commands

這份文件整理目前 `SkillMimic-V2` 桌球 replay 常用指令，目的是之後你自己就能直接跑：

- `serve`
- `forehand`
- 一般版球拍 asset
- `nohand` 版球拍 asset
- viewer 大小調整
- 握拍補正開關

## 1. 啟動前準備

每次新開 terminal 先跑：

```bash
source /home/zane82128/miniconda3/etc/profile.d/conda.sh
conda activate skillmimic
cd /home/zane82128/EV/FP/SkillMimic-V2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
```

## 2. 可選環境變數

如果 viewer 太大或位置容易跑掉，可以先設小一點：

```bash
export SKILLMIMIC_VIEWER_WIDTH=960
export SKILLMIMIC_VIEWER_HEIGHT=640
```

如果要調整 replay 時的右手握拍補正：

```bash
export SKILLMIMIC_RACKET_GRIP_SCALE=1.0
```

- `0.7`：握得比較鬆
- `1.2`：握得比較緊

如果要關掉 replay 時加上的右手握拍 preset：

```bash
export SKILLMIMIC_ENABLE_RACKET_GRIP=0
```

恢復預設開啟：

```bash
export SKILLMIMIC_ENABLE_RACKET_GRIP=1
```

## 3. `serve` Replay

一般版球拍 asset：

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

右手隱藏、只留球拍的 `nohand` 版：

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

## 4. `forehand` Replay

一般版球拍 asset：

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

右手隱藏、只留球拍的 `nohand` 版：

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
  --state_init 2 \
  --asset_file_name mjcf/mocap_humanoid_racket_nohand.xml
```

## 5. 存圖版本

如果要把 viewer frame 存成圖片序列，就在命令最後加：

```bash
--save_images
```

圖片會存到：

```bash
skillmimic/data/images/<timestamp>/
```

## 6. 目前可用的 asset

一般版：

```text
skillmimic/data/assets/mjcf/mocap_humanoid.xml
```

右手隱藏、只留球拍版：

```text
skillmimic/data/assets/mjcf/mocap_humanoid_racket_nohand.xml
```

## 7. viewer 行為

目前 `--play_dataset` 已經改成會一直 loop 播放，不會播一輪就自動結束。

停止方式：

- 關閉 Isaac Gym viewer
- terminal 按 `Ctrl+C`

## 8. 常見情況

### viewer 跑太大或位置怪

先試：

```bash
export SKILLMIMIC_VIEWER_WIDTH=960
export SKILLMIMIC_VIEWER_HEIGHT=640
```

### 右手 `nohand` 版出現 `Could not load visual geom`

這是目前 Isaac Gym 對隱藏右手 visual 的 warning。  
目前 replay 仍可正常播放，如果畫面正常，可以先忽略。

### 想切回一般版

只要把：

```bash
--asset_file_name mjcf/mocap_humanoid_racket_nohand.xml
```

整段拿掉即可。

## 9. 相關檔案

- 一般球拍 asset: [mocap_humanoid.xml](skillmimic/data/assets/mjcf/mocap_humanoid.xml)
- `nohand` asset: [mocap_humanoid_racket_nohand.xml](skillmimic/data/assets/mjcf/mocap_humanoid_racket_nohand.xml)
- 球拍 mesh: [r_racket.obj](skillmimic/data/assets/obj/r_racket.obj)
- replay task: [skillmimic1.py](skillmimic/env/tasks/skillmimic1.py)

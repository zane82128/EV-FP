# Table Tennis Current Progress

更新日期: `2026-05-24`

這份文件整理目前 `SkillMimic-V2` 桌球專案的實際進度，重點放在：

- `SMPL-X -> SkillMimic` 轉檔
- `serve / forehand` replay
- 球拍 asset
- 目前還沒做的事

## 1. 目前 scope

目前只做兩個動作：

- `serve`
- `forehand`

`backhand` 先不做。

## 2. 原始資料狀態

### `serve`

來源資料：

- `data/Serve/01/tabletennis_serve/subject-1.smpl`
- `data/Serve/01/tabletennis_serve/subject-1_serve_segments.json`

已切出 clip：

- `data/Serve/01/extracted_clips_subject1/serve/001_serve_0001.npz`

### `forehand`

來源資料：

- `data/Forehand/match4_001_smplx/match4_001/subject-1.smpl`
- `data/Forehand/match4_001_smplx/match4_001/subject-1_forehand_segments.json`

已切出 clip：

- `data/Forehand/match4_001_smplx/extracted_clips_subject1/forehand/002_forehand_0001.npz`

## 3. 轉檔 pipeline 狀態

目前 `serve` 和 `forehand` 都已經走到可 replay 的 `.pt`。

### Step 3: source joints

已完成 `source_joints [T, 22, 3]` 輸出：

- `data/source_joints_subject1/001_serve_0001.npy`
- `data/source_joints_subject1/002_forehand_0001.npy`

使用工具：

- `skillmimic/utils/results_pkl_to_source_joints.py`

### Step 4-7: canonical clip

已完成 canonical human-only clip：

- `data/canonical_subject1/serve/001_serve_0001.npz`
- `data/canonical_subject1/forehand/002_forehand_0001.npz`

使用工具：

- `skillmimic/utils/smpl_to_canonical.py`

目前 `smpl_to_canonical.py` 已不是舊版的直接 copy rotvec，而是有做：

- 座標系轉換
- source joints 到 target skeleton 的 retarget
- target skeleton FK

### Step 8: final SkillMimic `.pt`

已完成最終訓練 / replay 格式：

- `skillmimic/data/motions/TableTennis/serve/001_serve_0001.pt`
- `skillmimic/data/motions/TableTennis/forehand/002_forehand_0001.pt`

使用工具：

- `skillmimic/utils/canonical_to_skillmimic_pt.py`

### 驗證工具

已補上 validator：

- `skillmimic/utils/validate_motion_pt.py`

用途：

- 檢查 `[T, 337]`
- 檢查 NaN / Inf
- 檢查 root height
- 檢查 pelvis / root 對齊
- 檢查 dummy object / contact 是否合理

## 4. Replay 狀態

### 已完成

- `serve` 可在 Isaac Gym replay
- `forehand` 可在 Isaac Gym replay
- `--play_dataset` 已改成會 loop，不會播一輪就自動結束

相關修改：

- `skillmimic/learning/skillmimic_players.py`

### 目前判斷

現在的 replay 已經不是早期那種：

- 躺平
- 翻轉
- 完全不像原始動作

目前大動作語義大致正確，可以看出：

- `serve`
- `forehand`

### 仍然存在的限制

- 手部細節仍不自然
- 右手持拍姿勢仍是近似修正，不是真實影片追出的精準手型
- 目前右手握拍姿勢有額外 replay 補正，還沒有正式 bake 回所有 training motion

## 5. 球拍 asset 狀態

### 已完成

目前球拍不是從真實影片追出來的，而是直接掛在 `R_Wrist` 上。

球拍已拆成兩個 mesh：

- 握把：`skillmimic/data/assets/obj/r_racket_handle.obj`
- 拍面：`skillmimic/data/assets/obj/r_racket_blade.obj`

掛載 asset：

- 一般版：`skillmimic/data/assets/mjcf/mocap_humanoid.xml`
- 右手隱藏版：`skillmimic/data/assets/mjcf/mocap_humanoid_racket_nohand.xml`

### 目前顏色

- 握把：木頭棕
- 拍面：亮桃紅

### 目前可用版本

1. 一般版

- 右手保留
- 球拍保留

2. `nohand` 版

- 右手幾何隱藏
- 球拍保留
- 用來避免手掌和球拍穿插太明顯

### 注意

`nohand` 版目前可能在 log 裡出現 `Could not load visual geom` warning。  
目前 replay 仍可正常播放，但這個 asset 還不是最乾淨的版本。

## 6. 目前常用文件

### 轉檔解釋

- `SMPLX_TO_SKILLMIMIC_FORMAT.md`

內容：

- 怎麼從 `SMPL-X / results.pkl` 轉成 SkillMimic `.pt`
- 做了哪些格式對齊
- retarget 的思路

### Replay 指令

- `TABLE_TENNIS_REPLAY_COMMANDS.md`

內容：

- conda 啟動方式
- `serve / forehand` replay 指令
- 一般版 / `nohand` 版 asset 切換
- viewer 大小設定
- 握拍補正開關

### 驗證工具說明

- `ADDED_VALIDATION_TOOLS.md`

## 7. 目前技術修改摘要

已改過的重要程式：

- `skillmimic/utils/results_pkl_to_source_joints.py`
- `skillmimic/utils/smpl_to_canonical.py`
- `skillmimic/utils/validate_motion_pt.py`
- `skillmimic/learning/skillmimic_players.py`
- `skillmimic/env/tasks/skillmimic1.py`
- `skillmimic/env/tasks/humanoid_task.py`

已加過的重要 asset / 文件：

- `skillmimic/data/assets/obj/r_racket_handle.obj`
- `skillmimic/data/assets/obj/r_racket_blade.obj`
- `skillmimic/data/assets/mjcf/mocap_humanoid_racket_nohand.xml`
- `TABLE_TENNIS_REPLAY_COMMANDS.md`
- `SMPLX_TO_SKILLMIMIC_FORMAT.md`

## 8. 目前尚未完成

### Step 10: training

還沒開始正式訓練。

也就是說目前完成的是：

- 轉檔
- replay
- asset / viewer 調整

但還沒有做：

- 正式 train
- reward / learning curve 檢查
- rollout 品質驗證

### 手部精度

目前最大的視覺限制還是右手：

- 握拍姿勢仍不夠自然
- 手指沒有真實 racket grasp 標註
- 目前是靠 replay 補正和 asset 遮蓋在改善觀感

### 球與球拍互動

目前球拍是固定掛在右手腕上的 asset。  
還沒有做：

- 真正獨立動力學球拍
- 真正球拍軌跡 supervision
- 真實球拍 contact reference

## 9. 建議下一步

如果繼續往前做，優先順序建議是：

1. 選定你要用哪個 asset 版本

- 一般版
- `nohand` 版

2. 把現在滿意的 viewer / replay 狀態固定下來

3. 開始做短訓練 smoke test

- 先看 train 會不會崩
- 再看 reward / reset / imitation 品質

4. 如果未來要更真實的持拍

- 再考慮把 grip pose 正式 bake 回 motion pipeline
- 或做更完整的 racket-specific retarget / annotation

## 10. 一句話總結

目前已經完成：

- `SMPL-X -> SkillMimic` 轉檔
- `serve / forehand` replay
- 桌球拍 asset
- 一般版與 `nohand` 版顯示方案

目前還沒完成：

- 正式訓練
- 高品質右手握拍姿勢
- 真實球拍 supervision

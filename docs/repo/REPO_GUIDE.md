# SkillMimic-V2 Repo Guide

這份文件的目標不是解釋論文方法細節，而是幫你先建立一個最基本的 repo 地圖：

- 這個專案的主要入口在哪
- 各個資料夾在做什麼
- 如果你想找訓練、資料、場景、模型，應該先看哪裡

---

## 1. 先用一句話理解這個 repo

`SkillMimic-V2` 是一個基於 `Isaac Gym + rl_games` 的 humanoid imitation / interaction learning 專案。

它主要做三件事：

1. 讀取 reference motion 資料
2. 在 simulator 中建立 humanoid 與 object 的互動場景
3. 用 RL / imitation reward 訓練 policy

---

## 2. 建議的閱讀順序

如果你是第一次看這個 repo，建議照這個順序：

1. `README.md`
2. `skillmimic/run.py`
3. `skillmimic/env/tasks/`
4. `skillmimic/utils/`
5. `skillmimic/data/`
6. `models/`、`hist_encoder/`

這樣你會先知道：

- 怎麼跑
- 跑的是哪個 task
- task 讀什麼資料
- 模型和 checkpoint 放哪裡

---

## 3. 頂層目錄在做什麼

### `README.md`
官方使用說明。

主要包含：

- 安裝方式
- dataset replay 指令
- BallPlay / Locomotion 的 inference 與 training 指令
- history encoder 與 offline preprocess 的流程

如果你想先 reproduce 原作者流程，先看這個檔。

### `Yu et al. - 2025 - SkillMimic-V2 ... .pdf`
這個 repo 對應的論文。

### `requirements.txt`
V2 額外依賴，主要只有：

- `six`
- `pytorch-lightning==1.9.0`

### `models/`
預訓練 policy checkpoint。

目前可以看到：

- `models/BallPlay/DeepMimic/model.pth`
- `models/BallPlay/SkillMimic/model.pth`
- `models/BallPlay/SkillMimic-V2/model.pth`
- `models/Locomotion/model.pth`
- `models/Matchup/model.pth`

### `hist_encoder/`
history encoder checkpoint。

目前有：

- `hist_encoder/BallPlay/hist_model.ckpt`
- `hist_encoder/Locomotion/hist_model.ckpt`

### `images/`
README 和專案展示圖。

### `data/`
這個不是原作者核心 training data 目錄，而是你們自己額外放的資料區。

目前最重要的是：

- `data/raw/match4_001_legacy/`

這是一段桌球影片經過 3D 重建之後的中間資料包，不是 SkillMimic 可直接訓練的 `.pt`。

### `docs/data/DATA_FORMAT.md`
你們目前自己整理的桌球資料格式規格。

主要在講：

- BallPlay-style `.pt` 要長什麼樣
- 桌球資料要如何轉成 SkillMimic 可吃的格式

### `docs/table_tennis/TABLE_TENNIS_WORKFLOW.md`
你們目前桌球專案的主 runbook。

---

## 4. `skillmimic/` 是整個專案的核心

這個資料夾裡面放的是：

- 入口程式
- task 定義
- 訓練 agent / network
- 資料 loader
- config

可以把它理解成真正的程式本體。

---

## 5. 最重要的入口：`skillmimic/run.py`

這是整個專案最常用的執行入口。

它負責：

- 解析 command line arguments
- 載入 env config 與 train config
- 建立 task / env
- 註冊 rl_games 的 agent、player、network
- 啟動 training 或 inference

你在 README 看到的大部分指令，最後都是進這支。

如果你想知道：

- `--task` 是怎麼接進程式的
- `--cfg_env` / `--cfg_train` 怎麼被讀
- training 為什麼會跑起來

先看這支。

如果你想直接看：

- `run.py` 的執行流程
- 哪些 arguments 最常用
- 哪些 arguments 可以先忽略

看：

- [run_process.md](./run_process.md)

---

## 6. `skillmimic/env/tasks/`：環境與 task 定義

這個資料夾決定：

- simulator 裡有哪些角色與物體
- observation 怎麼組
- reward 怎麼算
- reset / replay / state init 怎麼做

這是你之後如果要加桌球 task，最可能要改的地方。

### 幾個重要檔案

#### `humanoid_task.py`
最底層 humanoid task 基底類別。

主要負責：

- humanoid asset 載入
- humanoid observation
- body / dof / contact 等基礎資訊

#### `humanoid_object_task.py`
在 humanoid 上再加 object 的版本。

如果你的 task 有球、桌子、物體互動，這支很重要。

桌球未來如果要加：

- 球
- 桌面
- 球網

通常會參考這支。

#### `skillmimic1.py`
SkillMimic baseline task 的核心。

主要負責：

- reference motion 載入後的使用
- observation 組合
- imitation reward 的主要邏輯

#### `skillmimic1_unified.py`
SkillMimic 的 unified HOI imitation reward 版本。

如果你在看：

- body reward
- object reward
- relative motion reward
- contact reward

這支值得看。

#### `skillmimic1_reweight.py`
加入 adaptive trajectory sampling / reweight 的版本。

#### `skillmimic1_rand.py`
加入 state randomization / dataset replay / reset 邏輯的重要檔。

#### `skillmimic1_hist.py`
加入 history encoder 的版本。

#### `skillmimic2.py`
V2 BallPlay task 的封裝入口。

#### `offline_state_search.py`
做 STG / offline preprocess 的 task。

README 裡建 graph 的步驟就是跑這個。

#### `skillmimic_parahome.py`
ParaHome 家務操作任務的主要 task。

#### `skillmimic_parahome_multiobj.py`
多物體版本的 ParaHome task。

---

## 7. `skillmimic/learning/`：agent、network、player

這裡主要是和 `rl_games` 對接的學習程式碼。

如果 `env/tasks` 決定的是「環境怎麼長」，那 `learning/` 決定的是：

- policy 怎麼建
- training loop 怎麼跑
- inference player 怎麼控制 env

### 幾個重要檔案

#### `skillmimic_agent.py`
SkillMimic 訓練用 agent。

#### `skillmimic_players.py`
SkillMimic inference / replay player。

像 `--play_dataset` 的邏輯，就是從這邊進去 dataset replay。

#### `skillmimic_network_builder.py`
policy / value network 的組裝。

#### `skillmimic_models.py`
和 network 對接的 model wrapper。

#### `hrl_*`
高階控制 / match policy 相關。

#### `amp_*`
AMP 相關 baseline。

如果你目前只是要理解 BallPlay / SkillMimic-V2 主線，先不用深讀 `hrl_*` 和 `amp_*`。

---

## 8. `skillmimic/utils/`：資料、config、輔助工具

這個資料夾非常重要，因為很多「資料到底怎麼被讀進來」都在這裡。

### 幾個最重要的檔案

#### `config.py`
定義 command line arguments 與 config 載入邏輯。

例如：

- `--play_dataset`
- `--motion_file`
- `--build_blender_motion`
- `--state_init`

都在這裡定義。

#### `parse_task.py`
把 `--task` 轉成實際 Python task 類別。

#### `motion_data_handler.py`
BallPlay / Locomotion 的資料 loader。

如果你要知道：

- `.pt` 的 shape 與欄位
- skill id 怎麼從檔名取出
- raw motion 怎麼轉成 internal `hoi_data`

這支最重要。

#### `paramotion_data_handler.py`
ParaHome 的資料 loader。

#### `paramotion_data_handler_multiobj.py`
多物體版本 loader。

#### `history_encoder.py`
history encoder 的模型定義。

#### `state_prediction_ballplay.py`
BallPlay history encoder 的訓練腳本。

#### `state_prediction_parahome.py`
ParaHome history encoder 的訓練腳本。

#### `make_video.py`
把儲存下來的圖片串成影片。

---

## 9. `skillmimic/data/`：config、assets、motions、preprocess

這是原作者專案內建資料最集中的地方。

### `skillmimic/data/cfg/`
環境 config。

常見檔案：

- `skillmimic.yaml`
- `skillmimic_test.yaml`
- `skillmimic_parahome.yaml`
- `amp.yaml`
- `deepmimic.yaml`

如果你想調：

- episode length
- object properties
- state init
- task 參數

通常要看這裡。

### `skillmimic/data/cfg/train/rlg/`
訓練 config。

主要是 `rl_games` 的超參數設定。

常見檔案：

- `skillmimic.yaml`
- `parahome.yaml`
- `amp.yaml`
- `hrl_humanoid_virtual.yaml`

如果你想調：

- learning rate
- PPO 參數
- batch / minibatch
- network 設定

看這裡。

### `skillmimic/data/assets/`
模擬環境用 asset。

分成兩大類：

- `assets/mjcf/`：humanoid skeleton / body asset
- `assets/urdf/`、`assets/obj/`：球或家務物體

例子：

- `assets/mjcf/mocap_humanoid.xml`
- `assets/urdf/ball.urdf`
- `assets/obj/book.urdf`
- `assets/obj/kettle.urdf`

如果你未來要加桌球：

- `pingpong_ball.urdf`
- 桌子 / 球網的 asset 或 static box

會和這區有關。

### `skillmimic/data/motions/`
reference motion 資料。

目前主要有：

- `BallPlay/`
- `BallPlay-Pick/`
- `Locomotion/`
- `ParaHome/`

`BallPlay/` 裡的 `.pt` 是最值得參考的 raw training data 形式。

例子：

- `BallPlay/shot/009_006pickle_dribble2shot.pt`
- `BallPlay/layup/031_015pickle_layup_001_006.pt`

### `skillmimic/data/preprocess/`
offline preprocess 輸出，例如 STG graph。

例如：

- `ballplay.pkl`
- `locomotion.pkl`

### `skillmimic/data/videos/`
由專案輸出的影片。

---

## 10. `skillmimic/metric/`：評估指標

這裡放的是不同 skill / task 的 metric。

例如：

- `shot_metric.py`
- `layup_metric.py`
- `run_metric.py`
- `drink_metric.py`

如果你要做正式 evaluation，而不是只看 reward，這裡很重要。

---

## 11. `data/raw/match4_001_legacy/` 在這個 repo 的角色

這是你們自己加入的桌球資料，不是原作者內建 training data。

裡面包含：

- `subject-1.smpl`
- `subject-2.smpl`
- `results.pkl`
- `world4d.mcs`
- `world4d.glb`

本質上它是：

`桌球影片 -> 3D 人體/場景重建結果`

不是：

`SkillMimic 可直接訓練的 .pt`

所以它目前比較像中間資料來源，而不是最終 dataset。

---

## 12. 如果你想找某件事，該先去哪裡

### 想知道怎麼跑訓練
先看：

- `README.md`
- `skillmimic/run.py`
- `skillmimic/data/cfg/train/rlg/skillmimic.yaml`

### 想知道 dataset 長什麼樣
先看：

- `skillmimic/data/motions/BallPlay/`
- `skillmimic/utils/motion_data_handler.py`
- `docs/data/DATA_FORMAT.md`

### 想知道 reward 怎麼算
先看：

- `skillmimic/env/tasks/skillmimic1.py`
- `skillmimic/env/tasks/skillmimic1_unified.py`

### 想知道 observation 是什麼
先看：

- `skillmimic/env/tasks/skillmimic1.py`
- `skillmimic/env/tasks/humanoid_task.py`
- `skillmimic/env/tasks/humanoid_object_task.py`

### 想知道 simulator 裡的球和物體怎麼建
先看：

- `skillmimic/env/tasks/humanoid_object_task.py`
- `skillmimic/data/assets/urdf/ball.urdf`
- `skillmimic/data/assets/obj/`

### 想知道 ParaHome 怎麼轉檔
先看：

- `skillmimic/data/motions/ParaHome/rot_data_from_parahome.py`
- `skillmimic/data/motions/ParaHome/pt_data_from_parahome.py`

### 想知道你們桌球資料目前缺什麼
先看：

- `docs/data/DATA_FORMAT.md`
- `data/raw/match4_001_legacy/`

---

## 13. 對你目前最有用的最短路線

如果你現在的目標是「做桌球 task」，最重要的不是把所有程式都看完，而是先抓住這幾條主線：

1. `README.md`
2. `skillmimic/run.py`
3. `skillmimic/utils/motion_data_handler.py`
4. `skillmimic/env/tasks/humanoid_object_task.py`
5. `skillmimic/env/tasks/skillmimic1.py`
6. `skillmimic/data/motions/BallPlay/*.pt`
7. `docs/data/DATA_FORMAT.md`

只要先把這七個區塊看懂，你就已經能掌握：

- 資料怎麼進
- 場景怎麼建
- reward 怎麼算
- BallPlay-style dataset 應該長什麼樣

---

## 14. 一句話總結

這個 repo 可以先粗分成三層：

- `skillmimic/data/`：資料、asset、config
- `skillmimic/env/tasks/`：環境、observation、reward
- `skillmimic/learning/`：policy、agent、training loop

而你目前做桌球最重要的工作，主要會落在：

- `data/raw/match4_001_legacy/` 與 `docs/data/DATA_FORMAT.md`
- `skillmimic/utils/motion_data_handler.py`
- `skillmimic/env/tasks/humanoid_object_task.py`
- `skillmimic/env/tasks/skillmimic1*.py`

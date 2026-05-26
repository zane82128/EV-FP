# run.py Process And Arguments

更新日期: `2026-05-26`

這份文件的目標是幫你回答兩個問題：

1. `skillmimic/run.py` 啟動之後，整個流程怎麼走
2. `run.py` / `config.py` 相關的 arguments 有哪些，該怎麼分組看

這份文件**只聚焦在實際會影響這個專案流程的參數**。  
不會把所有底層 Isaac Gym 內建參數都混進來。

---

## 1. 一句話理解 `run.py`

[run.py](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/run.py:1) 是整個專案的**命令列總入口**。

它主要做 6 件事：

1. 解析 command line arguments
2. 載入 `cfg_env` 與 `cfg_train`
3. 用 CLI 參數覆蓋 yaml 內容
4. 建立 simulator / task / env
5. 註冊 agent / player / network
6. 啟動 training、testing 或 dataset replay

---

## 2. `run.py` 的執行流程

### Step 1. 讀 arguments

- 入口在 [get_args()](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/utils/config.py:221)
- 專案自訂參數大多也定義在這裡

### Step 2. 載入 yaml

- [load_cfg()](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/utils/config.py:80)

它會讀：

- `--cfg_env`
- `--cfg_train`

並先做一批基本覆蓋，例如：

- `--num_envs`
- `--episode_length`
- `--seed`
- `--max_epochs`

### Step 3. `run.py` 再做第二次覆蓋

在 [main()](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/run.py:226) 裡，`run.py` 會把一些專案常用設定再寫回 config，例如：

- `--motion_file`
- `--asset_file_name`
- `--play_dataset`
- `--op`
- `--ig`
- `--cg1`
- `--cg2`
- `--state_init`
- `--output_path`

### Step 4. 建立 env

- [create_rlgpu_env()](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/run.py:71)
- [parse_sim_params()](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/utils/config.py:184)
- [parse_task()](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/utils/parse_task.py:82)

這一段會決定：

- 用哪個 task
- 用哪個 asset
- observation / action 維度是多少
- simulator 物理參數是什麼

### Step 5. 建立 runner

- [build_alg_runner()](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/run.py:190)

這裡會註冊：

- `skillmimic`
- `hrl_discrete`
- `amp`

以及它們對應的：

- agent
- player
- model
- network builder

### Step 6. 執行

最後在 [runner.run(vargs)](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/run.py:323) 啟動。

---

## 3. 先怎麼看 arguments

不要把所有參數一起看。  
先分成這 4 群：

1. **模式切換**
2. **資料 / task / asset**
3. **訓練規模 / reward**
4. **進階功能**

如果你現在只是在做桌球 `serve`：

- 先只看前 3 群
- 第 4 群大多可以先略過

---

## 4. 模式切換參數

| 參數 | 作用 | 你現在常用嗎 |
|---|---|---|
| `--test` | 測試模式，不訓練 policy | 常用 |
| `--play` | 與 `--test` 類似，偏 `rl_games` 介面 | 少用 |
| `--play_dataset` | 不跑 policy，直接 replay dataset | 常用 |
| `--resume` | 從 checkpoint 恢復訓練/測試 | 偶爾 |
| `--checkpoint` | 指定載入的權重檔 | 常用 |
| `--headless` | 關閉 viewer | 訓練常用 |
| `--output_path` | 指定輸出資料夾 | 常用 |
| `--experiment` | 覆蓋實驗名稱 | 偶爾 |
| `--metadata` | 在實驗名稱上附加裝置/物理資訊 | 少用 |

補充：

- `--test` 會讓 `args.train = False`
- `--play_dataset` 會把 `cfg['env']['playdataset'] = True`

---

## 5. task / 資料 / asset 參數

| 參數 | 作用 | 寫到哪裡 |
|---|---|---|
| `--task` | 選 task 類別，例如 `SkillMimic1BallPlay` | `cfg["name"]` |
| `--cfg_env` | env yaml 路徑 | `load_cfg()` 直接讀 |
| `--cfg_train` | train yaml 路徑 | `load_cfg()` 直接讀 |
| `--motion_file` | 指定 reference motion 路徑 | `cfg['env']['motion_file']` |
| `--switch_motion_file` | 切換 reference motion | `cfg['env']['switch_motion_file']` |
| `--asset_file_name` | 覆蓋 humanoid asset XML | `cfg['env']['asset']['assetFileName']` |
| `--state_init` | 指定初始化模式或 frame | `cfg['env']['stateInit']` |
| `--state_init_random_prob` | reference random init 機率 | `cfg['env']['state_init_random_prob']` |
| `--state_switch_prob` | skill switch 機率 | `cfg['env']['state_switch_prob']` |

桌球專案最常用的組合是：

- `--task SkillMimic1BallPlay`
- `--motion_file skillmimic/data/motions/TableTennis/serve`
- `--asset_file_name mjcf/mocap_humanoid_hand_only.xml`

---

## 6. 訓練規模 / PPO 參數

| 參數 | 作用 | 什麼時候改 |
|---|---|---|
| `--num_envs` | 環境數量 | 幾乎一定會改 |
| `--episode_length` | episode 長度 | 幾乎一定會改 |
| `--max_epochs` | 最大訓練 epochs | 常改 |
| `--horizon_length` | 每輪 PPO rollout 步數 | 有需要才改 |
| `--minibatch_size` | PPO optimization minibatch size | 有需要才改 |
| `--seed` | 隨機種子 | 重現結果時會改 |
| `--torch_deterministic` | 提高可重現性 | debug/reproduce 時改 |

你現在最常會碰的是：

- `--num_envs`
- `--episode_length`
- `--max_epochs`

---

## 7. reward / 參考動作調整參數

| 參數 | 作用 | 你現在 serve 會不會用 |
|---|---|---|
| `--op` | object position reward 權重 | 會 |
| `--ig` | interaction graph reward 權重 | 會 |
| `--cg1` | body contact reward 權重 | 偶爾 |
| `--cg2` | object contact reward 權重 | 會 |
| `--frames_scale` | reference 動作時間縮放 | 偶爾 |
| `--ball_size` | ball 大小縮放 | 偶爾 |
| `--init_vel` | 第一幀初始化物件速度 | 少用 |

桌球 `Phase 1` 最常見的設定是：

- `--op 0`
- `--ig 0`
- `--cg2 0`

原因：

- 你現在的 `serve` reference 是 human-only / dummy ball
- 不希望假球進 reward

---

## 8. replay / viewer 相關參數

| 參數 | 作用 | 你現在常用嗎 |
|---|---|---|
| `--play_dataset` | 直接播放 reference dataset | 常用 |
| `--save_images` | 存 viewer image frames | 偶爾 |
| `--projtype` | 投射物模式，`None/Auto/Mouse` | 幾乎不用 |

補充：

- viewer 大小現在不是用 `run.py` 參數控制  
- 而是用環境變數：
  - `SKILLMIMIC_VIEWER_WIDTH`
  - `SKILLMIMIC_VIEWER_HEIGHT`

---

## 9. history / state-search / SkillMimic2 參數

這一組是**進階功能**。  
如果你現在還在做 `serve phase 1`，大多可以先略過。

| 參數 | 作用 | 什麼時候碰 |
|---|---|---|
| `--hist_ckpt` | history encoder checkpoint | SkillMimic2 時 |
| `--hist_length` | history 長度 | SkillMimic2 時 |
| `--history_embedding_size` | history embedding 維度 | SkillMimic2 時 |
| `--graph_file` | 載入 state search graph | state-search 時 |
| `--graph_save_path` | 輸出 offline graph | offline preprocess 時 |
| `--state_search_to_align_reward` | 用 state search 對齊 reward | SkillMimic2 / randomized init |

---

## 10. reweight / 特殊實驗參數

這些也是進階選項，不是你現在最先要看的。

| 參數 | 作用 |
|---|---|
| `--reweight` | 開啟 motion/time reweight |
| `--reweight_alpha` | reweight 強度 |
| `--use_old_reweight` | 舊版 reweight 方法 |
| `--disable_time_reweight` | 關掉 time-level reweight |
| `--local_reward` | 使用 local reward |
| `--local_reward_randskill` | random skill 用 local reward |
| `--eval_randskill` | replay 時評估 random skill |
| `--test_random_pick` | 測試隨機初始化物件 |
| `--enable_buffernode` | 啟用 buffernode |
| `--regloss_weight` | regularization loss 權重 |
| `--build_blender_motion` | 匯出 blender motion |
| `--NR` | 計算 normalized accumulated reward |
| `--adapt_prob` | 特定 skill switch / adapt 實驗開關 |

這一組通常只在：

- 重現作者特定實驗
- 做 ablation
- 做 SkillMimic2 / graph / history pipeline

才需要碰。

---

## 11. HRL / 多 GPU 相關參數

| 參數 | 作用 | 什麼時候碰 |
|---|---|---|
| `--llc_checkpoint` | HRL low-level controller checkpoint | HRL 時 |
| `--horovod` | 多 GPU 分散式訓練 | 多 GPU 時 |

如果你現在只是單技能桌球 imitation：

- 幾乎不用看這一組

---

## 12. 這份表格沒列，但實際也存在的 Isaac Gym 內建參數

`gymutil.parse_arguments()` 還會提供一批 Isaac Gym 內建參數。  
它們不是這個 repo 自己宣告的，但 `run.py` / `config.py` 會用到，例如：

- `--sim_device`
- `--pipeline`
- `--graphics_device_id`
- `--physics_engine`
- `--use_gpu`
- `--use_gpu_pipeline`
- `--subscenes`
- `--num_threads`
- `--slices`

這些主要影響：

- 物理引擎
- GPU/CPU pipeline
- physics thread / subscene 數量

如果你現在只是正常訓練 serve：

- 先維持預設就好

---

## 13. 你現在最應該記住的最小參數集合

如果你只想做桌球 `serve`，先只記這一組：

| 類型 | 參數 |
|---|---|
| task | `--task` |
| config | `--cfg_env`, `--cfg_train` |
| data | `--motion_file` |
| asset | `--asset_file_name` |
| train scale | `--num_envs`, `--episode_length`, `--max_epochs` |
| init | `--state_init` |
| mode | `--headless`, `--test`, `--play_dataset`, `--checkpoint` |
| reward | `--op`, `--ig`, `--cg2` |
| output | `--output_path` |

這些已經足夠應付：

- replay
- `Phase 1A` serve 訓練
- checkpoint test

---

## 14. 三種最常見情境

### A. Replay Dataset

```bash
python skillmimic/run.py \
  --play_dataset \
  --test \
  --task SkillMimic1BallPlay \
  --motion_file skillmimic/data/motions/TableTennis/serve \
  --episode_length 60
```

你在用的核心參數只有：

- `--play_dataset`
- `--test`
- `--task`
- `--motion_file`
- `--episode_length`

### B. Phase 1A Train

```bash
python skillmimic/run.py \
  --task SkillMimic1BallPlay \
  --motion_file skillmimic/data/motions/TableTennis/serve \
  --asset_file_name mjcf/mocap_humanoid_hand_only.xml \
  --num_envs 128 \
  --episode_length 60 \
  --state_init Random \
  --op 0 \
  --ig 0 \
  --cg2 0 \
  --max_epochs 200 \
  --output_path output/serve_phase1a \
  --headless
```

### C. Test Checkpoint

```bash
python skillmimic/run.py \
  --test \
  --task SkillMimic1BallPlay \
  --motion_file skillmimic/data/motions/TableTennis/serve \
  --episode_length 60 \
  --checkpoint output/serve_phase1a/model.pth
```

---

## 15. 最後的閱讀建議

不要試圖一次記住全部 arguments。

建議順序是：

1. 先只會用 `replay`
2. 再會用 `phase 1 train`
3. 再學會 `checkpoint test`
4. 最後才去碰 history / graph / reweight / HRL

如果你現在還在桌球 `serve` 階段，請優先熟悉：

- `--motion_file`
- `--asset_file_name`
- `--num_envs`
- `--episode_length`
- `--state_init`
- `--op`
- `--ig`
- `--cg2`
- `--headless`
- `--checkpoint`

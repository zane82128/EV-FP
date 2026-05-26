# Docs Index

這個 `docs/` 目錄是目前專案文件的正式入口。

目標是把文件收斂成少量、用途清楚的幾份：

- repo 地圖
- data 佈局
- data format 規格
- SMPL-X -> SkillMimic 轉檔流程
- 桌球專案的實際 runbook

---

## 1. Repo

- [repo/REPO_GUIDE.md](./repo/REPO_GUIDE.md)
- [repo/run_process.md](./repo/run_process.md)

用途：

- 快速理解整個 repo 的結構
- 找入口、task、loader、asset、training loop
- `run_process.md`
  - 專門整理 `skillmimic/run.py` 的執行流程
  - 把 arguments 按用途分類成表格

---

## 2. Data

- [data/DATA_LAYOUT.md](./data/DATA_LAYOUT.md)
- [data/DATA_FORMAT.md](./data/DATA_FORMAT.md)

用途：

- `DATA_LAYOUT.md`
  - 定義本地 `data/` 的資料夾結構
  - 說明 raw / annotations / clips / canonical 各自放什麼

- `DATA_FORMAT.md`
  - 定義桌球資料最後要轉成什麼格式
  - 說明 final `.pt` 的 `[T, 337]` layout

---

## 3. Pipeline

- [pipeline/SMPLX_TO_SKILLMIMIC_FORMAT.md](./pipeline/SMPLX_TO_SKILLMIMIC_FORMAT.md)
- [pipeline/SMPLX_TO_SKILLMIMIC_PIPELINE_README.md](./pipeline/SMPLX_TO_SKILLMIMIC_PIPELINE_README.md)

用途：

- 說明目前實際怎麼把 SMPL-X / 4D-Humans 輸出轉成 SkillMimic 可 replay / train 的格式
- 包含 clip、source joints、retarget、canonical、final `.pt`
- 也包含 validator / replay 這些驗證方法
- `SMPLX_TO_SKILLMIMIC_PIPELINE_README.md`
  - 給資料前處理組員的操作版 README
  - 只講要準備什麼、跑什麼命令、看什麼輸出
  - 對應 wrapper script:
    `scripts/run_smplx_to_skillmimic.sh`

---

## 4. Table Tennis

- [table_tennis/TABLE_TENNIS_WORKFLOW.md](./table_tennis/TABLE_TENNIS_WORKFLOW.md)
- [table_tennis/TRAINING_PHASES.md](./table_tennis/TRAINING_PHASES.md)

用途：

- 桌球專案目前的總入口
- 包含目前 scope、實際輸出、replay 指令、asset 狀態、serve 學習路線
- `TRAINING_PHASES.md`
  - 專門說明 phase 1 / phase 2 的訓練設計
  - 包含 reference、reward、物件需求、contact graph 取捨

---

## 5. Consolidation Notes

以下舊文件已被合併或淘汰，不再作為主要入口：

- `ADDED_VALIDATION_TOOLS.md`
  -> 內容已收斂到 `pipeline/SMPLX_TO_SKILLMIMIC_FORMAT.md`

- `MATCH4_001_TO_TRAINING_DATA.md`
  -> 內容和轉檔流程高度重疊，已被 `DATA_FORMAT.md` 與 `SMPLX_TO_SKILLMIMIC_FORMAT.md` 吸收

- `SERVE_LEARNING_PLAN.md`
- `TABLE_TENNIS_CURRENT_PROGRESS.md`
- `TABLE_TENNIS_REPLAY_COMMANDS.md`
  -> 已合併成 `table_tennis/TABLE_TENNIS_WORKFLOW.md`

- `EXECUTION_PLAN.md`
  -> 時間性太強，且多數內容已過期，暫時移除

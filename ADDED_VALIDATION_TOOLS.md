# Added Validation Tools

這份文件說明這次為了驗證 `SMPL/SMPL-X -> SkillMimic` 資料流程，我新增了哪些程式，以及它們各自的目的。

也會順便標出幾支 repo 內原本就存在、但和這次驗證流程密切相關的工具，避免之後回來看時搞混「新增」和「既有」。

---

## 1. 這次新增的程式

### `skillmimic/utils/validate_motion_pt.py`

**狀態**

- 這次新加入

**目的**

- 驗證最終輸出的 SkillMimic / BallPlay 格式 `.pt` 是否合理
- 幫助在正式訓練前，先抓出容易導致訓練壞掉的資料問題
- 提供比「只看 shape 對不對」更實際的 sanity check

**它在檢查什麼**

- `.pt` 是否為 `torch.Tensor`
- shape 是否為 `[T, 337]`
- 是否存在 `NaN` / `Inf`
- reserved 欄位是否接近 0
- `contact` 是否為合理的 binary 值
- `body_pos[:, 0]`（Pelvis）是否和 `root_pos` 一致
- `root_pos[:, 2]` 是否低於 env 的 `terminationHeight`
- root 平移速度是否有不合理 spike
- root 旋轉速度是否超過 replay 異常門檻
- `dof_pos` 的 rotvec 大小是否離譜
- `body_pos` 是否整體塌縮到 `root_pos`
- 骨長在時間上是否不穩定
- human-only dummy object 是否保持常數

**為什麼需要它**

因為目前 SkillMimic 的 loader 不只讀取位置和旋轉，還會從 motion 資料自行計算：

- `root_pos_vel`
- `dof_pos_vel`
- `root_rot_vel`

如果你的 `.pt` 雖然 shape 正確，但數值不連續、骨架塌掉、root 高度錯誤、或 rotation 跳變太大，訓練仍然很容易失敗。

**使用方式**

```bash
cd /home/zane82128/EV/FP/SkillMimic-V2

python skillmimic/utils/validate_motion_pt.py \
  skillmimic/data/motions/TableTennis/serve/001_serve_0001.pt \
  --expect-dummy-object \
  --fps 60 \
  --termination-height 0.25 \
  --strict-warn
```

**輸出風格**

- `[PASS]` 代表檢查通過
- `[WARN]` 代表資料可疑，建議先人工確認
- `[FAIL]` 代表不建議直接拿去訓練

**特別重要的一點**

如果這支 validator 報出：

- `body-collapse`

通常代表你的 `body_pos` 幾乎全部等於 `root_pos`，也就是 canonical 資料很可能來自 fallback，而不是真正的 source joints retarget。

這種資料可以拿來測試 pipeline，但不適合正式訓練。

---

## 2. 這次沒有新增，但和流程密切相關的既有程式

下面這些檔案不是這次新增的，但在整個資料轉換與驗證流程中很重要。

### `skillmimic/utils/extract_smpl_clip.py`

**狀態**

- repo 原本已有

**目的**

- 從 `.smpl` 依照 `start_frame / end_frame` 切出 skill clips
- 例如切出 `serve`、`forehand`、`backhand`

**輸出**

- 每段 clip 對應一個 `.npz`

---

### `skillmimic/utils/smpl_to_canonical.py`

**狀態**

- repo 原本已有

**目的**

- 將 clip `.npz` 轉成 canonical human-only clip
- 產出：
  - `root_pos`
  - `root_rot_3d`
  - `dof_pos`
  - `body_pos`

**特別注意**

這支腳本支援 fallback 模式：

- 若沒有提供 `source_joints`
- 且使用 `--allow-missing-source-joints`

那麼它會把：

```python
body_pos[:] = root_pos[:, None, :]
```

這表示：

- 所有 body joints 都會疊在 root 上
- 只能用來測 pipeline
- 不適合正式訓練

---

### `skillmimic/utils/canonical_to_skillmimic_pt.py`

**狀態**

- repo 原本已有

**目的**

- 把 canonical clip 打包成最終的 `[T, 337]` `.pt`
- 補齊 human-only phase 需要的 dummy object / contact

**它負責的事**

- 寫入 `root_pos`
- 寫入 `root_rot_3d`
- reshape `dof_pos`
- reshape `body_pos`
- 補 dummy `obj_pos`
- 補 dummy `obj_rot_3d`
- 補 dummy `contact`

---

### `skillmimic/utils/inspect_motion_pt.py`

**狀態**

- repo 原本已有

**目的**

- 讀取 `.pt`
- 以欄位切片方式顯示內容
- 幫你快速確認 layout 是否符合 BallPlay 格式

**它和 `validate_motion_pt.py` 的差別**

- `inspect_motion_pt.py` 比較像「看資料」
- `validate_motion_pt.py` 比較像「判斷資料是否可拿去訓練」

---

## 3. 建議的使用順序

如果之後你又要從新的 SMPLX 資料做一次流程，建議順序如下：

1. 用 `extract_smpl_clip.py` 切出 skill clip
2. 用 `smpl_to_canonical.py` 產生 canonical clip
3. 用 `canonical_to_skillmimic_pt.py` 打包成 `[T, 337]`
4. 用 `validate_motion_pt.py` 做數值與骨架檢查
5. 用 `--play_dataset` 做 replay 檢查
6. 確認沒問題後再開始 train

---

## 4. 這次新增內容的核心價值

這次真正新增的重點不是新的轉檔器，而是：

- 一個專門檢查最終 `.pt` 是否會把訓練餵壞的 validator

它的定位是：

- 在正式訓練之前
- 幫你先抓出 shape 正確但語義錯誤的 motion data

也就是說，它不是取代 replay，而是讓你在 replay 之前就先淘汰掉最危險的資料。

---

## 5. 目前你可以怎麼用

你之後如果轉出新的桌球 motion 檔，可以先做：

```bash
python skillmimic/utils/validate_motion_pt.py <your_motion.pt> --expect-dummy-object --strict-warn
```

如果出現：

- `FAIL`

先不要 train。

如果只有：

- `WARN`

建議先用 `--play_dataset` 人眼檢查。

如果大多數都：

- `PASS`

再進一步做 replay 和短時間 smoke training。

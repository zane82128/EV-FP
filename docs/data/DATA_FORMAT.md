# SkillMimic-V2 Data Format for Table Tennis

這份文件是給負責資料轉換的人看的。

目標不是先講完整論文背景，而是直接回答這件事：

**桌球資料最後要轉成什麼格式，SkillMimic-V2 才能吃。**

---

## TL;DR

如果你想用**現有 SkillMimic-V2 code，且先盡量少改 code**，目前最推薦的做法是：

- **Phase 1 先做 human-only imitation**
- **Phase 1 不做 ball tracking**
- **Phase 1 先只學人體動作**
  - `serve`
  - `forehand`
  - `backhand`

在這個前提下，第一版請把桌球資料轉成：

- **BallPlay-style 單物體格式**
- 每個 clip 一個 `.pt` 檔
- 每個 `.pt` 是 `torch.Tensor`
- shape 是 **`[T, 337]`**
- `T` 是這段動作的幀數

但要注意：

- 這裡沿用 `BallPlay-style [T, 337]`
- 主要是為了先和現有 loader / task 相容
- 不代表 Phase 1 真的已經要把真實球資料塞進來

先把問題簡化成：

- 1 個 humanoid
- 暫時不依賴真實影片中的球
- 球拍先當成手的一部分，不先做成獨立物體
- object 欄位先用 dummy 值保留相容性

這樣最接近 repo 現有的 `BallPlay` pipeline。

---

## 目前推薦路線：Human-Only First

你們現在最大的瓶頸不是 policy，而是：

- 很難從真實比賽影片中穩定做 ball segmentation / tracking

所以目前最實際的路線不是先追球，而是：

1. 先從 `match4_001` 取出單人的人體 motion
2. 先切成 `serve / forehand / backhand`
3. 先做 `SMPL -> SkillMimic` retarget
4. 先讓 humanoid 學會人體動作 imitation
5. 等 Phase 1 跑通後，再在 Phase 2 把球交給 Isaac Gym 模擬

也就是：

```text
Phase 1:
  真實影片 -> SMPL motion -> human-only .pt -> imitation learning

Phase 2:
  human-only policy -> simulator ball -> RL 擊球到對面桌上
```

這份文件後面仍然保留完整的 `337` 維格式說明，但你目前最該優先做的是：

- 把人體欄位做好
- 先不要被球軌跡卡住

---

## 最後要交付什麼

建議最後交付的資料長這樣：

```text
skillmimic/data/motions/TableTennis/
  serve/
    001_serve_0001.pt
    001_serve_0002.pt
  forehand/
    002_forehand_0001.pt
  backhand/
    003_backhand_0001.pt
```

命名規則：

- 檔名前 3 碼是 `skill_id`
- 例如 `001`、`002`、`003`
- 這個 repo 會直接用檔名前 3 碼當 skill label

建議：

- `001` = serve
- `002` = forehand
- `003` = backhand

如果之後進到 Phase 2 或 Phase 3，再加：

- `004` = receive

---

## 最終 `.pt` 格式

每個 `.pt` 應該是一個：

```python
torch.Tensor  # dtype=float32
shape = [T, 337]
```

其中每一幀 `frame[t]` 的 337 維內容如下：

| 區段 | 維度 | index | 意義 |
|---|---:|---|---|
| `root_pos` | 3 | `0:3` | humanoid root / pelvis 的世界座標位置 |
| `root_rot_3d` | 3 | `3:6` | humanoid root 的世界旋轉，rotvec / exp-map，單位 rad |
| `reserved_a` | 3 | `6:9` | 保留欄位，先全填 `0` |
| `dof_pos` | 156 | `9:165` | `52 x 3` 個關節 local rotation，rotvec，單位 rad |
| `body_pos` | 159 | `165:324` | `53 x 3` 個 body 的世界座標位置，單位 meter |
| `obj_pos` | 3 | `324:327` | 桌球的世界座標位置 |
| `obj_rot_3d` | 3 | `327:330` | 桌球旋轉，rotvec；第一版可全填 `0` |
| `reserved_b` | 6 | `330:336` | 保留欄位，先全填 `0` |
| `contact` | 1 | `336:337` | 接觸標記；擊球幀填 `1`，其他填 `0` |

也就是：

```text
frame =
[root_pos(3),
 root_rot_3d(3),
 reserved_a(3),
 dof_pos(52*3=156),
 body_pos(53*3=159),
 obj_pos(3),
 obj_rot_3d(3),
 reserved_b(6),
 contact(1)]
```

總共：

```text
3 + 3 + 3 + 156 + 159 + 3 + 3 + 6 + 1 = 337
```

這個格式對應 repo 內的 loader：
[motion_data_handler.py](/home/zane82128/EV/FP/SkillMimic-V2/skillmimic/utils/motion_data_handler.py:147)

---

## Phase 1 的實際用法

雖然 raw `.pt` 仍然是 `[T, 337]`，但在 **Phase 1: human-only imitation** 裡：

- `root_pos`
- `root_rot_3d`
- `dof_pos`
- `body_pos`

這四段是你真正要認真做的資料。

下面這三段在 Phase 1 可以先用 dummy 值：

- `obj_pos`
- `obj_rot_3d`
- `contact`

建議的最小做法：

```python
obj_pos[:] = [2.0, 0.0, 1.0]
obj_rot_3d[:] = 0.0
contact[:] = 0.0
```

也就是說，Phase 1 的 `.pt` 雖然形狀和 BallPlay 一樣，但語意其實是：

- human-only motion
- object 欄位只是為了先和現有 code 相容

這樣做的好處是：

- 不用先改 loader
- 不用先改 observation pipeline
- 可以直接用 `--play_dataset` 做 replay 檢查

---

## 你可以把這個格式想成什麼

每一幀就是：

- 人在哪裡
- 人朝哪裡
- 每個關節怎麼轉
- 身體每個 body 在世界座標哪裡
- Phase 1 裡 object 可以先是 dummy
- Phase 2 才真的把球放進來

SkillMimic 不是直接吃影片，也不是直接吃 SMPL 參數。

它吃的是：

- **已經 retarget 到 SkillMimic 骨架上的 humanoid motion**
- 加上一顆球的 3D 軌跡

---

## 每個欄位怎麼理解

### 1. `root_pos`

```python
root_pos.shape == (T, 3)
```

代表 pelvis / root 在世界座標系下的位置。

單位：

- meter

範例：

```python
root_pos[0] = [0.12, 0.00, 0.98]
```

---

### 2. `root_rot_3d`

```python
root_rot_3d.shape == (T, 3)
```

代表 root 的世界旋轉，用 **rotation vector / exponential map** 表示。

單位：

- rad

注意：

- 不是 quaternion
- 不是 Euler angle

範例：

```python
root_rot_3d[0] = [0.00, 0.15, 0.00]
```

---

### 3. `dof_pos`

```python
dof_pos.shape == (T, 52, 3)
```

這是最重要的欄位。

代表：

- 52 個關節
- 每個關節一個 local rotation
- 每個 rotation 用 3 維 rotvec 表示

單位：

- rad

注意：

- **一定要是 local rotation**
- **一定要對齊 SkillMimic 骨架順序**

---

### 4. `body_pos`

```python
body_pos.shape == (T, 53, 3)
```

代表 53 個 body 在世界座標的位置。

單位：

- meter

注意：

- 這 53 個 body 包含 `Pelvis`
- 建議這個欄位由 **retarget 後的目標骨架做 FK** 算出來
- 不要直接把來源 skeleton 的 joint xyz 硬塞進來

---

### 5. `obj_pos`

```python
obj_pos.shape == (T, 3)
```

代表桌球的位置。

但要注意：

- **Phase 1**：這一欄可以先放 dummy 值
- **Phase 2**：這一欄才真的改成 simulator ball 的位置

---

### 6. `obj_rot_3d`

```python
obj_rot_3d.shape == (T, 3)
```

代表桌球旋轉。

- **Phase 1**：直接全填 `0`
- **Phase 2**：如果真的需要，再接 simulator ball rotation

---

### 7. `contact`

```python
contact.shape == (T, 1)
```

代表這一幀是否發生接觸。

但 Phase 1 和 Phase 2 要分開看：

- **Phase 1**：先全填 `0`
- **Phase 2**：再定義成擊球事件標記

如果之後進到 Phase 2，最簡單的定義是：

- 擊球幀：`1`
- 其他幀：`0`

---

## `dof_pos` 的 joint 順序

`dof_pos` 的 52 個 joint 必須依照這個順序展平：

```python
DOF_ORDER = [
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Spine2", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    "L_Index1", "L_Index2", "L_Index3",
    "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3",
    "L_Ring1", "L_Ring2", "L_Ring3",
    "L_Thumb1", "L_Thumb2", "L_Thumb3",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    "R_Index1", "R_Index2", "R_Index3",
    "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3",
    "R_Ring1", "R_Ring2", "R_Ring3",
    "R_Thumb1", "R_Thumb2", "R_Thumb3",
]
```

也就是：

```python
dof_pos[t].shape == (52, 3)
dof_pos_flat[t].shape == (156,)
```

最後存檔時要做：

```python
motion[:, 9:165] = dof_pos.reshape(T, 156)
```

---

## `body_pos` 的 body 順序

`body_pos` 的 53 個 body 必須依照這個順序：

```python
BODY_ORDER = [
    "Pelvis",
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Spine2", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    "L_Index1", "L_Index2", "L_Index3",
    "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3",
    "L_Ring1", "L_Ring2", "L_Ring3",
    "L_Thumb1", "L_Thumb2", "L_Thumb3",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    "R_Index1", "R_Index2", "R_Index3",
    "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3",
    "R_Ring1", "R_Ring2", "R_Ring3",
    "R_Thumb1", "R_Thumb2", "R_Thumb3",
]
```

也就是：

```python
body_pos[t].shape == (53, 3)
body_pos_flat[t].shape == (159,)
```

最後存檔時要做：

```python
motion[:, 165:324] = body_pos.reshape(T, 159)
```

---

## 建議先做一個中間格式

不要一開始就直接拼 `.pt`。

建議先做一個中間格式，例如 `.npz` 或 `.pkl`，長這樣：

```python
canonical_clip = {
    "fps": 30.0,
    "skill_id": 1,
    "skill_name": "serve",
    "root_pos": np.ndarray((T, 3), dtype=np.float32),
    "root_rot_3d": np.ndarray((T, 3), dtype=np.float32),
    "dof_pos": np.ndarray((T, 52, 3), dtype=np.float32),
    "body_pos": np.ndarray((T, 53, 3), dtype=np.float32),
    "obj_pos": np.ndarray((T, 3), dtype=np.float32),
    "obj_rot_3d": np.ndarray((T, 3), dtype=np.float32),
    "contact": np.ndarray((T, 1), dtype=np.float32),
}
```

這樣好處是：

- 比較容易 debug
- 比較容易畫圖檢查
- 之後要改 `.pt` 打包器比較簡單

---

## 從桌球影片到這個格式的步驟

如果你現在手上是**現實世界桌球比賽影片**，建議 pipeline 是：

### Step 1. 切 clip

先把長影片切成短片段。

建議每段：

- 1 到 5 秒
- 單一技能為主
- 少鏡頭切換
- 角色盡量完整可見

例如：

- 發球
- 正手
- 反手
- 接發球

---

## 桌球資料怎麼切成 Skill

如果你要照 **現有 SkillMimic-V2** 的流程訓練，建議一定要把長影片切成多個 **skill clips**，不要直接拿整段長 rally 去訓練。

原因是：

- 這個 repo 本來就是用很多短 motion clips 訓練
- 檔名前 3 碼會被當成 `skill_id`
- policy observation 會接 skill condition
- skill clip 比較容易做 reset、reweight、switching 和 debug

你可以把這裡的 `skill` 理解成：

- 一段**語意單一**
- 有明確開始和結束
- 最多只包含 **1 個主要動作目標**

不是整段比賽，不是整個來回球。

---

## 第一版建議的 Skill 類別

先不要切太細，也不要一開始就追求完整桌球技術 taxonomy。

`2026-06-10` 前的第一版，建議只切下面 3 類：

| skill_id | skill_name | 說明 |
|---:|---|---|
| `001` | `serve` | 發球 |
| `002` | `forehand` | 正手進攻或正手回擊 |
| `003` | `backhand` | 反手進攻或反手回擊 |

如果之後進到 Phase 2 或時間還夠，再加：

| skill_id | skill_name | 說明 |
|---:|---|---|
| `004` | `receive` | 接發球 |
| `005` | `forehand_loop` | 正手連續腳步或拉球 |
| `006` | `backhand_block` | 反手擋球 |

第一版不要一開始就切：

- `long_rally`
- `full_point`
- `forehand_topspin_crosscourt`
- `serve_receive_counterloop`

這些太長、太混、太難標。

---

## 切分原則

### 一段 clip 最好只包含 1 個主要擊球事件

最理想的是：

- 1 段 `serve` clip 只有 1 次發球
- 1 段 `forehand` clip 只有 1 次正手主擊球
- 1 段 `backhand` clip 只有 1 次反手主擊球

也就是：

- 可以有準備動作
- 可以有 follow-through
- 但不要在同一段裡同時塞兩三個不同擊球

---

### 切點要以動作事件為中心，不要只看固定幀數

正確思路是：

- 先找到主要擊球幀
- 再往前保留準備動作
- 再往後保留收拍與恢復

不是：

- 每段固定切 60 幀
- 不管動作語意直接硬切

---

### 開始幀怎麼抓

建議從下面時點附近開始：

- `serve`
  - 拋球前或準備發球前的短暫穩定姿勢
- `forehand / backhand`
  - 開始轉肩、引拍、跨步的前幾幀
- `receive`
  - 對手發球後、自己開始準備接發的前幾幀

30 fps 下的經驗值：

- 可以先從擊球前 `8 ~ 15` 幀開始試

---

### 結束幀怎麼抓

建議在下面時點附近結束：

- 擊球後 follow-through 結束
- 重心回到相對穩定狀態
- 下一個明確 skill 還沒開始前

30 fps 下的經驗值：

- 可以先從擊球後 `10 ~ 25` 幀結束試

---

### 每段長度建議

第一版建議長度：

- `serve`: `1.0 ~ 2.5` 秒
- `forehand`: `0.5 ~ 2.0` 秒
- `backhand`: `0.5 ~ 2.0` 秒
- `receive`: `0.5 ~ 1.5` 秒

如果超過 `3` 秒，通常代表：

- 切太長
- 包進太多 skill
- 不適合當第一版資料

---

## 什麼情況不要拿來切

下面這些片段，第一版先不要用：

- 有鏡頭切換
- 有明顯 zoom / pan 導致相機不穩
- 球看不到或球軌跡斷掉
- 球員嚴重遮擋
- 同一段裡包含多個主技能，難以定義單一 label
- 3D 重建很明顯炸掉
- 球和人體不在同一世界座標系

---

## 實際切分流程

建議照下面流程做：

1. 先看原始影片或回放，標出每一次擊球事件。
2. 對每次擊球，先判斷是哪一類：
   - `serve`
   - `forehand`
   - `backhand`
   - `receive`
3. 以主要擊球幀為中心，往前取準備動作，往後取收拍。
4. 檢查這段裡是否只包含 1 個主要 skill。
5. 如果這段混了多個 skill，就重新縮短。
6. 如果球資料或人體資料在這段不穩，就先丟掉這段。

---

## 建議保存的切分標註

每段 clip 在轉成 `.pt` 前，建議先有一份 metadata，例如：

```python
segment_meta = {
    "source_video": "match4_001",
    "subject": "subject-1",
    "skill_id": 2,
    "skill_name": "forehand",
    "start_frame": 84,
    "hit_frame": 96,
    "end_frame": 114,
    "fps": 30.0,
    "notes": "clean forehand, ball visible",
}
```

最少要保存：

- `source_video`
- `subject`
- `skill_id`
- `skill_name`
- `start_frame`
- `hit_frame`
- `end_frame`

這樣之後出問題才知道是哪一段切錯。

---

## 命名方式

命名建議固定為：

```text
{skill_id}_{skill_name}_{clip_id}.pt
```

例如：

```text
001_serve_0001.pt
001_serve_0002.pt
002_forehand_0001.pt
003_backhand_0001.pt
004_receive_0001.pt
```

如果你還在中間格式階段，也可以先用：

```text
001_serve_0001.npz
002_forehand_0001.npz
```

---

## `match4_001` 的切分建議

對於 [match4_001_legacy](/home/zane82128/EV/FP/SkillMimic-V2/data/raw/match4_001_legacy:1) 這種單段影片重建資料，建議不要一開始就把整段 `250` 幀直接轉成一個 skill。

更合理的做法是：

1. 先選一位球員，例如 `subject-1`
2. 找出這位球員在整段片段中的主要擊球時刻
3. 每個擊球時刻各自切一小段
4. 先保留最乾淨的 3 段

第一版建議至少切出：

- 1 段 `serve`
- 1 段 `forehand`
- 1 段 `backhand`

如果這段影片裡沒有發球，就改成：

- 2 段 `forehand`
- 1 段 `backhand`

---

## 第一版的最低切分標準

如果時間真的不夠，最低要求是：

- 每個 skill 類別至少 1 段
- 總共至少 `3` 段 clip
- 每段 clip 都有：
  - `skill_id`
  - `start_frame`
  - `end_frame`
  - `hit_frame`
  - `subject`

只要這個最低標準達成，你就可以開始做轉檔與 replay。

---

## 切分完成後，下一步是什麼

切完 skill 之後，不是直接訓練。

正確順序是：

1. 先切 skill
2. 再對每段 skill 做：
   - root / retarget
   - `dof_pos`
   - `body_pos`
3. 先生成 human-only canonical clip
4. 最後才打包成 `.pt`
5. 之後進到 Phase 2，再把 simulator ball 接回來

也就是：

```text
長影片
  -> skill clips
  -> human-only canonical clips
  -> SkillMimic .pt
  -> replay
  -> training
```

---

### Step 2. 做相機標定

你需要知道世界座標系，否則沒辦法得到穩定的 3D 人和 3D 球。

桌球桌尺寸是固定的，所以可以用：

- 桌角
- 球網
- 邊線

去估相機。

---

### Step 3. 估 3D 人體

你需要得到：

- root 世界位置
- root 世界旋轉
- 每個關節的姿態

來源可以是：

- SMPL / SMPL-X fitting
- 3D pose estimation
- mocap

但最後都要 **retarget 到 SkillMimic 骨架**。

---

### Step 4. Phase 1 先跳過球軌跡

如果你現在的目標是 **human-only imitation**，這一步可以先跳過。

原因是：

- 真實比賽影片的球很小
- 運動模糊嚴重
- ball segmentation / tracking 很容易成為整條管線瓶頸

所以目前更推薦：

- **Phase 1 不做這一步**
- **Phase 2 再用 simulator ball 取代真實球資料**

也就是說，這一步先不需要得到：

```python
obj_pos.shape == (T, 3)
```

Phase 1 先用 dummy object：

- `obj_pos = constant`
- `obj_rot_3d = 0`
- `contact = 0`

---

### Step 5. Retarget 到 SkillMimic 骨架

這一步最重要。

你最後一定要得到：

- `dof_pos: (T, 52, 3)`
- `body_pos: (T, 53, 3)`

做法是：

1. 建 `source skeleton -> SkillMimic skeleton` 的 joint mapping
2. 把來源姿態轉到目標骨架
3. 求每個 joint 的 **local rotation rotvec**
4. 用目標骨架做 FK，算出 `body_pos`

注意：

- `dof_pos` 要是 local rotvec
- `body_pos` 要是 retarget 後骨架的 world positions

---

### Step 6. Phase 1 先跳過 `contact`

如果你是 **Phase 1: human-only imitation**：

- `contact` 先全填 `0`

之後如果進到 **Phase 2: simulator ball interaction**，再定義：

- 球拍擊球那一幀標 `1`
- 其他標 `0`

---

### Step 7. 打包成 `.pt`

把資料塞進 `[T, 337]`：

```python
import torch

motion = torch.zeros(T, 337, dtype=torch.float32)
motion[:, 0:3] = root_pos
motion[:, 3:6] = root_rot_3d
motion[:, 9:165] = dof_pos.reshape(T, 156)
motion[:, 165:324] = body_pos.reshape(T, 159)
motion[:, 324:327] = obj_pos          # Phase 1 可先用 dummy constant
motion[:, 327:330] = obj_rot_3d       # Phase 1 可先全 0
motion[:, 336:337] = contact          # Phase 1 可先全 0

torch.save(motion, "001_serve_0001.pt")
```

---

### Step 8. 用 `--play_dataset` 驗證

不要先急著 train。

先 replay 你的資料，確認：

- 人沒有炸掉
- 關節沒有反折
- 人體姿態合理
- root 和桌面座標系沒有明顯錯位

如果 replay 不對，train 幾乎一定學不好。

---

## 一個最小範例

假設你已經有一個 `serve` clip：

```python
T = 120

root_pos.shape      == (120, 3)
root_rot_3d.shape   == (120, 3)
dof_pos.shape       == (120, 52, 3)
body_pos.shape      == (120, 53, 3)
obj_pos.shape       == (120, 3)      # Phase 1 可先 dummy
obj_rot_3d.shape    == (120, 3)      # Phase 1 可先全 0
contact.shape       == (120, 1)      # Phase 1 可先全 0
```

那最後存檔可以直接這樣做：

```python
import torch

motion = torch.zeros(120, 337, dtype=torch.float32)
motion[:, 0:3] = torch.from_numpy(root_pos)
motion[:, 3:6] = torch.from_numpy(root_rot_3d)
motion[:, 9:165] = torch.from_numpy(dof_pos.reshape(120, 156))
motion[:, 165:324] = torch.from_numpy(body_pos.reshape(120, 159))
motion[:, 324:327] = torch.from_numpy(obj_pos)
motion[:, 327:330] = torch.from_numpy(obj_rot_3d)
motion[:, 336:337] = torch.from_numpy(contact)

torch.save(motion, "001_serve_0001.pt")
```

---

## 一個假的數值範例

下面這個只是幫助理解欄位，不是真實桌球動作：

```python
frame0 = [
    0.10, 0.00, 0.98,         # root_pos
    0.00, 0.10, 0.00,         # root_rot_3d
    0.00, 0.00, 0.00,         # reserved
    ... 156 values ...,       # dof_pos
    ... 159 values ...,       # body_pos
    2.00, 0.00, 1.00,         # obj_pos (Phase 1 可先 dummy)
    0.00, 0.00, 0.00,         # obj_rot_3d
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # reserved
    0.00                      # contact (Phase 1 可先 0)
]
```

意思是：

- 人站在 `(0.10, 0.00, 0.98)`
- 身體稍微朝某方向轉
- Phase 1 裡 object 可以只是 dummy
- 這一幀先不需要真的代表擊球

---

## 成功標準

你交付的資料至少要滿足這些條件：

- 每個檔案都是 `torch.Tensor`
- 每個檔案 shape 都是 `[T, 337]`
- `dtype=float32`
- 沒有 `NaN`
- 沒有 `Inf`
- 長度 `T > 1`
- `root_pos / body_pos / obj_pos` 單位是 meter
- `root_rot_3d / dof_pos / obj_rot_3d` 單位是 rad
- 可以被 `--play_dataset` 正常 replay

---

## 不建議第一版就做的事

第一版不建議一開始就做：

- 球拍作為獨立物體
- 球拍 + 球 + 桌子 + 球網 多物體建模
- 直接照 ParaHome multi-object 路線改整個 env

因為這會讓資料格式、asset、loader、task 一起變複雜。

先做：

- human-only imitation
- BallPlay-style `[T, 337]`
- object 欄位先用 dummy 保持相容

這是最快能驗證 SkillMimic 能不能學桌球動作的路。

---

## ParaHome 和這份格式的關係

ParaHome 那條線是：

- 人
- 手
- 物體
- 更複雜的 household interaction

repo 的 ParaHome 轉檔腳本比較像是：

- 「結構化 3D 資料 -> SkillMimic `.pt`」的範例

不是桌球第一版最推薦直接照抄的格式。

對桌球來說，**第一版請先瞄準 BallPlay-style**，不是 ParaHome-style。

---

## `match4_001` 對照到 SkillMimic 欄位

下面用你們目前這包資料：

- [match4_001_legacy](/home/zane82128/EV/FP/SkillMimic-V2/data/raw/match4_001_legacy:1)

直接對照 SkillMimic 最後要的 `[T, 337]` 格式。

先講結論：

- 這包資料**不是最終 `.pt`**
- 這包資料是**很好的中間格式**
- 目前最關鍵的缺口是：
  - **SMPL -> SkillMimic 骨架 retarget**
  - **Phase 1 先把人體欄位跑通**
  - **Phase 2 再補球與 contact**

---

### `match4_001` 裡目前有什麼

這包資料裡有：

- [subject-1.smpl](/home/zane82128/EV/FP/SkillMimic-V2/data/raw/match4_001_legacy/smpl/subject-1.smpl)
- [subject-2.smpl](/home/zane82128/EV/FP/SkillMimic-V2/data/raw/match4_001_legacy/smpl/subject-2.smpl)
- [results.pkl](/home/zane82128/EV/FP/SkillMimic-V2/data/raw/match4_001_legacy/world/results.pkl)
- [world4d.mcs](/home/zane82128/EV/FP/SkillMimic-V2/data/raw/match4_001_legacy/world/world4d.mcs)
- [world4d.glb](/home/zane82128/EV/FP/SkillMimic-V2/data/raw/match4_001_legacy/world/world4d.glb)

我檢查到的重點如下：

- `subject-1.smpl`
  - `frameCount = 250`
  - `frameRate = 30.0`
  - `bodyTranslation.shape = (250, 3)`
  - `bodyPose.shape = (250, 22, 3)`
- `subject-2.smpl`
  - `frameCount = 250`
  - `frameRate = 30.0`
  - `bodyTranslation.shape = (250, 3)`
  - `bodyPose.shape = (250, 22, 3)`
- `world4d.mcs`
  - 場景裡有 `2` 個 SMPL bodies
  - 相機有 `250` 幀 animation
- `results.pkl`
  - 內容字串顯示有 `camera`、`people`、`track_id`、`bboxes`、`smplx_pose`、`smplx_transl`、`smplx_world`
  - 代表這是主要的追蹤與重建結果總表

也就是說，你們現在已經有：

- 兩個人的 3D 人體動作
- 世界座標中的相機/場景
- 雙人追蹤結果

---

### 這些檔案怎麼對應到 SkillMimic 最後欄位

| SkillMimic 欄位 | 目前來源 | 目前是否直接可用 | 說明 |
|---|---|---|---|
| `root_pos (T, 3)` | `subject-*.smpl -> bodyTranslation` | `部分可用` | 這很像 root/pelvis 世界位置，但仍要確認座標軸方向是否和目標世界一致 |
| `root_rot_3d (T, 3)` | `subject-*.smpl -> bodyPose[:, 0, :]` 或另外解 root orientation | `不可直接用` | 要先確認 `.smpl` 的第 0 個 joint 是否就是你要的 root 定義，且可能仍需座標系轉換 |
| `dof_pos (T, 52, 3)` | `subject-*.smpl -> bodyPose (250, 22, 3)` | `不可直接用` | 目前只有 22 個 SMPL body joints，不是 SkillMimic 的 52 joints，必須 retarget |
| `body_pos (T, 53, 3)` | `subject-*.smpl + SMPL model FK` | `不可直接用` | 要先從 SMPL 算 joint/body world positions，再轉到 SkillMimic 骨架 |
| `obj_pos (T, 3)` | `目前沒看到` | `Phase 1 可跳過` | Phase 1 可先用 dummy constant；Phase 2 才需要真實或 simulator 球位置 |
| `obj_rot_3d (T, 3)` | `目前沒看到` | `Phase 1 可跳過` | Phase 1 先全填 `0` |
| `contact (T, 1)` | `目前沒看到` | `Phase 1 可跳過` | Phase 1 先全填 `0`；Phase 2 再補擊球事件 |

---

### 哪些欄位已經有雛形

如果你只問：

**這包資料離 SkillMimic 最近的是哪些欄位？**

答案是：

1. `root_pos`
2. `root_rot_3d`
3. `dof_pos` 的來源姿態
4. `body_pos` 的來源姿態

因為 `.smpl` 已經給了：

- root translation
- body pose
- 每幀時間序列

所以人體部分已經有基礎，只是**還不是 SkillMimic 骨架格式**。

---

### 哪些欄位目前明確缺少

這包資料目前明確還缺：

1. SkillMimic 52-joint target skeleton 的 `dof_pos`
2. SkillMimic 53-body target skeleton 的 `body_pos`
3. 如果要進入 Phase 2，再補 `obj_pos`
4. 如果要進入 Phase 2，再補 `contact`

其中最重要的是：

- **人體 retarget 先跑通**

因為你們現在的第一階段本來就不是在做球互動，而是在做 human-only imitation。

---

### 這包資料現在最合理的角色

`match4_001` 最適合扮演的是：

- `video -> structured 3D human motion` 的輸出

不是：

- `SkillMimic-ready .pt`

所以你們現在的資料流程應該理解成：

```text
原始桌球影片
  -> match4_001 這種中間格式
  -> retarget 到 SkillMimic 骨架
  -> 打包成 human-only [T, 337] 的 .pt
  -> Phase 2 再加入 simulator ball
```

---

### 如果拿 `subject-1.smpl` 做第一版，實際要補什麼

假設第一版只學一位球員，可以先選：

- `subject-1.smpl`

那接下來至少要做：

1. 從 `bodyTranslation` 取出 `root_pos`
2. 從 `bodyPose` 和 SMPL model 解出 root orientation
3. 用 SMPL skeleton -> SkillMimic skeleton mapping 做 retarget
4. 生出 `dof_pos (T, 52, 3)`
5. 用目標骨架 FK 生出 `body_pos (T, 53, 3)`
6. Phase 1 先用 dummy `obj_pos (T, 3)`
7. Phase 1 先讓 `contact (T, 1)` 全 0
8. `obj_rot_3d` 第一版全填 `0`
9. 打包成 human-only `[T, 337]`

---

### 一句話判斷 `match4_001` 能不能用

可以用，但它現在是：

- **能拿來轉 SkillMimic 的中間資料**

不是：

- **能直接丟給 SkillMimic 訓練的最終資料**

---

## 一句話總結

如果你負責把桌球資料轉成 SkillMimic-V2 能吃的格式，請先把每段桌球 clip 轉成：

- 一個 `.pt`
- shape = **`[T, 337]`**
- 內容是：
  - `root_pos`
  - `root_rot_3d`
  - `dof_pos(52x3)`
  - `body_pos(53x3)`
  - `obj_pos`
  - `obj_rot_3d`
  - `contact`

先讓 SkillMimic 能 replay 你的資料，再談 training。

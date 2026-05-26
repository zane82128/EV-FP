# SMPL-X 轉 SkillMimic 格式說明

這份文件說明我目前是怎麼把你的 SMPL-X / 4D-Humans 輸出，轉成 SkillMimic-V2 可以 replay / train 的格式。

如果你要的是**組員可以直接照抄命令操作**的版本，請看：

- [SMPLX_TO_SKILLMIMIC_PIPELINE_README.md](./SMPLX_TO_SKILLMIMIC_PIPELINE_README.md)

重點不是「把檔案副檔名改掉」，而是把下面這幾件事對齊：

- 時間範圍對齊：先切出單一 skill clip
- 世界座標對齊：把來源資料轉成 Isaac Gym 可用的座標系
- 骨架語義對齊：把 SMPL-X 的人體骨架轉成 SkillMimic 的 target skeleton
- tensor layout 對齊：最後打包成 BallPlay loader 會讀的 `[T, 337]`

---

## 1. 整體流程

我現在實際用的流程是：

```text
subject-1.smpl
  -> extract_smpl_clip.py
  -> 001_serve_0001.npz / 002_forehand_0001.npz

results.pkl + SMPL-X model
  -> results_pkl_to_source_joints.py
  -> source_joints [T, 22, 3]

clip .npz + source_joints
  -> smpl_to_canonical.py
  -> canonical .npz
     root_pos [T, 3]
     root_rot_3d [T, 3]
     dof_pos [T, 52, 3]
     body_pos [T, 53, 3]

canonical .npz
  -> canonical_to_skillmimic_pt.py
  -> motion .pt [T, 337]
```

對應腳本：

- `skillmimic/utils/extract_smpl_clip.py`
- `skillmimic/utils/results_pkl_to_source_joints.py`
- `skillmimic/utils/smpl_to_canonical.py`
- `skillmimic/utils/canonical_to_skillmimic_pt.py`

### 自動化入口

如果不想手動串 4 支腳本，現在可以直接用：

- `skillmimic/utils/smplx_to_skillmimic_pipeline.py`

它會自動串起：

1. clip extraction
2. `results.pkl -> source_joints`
3. canonical retarget
4. final `.pt` packing
5. optional validation

注意：

- 這支自動化入口目前對應的是**現有 human-only / Phase 1 流程**
- final packing 仍然使用目前的 `canonical_to_skillmimic_pt.py`
- 也就是說它現在產生的是 **dummy object + zero contact** 的 BallPlay `.pt`

如果之後要做 `Phase 2` 的真實球軌跡、桌子、contact timeline，
還需要另外擴充 final packing 和 env/task。

最常見的用法是：

```bash
python skillmimic/utils/smplx_to_skillmimic_pipeline.py \
  --smpl-path /path/to/subject-1.smpl \
  --results-pkl /path/to/results.pkl \
  --segments-json /path/to/subject-1_serve_segments.json \
  --output-root /path/to/output_bundle \
  --subject subject-1 \
  --coord-transform y_up_to_z_up \
  --phase1-human-only
```

如果影片本身就只有一段 skill，也可以不先手寫 segment JSON，改用 full-clip mode：

```bash
python skillmimic/utils/smplx_to_skillmimic_pipeline.py \
  --smpl-path /path/to/subject-1.smpl \
  --results-pkl /path/to/results.pkl \
  --output-root /path/to/output_bundle \
  --subject subject-1 \
  --full-clip-skill-id 1 \
  --full-clip-skill-name serve \
  --coord-transform y_up_to_z_up \
  --phase1-human-only
```

輸出會集中在：

- `<output-root>/clips`
- `<output-root>/source_joints`
- `<output-root>/canonical`
- `<output-root>/motions`
- `<output-root>/manifest.json`

---

## 2. Step 1: 先把整段 `.smpl` 切成單一 skill clip

### 做了什麼

`extract_smpl_clip.py` 會從 `.smpl` 壓縮檔中讀出：

- `bodyPose.npy`
- `bodyTranslation.npy`
- `frameRate.npy`
- `shapeParameters.npy`

然後按照你標的 segment JSON，把某一段 frame 切成一個 clip。對應程式在：

- `build_clip_payload()`：
  `skillmimic/utils/extract_smpl_clip.py:56`

### 例子

`serve` 的 clip 目前是：

- `data/clips/serve_01/subject-1/serve/001_serve_0001.npz`

裡面的 shape 是：

- `body_pose`: `(60, 22, 3)`
- `body_translation`: `(60, 3)`

`forehand` 的 clip 目前是：

- `data/clips/match4_001_forehand/subject-1/forehand/002_forehand_0001.npz`

裡面的 shape 是：

- `body_pose`: `(250, 22, 3)`
- `body_translation`: `(250, 3)`

### 這一步解決了什麼

這一步不是 retarget，只是先把「整場資料」切成「單一技能片段」。

也就是：

- `serve` 只留下發球那段
- `forehand` 只留下正手那段

---

## 3. Step 2: 從 `results.pkl` 解出真正的世界座標 joints

### 為什麼不能只用 `body_pose`

clip 裡的 `body_pose [T, 22, 3]` 是 local rotation，不是世界座標 joints。

如果你直接拿這個去當 target skeleton 的 joints，會有兩個問題：

- 它不是 joint position
- 它的 local frame 屬於 SMPL-X，不屬於 SkillMimic 的 MJCF 骨架

所以我另外用 `results.pkl` 裡的：

- `smplx_world.pose`
- `smplx_world.trans`
- `smplx_world.shape`

加上 SMPL-X body model，重新 decode 出每一幀的 joints。

### 這一步怎麼做

對應腳本：

- `skillmimic/utils/results_pkl_to_source_joints.py`

關鍵邏輯：

- 先驗證 clip 跟 `results.pkl` 的 frame slice 完全一致：
  `validate_clip_against_results()`
  `skillmimic/utils/results_pkl_to_source_joints.py:150`
- 再把 `smplx_world.pose [T, 165]` 拆成：
  - `global_orient`
  - `body_pose`
  - `jaw_pose`
  - `eye pose`
  - `left/right hand pose`
  `skillmimic/utils/results_pkl_to_source_joints.py:135`
- 然後丟給 `smplx.create(...)` 建出的 body model，拿回 joints：
  `decode_source_joints()`
  `skillmimic/utils/results_pkl_to_source_joints.py:180`

### 這一步的輸出

目前輸出的 `source_joints` 是：

- `data/source_joints/serve_01/subject-1/001_serve_0001.npy`
- `data/source_joints/match4_001_forehand/subject-1/002_forehand_0001.npy`

shape 分別是：

- `serve`: `(60, 22, 3)`
- `forehand`: `(250, 22, 3)`

### 這 22 個 joints 是什麼

目前固定順序在：

- `skillmimic/utils/results_pkl_to_source_joints.py:10`
- `skillmimic/utils/smpl_to_canonical.py:68`

也就是：

```text
Pelvis,
L_Hip, R_Hip,
Spine1,
L_Knee, R_Knee,
Spine2,
L_Ankle, R_Ankle,
Spine3,
L_Foot, R_Foot,
Neck,
L_Collar, R_Collar,
Head,
L_Shoulder, R_Shoulder,
L_Elbow, R_Elbow,
L_Wrist, R_Wrist
```

這是我後面 retarget 的 source skeleton。

---

## 4. Step 3: 轉成 canonical human-only clip

這一步是整個轉檔最重要的地方。

### canonical clip 長什麼樣

`smpl_to_canonical.py` 的輸出不是最終 `.pt`，而是一個中間 canonical `.npz`，包含：

- `root_pos [T, 3]`
- `root_rot_3d [T, 3]`
- `dof_pos [T, 52, 3]`
- `body_pos [T, 53, 3]`

例如目前：

- `data/canonical/serve_01/subject-1/001_serve_0001.npz`
- `data/canonical/match4_001_forehand/subject-1/002_forehand_0001.npz`

### 4.1 先把座標系對齊

來源資料的世界座標看起來是 Y-up，但 Isaac Gym / SkillMimic 這邊需要 Z-up。

所以我在 `smpl_to_canonical.py` 加了：

- `--coord-transform y_up_to_z_up`
- 對應矩陣在：
  `skillmimic/utils/smpl_to_canonical.py:181`

矩陣是：

```text
[ 1,  0,  0]
[ 0,  0, -1]
[ 0,  1,  0]
```

這一步會同時作用在：

- `body_pose`
- `body_translation`
- `source_joints`

對應邏輯：

- `skillmimic/utils/smpl_to_canonical.py:527`

如果不做這一步，最常見的症狀就是：

- root height 很怪
- 人會躺地或埋地

### 4.2 先讀 SkillMimic 的 target skeleton

我不是手寫一個 target skeleton，而是直接讀 asset：

- `skillmimic/data/assets/mjcf/mocap_humanoid.xml`

`load_target_skeleton()` 會把每個 body 的：

- `parent`
- `offset`
- `children`

解析出來。對應程式：

- `skillmimic/utils/smpl_to_canonical.py:283`

這樣做的好處是：

- target 骨架以實際 MJCF 為準
- 不會把你 replay 的骨架和轉檔時假設的骨架分離

### 4.3 骨架語義怎麼對齊

source skeleton 是 22 個 joints，但 target skeleton 是 53 個 bodies / 52 個 DOFs。

我做了兩層對齊：

#### A. anchor 對齊

先定義 major target joint 對應到哪個 source joint，例如：

- `Pelvis -> Pelvis`
- `Torso -> Spine1`
- `Spine -> Spine2`
- `Chest -> Spine3`
- `Neck -> Neck`
- `L_Shoulder -> L_Shoulder`
- `R_Wrist -> R_Wrist`

對應表在：

- `TARGET_SOURCE_ANCHOR`
  `skillmimic/utils/smpl_to_canonical.py:151`

#### B. child override

某些 target body 光靠預設 child mapping 不夠，例如：

- `Spine2 -> Chest`

這裡我特別讓它看 source 的 `Neck` 方向，而不是只看 `Spine3` 自己：

- `CHILD_SOURCE_OVERRIDE`
  `skillmimic/utils/smpl_to_canonical.py:177`

### 4.4 為什麼一開始 replay 會錯

一開始錯的原因是：我如果直接做

```text
dof_pos[:, target_idx] = body_pose[:, source_idx]
```

格式上雖然能湊出 `[T, 337]`，但語義是錯的。

因為：

- SMPL-X 的 local rotation 定義在 SMPL-X 自己的骨架 frame
- SkillMimic 的 DOF 定義在 MJCF target skeleton 的 joint frame

同一個 rotvec，放在不同骨架上，姿態不會相同。

這就是為什麼之前雖然能在 Isaac Gym 載入，但人物不像在做發球或 forehand。

### 4.5 我現在怎麼 retarget

現在用的是 `retarget_from_source_joints()`：

- `skillmimic/utils/smpl_to_canonical.py:397`

做法是：

1. 對每個 target body，先找到它對應的 source joint
2. 看這個 joint 指向它 children 的方向向量
3. 用 target skeleton 的 rest offset，去擬合一個 rotation
4. 把這個 rotation 當作 target body 的姿態
5. 再由 parent-child FK 算出 target `body_pos`

關鍵幾個 helper：

- `rotation_between_vectors()`
  `skillmimic/utils/smpl_to_canonical.py:334`
- `fit_rotation()`
  `skillmimic/utils/smpl_to_canonical.py:362`

這等於是：

- source 用 joint positions 提供「骨頭朝哪裡」
- target 用自己的 rest skeleton 去重建 local/global rotation

### 4.6 head / wrist 這種末端關節怎麼辦

像 `Head`、`L_Wrist`、`R_Wrist` 這種末端 joint，單看 child direction 時資訊比較少。

所以我加了一個 fallback：

- 如果幾何方向不夠，就退回 source 的 local pose

對應邏輯：

- `source_local_pose=body_pose`
- `skillmimic/utils/smpl_to_canonical.py:431`

這能避免：

- 頭整段僵死
- 手腕完全沒旋轉

### 4.7 手指怎麼處理

目前 source 沒有完整 53-body 對應到 SkillMimic 手指骨架，所以手指先用 proxy：

- 左手所有 finger body 先跟 `L_Wrist`
- 右手所有 finger body 先跟 `R_Wrist`

對應表在：

- `BODY_PROXY`
  `skillmimic/utils/smpl_to_canonical.py:118`

這代表：

- 大動作會對
- 手指細節目前只是近似

---

## 5. Step 4: 打包成 SkillMimic BallPlay 的 `[T, 337]`

canonical 只是中間格式，最後還要打包成 `.pt`。

對應腳本：

- `skillmimic/utils/canonical_to_skillmimic_pt.py`

### 打包規則

`pack_human_only_pt()` 會建立：

- `motion = np.zeros((T, 337), dtype=np.float32)`

然後把欄位塞進固定位置：

```text
0:3      root_pos
3:6      root_rot_3d
6:9      reserved
9:165    dof_pos         = 52 * 3
165:324  body_pos        = 53 * 3
324:327  obj_pos
327:330  obj_rot_3d
330:336  reserved
336:337  contact
```

對應程式：

- `skillmimic/utils/canonical_to_skillmimic_pt.py:54`

這個 layout 也和 loader 對得上：

- `skillmimic/utils/motion_data_handler.py:147`

### 為什麼 human-only 還要放 object/contact

因為你現在跑的是：

- `SkillMimic1BallPlay`

它的 motion loader 仍然會去 slice：

- `obj_pos`
- `obj_rot`
- `contact`

所以就算目前只做 human-only，我還是補了 dummy 欄位：

- `obj_pos = [2.0, 0.0, 1.0]`
- `obj_rot_3d = [0, 0, 0]`
- `contact = 0`

對應程式：

- `skillmimic/utils/canonical_to_skillmimic_pt.py:58`

---

## 6. 目前兩個實際例子

### serve

來源：

- clip:
  `data/clips/serve_01/subject-1/serve/001_serve_0001.npz`
- source joints:
  `data/source_joints/serve_01/subject-1/001_serve_0001.npy`
- canonical:
  `data/canonical/serve_01/subject-1/001_serve_0001.npz`
- final pt:
  `skillmimic/data/motions/TableTennis/serve/001_serve_0001.pt`

shape：

- clip `body_pose`: `(60, 22, 3)`
- source_joints: `(60, 22, 3)`
- canonical `dof_pos`: `(60, 52, 3)`
- canonical `body_pos`: `(60, 53, 3)`
- final `.pt`: `(60, 337)`

### forehand

來源：

- clip:
  `data/clips/match4_001_forehand/subject-1/forehand/002_forehand_0001.npz`
- source joints:
  `data/source_joints/match4_001_forehand/subject-1/002_forehand_0001.npy`
- canonical:
  `data/canonical/match4_001_forehand/subject-1/002_forehand_0001.npz`
- final pt:
  `skillmimic/data/motions/TableTennis/forehand/002_forehand_0001.pt`

shape：

- clip `body_pose`: `(250, 22, 3)`
- source_joints: `(250, 22, 3)`
- canonical `dof_pos`: `(250, 52, 3)`
- canonical `body_pos`: `(250, 53, 3)`
- final `.pt`: `(250, 337)`

---

## 7. 怎麼驗證這個轉檔不是只有 shape 對

我目前用兩層驗證。

### A. 靜態 validator

腳本：

- `skillmimic/utils/validate_motion_pt.py`

會檢查：

- shape 是否為 `[T, 337]`
- reserved 欄位是否近似 0
- pelvis 和 `root_pos` 是否一致
- root height 是否合理
- object/contact 是否符合 human-only 假設
- 骨長是否穩定

### B. Isaac Gym replay

命令是：

```bash
python skillmimic/run.py \
  --play_dataset \
  --test \
  --task SkillMimic1BallPlay \
  --num_envs 1 \
  --motion_file skillmimic/data/motions/TableTennis/serve \
  --state_init 2
```

以及：

```bash
python skillmimic/run.py \
  --play_dataset \
  --test \
  --task SkillMimic1BallPlay \
  --num_envs 1 \
  --motion_file skillmimic/data/motions/TableTennis/forehand \
  --state_init 2
```

這一步看的是：

- 人會不會炸掉
- root 高度是否合理
- 朝向對不對
- 動作語義是不是像 `serve / forehand`

---

## 8. 目前這個版本的限制

這版已經能做到：

- 資料格式正確
- replay 可以看
- `serve / forehand` 大動作語義大致合理

但還有這些限制：

- 手指細節還是 proxy，不是完整 SMPL-X hand retarget
- 頭部與上半身的 twist 仍然是近似重建，不是完整 global rotation solve
- 目前是 human-only 流程，object/contact 還是 dummy 值

所以這版比較準確的定位是：

- 已經是可 replay、可當 Phase 1 human-only 訓練基礎的版本
- 但不是最終高保真動作重建版

---

## 9. 一句話總結

我不是把 SMPL-X 的 rotvec 直接塞進 SkillMimic。

我實際做的是：

1. 先把整段 `.smpl` 切成 skill clip
2. 用 `results.pkl + SMPL-X model` 解出 source world joints
3. 把 source joints 轉到 SkillMimic target skeleton 的座標系和骨架語義
4. 重建 target 的 `root_pos / root_rot_3d / dof_pos / body_pos`
5. 最後打包成 BallPlay loader 可讀的 `[T, 337]`

這就是目前 `serve` 和 `forehand` 能在 Isaac Gym 裡 replay 的原因。

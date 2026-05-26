# Table Tennis Training Phases

更新日期: `2026-05-26`

這份文件專門回答一個問題:

**目前這個 repo 裡，桌球 skill 應該分成哪幾個 phase 訓練，以及每個 phase 的 reference、reward、物件配置應該怎麼設計。**

---

## 1. Executive Summary

建議把訓練拆成兩個 phase:

- `Phase 1`: 學 **人體 skill 動作**  
  不追求真實發球結果，先學站姿、揮拍、重心轉移。

- `Phase 2`: 學 **人體 + 球 + 桌子** 的完整 skill  
  這時才要求球在手邊、拋球、擊球、落桌或飛行。

這兩個 phase 的差別，不只是 reward 不同，而是:

- reference 的內容不同
- environment 內需要的物件不同
- contact 訊號是否值得打開也不同

另外要先講清楚一個現實限制:

**你目前 repo 裡的 `SkillMimic1BallPlay` 會固定生成一顆 ball actor。**

也就是說，理想上的 `Phase 1 = 無球`，在現有實作裡更準確的說法是:

**Phase 1 = 不學球，只學人體。球可以保留成 dummy actor，但不讓它進 reward。**

---

## 2. Recommended Training Mode

### Phase 1

先用:

- `SkillMimic1BallPlay`

不要一開始就用:

- `SkillMimic2BallPlay`
- `AMP`

原因:

- `SkillMimic1BallPlay` 最直接，最容易 debug
- 你現在只有單一 `serve` reference
- 你現在的 `serve` 還不是完整人球互動 reference

### Phase 2

第一版仍然建議先用:

- `SkillMimic1BallPlay`

等到以下條件成立，再考慮升級到:

- `SkillMimic2BallPlay`

適合升級的時機:

- 有多段 `serve` 或多個 skill clip
- 單一 reference tracking 已經穩
- 想加入 history encoder / state graph 來提升初始化與追蹤穩定性

結論:

- `Phase 1`: `SkillMimic1BallPlay`
- `Phase 2`: 先 `SkillMimic1BallPlay`，成熟後再考慮 `SkillMimic2BallPlay`

---

## 3. Phase Comparison

| 項目 | Phase 1 | Phase 2 |
|---|---|---|
| 目標 | 學人體 skill 動作 | 學完整人球桌互動 |
| 主要能力 | 姿態、節奏、揮拍、重心轉移 | 持球、拋球、擊球、球飛行/落桌 |
| task | `SkillMimic1BallPlay` | `SkillMimic1BallPlay` 起步，之後可升 `SkillMimic2BallPlay` |
| reference | human-only 為主 | human + ball trajectory |
| 球 | dummy 或不參與 reward | 真實 ball trajectory |
| 桌子 | 不需要 | 需要 |
| contact graph | `cg2` 不需要 | 可用，但前提是 contact 標註可信 |
| 程式狀態 | 現在就能做 | 還需要補資料與 env |

---

## 4. Phase 1

### 4.1 目標

Phase 1 的目標不是讓 agent 真正把球發出去，而是先學會:

- 發球預備姿勢
- 上肢揮拍動作
- 軀幹旋轉
- 下肢支撐與重心轉移

這一階段你在訓練的是:

**skill body motion imitation**

---

### 4.2 Phase 1 需要的物件

最小配置:

- humanoid
- 固定掛在 `R_Wrist` 上的 racket asset
- 地板

在理想設計上:

- 不需要球
- 不需要桌子

但在你目前這個 repo 的現實做法裡:

- `BallPlay` task 仍然會生成一顆 ball actor

所以實際上是:

- humanoid
- racket
- floor
- **dummy ball actor**

這顆 dummy ball 的處理原則是:

- 不讓它成為學習目標
- 不讓它主導 reward
- 可以放在固定位置，或維持 dummy reference

### Phase 1A 建議 asset

如果你要先做「有手、無球拍」版本，現在可以直接用:

- `mjcf/mocap_humanoid_hand_only.xml`

這個 asset 的特性是:

- 保留完整右手
- 不包含球拍 mesh
- DOF 和 body 結構不變

很適合拿來做:

- `Phase 1A`: pure body imitation

之後再切回:

- `mjcf/mocap_humanoid.xml`

做有球拍的 fine-tune。

---

### 4.3 Phase 1 的 Reference / Ground Truth 怎麼設計

Phase 1 的 final `.pt` 仍然要符合 SkillMimic loader 期待的 `[T, 337]` 格式。

但只有人體欄位是有語義的:

- `root_pos`
- `root_rot_3d`
- `dof_pos`
- `body_pos`

球相關欄位可以先是 dummy:

- `obj_pos`: 固定值
- `obj_rot`: 固定 identity
- `obj_pos_vel`: 0
- `obj_rot_vel`: 0
- `contact`: 0

你現在的 `serve` 就屬於這一類。實際檢查結果是:

- `obj_pos` 全 clip 只有一個固定值
- `contact` 全部是 `0`

所以這份資料適合拿來學:

- body motion

不適合直接拿來學:

- 真實發球互動

---

### 4.4 Phase 1 的 Reward 怎麼設計

### 核心原則

Phase 1 要把 reward 集中在 **人體 tracking**，把球相關 reward 關掉或降到接近 0。

目前 `SkillMimic1BallPlay` 的 reward 是乘法結構:

`body * object * interaction_graph * contact`

這代表:

**只要 object 或 contact 分支有一個很差，總 reward 就會一起被拉低。**

因此 Phase 1 不適合讓 dummy ball 相關項目參與學習。

### 建議打開的項目

- `p`: body position
- `r`: body rotation
- `cg1`: body contact regularization

`cg1` 的作用不是「學接觸圖」，而是避免整個人亂撞地面、亂倒。

### 建議先關掉的項目

- `op`: object position
- `opv`: object velocity
- `ig`: interaction graph
- `cg2`: object contact

如果球是 dummy:

- `op` 沒有意義
- `ig` 沒有意義
- `cg2` 沒有意義

### Phase 1 實務設定

最簡單的做法是直接用 CLI 蓋掉權重:

```bash
python skillmimic/run.py \
  --task SkillMimic1BallPlay \
  --num_envs 128 \
  --episode_length 60 \
  --cfg_env skillmimic/data/cfg/skillmimic.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
  --motion_file skillmimic/data/motions/TableTennis/serve \
  --state_init Random \
  --op 0 \
  --ig 0 \
  --cg2 0 \
  --max_epochs 200 \
  --output_path output/serve_phase1 \
  --headless
```

這條命令代表:

- 只學 `serve`
- 只看人體 imitation
- dummy ball 不進主要 reward

如果你要明確指定 `Phase 1A` 的 hand-only asset，可以再加:

```bash
--asset_file_name mjcf/mocap_humanoid_hand_only.xml
```

---

### 4.5 Phase 1 需不需要 Contact Graph

### 不需要的部分

- `cg2` 不需要

因為它是 object contact imitation。

### 可以保留的部分

- `cg1` 可以保留

因為它更像是 anti-fall / anti-body-collision regularizer，不是要求你模仿真實球接觸。

### 結論

Phase 1:

- **不需要 object contact graph**
- `cg1` 可以留
- `cg2` 建議關掉

---

## 5. Phase 2

### 5.1 目標

Phase 2 才是真正學:

- 球在手邊
- 拋球
- 擊球
- 球飛行
- 球落桌或與桌面互動

這一階段你在訓練的是:

**skill + ball + table interaction**

---

### 5.2 Phase 2 需要的物件

最小配置:

- humanoid
- racket
- dynamic ball
- static table
- floor

這裡有個重點:

**桌子不是 reference tensor 裡一定要新增的一個欄位。**

如果桌子是靜態且固定的，它可以存在於 environment 裡，不一定要寫進 final `.pt`。

但要滿足一個條件:

**reference ball trajectory 必須和 table 的位置對齊。**

也就是:

- reference 說球會落在桌上
- env 裡的桌子也必須在那個位置

否則球的 reference 和真實物理場景會互相打架。

### 目前 repo 的現況

目前 `SkillMimic1BallPlay` 這條線天然有:

- humanoid
- ball

但 **沒有桌子**。

所以要做真正的 Phase 2，你還需要:

- 擴充目前的 BallPlay env
- 或新增一個 table-tennis task，讓它在 scene 裡載入 table

這一點和 `Parahome` 那條有靜態場景物件的做法相似，但桌球這邊目前還沒接上。

---

### 5.3 Phase 2 的 Reference / Ground Truth 怎麼設計

Phase 2 的關鍵不只是人體 motion，還要讓 ball 的 reference 變成真的。

### 最低限度要有的球軌跡

以 `serve` 為例，reference 最少應該包含這些階段:

1. 球一開始在手附近
2. 球被拋起來
3. 球接近拍面
4. 球被擊出
5. 球往前飛
6. 如果 clip 包含落桌，球和桌面的互動也要合理

### 哪些欄位要變成有語義

Phase 2 至少要讓這些欄位有真實內容:

- `obj_pos`
- `obj_pos_vel`
- `contact`

對球來說，`obj_rot` 常常不重要，因為球是近似球體，旋轉對觀察和 reward 的重要性比位置小得多。

### Table 要不要進 reference

如果桌子是:

- 固定位置
- 固定尺寸
- 所有 clip 都用同一張桌子設定

那通常:

- table 不必直接進 `[T, 337]`
- 但 env 裡一定要有它

如果之後你要做:

- 多桌型
- 多桌位置
- 場景可變

那才需要重新思考 table state 是否要進 observation / dataset。

---

### 5.4 Phase 2 的 Reward 怎麼設計

### 核心原則

Phase 2 才讓 object reward 和 interaction reward 回來。

但不要一次把所有項目都打滿，因為目前 reward 是乘法結構，太早打開 noisy 的 object/contact 項，訓練很容易變脆弱。

### Phase 2 建議的開啟順序

#### Step A: 先讓球軌跡對上

優先打開:

- `p`
- `r`
- `op`
- `opv`
- `cg1`

`opv` 在 `serve` 很重要，因為只看球位置，無法很好約束:

- 拋球速度
- 擊球後飛行速度

#### Step B: 再把 interaction 加回來

當 ball trajectory 已經合理之後，再逐步打開:

- `ig`

`ig` 是人體 keypoints 相對球的位置關係。  
它比單純的球位置更接近「人和球互動方式」。

#### Step C: 最後才加 contact

最後再考慮打開:

- `cg2`

前提是:

- `contact` 標註本身可信
- reference 和模擬中的接觸時序大致對得上

如果 contact 標註不準，`cg2` 很容易變成噪聲。

---

### 5.5 Phase 2 需不需要 Contact Graph

### 短答案

- 不是一開始就必須
- 但後期值得加

### 為什麼

如果你的 ball trajectory 已經很好，agent 常常可以先靠:

- `op`
- `opv`
- `ig`

把大部分的人球互動學起來。

這時 contact graph 不是唯一必要條件。

### 但目前 contact graph 的表達能力有限

目前 final `.pt` 只有:

- 一個 scalar `contact`

這代表它不能區分:

- 球碰手
- 球碰拍
- 球碰桌

它只是一個:

- **球此刻是否在接觸某個相關物體**

所以如果你未來真的想精準約束:

- toss release
- racket hit
- table bounce

那現在這個一維 `contact` 其實不夠，需要擴充資料格式或 reward。

### 結論

Phase 2:

- **最小可行版不一定要先靠 contact graph**
- 先把 ball trajectory 做對更重要
- `cg2` 應該放在後期增強，不是起手式

---

## 6. Recommended Ground Truth Design

### Phase 1 Ground Truth

建議設計成:

- 人體 motion 正確
- 球欄位是 dummy
- `contact = 0`
- 沒有桌子

這樣你的 reference 表示的是:

**純人體 serve 動作**

### Phase 2 Ground Truth

建議設計成:

- 人體 motion 正確
- 球 trajectory 真實
- 球速度真實
- `contact` 至少是合理的 binary timeline
- table 放在 env，位置和 reference ball trajectory 對齊

這樣你的 reference 表示的是:

**完整人球桌 serve**

---

## 7. Recommended Roadmap

### Phase 1 先做什麼

1. 用目前 `serve` 先跑人體 imitation
2. 關掉 `op`, `ig`, `cg2`
3. 確認 agent 能學到 serve body motion

### Phase 2 再做什麼

1. 補第一版真實 ball trajectory
2. 為桌球 env 加入 table
3. 先用 `p`, `r`, `op`, `opv`, `cg1`
4. 再逐步加入 `ig`
5. 最後再決定是否打開 `cg2`

---

## 8. What You Need Before Training

### 現在就能開始的

你現在已經可以開始:

- `Phase 1`

因為你已經有:

- 可 replay 的 `serve` motion
- 合法的 final `.pt`
- 可跑的 `SkillMimic1BallPlay`

### 還不能直接開始的

你現在還不能完整開始:

- `Phase 2`

因為你還缺:

1. 真實的 serve ball trajectory
2. 合理的 ball contact 標註
3. BallPlay env 裡的 table

---

## 9. Final Recommendation

對你目前這個專案，最穩的訓練策略是:

1. **先把 Phase 1 跑通**
2. 確認 agent 能學到 `serve body motion`
3. 再開始補 `Phase 2` 的球和桌子

不要一開始就把目標定成:

- 真實拿球
- 真實擊球
- 真實落桌

因為你目前的 reference 和 env 還沒同時滿足這些條件。

正確順序是:

- 先學 **body skill**
- 再學 **body + ball + table**

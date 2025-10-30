# 🤖 AI 交接手冊 - Nova Carter 導航訓練專案

> **版本**：2025-10-30  
> **專案**：Nova Carter 本地規劃器強化學習環境  
> **目的**：讓下一位 AI 快速接手並繼續工作

---

## 📋 目錄

1. [專案概述](#專案概述)
2. [當前狀態](#當前狀態)
3. [關鍵歷史決策](#關鍵歷史決策)
4. [代碼結構](#代碼結構)
5. [訓練配置](#訓練配置)
6. [已知問題與解決方案](#已知問題與解決方案)
7. [用戶偏好與原則](#用戶偏好與原則)
8. [下一步建議](#下一步建議)

---

## 🎯 專案概述

### 任務目標
訓練 Nova Carter 移動機器人在有障礙物的環境中，使用 360° LiDAR 進行自主導航，從起點到達隨機生成的目標點。

### 技術棧
- **模擬器**：NVIDIA Isaac Sim 5.0
- **框架**：Isaac Lab v2.2
- **機器人**：Nova Carter（差速驅動）
- **演算法**：PPO (Proximal Policy Optimization)
- **RL 庫**：RSL-RL
- **Python**：3.11
- **CUDA**：12.x

### 環境規格
- **觀測空間**：369 維
  - LiDAR [360]：每度 1 條射線，標準化到 [0, 1]
  - 線速度 [3]：`[vx, vy, vz]` 機器人座標系
  - 角速度 [3]：`[wx, wy, wz]` 機器人座標系
  - 目標相對位置 [2]：`[dx, dy]` 機器人座標系
  - 目標距離 [1]：歐幾里得距離

- **動作空間**：2 維
  - 線速度指令 [1]：範圍 0.8 m/s
  - 角速度指令 [1]：範圍 0.8 rad/s
  - 通過差速驅動轉換為左右輪速度

- **獎勵函數**（當前最佳配置）：
  ```
  總獎勵 = 
    + progress_to_goal × 15.0       # 接近目標
    + near_goal_shaping × 10.0      # 近距離塑形（1.5m 內）
    + heading_alignment × 5.0       # 朝向對齊
    + reached_goal × 200.0          # 到達目標
    - standstill × 2.0              # 靜止不動
  ```

---

## 📊 當前狀態

### 最新訓練結果
- **日期**：2025-10-27 22:13
- **位置**：`logs/rsl_rl/local_planner_carter/2025-10-27_20-56-40/`
- **迭代次數**：3000 次
- **結果**：訓練尚未收斂
  - Mean Reward：波動劇烈，未穩定上升
  - Success Rate：< 30%
  - Timeout Rate：70-90%
  - Position Error：約 1.4m（未收斂）
  - Collision Rate：低（安全性佳）

### 診斷結論（基於 TensorBoard 分析）
1. **Entropy 持續上升** → 探索過強，策略不穩定
2. **Value/Surrogate Loss 劇烈波動** → PPO 更新幅度過大
3. **Position Error 停滯** → 缺乏中間層次的 reward shaping
4. **方向控制良好，位置不準** → Agent 能朝向目標但無法抵達

### 已實施的修正（2025-10-30）

#### 1. PPO 穩定化調整
**文件**：`agents/rsl_rl_ppo_cfg.py`

```python
# 修改前（不穩定）
init_noise_std = 1.0
entropy_coef = 0.01
clip_param = 0.2
num_learning_epochs = 5

# 修改後（穩定化）✅
init_noise_std = 0.5
entropy_coef = 0.001      # 降低 10 倍
clip_param = 0.1          # 降低更新幅度
num_learning_epochs = 3   # 減少過擬合
```

**原因**：
- 降低 entropy 避免探索過強
- 縮小 clip_param 避免策略劇烈變化
- 減少 epochs 避免在小批次上過度更新

#### 2. 獎勵 Shaping 改進
**文件**：`mdp/rewards.py` + `local_planner_env_cfg_min.py`

**新增兩項獎勵**：

1. **near_goal_shaping**（近距離塑形）
   - 位置：`rewards.py` 第 196-246 行
   - 功能：在 1.5m 範圍內給予額外正獎勵，隨距離線性遞減
   - 公式：`reward = max(0, (1.5 - distance) / 1.5)`
   - 權重：10.0
   - 作用：解決「最後一公里」問題，引導 Agent 逼近目標

2. **heading_alignment_reward**（朝向對齊）
   - 位置：`rewards.py` 第 249-276 行
   - 功能：鼓勵機器人朝向目標方向
   - 公式：`cos(heading_error) = goal_x / ||goal_xy||`
   - 權重：5.0
   - 作用：避免機器人側身或倒退接近目標

**調整後的獎勵權重**：
```python
progress_to_goal: 15.0      # 從 50.0 降低（避免壓倒其他信號）
near_goal_shaping: 10.0     # 新增
heading_alignment: 5.0      # 新增
reached_goal: 200.0         # 從 500.0 降低（更溫和）
standstill: -2.0            # 從 -0.1 增強（強制移動）
```

#### 3. 動作與成功條件調整
**文件**：`local_planner_env_cfg_min.py`

```python
# 降低最大速度（提升穩定性）
max_linear_speed: 2.0 → 0.8
max_angular_speed: π → 0.8

# 放寬成功閾值（促進早期成功）
goal_reached threshold: 0.5m → 0.8m
```

**原因**：
- 降低速度減少高速震盪，利於早期訓練
- 放寬閾值讓 Agent 更容易獲得成功獎勵，建立正向學習循環

#### 4. 環境精簡
**刪除的配置檔**（保持簡潔）：
- `local_planner_env_cfg_cpu.py`
- `local_planner_env_cfg_debug.py`
- `local_planner_env_cfg_demo.py`
- `local_planner_env_cfg_dynamic.py`
- `local_planner_env_cfg_easy.py`
- `local_planner_env_cfg_gpu_optimized.py`
- `local_planner_env_cfg_gpu_optimized_fixed.py`
- `local_planner_env_cfg_gui_fixed.py`
- `local_planner_env_cfg_isaac_sim_5_fixed.py`
- `local_planner_env_cfg_pccbf.py`
- `local_planner_env_cfg_pccbf_simple.py`
- `local_planner_env_cfg_simple_v2.py`

**保留的配置檔**：
- `local_planner_env_cfg.py`：標準版（含動態障礙物）
- `local_planner_env_cfg_min.py`：最小版（推薦，僅 Terrain + Robot + LiDAR + Goal）⭐

#### 5. Gym 環境註冊簡化
**文件**：`__init__.py`

**保留的環境**：
- `Isaac-Navigation-LocalPlanner-Carter-v0`（標準版）
- `Isaac-Navigation-LocalPlanner-Carter-Simple-v0`（簡化版）
- `Isaac-Navigation-LocalPlanner-Min-v0`（最小版，推薦）⭐

---

## 🗂️ 代碼結構

### 核心文件位置

```
IsaacLab/
├─ source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/local_planner/
│  ├─ __init__.py                          # 環境註冊（已精簡）
│  ├─ local_planner_env_cfg.py            # 標準環境配置
│  ├─ local_planner_env_cfg_min.py        # 最小環境配置 ⭐
│  │
│  ├─ agents/
│  │  ├─ rsl_rl_ppo_cfg.py                # PPO 配置（已穩定化）⭐
│  │  └─ sb3_ppo_cfg.py                   # Stable Baselines3 配置
│  │
│  └─ mdp/                                 # MDP 組件實現 ⭐
│     ├─ observations.py                  # 觀測函數（含 ray_hits_w API）
│     ├─ actions.py                       # 動作轉換（差速驅動）
│     ├─ rewards.py                       # 獎勵函數（含新增 shaping）⭐
│     └─ terminations.py                  # 終止條件
│
├─ scripts/reinforcement_learning/rsl_rl/
│  ├─ train.py                            # 訓練腳本 ⭐
│  └─ play.py                             # 測試腳本
│
└─ md/                                     # 文檔目錄
   ├─ 訓練架構完整說明.md                  # 技術詳細文檔 ⭐
   ├─ 指令快速參考.md                      # 命令速查表 ⭐
   └─ AI交接手冊.md                        # 本文檔 ⭐
```

### 重要代碼段落

#### 1. PPO 配置（rsl_rl_ppo_cfg.py）
- **第 34-39 行**：Actor-Critic 網路架構
  - `[256, 256, 128]`，ELU 激活
  - `init_noise_std=0.5`（已降低）

- **第 42-55 行**：PPO 超參數
  - `learning_rate=3e-4`（穩定）
  - `entropy_coef=0.001`（已降低）⭐
  - `clip_param=0.1`（已降低）⭐
  - `num_learning_epochs=3`（已降低）⭐

#### 2. 最小環境配置（local_planner_env_cfg_min.py）
- **第 41-55 行**：地形（平面）
- **第 57-74 行**：Nova Carter 機器人
- **第 76-88 行**：LiDAR（360° 2D，必須包含 `mesh_prim_paths`）
- **第 97-110 行**：動作配置（差速驅動，0.8 m/s）
- **第 152-171 行**：獎勵配置（5 項極簡設計）⭐
- **第 175-180 行**：終止條件（0.8m 閾值）

#### 3. 獎勵函數實現（mdp/rewards.py）
- **第 20-62 行**：`progress_to_goal_reward`（基礎進度）
- **第 65-84 行**：`reached_goal_reward`（成功獎勵）
- **第 179-193 行**：`standstill_penalty`（靜止懲罰）
- **第 196-246 行**：`near_goal_shaping`（新增，近距離塑形）⭐
- **第 249-276 行**：`heading_alignment_reward`（新增，朝向對齊）⭐

#### 4. 觀測函數（mdp/observations.py）
- **第 21-66 行**：`lidar_obs`（支援 Isaac Sim 5.0 `ray_hits_w` API）⭐
- **第 35-44 行**：重點！Isaac Sim 5.0 需要手動計算距離
  ```python
  if hasattr(data, "ray_hits_w") and hasattr(data, "pos_w"):
      hit_points = data.ray_hits_w
      sensor_pos = data.pos_w.unsqueeze(1)
      distances = torch.norm(hit_points - sensor_pos, dim=-1)
  ```

---

## ⚙️ 訓練配置

### 推薦訓練指令

#### 基礎訓練（最小環境，推薦）⭐
```bash
cd /home/aa/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Min-v0 \
    --num_envs 24 \
    --max_iterations 10000 \
    --headless
```

**說明**：
- 使用最小環境（已包含所有穩定化改進）
- 24 個並行環境（平衡速度與穩定性）
- 10000 次迭代（至少需要這麼多才能收斂）
- Headless 模式（更快，無 GUI）

#### 測試模型
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Min-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/local_planner_carter/[日期]/model_[iteration].pt
```

#### 觀察訓練曲線
```bash
tensorboard --logdir logs/rsl_rl/local_planner_carter/
# 瀏覽器：http://localhost:6006
```

### 訓練參數總結

| 類別 | 參數 | 值 | 說明 |
|------|------|-----|------|
| **環境** | num_envs | 24 | 並行環境數 |
| | episode_length_s | 30.0 | 回合時間（秒）|
| | decimation | 2 | 物理步/RL步 |
| | sim.dt | 0.01 | 物理時間步 |
| **PPO** | learning_rate | 3e-4 | 學習率 |
| | entropy_coef | 0.001 | 探索強度（已降低）⭐ |
| | clip_param | 0.1 | PPO clip（已降低）⭐ |
| | num_epochs | 3 | 每輪更新次數（已降低）⭐ |
| | num_mini_batches | 4 | Mini-batch 數量 |
| | gamma | 0.99 | 折扣因子 |
| | lam | 0.95 | GAE lambda |
| **網路** | actor_dims | [256,256,128] | Actor 隱藏層 |
| | critic_dims | [256,256,128] | Critic 隱藏層 |
| | activation | ELU | 激活函數 |
| | init_noise_std | 0.5 | 初始噪音（已降低）⭐ |
| **動作** | max_linear | 0.8 m/s | 最大線速度（已降低）⭐ |
| | max_angular | 0.8 rad/s | 最大角速度（已降低）⭐ |
| **成功** | threshold | 0.8m | 成功距離（已放寬）⭐ |

### 預期訓練時間
- **24 envs × 10000 iterations**：約 3-4 小時（RTX 3090/4090）
- **檢查點保存**：每 100 次迭代
- **總訓練步數**：24 × 24 × 10000 = 5,760,000 步

---

## ❗ 已知問題與解決方案

### 1. Isaac Sim 5.0 API 變更
**問題**：`ray_hits_w` API 不再直接提供距離數據

**解決方案**：在 `observations.py` 手動計算
```python
if hasattr(data, "ray_hits_w") and hasattr(data, "pos_w"):
    hit_points = data.ray_hits_w  # (num_envs, num_rays, 3)
    sensor_pos = data.pos_w.unsqueeze(1)  # (num_envs, 1, 3)
    distances = torch.norm(hit_points - sensor_pos, dim=-1)
```

**位置**：`mdp/observations.py` 第 35-44 行

### 2. Isaac Lab v2.2 模組重命名
**問題**：`omni.isaac.lab` → `isaaclab`

**解決方案**：所有導入已更新為新 API
```python
# ✅ 正確
from isaaclab.utils import configclass

# ❌ 舊版（不再支援）
from omni.isaac.lab.utils import configclass
```

### 3. LiDAR mesh_prim_paths 缺失
**問題**：`TypeError: Missing values detected... scene.lidar.mesh_prim_paths`

**解決方案**：LiDAR 配置必須包含 `mesh_prim_paths`
```python
lidar = RayCasterCfg(
    prim_path="/World/envs/.*/Robot/Robot/chassis_link/base_link",
    mesh_prim_paths=["/World/ground"],  # ⭐ 必須！
    ...
)
```

**已修復**：`local_planner_env_cfg_min.py` 第 78 行

### 4. Play 腳本地形初始化錯誤
**問題**：`Stage.GetPrimAtPath(Stage, NoneType)`

**狀態**：已知問題，但不影響訓練

**暫時方案**：
1. 優先使用 TensorBoard 查看結果
2. 如果需要可視化，清理進程後重試：
   ```bash
   killall -r "kit"
   sleep 5
   # 然後重新執行 play.py
   ```

### 5. 訓練不收斂（原始配置）
**問題**：Entropy 上升、Loss 波動、Success Rate 低

**原因**：
- PPO 參數過於激進（entropy_coef=0.01, clip=0.2）
- 缺乏中間層次的 reward shaping
- 動作速度過快（2.0 m/s）
- 成功閾值過嚴（0.5m）

**解決方案**：已全部實施（見「當前狀態 - 已實施的修正」）

---

## 👤 用戶偏好與原則

### 溝通風格
1. **使用繁體中文**：所有回覆與文檔
2. **簡潔直接**：不需要過多客套，直接給解決方案
3. **代碼優先**：直接修改代碼而非只提建議
4. **重複確認需求**：在執行前重述用戶需求，確保理解正確

### 工作原則
1. **主動完成**：不等待用戶多次提點，一次做完
2. **保持整潔**：刪除多餘檔案，只保留必要的
3. **文檔齊全**：重要決策都要記錄在文檔中
4. **Git 管理**：定期提交並推送到 GitHub

### 技術偏好
1. **極簡設計**：寧可簡單有效，不要複雜花俏
2. **理論驗證**：參考學術論文（如 PCCBF-MPC）但不過度實現
3. **實用主義**：如果簡單方法有效，不必追求完美
4. **穩定優先**：訓練穩定性比最高性能更重要

### 專案管理
1. **版本控制**：所有重要變更都要 commit
2. **文檔同步**：代碼改動後立即更新說明文檔
3. **清理冗餘**：定期刪除無用的實驗代碼
4. **交接友好**：為下一位維護者著想

---

## 🔄 關鍵歷史決策

### 決策 1：極簡獎勵設計（2025-10-30）
**背景**：複雜獎勵（7 項）導致訓練不穩定

**決策**：採用極簡設計（5 項）
- 保留：progress、reached_goal、standstill
- 新增：near_goal_shaping、heading_alignment
- 移除：obstacle_proximity（在無障礙物環境中無意義）、collision（同上）

**結果**：待驗證（需要新的訓練結果）

**理由**：
- 減少獎勵項之間的衝突
- 每項獎勵都有明確的學習目標
- 避免過度工程化

### 決策 2：環境配置精簡（2025-10-30）
**背景**：多達 13 個環境配置檔案，難以維護

**決策**：刪除 11 個，僅保留 2 個
- 保留：`local_planner_env_cfg.py`（標準）、`local_planner_env_cfg_min.py`（最小）
- 刪除：CPU、GPU 優化、GUI 修復、PCCBF、Debug 等變體

**結果**：代碼庫清晰，易於理解

**理由**：
- 多數變體是實驗性質，不需要長期維護
- 用戶希望「只保留要用的」
- 降低新 AI 理解專案的門檻

### 決策 3：PPO 穩定化（2025-10-30）
**背景**：TensorBoard 顯示 entropy 上升、loss 波動

**決策**：降低探索強度與更新幅度
- `entropy_coef: 0.01 → 0.001`
- `clip_param: 0.2 → 0.1`
- `num_epochs: 5 → 3`
- `init_noise_std: 1.0 → 0.5`

**結果**：待驗證

**理由**：
- 參考標準 PPO 實踐（OpenAI Spinning Up）
- 用戶訓練結果顯示明顯的過度探索症狀
- 業界共識：導航任務不需要高 entropy

### 決策 4：放寬成功閾值（2025-10-30）
**背景**：Agent 在 1.4m 附近徘徊，永遠無法觸發成功

**決策**：`goal_reached threshold: 0.5m → 0.8m`

**結果**：待驗證

**理由**：
- 早期訓練需要正向反饋建立學習循環
- 等模型穩定後再逐步收緊閾值
- 課程學習的常見做法

### 決策 5：降低最大速度（2025-10-30）
**背景**：高速運動導致震盪和不穩定

**決策**：`max_speed: 2.0 → 0.8 m/s`

**結果**：待驗證

**理由**：
- 降低速度減少慣性，易於控制
- 符合真實機器人安全操作習慣
- 訓練穩定後可以逐步提高

---

## 📈 訓練監控指標

### TensorBoard 關鍵曲線

#### 必看指標（判斷收斂）
1. **Mean Reward**
   - 應該：持續上升，最終穩定在正值
   - 警訊：持續為負、劇烈波動、長期停滯

2. **Entropy**
   - 應該：逐步下降並穩定（0.5-1.5 範圍）
   - 警訊：持續上升（探索過強）、過快歸零（過早收斂）

3. **Episode_Termination/goal_reached**
   - 應該：從 0% 逐步上升到 >30%
   - 警訊：長期停在 0%、上升後又下降

4. **Episode_Termination/time_out**
   - 應該：從 90% 逐步下降到 <50%
   - 警訊：持續 >80%（Agent 無法完成任務）

5. **Metrics/goal_command/position_error**
   - 應該：從 >2.0m 下降到 <1.0m
   - 警訊：停滯在 >1.5m（無法接近目標）

#### 次要指標（調試用）
6. **Loss/value_function**：應該收斂，不應劇烈波動
7. **Loss/surrogate**：應該在小範圍內波動
8. **Loss/entropy**：應該逐步下降
9. **Episode_Reward/near_goal_shaping**：平均 >0.3 表示進入 1.5m 範圍
10. **Episode_Reward/heading_alignment**：平均 >0.5 表示朝向正確

### 成功訓練的標準
- ✅ Mean Reward > 0（理想 >50）
- ✅ Success Rate > 30%（理想 >50%）
- ✅ Timeout Rate < 60%
- ✅ Position Error < 1.0m
- ✅ Entropy 逐步下降並穩定
- ✅ Loss 曲線平滑，無劇烈震盪

### 失敗訓練的徵兆
- ❌ Mean Reward < -1000
- ❌ Success Rate = 0%（長時間）
- ❌ Timeout Rate > 90%
- ❌ Entropy 持續上升
- ❌ Loss 劇烈震盪
- ❌ Position Error 無改善

---

## 🚀 下一步建議

### 立即行動
1. **啟動新訓練**（使用穩定化配置）
   ```bash
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
       --task Isaac-Navigation-LocalPlanner-Min-v0 \
       --num_envs 24 \
       --max_iterations 10000 \
       --headless
   ```

2. **監控訓練進度**
   ```bash
   tensorboard --logdir logs/rsl_rl/local_planner_carter/
   ```

3. **記錄實驗結果**
   - 在 `md/` 目錄創建實驗日誌
   - 記錄關鍵指標（1000/3000/5000/10000 iterations）
   - 截圖保存 TensorBoard 曲線

### 短期目標（1-3 天）
1. **驗證穩定化改進**
   - 確認 Entropy 是否下降
   - 確認 Success Rate 是否上升
   - 確認 Position Error 是否改善

2. **如果仍不收斂**
   - 進一步降低 `entropy_coef` 到 0.0005
   - 增加 `progress_to_goal` 權重到 20.0
   - 減少環境數到 16（降低訓練噪音）

3. **如果收斂良好**
   - 逐步收緊成功閾值（0.8m → 0.6m → 0.5m）
   - 逐步提高最大速度（0.8 → 1.2 → 2.0）
   - 加入障礙物（使用標準環境）

### 中期目標（1-2 週）
1. **課程學習**
   - Stage 1：簡單目標（2-5m），無障礙物
   - Stage 2：中等目標（5-8m），2-3 個靜態障礙物
   - Stage 3：困難目標（3-10m），完整障礙物

2. **性能優化**
   - 嘗試更大的網路（[512, 512, 256]）
   - 嘗試更多環境（48-64）
   - 嘗試更長訓練（20000 iterations）

3. **部署準備**
   - 導出模型為 ONNX
   - 準備 ROS2 接口
   - 真實機器人測試規劃

### 長期目標（1 個月+）
1. **動態障礙物**
   - 加入移動障礙物
   - 預測未來軌跡（PCCBF）
   - 高速導航測試

2. **多任務擴展**
   - 路徑跟隨
   - 動態目標追蹤
   - 多機器人協作

3. **論文發表**
   - 整理實驗數據
   - 撰寫技術報告
   - 投稿相關會議/期刊

---

## 🔧 故障排除速查

### 問題：訓練啟動失敗
```
TypeError: Missing values detected... scene.lidar.mesh_prim_paths
```

**解決**：檢查 LiDAR 配置是否包含 `mesh_prim_paths`

---

### 問題：模組導入錯誤
```
ModuleNotFoundError: No module named 'omni.isaac.lab'
```

**解決**：更新為新 API `from isaaclab...`

---

### 問題：Play 腳本錯誤
```
Boost.Python.ArgumentError: Stage.GetPrimAtPath(Stage, NoneType)
```

**解決**：
1. 清理進程：`killall -r "kit"`
2. 優先使用 TensorBoard 而非 play.py

---

### 問題：訓練速度慢
**檢查**：
1. 是否使用 `--headless`？
2. `num_envs` 是否過多（>48 可能超出 GPU 記憶體）？
3. 是否有其他進程佔用 GPU？

---

### 問題：Success Rate = 0%
**可能原因**：
1. 目標太遠（檢查 `CommandsCfg.ranges`）
2. 閾值太嚴（檢查 `TerminationsCfg.goal_reached.threshold`）
3. 獎勵設計問題（檢查 `progress_to_goal` 是否生效）

**調試**：
1. TensorBoard 查看 `Episode_Reward/progress_to_goal` 是否為正
2. TensorBoard 查看 `Metrics/goal_command/position_error` 是否下降
3. 嘗試放寬閾值到 1.0m

---

## 📞 聯絡與資源

### 用戶資訊
- **工作目錄**：`/home/aa/IsaacLab`
- **Conda 環境**：`env_isaaclab`
- **Isaac Sim 路徑**：`/home/aa/isaacsim/`
- **Nova Carter USD**：`/home/aa/isaacsim/usd/nova_carter.usd`

### 重要指令
```bash
# 進入專案目錄
cd /home/aa/IsaacLab

# 激活環境
conda activate env_isaaclab

# 使用正確的 Python（重要！）
./isaaclab.sh -p [script.py]

# 列出環境
./isaaclab.sh -p scripts/tools/list_envs.py

# 清理進程
killall -r "kit"

# Git 操作
git status
git add -A
git commit -m "message"
git push origin main
```

### 外部資源
- **Isaac Lab 文檔**：https://isaac-sim.github.io/IsaacLab/
- **Isaac Sim 文檔**：https://docs.omniverse.nvidia.com/isaacsim/
- **RSL-RL GitHub**：https://github.com/leggedrobotics/rsl_rl
- **PCCBF 論文**：https://www.arxiv.org/pdf/2510.02885

---

## 📝 維護記錄

### 2025-10-30
- **AI 交接手冊創建**
- **環境配置精簡**：13 → 2 個配置檔
- **PPO 穩定化**：降低 entropy、clip、epochs
- **獎勵 Shaping**：新增 near_goal_shaping、heading_alignment
- **動作調整**：降低最大速度到 0.8 m/s
- **成功閾值**：放寬到 0.8m
- **文檔更新**：訓練架構完整說明、指令快速參考

### 2025-10-27
- **訓練完成**：3000 iterations（未收斂）
- **問題診斷**：Entropy 上升、Loss 波動、Success Rate 低
- **TensorBoard 分析**：確認需要 PPO 穩定化

### 2025-10-24
- **環境建立**：最初的環境配置
- **Isaac Sim 5.0 適配**：解決 ray_hits_w API 問題
- **Isaac Lab v2.2 遷移**：模組重命名

---

## ✅ 快速檢查清單（給下一位 AI）

在接手專案時，請確認以下項目：

- [ ] 用戶環境：`conda activate env_isaaclab`
- [ ] 專案目錄：`cd /home/aa/IsaacLab`
- [ ] 最新代碼：`git pull origin main`（如果有遠端更新）
- [ ] 環境註冊：`./isaaclab.sh -p scripts/tools/list_envs.py | grep Min`
- [ ] 配置檔案：`local_planner_env_cfg_min.py` 存在且包含 `mesh_prim_paths`
- [ ] PPO 配置：`rsl_rl_ppo_cfg.py` 已套用穩定化參數
- [ ] 獎勵函數：`rewards.py` 包含 `near_goal_shaping` 和 `heading_alignment_reward`
- [ ] 訓練日誌：`logs/rsl_rl/local_planner_carter/` 可訪問
- [ ] TensorBoard：可正常啟動

**如果以上都確認無誤，可以開始新的訓練了！**

---

## 🎯 總結

### 專案核心價值
1. **實用性**：訓練真實可部署的導航策略
2. **可維護性**：代碼精簡，文檔齊全
3. **可擴展性**：架構清晰，易於添加新功能
4. **教學性**：詳細註釋，適合學習 RL 與機器人導航

### 當前挑戰
1. **訓練收斂**：需要驗證穩定化改進是否有效
2. **性能提升**：從 30% 成功率提升到 >50%
3. **泛化能力**：在不同環境和障礙物配置下測試

### 下一位 AI 的任務
1. 啟動並監控新訓練
2. 根據結果調整超參數
3. 實驗課程學習策略
4. 準備部署到真實機器人

---

**祝下一位 AI 工作順利！如有問題，參考本手冊或聯絡用戶。** 🤖✨

**最後更新**：2025-10-30  
**文檔版本**：v1.0  
**維護者**：AI Assistant（Claude）


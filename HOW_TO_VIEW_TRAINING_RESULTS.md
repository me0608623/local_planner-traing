# 📊 如何查看訓練結果和訓練架構

## 🎉 您的訓練結果

**已發現您的訓練結果！** 最近一次訓練已完成 3000 次迭代。

---

## 📁 訓練結果位置

### 主要目錄

```
logs/rsl_rl/local_planner_carter/
├─ 2025-10-22_09-40-12/     # 第1次訓練
├─ 2025-10-22_12-00-30/     # 第2次訓練
├─ ...
└─ 2025-10-23_00-43-53/     # 最新訓練 ⭐
   ├─ model_0.pt            # 初始模型
   ├─ model_100.pt          # 第100次迭代
   ├─ model_200.pt          # 第200次迭代
   ├─ ...
   ├─ model_2999.pt         # 最終模型（3000次）✅
   ├─ events.out.tfevents.* # TensorBoard 日誌
   ├─ params/               # 訓練參數
   └─ git/                  # Git 信息
```

### 最新訓練結果

```bash
cd /home/aa/IsaacLab
ls -lh logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/
```

**包含**：
- ✅ **30個模型檢查點**（每100次迭代保存一次）
- ✅ **TensorBoard 日誌**（訓練曲線數據）
- ✅ **訓練參數**（配置信息）

---

## 📊 查看訓練結果的方法

### 方法 1: TensorBoard（推薦）⭐

**啟動 TensorBoard**：

```bash
cd /home/aa/IsaacLab

# 查看所有訓練
tensorboard --logdir logs/rsl_rl/

# 或只查看最新訓練
tensorboard --logdir logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/
```

**然後在瀏覽器打開**：`http://localhost:6006`

**您會看到**：
- 📈 **Mean Reward** 曲線（是否上升？）
- 📊 **Episode Length** 曲線
- 🎯 **Success Rate** (reached_goal)
- ❌ **Collision Rate** (collision)
- ⏱️ **Time Out Rate** (time_out)
- 📉 **Loss** 曲線（Value、Policy、Entropy）
- 🎮 各項獎勵分量曲線

**這是最直觀的方式！**

---

### 方法 2: 查看模型文件

```bash
# 查看所有保存的模型
ls -lh logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/model_*.pt

# 查看最終模型
ls -lh logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/model_2999.pt
```

**模型文件包含**：
- 神經網路權重（Actor 和 Critic）
- 優化器狀態
- 訓練迭代次數

---

### 方法 3: 使用 Play 腳本測試模型

**可視化訓練好的策略**：

```bash
cd /home/aa/IsaacLab

# 使用最終模型
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/model_2999.pt

# 或測試不同迭代的模型（比較進步）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/model_500.pt
```

**您會看到**：
- 🤖 機器人實際的導航行為
- 🎯 是否成功到達目標
- 🚧 如何避開障礙物
- 💡 策略是否學會了任務

---

### 方法 4: 查看訓練參數

```bash
# 查看訓練配置
cat logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/params/env.json
cat logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/params/agent.json

# 查看 Git 信息（訓練時的代碼版本）
cat logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/git/git_diff.txt
```

---

## 🏗️ 訓練架構在哪裡看？

### 架構總覽

訓練架構由以下部分組成：

```
訓練架構
├─ 環境配置
│  └─ local_planner_env_cfg.py
├─ 演算法配置
│  └─ agents/rsl_rl_ppo_cfg.py
├─ MDP 組件
│  ├─ mdp/observations.py
│  ├─ mdp/actions.py
│  ├─ mdp/rewards.py
│  └─ mdp/terminations.py
└─ 訓練腳本
   └─ scripts/reinforcement_learning/rsl_rl/train.py
```

---

### 核心架構文件

#### 1. 神經網路架構

**位置**：`agents/rsl_rl_ppo_cfg.py` 第 34-39 行

```bash
vim source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
    local_planner/agents/rsl_rl_ppo_cfg.py +34
```

**您會看到**：
```python
policy = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    actor_hidden_dims=[256, 256, 128],      # Actor 網路架構
    critic_hidden_dims=[256, 256, 128],     # Critic 網路架構
    activation="elu",
)
```

**架構**：
```
Actor Network (策略網路):
  輸入[369] → FC[256] → ELU → FC[256] → ELU → FC[128] → ELU → 輸出[2]
  
Critic Network (價值網路):
  輸入[369] → FC[256] → ELU → FC[256] → ELU → FC[128] → ELU → 輸出[1]
```

#### 2. 觀測空間架構

**位置**：`local_planner_env_cfg.py` 第 163-194 行

```bash
vim +163 source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
    local_planner/local_planner_env_cfg.py
```

**觀測維度**：
```
觀測空間 = [
    LiDAR 距離[360]       ← RayCaster 感測器
    線速度[3]             ← root_lin_vel_b
    角速度[3]             ← root_ang_vel_b  
    目標相對位置[2]       ← goal_position_in_robot_frame
    目標距離[1]           ← distance_to_goal
]
總維度：369 維
```

#### 3. 動作空間架構

**位置**：`local_planner_env_cfg.py` 第 142-156 行

```python
動作空間 = [
    線速度指令,          # 範圍: -2.0 到 +2.0 m/s
    角速度指令           # 範圍: -π 到 +π rad/s
]
總維度：2 維

通過差速驅動轉換為左右輪速度
```

#### 4. 獎勵架構

**位置**：`local_planner_env_cfg.py` 第 219-261 行

```bash
vim +219 source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
    local_planner/local_planner_env_cfg.py
```

**獎勵組成**：
```
總獎勵 = 
  + progress_to_goal × 10.0      # 接近目標
  + reached_goal × 100.0         # 到達目標
  - obstacle_proximity × 5.0     # 靠近障礙物
  - collision × 50.0             # 碰撞
  - ang_vel_penalty × 0.01       # 角速度過大
  - standstill × 0.1             # 靜止不動
```

---

## 📈 如何分析訓練結果

### 步驟 1: 啟動 TensorBoard

```bash
cd /home/aa/IsaacLab
tensorboard --logdir logs/rsl_rl/
```

瀏覽器打開：`http://localhost:6006`

### 步驟 2: 查看關鍵曲線

#### 性能指標
- **Mean Reward**: 應該從負值逐漸上升
- **Episode_Reward/reached_goal**: 成功獎勵（應該增加）
- **Episode_Termination/goal_reached**: 成功率（應該增加）
- **Episode_Termination/time_out**: 超時率（應該減少）

#### 訓練指標
- **Mean Value Function Loss**: 應該逐漸收斂
- **Mean Entropy Loss**: 探索程度（逐漸減少）
- **Learning Rate**: 可能自適應調整

### 步驟 3: 測試模型

```bash
# 測試最終模型
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/model_2999.pt
```

**觀察**：
- 機器人是否能到達目標？
- 避障行為是否合理？
- 運動是否平滑？

### 步驟 4: 比較不同迭代

```bash
# 測試早期模型（第500次）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/model_500.pt

# 測試中期模型（第1500次）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/model_1500.pt

# 測試最終模型（第2999次）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/model_2999.pt
```

**對比學習進度！**

---

## 🔍 查看訓練架構

### 完整架構文檔位置

| 文檔 | 內容 | 鏈接 |
|------|------|------|
| **訓練策略總結** | 核心訓練策略和快速參考 | [TRAINING_STRATEGY_SUMMARY.md](TRAINING_STRATEGY_SUMMARY.md) |
| **代碼架構指南** | 詳細的代碼結構說明 | [md/CODE_ARCHITECTURE_GUIDE.md](md/CODE_ARCHITECTURE_GUIDE.md) |
| **場景設計** | 模擬場景和USD模型 | [md/SIMULATION_SCENE_DESIGN.md](md/SIMULATION_SCENE_DESIGN.md) |
| **Agent感知** | 觀測空間和目標感知 | [md/HOW_AGENT_SEES_GOAL.md](md/HOW_AGENT_SEES_GOAL.md) |

### 快速查看架構

#### 神經網路架構

```bash
# Actor-Critic 網路配置
cat << 'EOF'
Actor Network (策略):
  輸入: 觀測[369維]
    ↓
  FC Layer 1: [369] → [256]
    ↓ ELU激活
  FC Layer 2: [256] → [256]
    ↓ ELU激活
  FC Layer 3: [256] → [128]
    ↓ ELU激活
  輸出: 動作[2維] = [線速度, 角速度]

Critic Network (價值估計):
  輸入: 觀測[369維]
    ↓
  FC Layer 1: [369] → [256]
    ↓ ELU激活
  FC Layer 2: [256] → [256]
    ↓ ELU激活
  FC Layer 3: [256] → [128]
    ↓ ELU激活
  輸出: State Value[1維]

總參數量: ~200K-300K 參數
EOF
```

#### 訓練架構流程

```bash
cat << 'EOF'
訓練架構流程圖：

┌─────────────────────────────────────────────────┐
│            1. 環境初始化                         │
│  ├─ 創建場景（地形、機器人、障礙物）            │
│  ├─ 初始化 LiDAR 感測器                         │
│  └─ 生成隨機目標                                 │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│            2. PPO 演算法初始化                   │
│  ├─ 創建 Actor-Critic 網路                      │
│  ├─ 初始化優化器（Adam）                        │
│  └─ 準備經驗緩衝區                               │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         3. 訓練循環（3000次迭代）                │
│                                                 │
│  For iteration = 1 to 3000:                    │
│                                                 │
│    3.1 收集經驗（Rollout）                      │
│    ├─ 運行24步/環境                             │
│    ├─ 獲取觀測（LiDAR + 目標 + 速度）           │
│    ├─ Actor輸出動作                             │
│    ├─ 環境執行動作                               │
│    ├─ 計算獎勵                                   │
│    └─ 存儲 (s,a,r,s')                           │
│                                                 │
│    3.2 計算優勢函數（GAE）                      │
│    └─ 使用Critic估計Value                       │
│                                                 │
│    3.3 更新策略（PPO）                          │
│    ├─ 5個epoch                                  │
│    ├─ 4個mini-batch                             │
│    ├─ 計算loss（policy + value + entropy）      │
│    └─ 反向傳播更新                               │
│                                                 │
│    3.4 保存模型（每100次）                      │
│    └─ model_[iteration].pt                     │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│            4. 訓練完成                           │
│  └─ 保存最終模型 model_2999.pt                  │
└─────────────────────────────────────────────────┘
EOF
```

---

## 📊 您的訓練結果分析

### 查看最新訓練

```bash
# 1. 啟動 TensorBoard
cd /home/aa/IsaacLab
tensorboard --logdir logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/

# 2. 瀏覽器打開
http://localhost:6006

# 3. 查看這些曲線
#    - SCALARS/Mean Reward
#    - SCALARS/Episode_Reward/reached_goal
#    - SCALARS/Episode_Termination/goal_reached
#    - SCALARS/Episode_Termination/time_out
```

### 快速評估

```bash
# 使用我們的分析工具（需要訓練日誌文本）
./isaaclab.sh -p scripts/analyze_training_log.py

# 或查看 TensorBoard 的最終數值
```

---

## 🎯 訓練架構核心文件位置

### 快速導航

| 組件 | 文件 | 行號 | 內容 |
|------|------|------|------|
| **神經網路** | `agents/rsl_rl_ppo_cfg.py` | 34-39 | Actor-Critic 架構 |
| **觀測空間** | `local_planner_env_cfg.py` | 163-194 | 輸入定義（369維） |
| **動作空間** | `local_planner_env_cfg.py` | 142-156 | 輸出定義（2維） |
| **獎勵函數** | `local_planner_env_cfg.py` | 219-261 | 獎勵權重 |
| **獎勵實現** | `mdp/rewards.py` | 全文件 | 獎勵計算邏輯 |
| **場景定義** | `local_planner_env_cfg.py` | 37-135 | 場景組件 |
| **訓練腳本** | `train.py` | 全文件 | 訓練主循環 |

### 架構可視化

```bash
# 查看完整的架構說明
cat md/CODE_ARCHITECTURE_GUIDE.md

# 或查看訓練策略總結
cat TRAINING_STRATEGY_SUMMARY.md

# 或查看項目架構
cat md/PROJECT_ARCHITECTURE_SUMMARY.md
```

---

## 🔬 深度分析訓練結果

### Python 腳本分析模型

```python
# analyze_model.py
import torch

# 加載模型
model_path = "logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/model_2999.pt"
checkpoint = torch.load(model_path)

print("模型內容:")
print("- 鍵:", list(checkpoint.keys()))
print("- Actor 參數:", sum(p.numel() for p in checkpoint['model_state_dict'].values() if 'actor' in str(p)))
print("- Critic 參數:", sum(p.numel() for p in checkpoint['model_state_dict'].values() if 'critic' in str(p)))
print("- 訓練迭代:", checkpoint.get('iter', 'N/A'))
```

運行：
```bash
./isaaclab.sh -p analyze_model.py
```

---

## 📚 完整訓練架構文檔索引

### 按主題分類

#### 訓練策略
- [訓練策略快速參考](TRAINING_STRATEGY_SUMMARY.md)
- [完整代碼架構指南](md/CODE_ARCHITECTURE_GUIDE.md)

#### 環境設計
- [模擬場景設計](md/SIMULATION_SCENE_DESIGN.md)
- [Agent 目標感知](md/HOW_AGENT_SEES_GOAL.md)
- [並行訓練機制](md/PARALLEL_TRAINING_AND_COLLISION.md)

#### 訓練結果
- 本文檔（如何查看結果）
- [訓練診斷指南](md/TRAINING_DIAGNOSIS_GUIDE.md)

---

## 💡 快速操作指令

### 一鍵查看最新訓練

```bash
# 1. TensorBoard 可視化
cd /home/aa/IsaacLab && tensorboard --logdir logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/

# 2. 測試最終模型
cd /home/aa/IsaacLab && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Navigation-LocalPlanner-Carter-v0 --num_envs 1 --checkpoint logs/rsl_rl/local_planner_carter/2025-10-23_00-43-53/model_2999.pt

# 3. 查看架構文檔
cat md/CODE_ARCHITECTURE_GUIDE.md
```

---

## 🎯 總結

### 訓練結果位置

```
logs/rsl_rl/local_planner_carter/
└─ 2025-10-23_00-43-53/  ← 您最新的訓練
   ├─ model_2999.pt      ← 最終模型（3000次迭代）
   ├─ events.out.*       ← TensorBoard 日誌
   └─ params/            ← 訓練配置
```

### 訓練架構位置

```
source/isaaclab_tasks/.../local_planner/
├─ agents/rsl_rl_ppo_cfg.py     ← 神經網路架構（第34-39行）
├─ local_planner_env_cfg.py     ← 環境架構
│  ├─ 觀測空間（第163-194行）
│  ├─ 動作空間（第142-156行）
│  └─ 獎勵函數（第219-261行）
└─ mdp/                         ← MDP 組件實現
   ├─ observations.py           ← 觀測計算
   ├─ actions.py                ← 動作轉換
   ├─ rewards.py                ← 獎勵計算
   └─ terminations.py           ← 終止判斷
```

### 最佳查看方式

1. **訓練結果** → 使用 **TensorBoard** ⭐
2. **模型表現** → 使用 **Play 腳本** ⭐
3. **訓練架構** → 閱讀 **代碼架構指南** ⭐

---

**現在您知道如何查看訓練結果和訓練架構了！** 🎯

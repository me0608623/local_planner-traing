# 🤖 Nova Carter 強化學習策略架構詳解

## 🎯 任務定義

### 核心任務
**動態避障導航**：Nova Carter 輪式機器人在包含靜態和動態障礙物的環境中，使用 LiDAR 感測器導航到隨機目標位置。

### 挑戰
- **實時避障**：避開靜態方塊和移動球體
- **路徑規劃**：在複雜環境中找到最優路徑
- **運動控制**：平滑的差速驅動控制
- **感測器融合**：整合 LiDAR 和本體感測

## 🏗️ 環境架構

### 場景組成
```
環境場景：
├── 地形：平坦地面（15m × 15m）
├── 機器人：Nova Carter（差速驅動輪式機器人）
│   ├── 主驅動輪：wheel_left, wheel_right
│   ├── 輔助滑輪：caster_* （被動支撐）
│   └── LiDAR：2D 360° 掃描
├── 障礙物：
│   ├── 靜態障礙物：方塊（固定位置）
│   └── 動態障礙物：球體（隨機運動，每3-5秒推力）
└── 目標：隨機生成位置（每10秒重新生成）
```

### 物理參數
```python
# Nova Carter 規格
wheel_radius = 0.125m      # 輪半徑
wheel_base = 0.413m        # 輪距
max_linear_speed = 2.0m/s  # 最大線速度
max_angular_speed = π rad/s # 最大角速度

# LiDAR 規格
horizontal_fov = 360°      # 水平視野
resolution = 1°/點         # 角度解析度
max_range = 10.0m         # 最大測距
points = 360              # 總點數
```

## 📊 狀態空間（觀察）

### 總維度：**369 維**

| 觀察項 | 維度 | 範圍 | 描述 |
|--------|------|------|------|
| **LiDAR 距離** | 360 | [0, 10.0] | 360° 雷射距離測量（m） |
| **機器人線速度** | 3 | [-2.0, 2.0] | [vₓ, vᵧ, vᵤ] 在世界座標系（m/s） |
| **機器人角速度** | 3 | [-π, π] | [ωₓ, ωᵧ, ωᵤ] 在世界座標系（rad/s） |
| **目標相對位置** | 2 | [-∞, ∞] | [Δx, Δy] 在機器人座標系（m） |
| **目標距離** | 1 | [0, ∞] | 到目標的歐氏距離（m） |

### 觀察處理
```python
class PolicyObservation:
    lidar_distances: torch.Tensor    # (N, 360) - 歸一化到 [0,1]
    base_lin_vel: torch.Tensor      # (N, 3)   - 當前線速度
    base_ang_vel: torch.Tensor      # (N, 3)   - 當前角速度  
    goal_position: torch.Tensor     # (N, 2)   - 機器人座標系下目標位置
    goal_distance: torch.Tensor     # (N, 1)   - 歸一化距離

    # 最終拼接：(N, 369)
    obs = torch.cat([lidar_distances, base_lin_vel, base_ang_vel, 
                    goal_position, goal_distance], dim=1)
```

## 🎮 動作空間

### 總維度：**2 維**（連續動作）

| 動作 | 範圍 | 實際映射 | 描述 |
|------|------|----------|------|
| **線速度** | [-1, 1] | [-2.0, 2.0] m/s | 前進/後退速度 |
| **角速度** | [-1, 1] | [-π, π] rad/s | 左轉/右轉速度 |

### 差速驅動轉換
```python
# 動作映射
linear_velocity = action[0] * max_linear_speed   # [-2.0, 2.0]
angular_velocity = action[1] * max_angular_speed # [-π, π]

# 差速驅動運動學
R = wheel_radius
L = wheel_base

# 輪速計算
v_left = (linear_velocity - angular_velocity * L/2) / R
v_right = (linear_velocity + angular_velocity * L/2) / R

# 應用到關節
left_wheel.set_velocity(v_left)
right_wheel.set_velocity(v_right)
```

## 🏆 獎勵函數設計

### 總獎勵 = Σ(權重 × 項目獎勵)

| 獎勵項 | 權重 | 描述 | 計算方式 |
|--------|------|------|----------|
| **接近目標** | +10.0 | 鼓勵向目標移動 | `10.0 × progress_ratio` |
| **到達目標** | +100.0 | 成功到達獎勵 | `100.0 if distance < 0.5m` |
| **避障懲罰** | -5.0 | 懲罰接近障礙物 | `-5.0 × proximity_factor` |
| **碰撞懲罰** | -50.0 | 懲罰碰撞 | `-50.0 if min_distance < 0.3m` |
| **角速度懲罰** | -0.01 | 鼓勵平滑運動 | `-0.01 × ω²` |
| **靜止懲罰** | -0.1 | 防止不動 | `-0.1 if speed < threshold` |

### 獎勵設計原理
```python
# 1. 稀疏 + 密集獎勵結合
dense_reward = progress_to_goal      # 密集引導
sparse_reward = reached_goal         # 稀疏成功信號

# 2. 安全優先
safety_penalty = collision + proximity  # 避障是首要任務

# 3. 運動品質
smooth_penalty = angular_velocity_penalty  # 鼓勵平滑運動

# 總獎勵平衡
total_reward = dense_reward + sparse_reward + safety_penalty + smooth_penalty
```

## 🧠 策略網路架構

### PPO Actor-Critic 架構

```python
# 網路配置
policy = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,           # 初始探索噪音
    actor_hidden_dims=[256, 256, 128],    # Actor 隱藏層
    critic_hidden_dims=[256, 256, 128],   # Critic 隱藏層
    activation="elu",             # 激活函數
)

# 網路結構
Input(369) → FC(256) → ELU → FC(256) → ELU → FC(128) → ELU
                                                        ↓
                              Actor: → FC(2) → Tanh → Actions
                              Critic: → FC(1) → Value
```

### PPO 超參數
```python
algorithm = RslRlPpoAlgorithmCfg(
    # 核心參數
    learning_rate=1e-3,          # 學習率
    num_learning_epochs=5,       # 每次更新的epoch數
    num_mini_batches=4,          # 小批次數量
    
    # PPO 特定參數
    clip_param=0.2,              # PPO裁剪參數
    entropy_coef=0.01,           # 熵係數（探索）
    value_loss_coef=1.0,         # 價值損失係數
    
    # 訓練穩定性
    gamma=0.99,                  # 折扣因子
    lam=0.95,                    # GAE lambda
    desired_kl=0.01,             # 目標KL散度
    max_grad_norm=1.0,           # 梯度裁剪
)
```

## ⏰ 訓練設定

### 時間結構
```python
# 物理模擬
dt = 0.01s                    # 物理時間步長（100 Hz）
decimation = 4                # RL決策頻率（25 Hz）
episode_length = 30s          # 單回合時長
max_steps_per_episode = 750   # 30s ÷ 0.04s

# 訓練規模
num_envs = 128               # 並行環境數（可調至1024）
num_steps_per_env = 24       # 每個環境的步數
total_samples = 128 × 24 = 3072  # 每次迭代的總樣本數
```

### 終止條件
```python
# 成功終止
goal_reached: distance_to_goal < 0.5m

# 失敗終止  
collision: min_lidar_distance < 0.3m

# 時間終止
time_out: episode_time >= 30s
```

## 🎯 訓練策略

### 課程學習（隱式）
1. **初期**：靜態障礙物 + 近距離目標
2. **中期**：加入動態障礙物
3. **後期**：遠距離目標 + 複雜場景

### 探索策略
- **高斯噪音**：初始 σ=1.0，逐漸衰減
- **熵正則化**：係數 0.01 維持探索
- **隨機重置**：機器人和障礙物位置隨機化

### 穩定性技術
- **梯度裁剪**：max_grad_norm=1.0
- **KL散度控制**：desired_kl=0.01
- **自適應學習率**：根據KL散度調整

## 📊 關鍵性能指標

### 成功指標
- **成功率**：到達目標的回合比例 (目標 >90%)
- **平均回報**：每回合累積獎勵 (目標 >50)
- **碰撞率**：發生碰撞的回合比例 (目標 <5%)

### 效率指標
- **平均步數**：到達目標的平均步數
- **路徑效率**：實際路徑/最短路徑比值
- **運動平滑度**：角速度變化的標準差

### 泛化能力
- **不同障礙物配置**的表現
- **不同目標距離**的適應性
- **動態環境**的魯棒性

## 🔬 實驗設定建議

### 訓練階段
```bash
# 第一階段：基礎訓練（1-500 迭代）
--num_envs 64 --max_iterations 500
# 學習基本避障和導航

# 第二階段：擴展訓練（500-2000 迭代）  
--num_envs 128 --max_iterations 2000
# 提升性能和穩定性

# 第三階段：大規模訓練（2000+ 迭代）
--num_envs 256 --max_iterations 5000  
# 最終性能優化
```

### 超參數調優
```python
# 關鍵調優參數
learning_rate: [1e-4, 5e-4, 1e-3]      # 學習率
clip_param: [0.1, 0.2, 0.3]            # PPO裁剪
entropy_coef: [0.005, 0.01, 0.02]      # 探索強度
reward_weights: 針對任務特性調整          # 獎勵平衡
```

## 🎯 Nova Carter 特色

### 機器人特性
- **輪式機器人**：差速驅動，適合室內導航
- **實際參數**：基於真實 Nova Carter 規格
- **感測器融合**：LiDAR + 里程計信息

### 任務特色
- **動態環境**：移動障礙物增加挑戰
- **實時性**：25Hz 決策頻率符合實際應用
- **可擴展**：支援多環境並行訓練

### 應用場景
- **服務機器人**：室內配送、巡邏
- **自動導引車(AGV)**：倉庫物流
- **研究平台**：導航算法驗證

---

這個架構設計平衡了**任務複雜性**、**訓練效率**和**實際應用**，是一個完整的端到端強化學習導航系統！

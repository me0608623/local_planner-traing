# 🤖 DRL-robot-navigation 專案架構分析

> **版本**：v1.0  
> **分析日期**：2025-10-30  
> **專案來源**：[GitHub - reiniscimurs/DRL-robot-navigation](https://github.com/reiniscimurs/DRL-robot-navigation)  
> **論文**：ICRA 2022 & IEEE RA-L - "Goal-Driven Autonomous Exploration Through Deep Reinforcement Learning"

---

## 📋 目錄

1. [專案概述](#專案概述)
2. [核心架構](#核心架構)
3. [訓練流程](#訓練流程)
4. [與當前系統對比](#與當前系統對比)
5. [整合策略](#整合策略)
6. [實作建議](#實作建議)

---

## 📖 專案概述

### 🎯 專案目標

使用 **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** 算法，訓練移動機器人在模擬環境中自主導航到隨機目標點，同時避障。

### 🛠️ 技術棧

| 組件 | 技術 | 當前專案 |
|------|------|----------|
| **模擬器** | ROS Gazebo | Isaac Sim |
| **RL 算法** | TD3（Actor-Critic） | PPO（Actor-Critic） |
| **感測器** | 3D Velodyne LiDAR | 3D LiDAR |
| **框架** | PyTorch（自定義） | RSL-RL（PPO 庫） |
| **記錄工具** | TensorBoard | TensorBoard / WandB |
| **通訊** | ROS Topics | Isaac Lab API |
| **機器人** | 自定義（r1） | Nova Carter |

### 📊 訓練參數

```python
max_timesteps = 5e6        # 500 萬步（vs 當前 240k）
buffer_size = 1e6          # 100 萬經驗（Replay Buffer）
batch_size = 40            # 小批次（vs 當前 576）
discount = 0.99999         # 幾乎無折扣（vs 0.99）
expl_noise = 1 → 0.1       # 探索噪音衰減
learning_rate = Adam 預設   # 未明確指定（vs 3e-4）
```

---

## 🏗️ 核心架構

### 1️⃣ 環境設計（`velodyne_env.py`）

#### 📡 觀測空間（State Space）

**維度**：`24` = `20 (LiDAR) + 4 (Robot State)`

```python
state = [
    # 1. LiDAR 掃描（20 個角度範圍的最小距離）
    velodyne_data[0:20],     # 每個角度範圍的最小障礙距離
    
    # 2. 機器人狀態（4 維）
    distance,                # 到目標的距離（歐氏距離）
    theta,                   # 相對目標的角度（-π 到 π）
    action[0],               # 上一步的線速度
    action[1]                # 上一步的角速度
]
```

**LiDAR 處理邏輯**：
- 掃描範圍：`-π/2` 到 `π/2`（180°前方）
- 分成 20 個角度範圍（每個 9°）
- 過濾高度 > -0.2m 的點（忽略地面）
- 每個範圍取最小距離（最近障礙物）
- 初始值：10m（無障礙物）

#### 🎮 動作空間（Action Space）

**維度**：`2` = `[linear_velocity, angular_velocity]`

```python
action = [
    linear_velocity,   # 線速度：[0, 1]（經過轉換）
    angular_velocity   # 角速度：[-1, 1]
]

# 實際執行時的轉換
a_in = [(action[0] + 1) / 2, action[1]]
```

**動作特性**：
- 輸出範圍：`[-1, 1]`（Actor 使用 Tanh 激活）
- 線速度轉換：從 `[-1, 1]` → `[0, 1]`
- 角速度直接使用

#### 🎁 獎勵函數（Reward Function）

```python
def get_reward(target, collision, action, min_laser):
    if target:
        return 100.0           # ✅ 到達目標
    elif collision:
        return -100.0          # ❌ 碰撞
    else:
        # 正常移動獎勵
        r3 = lambda x: 1 - x if x < 1 else 0.0
        return action[0] / 2           # 鼓勵前進（+0.5 max）
               - abs(action[1]) / 2    # 懲罰旋轉（-0.5 max）
               - r3(min_laser) / 2     # 懲罰靠近障礙（-0.5 max）
```

**獎勵設計特點**：
- ✅ **極簡設計**：只有 3 項（vs 當前 8 項）
- ✅ **稀疏 + 密集結合**：
  - 稀疏：目標 +100、碰撞 -100
  - 密集：速度鼓勵、旋轉懲罰、避障懲罰
- ✅ **無時間懲罰**：不懲罰耗時（discount 接近 1）

**與當前系統對比**：

| 項目 | DRL-robot-navigation | 當前系統（v4） |
|------|---------------------|--------------|
| 獎勵項數量 | 3 | 8 |
| Progress | `action[0] / 2` | `weight=60.0` |
| 避障 | `-r3(min_laser) / 2` | 無直接獎勵 |
| 旋轉 | `-abs(action[1]) / 2` | `spin_penalty=0.1` |
| 時間 | 無 | `time_penalty=0.005` |
| 朝向 | 無 | `heading_alignment=1.0` |
| 靜止 | 無 | `standstill=1.0` |

#### 🔄 終止條件（Termination）

```python
# 1. 到達目標
if distance < GOAL_REACHED_DIST:  # 0.3m
    target = True
    done = True

# 2. 碰撞
if min_laser < COLLISION_DIST:  # 0.35m
    collision = True
    done = True

# 3. 超時（訓練腳本中）
if episode_timesteps >= max_ep:  # 500 步
    done = True
```

#### 🎯 目標生成（Goal Generation）

```python
def change_goal(self):
    # 課程學習：逐漸擴大目標範圍
    if self.upper < 10:
        self.upper += 0.004     # 從 5.0 慢慢增加
    if self.lower > -10:
        self.lower -= 0.004     # 從 -5.0 慢慢減少
    
    # 隨機生成目標（相對於機器人當前位置）
    self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
    self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
    
    # 確保目標不在障礙物上
    goal_ok = check_pos(self.goal_x, self.goal_y)
```

**課程學習策略**：
- 初始範圍：`[-5, 5]`（10m × 10m）
- 最終範圍：`[-10, 10]`（20m × 20m）
- 每次 reset 增加 0.004（約 10000 次後達到最大）

#### 🧱 動態障礙物（Randomization）

```python
def random_box(self):
    # 每次 reset 隨機移動 4 個箱子
    for i in range(4):
        x = np.random.uniform(-6, 6)
        y = np.random.uniform(-6, 6)
        # 確保箱子：
        # 1. 不在固定障礙物上
        # 2. 距離機器人 > 1.5m
        # 3. 距離目標 > 1.5m
```

**環境隨機化**：
- 每個 episode 重置機器人位置（隨機）
- 每個 episode 重置機器人朝向（隨機）
- 每個 episode 重置目標位置（課程學習範圍內）
- 每個 episode 重置 4 個動態箱子位置

---

### 2️⃣ TD3 算法（`train_velodyne_td3.py`）

#### 🏛️ 網路架構

**Actor 網路**（策略網路）：
```python
Input: state (24)
  ↓
Layer 1: Linear(24, 800) + ReLU
  ↓
Layer 2: Linear(800, 600) + ReLU
  ↓
Layer 3: Linear(600, 2) + Tanh
  ↓
Output: action (2) in [-1, 1]
```

**Critic 網路**（雙 Q 網路）：
```python
# Q1 網路
Input: state (24)
  ↓
Layer 1: Linear(24, 800) + ReLU
  ↓
Layer 2_s: Linear(800, 600)  ←─┐
Action: (2)                     │ 相加
  ↓                             │
Layer 2_a: Linear(2, 600)  ─────┘ + ReLU
  ↓
Layer 3: Linear(600, 1)
  ↓
Output: Q1 value

# Q2 網路（相同架構）
...
Output: Q2 value
```

**參數量對比**：

| 網路 | DRL-robot-navigation | 當前系統（PPO） |
|------|---------------------|----------------|
| Actor | 24→800→600→2 ≈ 500k | 24→256→256→128→2 ≈ 100k |
| Critic | 24→800→600→1 × 2 ≈ 1M | 24→256→256→128→1 ≈ 100k |
| **總計** | **≈ 1.5M** | **≈ 200k** |

**關鍵差異**：
- TD3 使用更大的網路（800, 600 vs 256, 256, 128）
- TD3 使用雙 Critic（減少 Q 值過估）
- PPO 使用單 Critic（Value Function）

#### 🔄 訓練循環

```python
# 主訓練循環（5M 步）
while timestep < max_timesteps:
    if done:
        # Episode 結束時訓練
        network.train(replay_buffer, episode_timesteps, ...)
        
        # 每 5000 步評估一次
        if timesteps_since_eval >= eval_freq:
            evaluations.append(evaluate(...))
            network.save(...)
    
    # 動作選擇（加噪音探索）
    action = network.get_action(state)
    action += np.random.normal(0, expl_noise, size=2)
    
    # 近障礙物特殊策略
    if random_near_obstacle and min(state[4:-8]) < 0.6:
        if np.random.uniform(0, 1) > 0.85:
            # 強制隨機後退
            action = random_action
            action[0] = -1
    
    # 執行動作
    next_state, reward, done, target = env.step(action)
    
    # 儲存經驗
    replay_buffer.add(state, action, reward, done, next_state)
```

#### 🎓 TD3 更新機制

```python
def train(replay_buffer, iterations, batch_size=100, ...):
    for it in range(iterations):
        # 1. 從 Replay Buffer 採樣
        states, actions, rewards, dones, next_states = replay_buffer.sample_batch(batch_size)
        
        # 2. 計算 Target Q（使用 Target Networks）
        next_action = actor_target(next_states)
        next_action += noise.clamp(-noise_clip, noise_clip)  # Smoothing
        target_Q1, target_Q2 = critic_target(next_states, next_action)
        target_Q = min(target_Q1, target_Q2)  # 取最小（減少過估）
        target_Q = rewards + discount * target_Q
        
        # 3. 更新 Critic
        current_Q1, current_Q2 = critic(states, actions)
        loss = MSE(current_Q1, target_Q) + MSE(current_Q2, target_Q)
        critic_optimizer.step()
        
        # 4. 延遲更新 Actor（每 2 步）
        if it % policy_freq == 0:
            actor_loss = -critic(states, actor(states)).mean()  # Maximize Q
            actor_optimizer.step()
            
            # 5. Soft Update Target Networks
            target_params = τ * params + (1 - τ) * target_params
```

**TD3 核心技巧**：
1. **雙 Critic**：減少 Q 值過估（取 min）
2. **Delayed Policy Update**：Actor 更新頻率 < Critic（避免不穩定）
3. **Target Policy Smoothing**：給 target action 加噪音（正則化）
4. **Soft Target Update**：目標網路緩慢跟隨（τ=0.005）

**vs PPO**：
- PPO：On-Policy（使用當前策略的數據）
- TD3：Off-Policy（使用 Replay Buffer 的舊數據）
- PPO：策略梯度 + Clipping
- TD3：Q-Learning + Actor-Critic

---

### 3️⃣ 經驗回放（`replay_buffer.py`）

```python
class ReplayBuffer:
    def __init__(self, buffer_size=1e6):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, done, next_state):
        self.buffer.append((state, action, reward, done, next_state))
    
    def sample_batch(self, batch_size=40):
        batch = random.sample(self.buffer, batch_size)
        return states, actions, rewards, dones, next_states
```

**關鍵特性**：
- **容量**：100 萬筆經驗（vs PPO 無 Buffer，直接用完即丟）
- **採樣**：隨機採樣（打破時間相關性）
- **資料利用率**：高（可重複使用舊經驗）

**vs PPO**：
- PPO：收集 `num_envs × num_steps_per_env` 步後立即訓練並丟棄
- TD3：持續累積經驗，隨機採樣訓練（更穩定但需更多記憶體）

---

### 4️⃣ 探索策略

#### 📉 噪音衰減

```python
expl_noise = 1.0           # 初始（100% 探索）
expl_min = 0.1             # 最終（10% 探索）
expl_decay_steps = 500000  # 衰減週期

# 每步衰減
expl_noise -= (1 - expl_min) / expl_decay_steps
```

**衰減曲線**：
```
Step 0:      expl_noise = 1.0
Step 250k:   expl_noise = 0.55
Step 500k:   expl_noise = 0.1
Step 500k+:  expl_noise = 0.1（保持）
```

#### 🚧 近障礙物策略（Random Near Obstacle）

```python
if random_near_obstacle:
    # 如果雷射掃描 < 0.6m 且 15% 機率
    if np.random.uniform(0, 1) > 0.85 and min(state[4:-8]) < 0.6:
        # 強制後退 8-15 步
        count_rand_actions = np.random.randint(8, 15)
        random_action = np.random.uniform(-1, 1, 2)
        action = random_action
        action[0] = -1  # 強制後退
```

**目的**：
- 增加近障礙物的探索
- 避免 Agent 學會「卡在牆邊」
- 打破局部最優（類似 Epsilon-Greedy）

---

## 📊 與當前系統對比

### 完整對比表

| 項目 | DRL-robot-navigation | 當前專案（Nova Carter） |
|------|---------------------|------------------------|
| **算法** | TD3（Off-Policy） | PPO（On-Policy） |
| **模擬器** | ROS Gazebo | Isaac Sim |
| **框架** | PyTorch 自定義 | RSL-RL 庫 |
| **觀測維度** | 24 | 548（20 LiDAR + 528 其他） |
| **動作維度** | 2 | 2 |
| **獎勵項數量** | 3（極簡） | 8（複雜） |
| **網路大小** | 1.5M 參數 | 200k 參數 |
| **Batch Size** | 40 | 576 (24 envs × 24 steps) |
| **訓練步數** | 5M | 240k（10000 iter） |
| **Replay Buffer** | 1M 經驗 | 無（On-Policy） |
| **探索策略** | 噪音衰減 + 近障礙隨機 | 動作噪音（固定 std） |
| **課程學習** | 目標距離逐漸增加 | 固定範圍 |
| **環境隨機化** | 動態箱子 + 機器人位置 | 固定場景 |
| **時間懲罰** | 無 | 有（0.005） |
| **並行環境** | 1 | 24 |
| **評估頻率** | 每 5000 步 | 每 100 iter |

---

## 🔄 整合策略

### 方案 A：直接移植 TD3 到 Isaac Lab

**目標**：用 TD3 替換 PPO，保持 Isaac Lab 環境

#### 優點
- ✅ Off-Policy 學習（更穩定）
- ✅ 更好的樣本效率（Replay Buffer）
- ✅ 適合連續動作空間（vs PPO）
- ✅ 論文實證成功（ICRA 2022）

#### 缺點
- ❌ 需要大量記憶體（1M Buffer）
- ❌ 單環境訓練（vs 當前 24 並行）
- ❌ 訓練時間更長（5M vs 240k 步）

#### 實作步驟

```python
# 1. 創建 TD3 Agent
td3_agent = TD3Agent(
    state_dim=24,          # 20 LiDAR + 4 robot state
    action_dim=2,
    actor_hidden=[800, 600],
    critic_hidden=[800, 600],
    max_action=1.0,
)

# 2. 簡化觀測空間（當前 548 → 24）
@configclass
class ObservationsCfg:
    policy = ObsTerm(func=observe_state)
    
    def observe_state(env):
        # LiDAR: 20 角度範圍的最小距離
        lidar_min_distances = process_lidar(env.scene["lidar"])
        
        # Robot state
        goal_distance = compute_distance_to_goal(env)
        goal_theta = compute_relative_angle_to_goal(env)
        last_linear_vel = env.action_manager.prev_actions[:, 0]
        last_angular_vel = env.action_manager.prev_actions[:, 1]
        
        return torch.cat([
            lidar_min_distances,    # (20,)
            goal_distance,          # (1,)
            goal_theta,             # (1,)
            last_linear_vel,        # (1,)
            last_angular_vel,       # (1,)
        ], dim=-1)  # Total: 24

# 3. 簡化獎勵函數（8 項 → 3 項）
@configclass
class RewardsCfg:
    # 1. 前進鼓勵
    forward_velocity = RewTerm(
        func=lambda env: env.action_manager.action[:, 0] / 2,
        weight=1.0,
    )
    
    # 2. 旋轉懲罰
    angular_penalty = RewTerm(
        func=lambda env: -torch.abs(env.action_manager.action[:, 1]) / 2,
        weight=1.0,
    )
    
    # 3. 避障懲罰
    obstacle_penalty = RewTerm(
        func=lambda env: -(1 - torch.min(env.scene["lidar"].data.ray_hits_w, dim=-1).values).clamp(0, 1) / 2,
        weight=1.0,
    )
    
    # 稀疏獎勵
    reached_goal = RewTerm(
        func=mdp.reached_goal_reward,
        weight=100.0,
    )
    
    collision = RewTerm(
        func=mdp.collision_penalty,
        weight=-100.0,
    )

# 4. TD3 訓練循環
replay_buffer = ReplayBuffer(1000000)

for timestep in range(5_000_000):
    if done:
        state = env.reset()
    
    # 選擇動作（加噪音）
    action = td3_agent.get_action(state)
    action += np.random.normal(0, expl_noise, size=2)
    
    # 執行
    next_state, reward, done, info = env.step(action)
    
    # 儲存經驗
    replay_buffer.add(state, action, reward, done, next_state)
    
    # 訓練（從 Buffer 採樣）
    if replay_buffer.size() > batch_size:
        td3_agent.train(replay_buffer, iterations=episode_timesteps)
```

---

### 方案 B：借鑑 TD3 設計改進 PPO

**目標**：保持 PPO 算法，但借鑑 TD3 的成功經驗

#### 可借鑑的設計

**1. 極簡獎勵函數**
```python
# 從 8 項減少到 3-4 項
@configclass
class RewardsCfg:
    # 核心：只保留最重要的
    progress = RewTerm(weight=1.0)         # 前進
    rotation = RewTerm(weight=-0.5)        # 旋轉懲罰
    obstacle = RewTerm(weight=-0.5)        # 避障
    reached_goal = RewTerm(weight=100.0)   # 成功
    # 刪除：standstill, anti_idle, time, heading, near_goal
```

**2. 課程學習**
```python
@configclass
class CommandsCfg:
    goal_command = UniformVelocityCommand2dCfg(
        resampling_time_range=(5.0, 5.0),
        ranges=GoalCommandRanges(
            # 初始範圍：5m
            pos_x=(self.curriculum_distance, self.curriculum_distance),
            pos_y=(self.curriculum_distance, self.curriculum_distance),
        ),
    )
    
    # 每 10000 iter 增加 0.5m
    def update_curriculum(self, iteration):
        if iteration % 10000 == 0:
            self.curriculum_distance = min(self.curriculum_distance + 0.5, 10.0)
```

**3. 環境隨機化**
```python
@configclass
class LocalPlannerSceneCfg:
    # 每次 reset 隨機移動障礙物
    dynamic_obstacles = AssetBaseCfg(
        prim_path="/World/Obstacles",
        spawn=sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5)),
        init_state=AssetInitialStateCfg(
            pos=RandomUniformDistribution(min=(-5, -5, 0), max=(5, 5, 0)),
        ),
    )
```

**4. 近障礙物探索**
```python
def compute_actions(self, obs):
    actions = self.policy(obs)
    
    # 如果雷射掃描 < 0.6m，15% 機率強制後退
    min_lidar = torch.min(obs[:, :20], dim=-1).values
    near_obstacle = min_lidar < 0.6
    force_backward = torch.rand(len(obs)) > 0.85
    
    actions[near_obstacle & force_backward, 0] = -1.0  # 後退
    
    return actions
```

**5. 更大的網路**
```python
policy = RslRlPpoActorCriticCfg(
    actor_hidden_dims=[800, 600],     # vs 當前 [256, 256, 128]
    critic_hidden_dims=[800, 600],    # vs 當前 [256, 256, 128]
    activation="relu",                # vs 當前 "elu"
)
```

---

### 方案 C：混合架構（最推薦）

**核心理念**：TD3 的環境設計 + PPO 的訓練效率

#### 混合配置

```python
# 環境設計（借鑑 TD3）
1. 觀測空間：24 維（20 LiDAR + 4 robot）
2. 獎勵函數：3-4 項（極簡）
3. 課程學習：目標距離逐漸增加
4. 環境隨機化：動態障礙物 + 機器人位置

# 訓練算法（保持 PPO）
1. 並行環境：24 個（效率）
2. On-Policy：不需要大 Buffer（省記憶體）
3. 穩定性：PPO Clipping（訓練穩定）
4. 成熟工具：RSL-RL（易用）
```

#### 具體實作

**階段 1：簡化獎勵（立即可做）**
```python
# 刪除 v4 的複雜懲罰，回歸 TD3 式極簡設計
@configclass
class RewardsCfg:
    forward = RewTerm(
        func=lambda env: env.action_manager.action[:, 0] / 2,
        weight=1.0,
    )
    angular = RewTerm(
        func=lambda env: -torch.abs(env.action_manager.action[:, 1]) / 2,
        weight=1.0,
    )
    obstacle = RewTerm(
        func=mdp.obstacle_penalty,  # 新增：基於 min_lidar 的懲罰
        weight=1.0,
    )
    reached_goal = RewTerm(
        func=mdp.reached_goal_reward,
        weight=200.0,  # vs TD3 的 100
    )
```

**階段 2：課程學習（下一版）**
```python
# 在訓練腳本中動態調整目標範圍
class CurriculumManager:
    def __init__(self):
        self.goal_range = 5.0  # 初始 5m
    
    def update(self, iteration):
        if iteration % 1000 == 0 and self.goal_range < 10.0:
            self.goal_range += 0.05
            env_cfg.commands.goal_command.ranges.pos_x = (
                -self.goal_range, self.goal_range
            )
```

**階段 3：環境隨機化（進階）**
```python
# 每次 reset 時移動障礙物
def reset_obstacles(env):
    for i in range(4):
        x = torch.rand(1) * 12 - 6  # [-6, 6]
        y = torch.rand(1) * 12 - 6
        env.scene[f"obstacle_{i}"].set_world_poses(
            positions=torch.tensor([[x, y, 0.25]]),
        )
```

---

## 💡 實作建議

### 🚀 快速驗證路徑（v5 建議）

**v5 配置：TD3 極簡獎勵 + PPO 訓練**

```python
# 1. 獎勵函數：從 8 項 → 4 項
@configclass
class RewardsCfg:
    """v5：TD3 啟發的極簡設計"""
    
    # 密集獎勵（TD3 style）
    forward_velocity = RewTerm(
        func=lambda env: env.action_manager.action[:, 0] / 2,
        weight=1.0,
    )
    angular_penalty = RewTerm(
        func=lambda env: -torch.abs(env.action_manager.action[:, 1]) / 2,
        weight=1.0,
    )
    obstacle_proximity = RewTerm(
        func=mdp.obstacle_proximity_penalty,  # 實作 TD3 的 r3(min_laser)
        weight=1.0,
    )
    
    # 稀疏獎勵
    reached_goal = RewTerm(
        func=mdp.reached_goal_reward,
        weight=200.0,
    )
    
    # 刪除：progress, standstill, anti_idle, spin, time, heading, near_goal

# 2. 實作 obstacle_proximity_penalty
def obstacle_proximity_penalty(env) -> torch.Tensor:
    """TD3 式避障懲罰：r3(x) = -(1-x) if x<1 else 0"""
    min_lidar = torch.min(env.scene["lidar"].data.ray_hits_w, dim=-1).values
    penalty = torch.where(
        min_lidar < 1.0,
        -(1.0 - min_lidar) / 2,  # 範圍 [-0.5, 0]
        torch.zeros_like(min_lidar),
    )
    return penalty

# 3. PPO 參數（保持 v4）
LocalPlannerPPORunnerCfg(
    learning_rate=3e-4,
    entropy_coef=0.001,
    max_iterations=10000,
    logger="wandb",
    ...
)
```

**理論預期**：
```
正常移動（假設 v=0.5, ω=0.2, min_lidar=2.0）：
  forward = 0.5 / 2 = 0.25
  angular = -0.2 / 2 = -0.1
  obstacle = 0（距離 > 1m）
  Total = 0.15 ✅（鼓勵前進）

靠近障礙（v=0.3, ω=0.1, min_lidar=0.5）：
  forward = 0.15
  angular = -0.05
  obstacle = -(1 - 0.5) / 2 = -0.25
  Total = -0.15 ❌（懲罰靠近）

到達目標：
  + 200 ✅（強烈正向）
```

### 📊 v5 驗收標準

| 指標 | v4 結果 | v5 目標 | 理由 |
|------|---------|---------|------|
| **訓練穩定性** | - | 曲線平滑 | 極簡獎勵減少衝突 |
| **Forward Reward** | - | > 0.2 | 鼓勵前進生效 |
| **Obstacle Penalty** | - | < -0.1 | 避障機制生效 |
| **Position Error** | 3.84m | < 2.5m | TD3 論文達到 <1m |
| **Success Rate** | 0% | > 10% | 論文最終 >80% |

### 🛠️ 長期整合計畫

**Phase 1：極簡獎勵驗證**（v5 - 當前）
- ✅ 實作 TD3 式獎勵函數
- ✅ 移除所有懲罰項（standstill, anti_idle, time 等）
- 📊 訓練 10000 iter + WandB 記錄
- 🎯 目標：Success Rate > 10%

**Phase 2：課程學習**（v6）
```python
# 目標距離從 5m 逐漸增加到 10m
curriculum = CurriculumScheduler(
    initial_range=5.0,
    final_range=10.0,
    update_freq=1000,  # 每 1000 iter 增加
    increment=0.05,
)
```

**Phase 3：環境隨機化**（v7）
```python
# 動態障礙物 + 機器人隨機初始位置
@configclass
class LocalPlannerSceneCfg:
    robot = AssetBaseCfg(
        init_state=AssetInitialStateCfg(
            pos=RandomUniformDistribution((-5, -5, 0), (5, 5, 0)),
            rot=RandomUniformDistribution((0, 0, -π), (0, 0, π)),
        ),
    )
```

**Phase 4：近障礙物策略**（v8）
```python
# 15% 機率強制後退（當 min_lidar < 0.6m）
def sample_actions_with_obstacle_strategy(policy_actions, lidar_data):
    ...
```

**Phase 5：TD3 完整移植**（v9 - 可選）
- 實作完整 TD3 算法
- Replay Buffer（1M）
- 雙 Critic 網路
- Off-Policy 訓練

---

## 📚 關鍵差異總結

### 哲學差異

| 項目 | DRL-robot-navigation（TD3） | 當前專案（PPO） |
|------|---------------------------|----------------|
| **獎勵哲學** | 極簡（3 項） | 多元（8 項） |
| **學習策略** | Off-Policy（Replay） | On-Policy（即時） |
| **探索策略** | 噪音衰減 + 特殊策略 | 固定噪音 |
| **課程設計** | 目標距離逐漸增加 | 固定範圍 |
| **網路規模** | 大（1.5M） | 小（200k） |
| **訓練步數** | 長（5M） | 短（240k） |

### 成功要素分析

**TD3 論文成功的關鍵**：
1. ✅ **極簡獎勵**：只有 3 項，避免獎勵衝突
2. ✅ **課程學習**：目標距離從簡到難
3. ✅ **環境隨機化**：動態障礙物打破過擬合
4. ✅ **近障礙策略**：強制探索困難區域
5. ✅ **長時間訓練**：5M 步（vs 當前 240k）
6. ✅ **大網路**：800-600 隱藏層

**當前專案的挑戰**：
1. ❌ 獎勵過於複雜（8 項互相衝突）
2. ❌ 缺乏課程學習（目標範圍固定）
3. ❌ 環境固定（無隨機化）
4. ❌ 訓練時間較短
5. ⚠️ 網路較小（可能表達能力不足）

---

## ✅ 行動建議

### 立即可做（v5）

```bash
# 1. 創建 v5 配置文件
cp local_planner_env_cfg_min.py local_planner_env_cfg_td3_style.py

# 2. 實作 TD3 式獎勵
# - 刪除 progress, standstill, anti_idle, spin, time, heading, near_goal
# - 新增 forward_velocity, angular_penalty, obstacle_proximity
# - 保留 reached_goal

# 3. 實作 obstacle_proximity_penalty 函數
# source/isaaclab_tasks/.../mdp/rewards.py

# 4. 訓練 v5
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-TD3Style-v0 \
    --num_envs 24 \
    --max_iterations 10000 \
    --headless

# 5. 對比 v4 vs v5
# WandB 監控：forward, angular, obstacle, success_rate
```

### 後續迭代（v6-v9）

1. **v6**：課程學習（目標距離動態增加）
2. **v7**：環境隨機化（動態障礙物）
3. **v8**：近障礙策略（強制探索）
4. **v9**：TD3 完整移植（可選）

---

## 📖 參考資料

1. **論文**：[Goal-Driven Autonomous Exploration Through Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9645287)
2. **GitHub**：[reiniscimurs/DRL-robot-navigation](https://github.com/reiniscimurs/DRL-robot-navigation)
3. **TD3 原論文**：[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
4. **Medium 教學**：[Deep Reinforcement Learning in Mobile Robot Navigation - Tutorial](https://medium.com/@reinis_86651/deep-reinforcement-learning-in-mobile-robot-navigation-tutorial-part1-installation-d62715722303)

---

**總結**：`DRL-robot-navigation` 專案提供了一個經過論文驗證的成功架構。建議先借鑑其極簡獎勵設計（v5），驗證效果後再逐步整合課程學習、環境隨機化等進階技術。TD3 算法的完整移植可作為長期目標（記憶體和時間成本較高）。



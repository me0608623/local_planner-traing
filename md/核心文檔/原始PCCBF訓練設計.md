# 🎯 原始 PCCBF 訓練設計詳解

> **PPO + PCCBF** Nova Carter 導航訓練的完整架構說明

---

## 📋 目錄

1. [State (觀測空間)](#state-觀測空間)
2. [Action (動作空間)](#action-動作空間)
3. [Agent (神經網路架構)](#agent-神經網路架構)
4. [Reward (獎勵函數)](#reward-獎勵函數)
5. [PCCBF 整合](#pccbf-整合)
6. [訓練參數](#訓練參數)

---

## 🔍 State (觀測空間)

### 觀測維度總覽

**總維度**：`548` = `360 (LiDAR) + 3 (線速度) + 3 (角速度) + 2 (目標位置) + 1 (目標距離) + 179 (預測障礙物)`

**但在 v4 最小配置中簡化為**：`367` = `360 (LiDAR) + 3 + 3 + 2 + 1`

---

### 1️⃣ LiDAR 距離掃描

**配置**：
```python
lidar = RayCasterCfg(
    prim_path="/World/envs/.*/Robot/Robot/chassis_link/base_link",
    mesh_prim_paths=["/World/ground"],
    pattern_cfg=patterns.LidarPatternCfg(
        channels=1,              # 單層 2D LiDAR
        vertical_fov_range=(0.0, 0.0),
        horizontal_fov_range=(-180.0, 180.0),  # 360° 掃描
        horizontal_res=1.0,      # 每 1° 一條射線
    ),
    max_distance=10.0,           # 最大探測距離 10m
    drift_range=(0.0, 0.0),
    debug_vis=False,
)
```

**輸出**：
- **形狀**：`(num_envs, 360)`
- **數值範圍**：`[0, 1]`（已正規化，0 = 最遠 10m，1 = 接觸）
- **物理意義**：360° 方向的障礙物距離
  - 0° = 機器人前方
  - 90° = 機器人左側
  - -90° = 機器人右側
  - ±180° = 機器人後方

**實作細節**：
```python
def lidar_obs(env, sensor_cfg):
    sensor = env.scene.sensors[sensor_cfg.name]
    data = sensor.data
    
    # Isaac Sim 5.0+ 需要手動計算距離
    if hasattr(data, "ray_hits_w"):
        hit_points = data.ray_hits_w  # (num_envs, 360, 3)
        sensor_pos = data.pos_w.unsqueeze(1)  # (num_envs, 1, 3)
        distances = torch.norm(hit_points - sensor_pos, dim=-1)
    
    # 正規化到 [0, 1]
    distances = distances / sensor.cfg.max_distance
    distances = torch.clamp(distances, 0.0, 1.0)
    
    return distances  # (num_envs, 360)
```

---

### 2️⃣ 機器人速度

#### 線速度（Linear Velocity）

**輸出**：
- **形狀**：`(num_envs, 3)`
- **座標系**：機器人座標系（Body Frame）
- **內容**：`[vx, vy, vz]`
  - `vx`：前進速度（前 +，後 -）
  - `vy`：橫向速度（左 +，右 -）
  - `vz`：垂直速度（通常為 0）

```python
def base_lin_vel(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b  # (num_envs, 3)
```

#### 角速度（Angular Velocity）

**輸出**：
- **形狀**：`(num_envs, 3)`
- **座標系**：機器人座標系
- **內容**：`[wx, wy, wz]`
  - `wx`：繞 X 軸旋轉（翻滾，通常為 0）
  - `wy`：繞 Y 軸旋轉（俯仰，通常為 0）
  - `wz`：繞 Z 軸旋轉（偏航，正 = 逆時針）

```python
def base_ang_vel(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b  # (num_envs, 3)
```

---

### 3️⃣ 目標資訊

#### 目標相對位置（Robot Frame）

**輸出**：
- **形狀**：`(num_envs, 2)`
- **座標系**：機器人座標系
- **內容**：`[dx, dy]`
  - `dx`：前後方向（前 +，後 -）
  - `dy`：左右方向（左 +，右 -）

**轉換流程**：
```python
def goal_position_in_robot_frame(env, command_name):
    # 1. 獲取目標世界座標
    command = env.command_manager.get_command(command_name)
    goal_pos_w = command[:, :3]  # (num_envs, 3)
    
    # 2. 獲取機器人世界座標
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w
    
    # 3. 計算相對位置（世界座標系）
    goal_pos_rel_w = goal_pos_w - robot_pos_w
    
    # 4. 轉換到機器人座標系
    goal_pos_rel_b = quat_apply_inverse(robot_quat_w, goal_pos_rel_w)
    
    # 5. 只返回 x, y（忽略 z）
    return goal_pos_rel_b[:, :2]
```

#### 目標距離（Scalar）

**輸出**：
- **形狀**：`(num_envs, 1)`
- **數值範圍**：`[0, ∞)`（單位：米）
- **計算**：2D 歐氏距離（忽略 Z 軸）

```python
def distance_to_goal(env, command_name):
    command = env.command_manager.get_command(command_name)
    goal_pos_w = command[:, :3]
    
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w
    
    # 2D 距離（忽略 z）
    distance = torch.norm(
        goal_pos_w[:, :2] - robot_pos_w[:, :2], 
        dim=-1, 
        keepdim=True
    )
    
    return distance  # (num_envs, 1)
```

---

### 4️⃣ PCCBF 預測觀測（可選）

**注意**：當前 v4 配置中**未啟用**，但代碼中已實作

#### 預測障礙物距離

**功能**：預測未來 N 步的障礙物分布

```python
def predicted_obstacle_distances(
    env, 
    sensor_cfg,
    prediction_horizon: int = 3  # 預測未來 3 步
):
    """
    基於 PCCBF-MPC 的「前瞻時域地圖（FTD Map）」概念
    預測未來 N 步機器人周圍的障礙物分布
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene["robot"]
    
    # 當前 LiDAR 數據
    current_distances = lidar_obs(env, sensor_cfg)
    
    # 機器人當前速度
    lin_vel = robot.data.root_lin_vel_b
    ang_vel = robot.data.root_ang_vel_b
    
    # 簡化：線性預測（等速模型）
    # 真實 PCCBF 使用卡爾曼濾波器
    dt = 0.1  # 時間步長
    predicted_positions = []
    
    for t in range(1, prediction_horizon + 1):
        # 預測機器人位置
        pred_x = lin_vel[:, 0] * t * dt
        pred_y = lin_vel[:, 1] * t * dt
        pred_theta = ang_vel[:, 2] * t * dt
        
        # 轉換 LiDAR 到預測位置
        # ... (簡化實作)
        
    return min_predicted_distances
```

---

### 📊 觀測空間總結

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 1. LiDAR 掃描 (360)
        lidar_distances = ObsTerm(
            func=mdp.lidar_obs,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
        )
        
        # 2. 線速度 (3)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        
        # 3. 角速度 (3)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        
        # 4. 目標相對位置 (2)
        goal_position = ObsTerm(
            func=mdp.goal_position_in_robot_frame,
            params={"command_name": "goal_command"},
        )
        
        # 5. 目標距離 (1)
        goal_distance = ObsTerm(
            func=mdp.distance_to_goal,
            params={"command_name": "goal_command"},
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  # 串接所有觀測

    policy: PolicyCfg = PolicyCfg()
```

**總維度**：360 + 3 + 3 + 2 + 1 = **369**

---

## 🎮 Action (動作空間)

### 差速驅動控制（Differential Drive）

**維度**：`2` = `[線速度指令, 角速度指令]`

**配置**：
```python
@configclass
class ActionsCfg:
    base_velocity = mdp.DifferentialDriveActionCfg(
        asset_name="robot",
        left_wheel_joint_names=["joint_wheel_left"],
        right_wheel_joint_names=["joint_wheel_right"],
        wheel_radius=0.125,       # 輪子半徑 12.5cm
        wheel_base=0.413,         # 輪距 41.3cm
        max_linear_speed=0.8,     # 最大線速度 0.8 m/s
        max_angular_speed=0.8,    # 最大角速度 0.8 rad/s
    )
```

### 動作轉換流程

```
神經網路輸出 (NN Output)
  ↓
[v_cmd, w_cmd]  ← 連續值在 [-1, 1] 之間
  ↓
縮放到物理範圍 (Scaling)
  ↓
v = v_cmd * max_linear_speed   # [-0.8, 0.8] m/s
w = w_cmd * max_angular_speed  # [-0.8, 0.8] rad/s
  ↓
差速驅動運動學 (Differential Drive Kinematics)
  ↓
v_left  = (v - w * wheel_base / 2) / wheel_radius
v_right = (v + w * wheel_base / 2) / wheel_radius
  ↓
輪子速度指令發送到模擬器
```

### 動作範圍

| 動作 | 神經網路輸出 | 物理範圍 | 物理意義 |
|------|-------------|----------|----------|
| **線速度** | `[-1, 1]` | `[-0.8, 0.8]` m/s | 前進（+）/ 後退（-） |
| **角速度** | `[-1, 1]` | `[-0.8, 0.8]` rad/s | 逆時針（+）/ 順時針（-） |

**典型動作**：
- `[1.0, 0.0]`：最快前進
- `[-1.0, 0.0]`：最快後退
- `[0.0, 1.0]`：原地逆時針旋轉
- `[0.0, -1.0]`：原地順時針旋轉
- `[0.5, 0.5]`：前進 + 左轉

---

## 🧠 Agent (神經網路架構)

### PPO Actor-Critic 架構

**配置**：
```python
policy = RslRlPpoActorCriticCfg(
    init_noise_std=0.5,          # 初始探索噪音
    actor_hidden_dims=[256, 256, 128],
    critic_hidden_dims=[256, 256, 128],
    activation="elu",            # ELU 激活函數
)
```

---

### 1️⃣ Actor Network（策略網路）

**功能**：輸出動作分布

```
Input: State (369)
  ↓
Dense(369 → 256) + ELU
  ↓
Dense(256 → 256) + ELU
  ↓
Dense(256 → 128) + ELU
  ↓
Dense(128 → 2)  # 均值 μ
  ↓
Output: Gaussian(μ, σ)
        σ 初始為 0.5，訓練中逐漸降低
  ↓
Sample: action ~ N(μ, σ²)
  ↓
Clip: action ∈ [-1, 1]
```

**參數量**：
```
Layer 1: 369 × 256 + 256 = 94,720
Layer 2: 256 × 256 + 256 = 65,792
Layer 3: 256 × 128 + 128 = 32,896
Output:  128 × 2 + 2 = 258
Total: ~193k 參數
```

---

### 2️⃣ Critic Network（價值網路）

**功能**：估計狀態價值 V(s)

```
Input: State (369)
  ↓
Dense(369 → 256) + ELU
  ↓
Dense(256 → 256) + ELU
  ↓
Dense(256 → 128) + ELU
  ↓
Dense(128 → 1)
  ↓
Output: V(s)  # 狀態價值（標量）
```

**參數量**：
```
Layer 1: 369 × 256 + 256 = 94,720
Layer 2: 256 × 256 + 256 = 65,792
Layer 3: 256 × 128 + 128 = 32,896
Output:  128 × 1 + 1 = 129
Total: ~193k 參數
```

---

### 🔧 網路設計特點

**1. ELU 激活函數**
```python
ELU(x) = {
    x,              if x > 0
    α(e^x - 1),     if x ≤ 0
}
```
- 優點：允許負值輸出，收斂更快
- vs ReLU：不會有"死神經元"問題

**2. 逐步衰減的探索**
```python
init_noise_std = 0.5  # 初始探索

# 訓練過程中：
σ(t) = σ₀ × decay_factor^t
# 逐漸從 0.5 降到 ~0.1
```

**3. 共享特徵提取**
- Actor 和 Critic 使用相同架構
- 可選：共享前幾層（當前未共享）

---

## 🎁 Reward (獎勵函數)

### v4 獎勵配置（當前版本）

**總獎勵項**：8 項（正向 4 項 + 負向 4 項）

---

### 1️⃣ 正向獎勵（Positive Rewards）

#### progress_to_goal（接近目標）

**權重**：`60.0`（最重要）

**計算**：
```python
def progress_to_goal_reward(env, command_name):
    # 當前距離
    current_distance = ||goal_pos - robot_pos||₂
    
    # 進度 = 上一步距離 - 當前距離
    progress = prev_distance - current_distance
    
    # 正值：接近目標
    # 負值：遠離目標
    # 零值：距離不變
    
    return progress
```

**物理意義**：
- 每步接近目標 0.1m → 獎勵 = 0.1 × 60 = 6.0
- 每步遠離目標 0.05m → 獎勵 = -0.05 × 60 = -3.0

---

#### reached_goal（到達目標）

**權重**：`200.0`

**計算**：
```python
def reached_goal_reward(env, command_name, threshold=0.8):
    distance = ||goal_pos - robot_pos||₂
    reward = 1.0 if distance < 0.8m else 0.0
    return reward
```

**物理意義**：
- 到達目標（< 0.8m）→ 獎勵 = 1.0 × 200 = 200
- 未到達 → 獎勵 = 0

---

#### near_goal_shaping（近目標塑形）

**權重**：`20.0`

**計算**：
```python
def near_goal_shaping(env, command_name, radius=3.0):
    distance = ||goal_pos - robot_pos||₂
    
    if distance < radius:
        # 指數衰減獎勵
        reward = exp(-distance / radius)
    else:
        reward = 0.0
    
    return reward
```

**物理意義**：
- 距離 0m → 獎勵 = 1.0 × 20 = 20
- 距離 1.5m → 獎勵 = exp(-0.5) × 20 ≈ 12
- 距離 3m → 獎勵 = exp(-1) × 20 ≈ 7.4
- 距離 > 3m → 獎勵 = 0

---

#### heading_alignment（朝向對齊）

**權重**：`1.0`（條件式）

**計算**：
```python
def heading_alignment_reward(env, command_name, v_min=0.1):
    # 計算朝向誤差
    goal_dir = goal_pos - robot_pos
    robot_heading = robot_quaternion_to_heading()
    
    heading_error = angle_between(robot_heading, goal_dir)
    
    # 條件：必須在前進時才獎勵
    is_moving = ||lin_vel|| > v_min
    
    if is_moving:
        reward = cos(heading_error)  # [1, -1]
    else:
        reward = 0.0
    
    return reward
```

**物理意義**：
- 完美朝向（0°）且前進 → 獎勵 = 1.0 × 1 = 1.0
- 垂直朝向（90°）且前進 → 獎勵 = 0.0
- 反向朝向（180°）且前進 → 獎勵 = -1.0
- 靜止不動 → 獎勵 = 0（避免原地轉圈）

---

### 2️⃣ 負向懲罰（Penalties）

#### standstill_penalty（靜止懲罰）

**權重**：`1.0`（v4 降低）

**計算**：
```python
def standstill_penalty(env):
    speed = ||lin_vel||₂
    
    # 速度越低，懲罰越大
    penalty = -exp(-speed / 0.1)
    
    return penalty  # 範圍 [-1, 0]
```

**物理意義**：
- 速度 0 m/s → 懲罰 = -1.0 × 1.0 = -1.0
- 速度 0.1 m/s → 懲罰 ≈ -0.37
- 速度 > 0.5 m/s → 懲罰 ≈ 0

---

#### anti_idle_penalty（反閒置懲罰）

**權重**：`0.5`（v4 降低）

**計算**：
```python
def anti_idle_penalty(env, v_threshold=0.05):
    speed = ||lin_vel||₂
    
    if speed < v_threshold:
        penalty = -1.0
    else:
        penalty = 0.0
    
    return penalty
```

**物理意義**：
- 速度 < 0.05 m/s → 懲罰 = -1.0 × 0.5 = -0.5
- 速度 ≥ 0.05 m/s → 懲罰 = 0

---

#### spin_penalty（旋轉懲罰）

**權重**：`0.1`（v4 降低）

**計算**：
```python
def spin_penalty(env, w_threshold=0.5, v_threshold=0.1):
    ang_speed = |ang_vel_z|
    lin_speed = ||lin_vel||₂
    
    # 高角速度 + 低線速度 = 原地打轉
    if ang_speed > w_threshold and lin_speed < v_threshold:
        penalty = -ang_speed
    else:
        penalty = 0.0
    
    return penalty
```

**物理意義**：
- 原地快速旋轉（w=0.8, v=0.05）→ 懲罰 = -0.8 × 0.1 = -0.08
- 前進中轉彎（w=0.3, v=0.5）→ 懲罰 = 0

---

#### time_penalty（時間懲罰）

**權重**：`0.005`（v4 降低）

**計算**：
```python
def time_penalty(env):
    return -1.0  # 每步固定懲罰
```

**物理意義**：
- 每步 → 懲罰 = -1.0 × 0.005 = -0.005
- 每 episode（1500 步）→ 總懲罰 = -7.5

---

### 📊 獎勵權重總覽（v4 配置）

| 獎勵項 | 權重 | 類型 | 範圍 | 目的 |
|--------|------|------|------|------|
| **progress_to_goal** | 60.0 | 正向 | (-∞, +∞) | 主要驅動力 |
| **reached_goal** | 200.0 | 正向 | {0, 200} | 成功獎勵 |
| **near_goal_shaping** | 20.0 | 正向 | [0, 20] | 近距離引導 |
| **heading_alignment** | 1.0 | 正向 | [-1, 1] | 方向輔助 |
| **standstill_penalty** | 1.0 | 負向 | [-1, 0] | 防止靜止 |
| **anti_idle** | 0.5 | 負向 | [-0.5, 0] | 防止閒置 |
| **spin_penalty** | 0.1 | 負向 | [-0.08, 0] | 防止打轉 |
| **time_penalty** | 0.005 | 負向 | -0.005 | 鼓勵快速 |

---

### 🎯 獎勵設計理念（v4）

**正向主導**：
```
理想情況（前進 0.1m/步）：
  progress: 0.1 × 60 = 6.0
  near_goal: ~0.3 × 20 = 6.0
  heading: ~0.5 × 1 = 0.5
  Total positive: ~12.5

負向約束：
  time: -0.005
  standstill: ~0 (在動)
  Total negative: ~-0.005

淨獎勵: ~12.5 ✅（強正向）
```

**vs v3（過度懲罰）**：
```
v3 配置：
  positive: ~12.5
  negative: standstill(4.0) + anti_idle(2.0) + time(0.01) ≈ -6
  淨獎勵: ~6.5（正向被壓制）
```

---

## 🛡️ PCCBF 整合

### PCCBF 概念

**PCCBF** = **Perception-aware Control Barrier Functions**（感知感知控制障礙函數）

**核心思想**：
1. **預測安全約束**：預測未來 N 步的障礙物分布
2. **軟約束**：不是硬性阻止動作，而是通過獎勵引導
3. **感知融合**：結合 LiDAR 和運動模型

---

### 當前實作狀態

**⚠️ 注意**：v4 配置中 PCCBF **未完全啟用**

**已實作**：
- ✅ 預測障礙物距離函數（`predicted_obstacle_distances`）
- ✅ PCCBF 安全獎勵函數（`pccbf_safety_reward`）
- ✅ PCCBF 違反懲罰函數（`pccbf_violation_penalty`）

**未啟用**：
- ❌ 未加入觀測空間（只用當前 LiDAR）
- ❌ 未加入獎勵配置（只用基本獎勵）

---

### PCCBF 獎勵函數（代碼中已實作）

```python
def pccbf_safety_reward(
    env,
    sensor_cfg,
    prediction_horizon: int = 3,
    safe_distance: float = 1.0,
) -> torch.Tensor:
    """PCCBF 啟發的安全獎勵
    
    原理：
    - 預測未來 N 步的障礙物分布
    - 如果預測路徑安全（距離 > safe_distance）→ 正獎勵
    - 如果預測路徑危險 → 負懲罰
    """
    # 預測未來障礙物距離
    pred_distances = predicted_obstacle_distances(
        env, sensor_cfg, prediction_horizon
    )
    
    # 計算安全裕度
    safety_margin = pred_distances - safe_distance
    
    # 安全：正獎勵；危險：負懲罰
    reward = torch.tanh(safety_margin)
    
    return reward
```

---

### 如何完整啟用 PCCBF

**步驟 1：加入觀測**
```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # ... 現有觀測 ...
        
        # 新增：PCCBF 預測
        predicted_obstacles = ObsTerm(
            func=mdp.predicted_obstacle_distances,
            params={
                "sensor_cfg": SceneEntityCfg("lidar"),
                "prediction_horizon": 3,
            },
        )
```

**步驟 2：加入獎勵**
```python
@configclass
class RewardsCfg:
    # ... 現有獎勵 ...
    
    # 新增：PCCBF 安全獎勵
    pccbf_safety = RewTerm(
        func=mdp.pccbf_safety_reward,
        weight=5.0,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "prediction_horizon": 3,
            "safe_distance": 1.0,
        },
    )
```

**步驟 3：調整網路**
```python
# 觀測維度增加
# 369 → 369 + (prediction_horizon * num_angles)
# 369 → 369 + (3 * 360) = 1449

policy = RslRlPpoActorCriticCfg(
    actor_hidden_dims=[512, 512, 256],  # 增大網路
    critic_hidden_dims=[512, 512, 256],
)
```

---

## ⚙️ 訓練參數

### PPO 算法參數

```python
algorithm = RslRlPpoAlgorithmCfg(
    # 損失函數係數
    value_loss_coef=1.0,              # Value loss 權重
    use_clipped_value_loss=True,      # 使用 Clipped Value Loss
    
    # PPO 核心參數
    clip_param=0.1,                   # PPO clip 範圍（v4 降低穩定）
    entropy_coef=0.001,               # 熵係數（v4 降低收斂）
    
    # 學習設置
    num_learning_epochs=3,            # 每次更新的 epoch 數（v4 降低）
    num_mini_batches=4,               # Mini-batch 數量
    learning_rate=3e-4,               # 學習率（v4 降低）
    schedule="adaptive",              # 自適應學習率
    
    # 折扣因子
    gamma=0.99,                       # 折扣因子
    lam=0.95,                         # GAE lambda
    
    # 約束
    desired_kl=0.01,                  # 目標 KL 散度
    max_grad_norm=1.0,                # 梯度裁剪
)
```

---

### 訓練流程參數

```python
runner = LocalPlannerPPORunnerCfg(
    # 基本設置
    seed=42,                          # 隨機種子
    device="cuda:0",                  # 訓練設備
    
    # 數據收集
    num_steps_per_env=24,             # 每個環境收集 24 步
    # 總樣本數 = num_envs(24) × num_steps_per_env(24) = 576
    
    # 訓練長度
    max_iterations=10000,             # 最大迭代數（v4）
    save_interval=100,                # 每 100 iter 保存
    
    # 日誌
    experiment_name="local_planner_carter",
    run_name="v4_balanced_penalties",
    logger="wandb",
    wandb_project="nova-carter-navigation",
)
```

---

### 環境參數

```python
env_cfg = LocalPlannerEnvCfgMin(
    # 場景設置
    scene=LocalPlannerSceneCfgMin(
        num_envs=24,                  # 並行環境數
        env_spacing=10.0,             # 環境間隔 10m
    ),
    
    # 時間設置
    decimation=2,                     # 控制頻率下採樣
    episode_length_s=30.0,            # Episode 長度 30 秒
    sim.dt=0.01,                      # 模擬時間步 0.01s
    sim.render_interval=2,            # 渲染間隔（decimation）
    
    # 設備
    sim.device="cuda:0",
)
```

**時間計算**：
```
物理時間步 dt = 0.01s
控制頻率 = 1 / (dt × decimation) = 1 / 0.02 = 50 Hz
Episode 步數 = episode_length_s / (dt × decimation) = 30 / 0.02 = 1500 步
```

---

### 目標生成參數

```python
goal_command = UniformPoseCommandCfg(
    resampling_time_range=(10.0, 10.0),  # 每 10 秒重新生成目標
    ranges=Ranges(
        pos_x=(2.0, 6.0),                # X 範圍 2-6m
        pos_y=(-3.0, 3.0),               # Y 範圍 -3-3m
        pos_z=(0.0, 0.0),                # Z 固定為 0
        yaw=(0.0, 0.0),                  # 不限制朝向
    ),
)
```

---

## 📊 完整數據流

```
訓練循環（每 Iteration）:

1. 數據收集 (Rollout)
   ├─ 24 個環境並行
   ├─ 每個環境收集 24 步
   └─ 總共 576 個 (s, a, r, s') 樣本

2. 優勢估計 (GAE)
   ├─ 使用 Critic 估計 V(s)
   ├─ 計算 TD error
   └─ 計算 Advantage A(s,a)

3. PPO 更新
   ├─ 3 個 Epoch
   ├─ 每個 Epoch 分 4 個 Mini-batch
   │   ├─ Mini-batch size = 576 / 4 = 144
   │   ├─ Actor loss (PPO clip)
   │   ├─ Critic loss (Value prediction)
   │   └─ Entropy loss (探索)
   └─ 梯度裁剪 + Adam 優化

4. 記錄 & 保存
   ├─ WandB 記錄曲線
   ├─ 每 100 iter 保存模型
   └─ 更新學習率（Adaptive）
```

---

## 🎯 關鍵設計決策

### 1️⃣ 為什麼用 PPO 而不是 TD3？

| 項目 | PPO | TD3 | 選擇 |
|------|-----|-----|------|
| **訓練穩定性** | ✅ 高（On-Policy + Clip） | ⚠️ 中（Off-Policy） | PPO |
| **樣本效率** | ⚠️ 低（用完即丟） | ✅ 高（Replay Buffer） | TD3 |
| **並行訓練** | ✅ 天生支持 | ⚠️ 需要額外設計 | PPO |
| **實作複雜度** | ✅ 簡單 | ⚠️ 複雜 | PPO |

**結論**：PPO 更適合 Isaac Lab 的並行環境架構

---

### 2️⃣ 為什麼觀測空間這麼大（369 維）？

**LiDAR 360 維**：
- 優點：完整的 360° 感知
- 缺點：維度高，訓練慢
- 替代方案：降採樣到 72 維（每 5°）

**當前保留原因**：
- 高解析度避障
- 論文中使用相同設置
- GPU 訓練可處理

---

### 3️⃣ 為什麼 v4 大幅降低懲罰權重？

**v3 問題**：
```
進度獎勵：+6.0
懲罰總和：-6.0
淨獎勵：≈0（互相抵消）
結果：Agent 被壓制，不敢動
```

**v4 修正**：
```
進度獎勵：+6.0
懲罰總和：-0.5
淨獎勵：+5.5（正向主導）
結果：Agent 敢於探索
```

---

### 4️⃣ 為什麼 PCCBF 未完全啟用？

**原因**：
1. **複雜度**：預測增加觀測維度和計算量
2. **階段性**：先驗證基本導航，再加安全約束
3. **調試友好**：簡單獎勵更易分析

**未來計劃**：
- v5：極簡獎勵（TD3 風格）
- v6：課程學習
- v7：完整 PCCBF 整合

---

## 📚 總結

### 核心設計

1. **State**：360° LiDAR + 速度 + 目標資訊（369 維）
2. **Action**：差速驅動（線速度 + 角速度，2 維）
3. **Agent**：3 層 MLP（256-256-128）
4. **Reward**：8 項（正向主導，懲罰為輔）

### 關鍵特點

- ✅ 並行訓練（24 環境）
- ✅ 高頻控制（50 Hz）
- ✅ WandB 記錄
- ✅ 模塊化設計（易於調整）
- ⚠️ PCCBF 預留但未啟用

### 優化方向

1. **短期**（v5）：極簡獎勵（3-4 項）
2. **中期**（v6-v7）：課程學習 + 環境隨機化
3. **長期**（v8-v9）：完整 PCCBF 整合

---

**這就是原始 PCCBF 訓練的完整設計！** 🎯


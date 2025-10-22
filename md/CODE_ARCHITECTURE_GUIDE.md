# Nova Carter 訓練代碼架構指南

## 🎯 訓練策略核心代碼

### 主要訓練腳本

```
scripts/reinforcement_learning/rsl_rl/train.py
```

**用途**: 訓練的入口點，負責啟動整個訓練流程

**核心功能**:
1. 初始化 Isaac Sim 環境
2. 創建並配置 RL 環境
3. 初始化 PPO 演算法
4. 執行訓練循環
5. 保存模型檢查點
6. 記錄訓練指標

**使用方法**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 4 --headless
```

---

## 📁 核心代碼文件結構

### 1. 訓練算法配置
```
source/isaaclab_tasks/.../local_planner/agents/rsl_rl_ppo_cfg.py
```

**用途**: 定義 PPO 演算法的超參數和網路架構

**關鍵配置**:

#### 基本設定
```python
seed: int = 42                    # 隨機種子（可重現性）
device: str = "cuda:0"            # 訓練設備
num_steps_per_env: int = 24       # 每個環境收集的步數
max_iterations: int = 3000        # 最大訓練迭代次數
save_interval: int = 100          # 每100次迭代保存一次模型
```

#### 網路架構
```python
actor_hidden_dims=[256, 256, 128]    # Actor網路：3層 MLP
critic_hidden_dims=[256, 256, 128]   # Critic網路：3層 MLP
activation="elu"                     # 激活函數
init_noise_std=1.0                   # 初始探索噪音
```

#### PPO 超參數
```python
learning_rate=1e-3        # 學習率
clip_param=0.2            # PPO裁剪參數
entropy_coef=0.01         # 熵係數（鼓勵探索）
gamma=0.99                # 折扣因子
lam=0.95                  # GAE lambda
num_learning_epochs=5     # 每批數據訓練5個epoch
num_mini_batches=4        # 小批次數量
```

**這個文件決定了**:
- 如何學習（演算法參數）
- 學習速度（學習率、更新頻率）
- 探索vs利用（熵係數、噪音）

---

### 2. 環境配置
```
source/isaaclab_tasks/.../local_planner/local_planner_env_cfg.py
```

**用途**: 定義整個 RL 環境的配置

#### 主要組件

##### A. 場景配置 (`LocalPlannerSceneCfg`)
```python
- 地形: 平坦地面
- 機器人: Nova Carter（差速驅動）
- LiDAR: 360度掃描（100束光線，10米範圍）
- 障礙物: 靜態和動態障礙物
- 環境數量: num_envs（並行訓練）
```

##### B. 觀測配置 (`ObservationsCfg`)
```python
policy觀測:
- LiDAR 數據: 360度障礙物距離（100維）
- 目標相對位置: 距離 + 角度（2維）
- 機器人速度: 線速度 + 角速度（2維）
- 前一動作: 歷史控制指令（2維）
總維度: ~106維
```

##### C. 動作配置 (`ActionsCfg`)
```python
動作空間: 2維連續動作
- 線速度指令: [-2.0, 2.0] m/s
- 角速度指令: [-3.14, 3.14] rad/s

通過差速驅動模型轉換為左右輪速度
```

##### D. 獎勵配置 (`RewardsCfg`) ⭐ **核心訓練策略**
```python
正向獎勵:
✅ progress_to_goal (權重: 10.0)
   - 獎勵接近目標的行為
   - 基於距離減少量

✅ reached_goal (權重: 100.0)
   - 到達目標的大獎勵
   - 閾值: 0.5米

負向獎勵（懲罰）:
❌ obstacle_proximity_penalty (權重: -5.0)
   - 懲罰靠近障礙物
   - 安全距離: 1.0米

❌ collision_penalty (權重: -50.0)
   - 嚴重懲罰碰撞
   - 碰撞閾值: 0.3米

❌ ang_vel_penalty (權重: -0.01)
   - 輕微懲罰過大角速度
   - 鼓勵平滑運動

❌ standstill_penalty (權重: -0.1)
   - 懲罰靜止不動
```

**獎勵設計哲學**:
- 大獎勵引導到達目標（100.0）
- 持續獎勵引導接近目標（10.0）
- 嚴重懲罰危險行為（-50.0碰撞）
- 輕微懲罰不良習慣（-0.1靜止）

##### E. 終止條件 (`TerminationsCfg`)
```python
Episode 結束條件:
1. time_out: 超時（30秒）
2. goal_reached: 到達目標
3. collision: 碰撞障礙物
```

##### F. 命令配置 (`CommandsCfg`)
```python
目標命令:
- 目標位置: 隨機生成在5-10米範圍
- 重採樣: 當到達目標或超時時
```

**這個文件決定了**:
- 機器人學什麼（觀測和動作）
- 如何評價行為好壞（獎勵函數）
- 什麼時候結束（終止條件）

---

### 3. MDP 組件實現

#### A. 觀測函數
```
source/isaaclab_tasks/.../local_planner/mdp/observations.py
```

**用途**: 實現具體的觀測獲取邏輯

**核心函數**:
```python
lidar_obs(): 
  - 從 RayCaster sensor 獲取 LiDAR 數據
  - 提取距離信息
  - 標準化處理

goal_relative_position():
  - 計算機器人到目標的相對位置
  - 返回距離和角度

robot_velocity():
  - 獲取當前速度狀態
```

#### B. 動作處理
```
source/isaaclab_tasks/.../local_planner/mdp/actions.py
```

**用途**: 定義動作如何轉換為機器人控制

**核心類**:
```python
DifferentialDriveAction:
  輸入: [線速度, 角速度]
  輸出: [左輪速度, 右輪速度]
  
  轉換公式:
  v_left = v_linear - v_angular * wheel_base / 2
  v_right = v_linear + v_angular * wheel_base / 2
```

#### C. 獎勵函數 ⭐ **訓練策略的核心**
```
source/isaaclab_tasks/.../local_planner/mdp/rewards.py
```

**用途**: 實現各個獎勵函數的計算邏輯

**核心函數**:

```python
progress_to_goal_reward():
  """接近目標獎勵"""
  當前距離與上次距離的差 * 權重
  鼓勵持續接近目標

reached_goal_reward():
  """到達目標獎勵"""
  if 距離 < threshold:
      return 大獎勵
  鼓勵完成任務

obstacle_proximity_penalty():
  """障礙物接近懲罰"""
  基於 LiDAR 最近距離
  距離越近，懲罰越大

collision_penalty():
  """碰撞懲罰"""
  if LiDAR最近距離 < 碰撞閾值:
      return 大懲罰
  防止危險行為

standstill_penalty():
  """靜止懲罰"""
  if 速度 < 閾值:
      return 懲罰
  鼓勵主動移動
```

**訓練策略實現細節**:
- 使用稠密獎勵（dense reward）而非稀疏獎勵
- 距離減少立即給予獎勵（即時反饋）
- 危險行為給予強懲罰（安全優先）
- 小懲罰引導良好習慣（平滑運動）

#### D. 終止條件
```
source/isaaclab_tasks/.../local_planner/mdp/terminations.py
```

**用途**: 判斷 episode 何時結束

**核心函數**:
```python
goal_reached(): 檢查是否到達目標
collision_termination(): 檢查是否碰撞
time_out(): 檢查是否超時
```

---

## 🔄 訓練流程

### 完整訓練循環

```
1. 初始化環境
   ├─ 創建場景（地形、機器人、障礙物）
   ├─ 初始化 LiDAR sensor
   └─ 生成隨機目標

2. 初始化 PPO 演算法
   ├─ 創建 Actor-Critic 網路
   ├─ 設定優化器（Adam）
   └─ 初始化經驗緩衝區

3. 訓練循環（max_iterations 次）:
   
   For each iteration:
   
   3.1 收集經驗（rollout）:
       For step = 1 to num_steps_per_env:
           ├─ 獲取觀測（LiDAR、目標位置等）
           ├─ Actor 輸出動作（線速度、角速度）
           ├─ 環境執行動作
           ├─ 計算獎勵（根據 RewardsCfg）
           ├─ 檢查終止條件
           └─ 存儲 transition (s, a, r, s')
   
   3.2 計算優勢函數（Advantage）:
       ├─ 使用 Critic 估計 Value
       └─ 計算 GAE (Generalized Advantage Estimation)
   
   3.3 更新策略（PPO）:
       For epoch = 1 to num_learning_epochs:
           For each mini_batch:
               ├─ 計算策略損失（PPO clip）
               ├─ 計算價值損失（MSE）
               ├─ 計算熵損失（探索）
               ├─ 總損失 = policy_loss + value_loss - entropy_loss
               └─ 反向傳播 + 梯度更新
   
   3.4 記錄指標:
       ├─ 平均獎勵
       ├─ Episode 長度
       ├─ 成功率
       └─ 各項獎勵分量
   
   3.5 保存模型（每 save_interval 次）

4. 訓練完成
   └─ 保存最終模型
```

---

## 🎓 訓練策略詳解

### 核心訓練策略: PPO (Proximal Policy Optimization)

#### 為什麼選擇 PPO？
1. ✅ **穩定性**: 裁剪機制防止策略更新過大
2. ✅ **樣本效率**: 可以多次使用同一批數據
3. ✅ **實現簡單**: 相比 TRPO 更容易實現
4. ✅ **效果好**: 在多種任務上表現優秀

#### PPO 的關鍵機制

##### 1. 裁剪目標函數
```python
ratio = π_new(a|s) / π_old(a|s)  # 新舊策略比率
clipped_ratio = clip(ratio, 1-ε, 1+ε)  # 裁剪，ε=0.2

L_clip = min(
    ratio * advantage,           # 未裁剪目標
    clipped_ratio * advantage    # 裁剪目標
)
```
**作用**: 防止策略更新太激進，保持訓練穩定

##### 2. 價值函數學習
```python
V(s) ≈ r + γ * V(s')  # Bellman 方程
L_value = MSE(V_predicted, V_target)
```
**作用**: 估計狀態的好壞，用於計算優勢函數

##### 3. 熵正則化
```python
H(π) = -Σ π(a|s) log π(a|s)  # 策略熵
L_entropy = -β * H(π)
```
**作用**: 鼓勵探索，防止過早收斂

##### 4. 優勢函數 (GAE)
```python
A_t = Σ (γλ)^k * δ_t+k
δ_t = r_t + γV(s_t+1) - V(s_t)
```
**作用**: 估計某個動作比平均好多少

### 獎勵設計策略

#### 稠密獎勵 vs 稀疏獎勵
本項目使用**稠密獎勵**設計：

```python
# 稠密獎勵：每步都有反饋
reward = (
    + progress_to_goal * 10.0        # 持續引導
    + reached_goal * 100.0           # 最終目標
    - obstacle_proximity * 5.0       # 即時危險警告
    - collision * 50.0               # 嚴重錯誤
    - ang_vel_penalty * 0.01         # 習慣養成
    - standstill * 0.1               # 鼓勵行動
)
```

**優點**:
- ✅ 學習更快（每步都有指導）
- ✅ 更容易調試（可以看到各項獎勵）
- ✅ 適合複雜任務（導航需要持續引導）

#### 獎勵權重設計原則
```
目標達成 (100.0) >> 碰撞 (-50.0) > 接近目標 (10.0) > 
接近障礙物 (-5.0) > 靜止 (-0.1) > 角速度 (-0.01)
```

**設計邏輯**:
1. **最重要**: 達成目標（生存 + 成功）
2. **次重要**: 避免碰撞（安全第一）
3. **持續引導**: 接近目標（方向正確）
4. **安全習慣**: 保持距離（預防性）
5. **運動品質**: 平滑移動（錦上添花）

---

## 📊 關鍵訓練指標

### 監控這些指標了解訓練狀態

```python
# 性能指標
Mean reward              # 平均累積獎勵（應該上升）
Episode length           # 平均 episode 長度
Success rate             # 到達目標的比率（goal_reached）

# 獎勵分解
progress_to_goal         # 接近目標獎勵（應該為正）
reached_goal            # 到達目標次數（應該增加）
collision_penalty       # 碰撞懲罰（應該減少）
obstacle_proximity      # 接近障礙物（應該減少）

# 終止原因
time_out                # 超時比率（應該減少）
goal_reached            # 成功比率（應該增加）
collision              # 碰撞比率（應該減少）

# 演算法指標
Value function loss     # 價值估計誤差（應該收斂）
Entropy loss           # 探索程度（逐漸減少）
Learning rate          # 當前學習率（可能自適應調整）
```

### 良好訓練的特徵

```
前期（0-500次）:
- Mean reward: 大幅波動，逐漸上升
- Success rate: 0-10%
- 主要學習基本導航

中期（500-1500次）:
- Mean reward: 持續上升，波動減小
- Success rate: 10-40%
- 學習避障和目標導向

後期（1500-3000次）:
- Mean reward: 穩定在較高水平
- Success rate: 40-70%+
- 精細化策略，提高成功率
```

---

## 🔧 調優建議

### 如果訓練效果不好

#### 1. 獎勵相關

**問題**: 機器人不移動
```python
# 增加運動獎勵
standstill_penalty: weight = -1.0  # 從 -0.1 增加
```

**問題**: 機器人碰撞太多
```python
# 增加碰撞懲罰
collision_penalty: weight = -100.0  # 從 -50.0 增加
obstacle_proximity_penalty: weight = -10.0  # 從 -5.0 增加
```

**問題**: 很難到達目標
```python
# 增加引導獎勵
progress_to_goal: weight = 20.0  # 從 10.0 增加
```

#### 2. 演算法超參數

**問題**: 訓練不穩定
```python
learning_rate = 3e-4  # 從 1e-3 減少
clip_param = 0.1      # 從 0.2 減少（更保守）
```

**問題**: 收斂太慢
```python
num_steps_per_env = 48  # 從 24 增加
num_mini_batches = 8    # 從 4 增加
```

**問題**: 探索不足
```python
entropy_coef = 0.05     # 從 0.01 增加
init_noise_std = 1.5    # 從 1.0 增加
```

#### 3. 環境設置

**問題**: 任務太難
```python
# 簡化環境
goal_command.ranges.distance = (2.0, 5.0)  # 從 (5.0, 10.0) 縮短
episode_length_s = 40.0  # 從 30.0 增加
```

---

## 📚 文件依賴關係

```
train.py (訓練入口)
  ├─ local_planner_env_cfg.py (環境配置)
  │   ├─ ObservationsCfg ──> observations.py (觀測實現)
  │   ├─ ActionsCfg ──> actions.py (動作實現)
  │   ├─ RewardsCfg ──> rewards.py (獎勵實現) ⭐
  │   ├─ TerminationsCfg ──> terminations.py (終止實現)
  │   └─ CommandsCfg (目標命令)
  │
  └─ rsl_rl_ppo_cfg.py (演算法配置)
      ├─ 網路架構 (Actor-Critic)
      ├─ PPO 超參數
      └─ 訓練設置
```

---

## 💡 總結

### 最重要的3個文件

1. **`rewards.py`** ⭐⭐⭐
   - 訓練策略的核心
   - 定義什麼行為是好的
   - 直接影響最終策略

2. **`rsl_rl_ppo_cfg.py`** ⭐⭐
   - 演算法超參數
   - 影響學習速度和穩定性
   - 決定探索vs利用

3. **`local_planner_env_cfg.py`** ⭐⭐
   - 環境整體設定
   - 觀測、動作、獎勵權重
   - 任務難度設定

### 快速修改指南

- **想改學習速度** → `rsl_rl_ppo_cfg.py` 的 `learning_rate`
- **想改行為偏好** → `local_planner_env_cfg.py` 的獎勵權重
- **想改任務難度** → `local_planner_env_cfg.py` 的目標距離、時間限制
- **想改探索程度** → `rsl_rl_ppo_cfg.py` 的 `entropy_coef`, `init_noise_std`

---

**記住**: 強化學習是實驗科學，需要反覆調整和測試！🧪

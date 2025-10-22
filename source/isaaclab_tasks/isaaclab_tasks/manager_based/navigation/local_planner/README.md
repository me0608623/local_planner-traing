# Nova Carter 本地規劃器環境

## 📋 任務描述

這是一個為 Nova Carter 輪式機器人設計的動態避障導航任務環境。機器人需要在包含靜態和動態障礙物的場景中，使用 LiDAR 感測器導航到目標位置並避開障礙物。

## 🏗️ 檔案結構

```
local_planner/
├── __init__.py                      # 環境註冊
├── local_planner_env_cfg.py         # 環境配置（場景、觀測、動作、獎勵）
├── README.md                        # 本文件
├── agents/                          # RL 演算法配置
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py           # RSL-RL PPO 配置
└── mdp/                             # MDP 函數（觀測、動作、獎勵）
    ├── __init__.py
    ├── actions.py                   # 差速驅動動作定義
    ├── observations.py              # LiDAR 和機器人狀態觀測
    ├── rewards.py                   # 導航任務獎勵函數
    └── terminations.py              # 終止條件（到達目標、碰撞）
```

## 🎯 環境特性

### 場景元素
- **機器人**: Nova Carter（差速驅動輪式機器人）
- **感測器**: 2D LiDAR（360度，360個點）
- **障礙物**: 
  - 靜態障礙物（方塊）
  - 動態障礙物（移動的球體）
- **目標**: 隨機生成的目標位置（綠色球體標記）

### 觀察空間
- LiDAR 距離數據（360 維）
- 機器人線速度（3 維）
- 機器人角速度（3 維）
- 目標相對位置（2 維）
- 目標距離（1 維）

**總維度**: 369 維

### 動作空間
- 線速度 v ∈ [-1, 1]（對應 [-2.0, 2.0] m/s）
- 角速度 ω ∈ [-1, 1]（對應 [-π, π] rad/s）

**總維度**: 2 維

### 獎勵函數
- ✅ **接近目標獎勵** (+10.0 × 進度)
- ✅ **到達目標獎勵** (+100.0)
- ❌ **障礙物接近懲罰** (-5.0 × 接近程度)
- ❌ **碰撞懲罰** (-50.0)
- ❌ **角速度過大懲罰** (-0.01 × ω²)
- ❌ **靜止不動懲罰** (-0.1)

### 終止條件
- ⏱️ 超時（30 秒）
- 🎯 到達目標（距離 < 0.5m）
- 💥 碰撞（LiDAR 最近距離 < 0.3m）

## 🚀 使用方法

### 1️⃣ 測試環境是否正確註冊

```bash
# 圖形化介面測試（1 個環境）
./isaaclab.sh -p scripts/test_local_planner_env.py

# 無頭模式測試（4 個環境）
./isaaclab.sh -p scripts/test_local_planner_env.py --num_envs 4 --headless

# 測試簡化版本
./isaaclab.sh -p scripts/test_local_planner_env.py --task Isaac-Navigation-LocalPlanner-Carter-Simple-v0
```

### 2️⃣ 使用 RSL-RL 訓練

```bash
# 開始訓練（建議使用較多環境數以加速訓練）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 128 \
    --headless

# 使用簡化版本快速測試
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-Simple-v0 \
    --num_envs 8 \
    --headless
```

### 3️⃣ 評估訓練好的策略

```bash
# 播放訓練好的策略
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 16 \
    --checkpoint /path/to/model.pt
```

### 4️⃣ 使用其他 RL 框架

目前配置僅包含 RSL-RL。如果您想使用其他框架（如 Stable-Baselines3），需要：

1. 在 `agents/` 目錄下建立對應的配置檔案
2. 在 `__init__.py` 中註冊時添加對應的 `*_cfg_entry_point`

例如，添加 SB3 支援：

```python
# agents/sb3_ppo_cfg.yaml
policy:
  class_name: ActorCriticPolicy
  net_arch: [256, 256, 128]
  activation_fn: elu

algorithm:
  class_name: PPO
  learning_rate: 0.001
  n_steps: 2048
  batch_size: 64
```

## ⚙️ 配置調整

### 調整環境數量

編輯 `local_planner_env_cfg.py` 中的 `LocalPlannerSceneCfg`:

```python
scene: LocalPlannerSceneCfg = LocalPlannerSceneCfg(
    num_envs=1024,  # 改為您想要的環境數量
    env_spacing=15.0  # 環境之間的間距（米）
)
```

### 調整任務難度

**更容易的任務**:
```python
# 增加安全距離閾值
obstacle_proximity_penalty.params.safe_distance = 2.0

# 增加目標到達閾值
reached_goal.params.threshold = 1.0

# 減少回合時間
self.episode_length_s = 20.0
```

**更困難的任務**:
```python
# 增加更多動態障礙物
# 減小安全距離閾值
obstacle_proximity_penalty.params.safe_distance = 0.5

# 縮小目標到達閾值
reached_goal.params.threshold = 0.3

# 增加回合時間
self.episode_length_s = 60.0
```

### 調整機器人參數

Nova Carter 的輪子參數在 `ActionsCfg` 中定義：

```python
base_velocity = mdp.DifferentialDriveActionCfg(
    asset_name="robot",
    left_wheel_joint_names=["wheel_left"],   # Nova Carter 左主驅動輪
    right_wheel_joint_names=["wheel_right"], # Nova Carter 右主驅動輪
    wheel_radius=0.125,  # 輪半徑 (m) - 根據實際規格調整
    wheel_base=0.413,    # 輪距 (m) - 根據實際規格調整
    max_linear_speed=2.0,  # 最大線速度
    max_angular_speed=math.pi,  # 最大角速度
)
```

**📝 Nova Carter 關節結構**：
- **主驅動輪**：`wheel_left`, `wheel_right`（用於差速驅動控制）
- **輔助滑輪**：`caster_*`（支撐用，無需主動控制）
  - `caster_frame_base`
  - `caster_swivel_left/right`（滑輪轉向關節）
  - `caster_wheel_left/right`（滑輪旋轉關節）

## 🐛 故障排除

### 問題 1: 環境註冊失敗

**錯誤**: `gymnasium.error.NameNotFound: Environment "Isaac-Navigation-LocalPlanner-Carter-v0" doesn't exist`

**解決方法**:
```python
# 確保在腳本開頭導入
import isaaclab_tasks  # 這會載入所有環境註冊
```

### 問題 2: LiDAR 無數據

**可能原因**: LiDAR 配置的 `mesh_prim_paths` 未正確設定

**解決方法**: 確保 `mesh_prim_paths` 包含場景中的所有障礙物路徑

### 問題 3: 機器人不移動

**可能原因**: 關節名稱錯誤

**解決方法**: 
1. 檢查 Nova Carter USD 檔案中的實際關節名稱
2. 更新 `ActionsCfg` 中的 `left_wheel_joint_names` 和 `right_wheel_joint_names`

### 問題 4: 訓練不收斂

**可能原因**: 
- 獎勵函數權重不平衡
- 觀察空間不足

**解決方法**:
1. 調整 `RewardsCfg` 中的權重
2. 檢查觀察數據是否正規化
3. 嘗試更小的學習率

## 📝 後續改進建議

### 功能擴展
- [ ] 添加多個目標點的導航任務
- [ ] 實現路徑跟蹤任務
- [ ] 添加人群導航場景
- [ ] 支援3D LiDAR
- [ ] 添加相機觀測（視覺導航）

### 優化
- [ ] 實現觀察歷史（時間序列觀測）
- [ ] 添加課程學習（從簡單到困難）
- [ ] 優化獎勵函數權重
- [ ] 實現更好的障礙物隨機化

### 多智能體
- [ ] 多機器人協作導航
- [ ] 多機器人避碰

## 📚 參考資源

- [Isaac Lab 官方文件](https://isaac-sim.github.io/IsaacLab/)
- [Nova Carter 文件](https://docs.nvidia.com/isaac/packages/isaac_ros/index.html)
- [RSL-RL 文件](https://github.com/leggedrobotics/rsl_rl)
- [Gymnasium 文件](https://gymnasium.farama.org/)

## 👥 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

BSD-3-Clause License（與 Isaac Lab 主專案相同）


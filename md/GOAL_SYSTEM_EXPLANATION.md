# 訓練目標系統詳細說明

## 🎯 訓練目標是什麼？

### 任務目標

**機器人的訓練目標**：從起點導航到隨機生成的目標位置，同時避開障礙物。

```
任務: 點到點導航（Point-to-Point Navigation）
起點: 隨機位置 x∈(-2,2)m, y∈(-2,2)m
終點: 隨機目標 x∈(3,10)m, y∈(-5,5)m
成功: 到達目標0.5米以內
失敗: 碰撞障礙物或超時30秒
```

---

## 🎯 目標在哪裡？如何顯示？

### 目標生成機制

**代碼位置**：`local_planner_env_cfg.py` 第 198-215 行

```python
@configclass
class CommandsCfg:
    """指令配置 - 目標位置生成"""
    
    goal_command = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="chassis_link",
        resampling_time_range=(10.0, 10.0),  # 每10秒重新生成
        debug_vis=True,  # ⭐ 啟用調試可視化
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(3.0, 10.0),   # 目標 X: 3-10米
            pos_y=(-5.0, 5.0),   # 目標 Y: -5到+5米
            pos_z=(0.0, 0.0),    # 目標 Z: 地面高度
        ),
    )
```

### 目標可視化標記

**代碼位置**：`local_planner_env_cfg.py` 第 118-127 行

```python
# 綠色球體標記目標位置
goal_marker = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/GoalMarker",
    spawn=sim_utils.SphereCfg(
        radius=0.3,  # 半徑 30cm
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0)  # 🟢 綠色
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(8.0, 0.0, 0.3)),
)
```

---

## 🔍 為什麼您看不到目標？

### 可能原因

#### 1. **使用 Headless 模式** ⚠️ **最可能**

```bash
# 如果您使用了 --headless
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 3 \
    --headless  # ← 無 GUI，看不到任何視覺化
```

**解決**: 移除 `--headless` 參數

```bash
# 啟用 GUI 模式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 2  # GUI 模式建議少用環境
```

#### 2. **debug_vis 可視化系統**

`debug_vis=True` 會顯示：
- 🔵 **藍色箭頭**：指向目標的方向
- 🟢 **綠色標記**：可能需要額外配置

但這依賴於：
- Isaac Sim 的可視化系統
- 必須在 **GUI 模式**下才能看到

#### 3. **目標標記可能在錯誤的高度**

```python
# 第 126 行
init_state=RigidObjectCfg.InitialStateCfg(pos=(8.0, 0.0, 0.3))
#                                                      Z軸 ↑
```

如果 Z=0.3 太低，可能被地面遮擋。

---

## 🎮 如何看到目標？

### 方法 1: 使用 GUI 模式（推薦）

```bash
cd /home/aa/IsaacLab

# 啟動 GUI 模式訓練
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 2 \
    --max_iterations 10  # 先測試10次
    # 注意：不使用 --headless
```

**您應該看到**:
- 🤖 兩個 Nova Carter 機器人
- 🟢 綠色球體（目標標記）
- 🔵 藍色箭頭（debug_vis 顯示）
- 🚧 障礙物（方塊和球體）

### 方法 2: 提高目標標記高度

修改第 126 行：

```python
# 原始
init_state=RigidObjectCfg.InitialStateCfg(pos=(8.0, 0.0, 0.3))

# 修改為更高
init_state=RigidObjectCfg.InitialStateCfg(pos=(8.0, 0.0, 1.0))
#                                           更明顯 ↑
```

### 方法 3: 使用 Play 腳本查看訓練好的策略

```bash
# 訓練後可視化
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/*/*/model_*.pt
```

Play 模式會清楚地顯示目標和機器人行為。

---

## 📊 目標系統如何工作？

### 完整流程

```
1. Episode 開始
   ├─ 隨機生成目標位置
   │  └─ X: 3-10米
   │  └─ Y: -5到+5米
   │  └─ Z: 0米（地面）
   │
   ├─ 更新目標標記（綠色球體）
   │  └─ 移動到新目標位置
   │
   └─ 機器人重置到隨機起點

2. 每個時間步
   ├─ 計算機器人到目標的距離
   ├─ 計算相對位置（觀測）
   ├─ 如果接近目標 → 給獎勵
   └─ 如果到達目標（<0.5m）→ 大獎勵 + 結束

3. 目標重新採樣
   ├─ 條件1: 到達目標
   ├─ 條件2: 超時
   └─ 條件3: 每10秒（resampling_time_range）
```

### 目標位置範圍

```
俯視圖（目標生成範圍）:

    Y (+5m)
      ↑
      │     ┌─────────────────┐
      │     │   目標範圍      │
      │     │  X: 3-10m       │
    0 ├─────┤  Y: -5 to +5m   ├──→ X
      │  🤖 │                 │
      │起點 └─────────────────┘
      │
    (-5m)
          0m   3m            10m

🤖 機器人起點: X∈(-2,2), Y∈(-2,2)
🎯 目標範圍:   X∈(3,10), Y∈(-5,5)
✅ 最近距離: ~1-2米
✅ 最遠距離: ~12-14米
```

---

## 🔧 目標相關配置位置

### 查看目標設定

```bash
# 打開配置文件
vim source/isaaclab_tasks/.../local_planner_env_cfg.py

# 目標命令配置（第 198-215 行）
:198

# 目標標記（第 118-127 行）
:118
```

### 關鍵參數

| 行號 | 參數 | 值 | 說明 |
|------|------|----|----|
| 206 | `debug_vis` | True | 啟用目標可視化（藍色箭頭） |
| 208 | `pos_x` | (3.0, 10.0) | 目標X範圍（3-10米） |
| 209 | `pos_y` | (-5.0, 5.0) | 目標Y範圍（-5到+5米） |
| 205 | `resampling_time_range` | (10.0, 10.0) | 每10秒重新生成 |
| 121 | `radius` | 0.3 | 目標標記半徑（30cm） |
| 124 | `diffuse_color` | (0.0, 1.0, 0.0) | 綠色 |
| 126 | 初始位置 | (8.0, 0.0, 0.3) | 初始在8米處 |

---

## 🧪 診斷目標系統

### 測試腳本

創建一個簡單的測試來確認目標系統工作：

```python
# test_goal_visualization.py
#!/usr/bin/env python3
"""
測試目標系統可視化

使用方法:
    ./isaaclab.sh -p test_goal_visualization.py
"""

import gymnasium as gym

# 創建環境（GUI模式）
env = gym.make("Isaac-Navigation-LocalPlanner-Carter-v0", num_envs=1)

# 重置環境
obs, info = env.reset()

print("=" * 80)
print("🎯 目標系統測試")
print("=" * 80)

# 獲取目標命令
if hasattr(env.unwrapped, 'command_manager'):
    cmd_manager = env.unwrapped.command_manager
    goal_cmd = cmd_manager.get_command("goal_command")
    
    print(f"\n目標位置 (世界座標):")
    print(f"  X: {goal_cmd[0, 0]:.2f}m")
    print(f"  Y: {goal_cmd[0, 1]:.2f}m")
    print(f"  Z: {goal_cmd[0, 2]:.2f}m")
    
    # 獲取機器人位置
    robot = env.unwrapped.scene["robot"]
    robot_pos = robot.data.root_pos_w[0]
    
    print(f"\n機器人位置:")
    print(f"  X: {robot_pos[0]:.2f}m")
    print(f"  Y: {robot_pos[1]:.2f}m")
    print(f"  Z: {robot_pos[2]:.2f}m")
    
    # 計算距離
    import torch
    distance = torch.norm(goal_cmd[0, :2] - robot_pos[:2])
    print(f"\n到目標距離: {distance:.2f}m")
    
    print("\n請在 Isaac Sim 視窗中查看:")
    print("  🟢 綠色球體 = 目標位置")
    print("  🔵 藍色箭頭 = 目標方向（debug_vis）")
    print("  🤖 機器人")

# 運行幾步
print("\n運行10步測試...")
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 5 == 0:
        print(f"步驟 {i}: reward={reward[0]:.2f}")

env.close()
print("\n✅ 測試完成")
```

---

## 🖼️ 目標可視化說明

### 在 GUI 模式下您會看到

```
場景視圖:
═══════════════════════════════════════════════════

    🎯 ←─────────────── 綠色球體（目標標記）
    │                   半徑 30cm，高度 30cm
    │
    │
    │         🔵 ←────── 藍色箭頭（debug_vis）
    │        ╱          從機器人指向目標
    │       ╱
    │      ╱
    🤖 ←────────────── Nova Carter 機器人
                       應該朝目標移動

    🚧 ←────────────── 障礙物（方塊/球體）
```

### debug_vis 的效果

當 `debug_vis=True` 時（第 206 行），系統會：
- 顯示一個**藍色箭頭**從機器人指向目標
- 箭頭長度代表距離
- 箭頭方向指示目標方向

**注意**: 這只在 **GUI 模式**下可見！

---

## ⚠️ 為什麼您可能看不到目標？

### 原因 1: Headless 模式 ⭐ **最可能**

```bash
# Headless 模式（無 GUI）
./isaaclab.sh -p ... --headless
└─ 沒有視覺化窗口
└─ 看不到任何場景
└─ 只有終端輸出

# GUI 模式
./isaaclab.sh -p ...
└─ 有 Isaac Sim 視窗
└─ 可以看到機器人、目標、障礙物
```

### 原因 2: 目標標記太小或顏色不明顯

```python
# 當前配置（第 121-124 行）
radius=0.3,  # 可能太小
diffuse_color=(0.0, 1.0, 0.0),  # 綠色，可能不夠明顯
```

**解決**: 增大標記或改變顏色：

```python
radius=0.5,  # 增大到 50cm
diffuse_color=(1.0, 1.0, 0.0),  # 改為黃色（更明顯）
```

### 原因 3: 目標在視野之外

如果相機視角不對，可能看不到目標。

**解決**: 在 Isaac Sim 中調整相機視角：
- 右鍵拖曳：旋轉視角
- 中鍵拖曳：平移視角
- 滾輪：縮放

---

## 🔧 讓目標更明顯的修改

### 修改 1: 增大目標標記

```python
# 第 118-127 行
goal_marker = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/GoalMarker",
    spawn=sim_utils.SphereCfg(
        radius=0.8,  # 從 0.3 增加到 0.8（更明顯）
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 1.0, 0.0),  # 改為黃色（更明顯）
            emissive_color=(1.0, 1.0, 0.0),  # 添加發光效果
            emissive_intensity=1.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(8.0, 0.0, 1.5)),  # 提高高度
)
```

### 修改 2: 添加目標更新事件

檢查是否有事件將目標標記更新到命令位置。如果沒有，需要添加：

```python
# 在 EventCfg 中添加（目前可能缺少）
update_goal_marker = EventTerm(
    func=mdp.update_goal_marker_position,
    mode="interval",
    interval_range_s=(0.1, 0.1),  # 每0.1秒更新一次
    params={
        "marker_cfg": SceneEntityCfg("goal_marker"),
        "command_name": "goal_command",
    },
)
```

---

## 📍 目標系統在訓練中的作用

### 觀測空間

機器人通過以下方式"知道"目標在哪裡：

```python
# 第 178-182 行
goal_position = ObsTerm(
    func=mdp.goal_position_in_robot_frame,  # 目標相對位置
    params={"command_name": "goal_command"},
)
# 輸出: [相對距離X, 相對距離Y]  (2維)
```

**機器人看到的目標信息**:
- 不是絕對位置（世界座標）
- 而是**相對位置**（機器人座標系）
- 例如：`[5.2, -2.3]` = 目標在機器人前方5.2米、左邊2.3米

### 獎勵計算

```python
# 接近目標獎勵（mdp/rewards.py）
def progress_to_goal_reward():
    distance = norm(goal_pos - robot_pos)  # 計算距離
    reward = -distance  # 距離越小，獎勵越大
    return reward

# 到達目標獎勵
def reached_goal_reward():
    distance = norm(goal_pos - robot_pos)
    if distance < 0.5m:  # 到達閾值
        return 100.0  # 大獎勵
    else:
        return 0.0
```

### 終止條件

```python
# 到達目標時終止（mdp/terminations.py）
def goal_reached():
    distance = norm(goal_pos - robot_pos)
    if distance < 0.5m:
        return True  # Episode 結束，成功！
```

---

## 🎯 目標系統配置總結

| 配置項 | 位置 | 當前值 | 說明 |
|-------|------|--------|------|
| **目標範圍X** | 第208行 | (3.0, 10.0) | 3-10米 |
| **目標範圍Y** | 第209行 | (-5.0, 5.0) | -5到+5米 |
| **到達閾值** | 第233行 | 0.5 | 0.5米內算到達 |
| **重採樣時間** | 第205行 | 10.0 | 每10秒新目標 |
| **debug_vis** | 第206行 | True | 藍色箭頭可視化 |
| **標記半徑** | 第121行 | 0.3 | 綠球半徑30cm |
| **標記顏色** | 第124行 | (0,1,0) | 綠色 |

---

## 💡 快速測試目標可視化

### 創建測試命令

```bash
# 1. 啟動GUI模式（短時間測試）
cd /home/aa/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1 \
    --max_iterations 5

# 2. 在 Isaac Sim 視窗中觀察
#    - 🟢 綠色球體 = 目標
#    - 🔵 藍色箭頭 = 目標方向
#    - 🤖 機器人應該朝目標移動
```

### 如果還是看不到

```bash
# 檢查環境是否正確加載
./isaaclab.sh -p -c "
import gymnasium as gym
env = gym.make('Isaac-Navigation-LocalPlanner-Carter-v0', num_envs=1)
print('場景組件:', list(env.unwrapped.scene.keys()))
env.close()
"
```

應該看到：
```
場景組件: ['terrain', 'robot', 'lidar', 'static_obstacles', 
           'dynamic_obstacles', 'goal_marker', 'dome_light']
```

---

## 📚 相關代碼

### 目標生成（命令管理器）

目標由 Isaac Lab 的命令管理器自動生成和管理：

```python
# Isaac Lab 內部會：
1. 根據 ranges 隨機生成目標位置
2. 每個環境獨立生成（所以3個環境有3個不同目標）
3. 到達目標或超時後重新生成
4. 通過 env.command_manager 訪問
```

### 目標觀測

```python
# mdp/observations.py
def goal_position_in_robot_frame(env, command_name):
    # 獲取目標世界位置
    goal_pos_w = env.command_manager.get_command(command_name)[:, :3]
    
    # 轉換到機器人座標系
    # ...
    
    return [相對X, 相對Y]  # 機器人用這個來導航
```

---

## 🎯 總結

### 訓練目標

**任務**: 從隨機起點導航到隨機目標點（3-10米遠），避開障礙物

**目標位置**: 每個 episode 隨機生成
- X: 3-10米
- Y: -5到+5米  
- 每10秒或到達後重新生成

**目標可視化**:
- 🟢 綠色球體（半徑30cm）
- 🔵 藍色箭頭（debug_vis）
- **僅在 GUI 模式可見**

**成功標準**: 到達目標 0.5 米以內

### 如何看到目標

```bash
# 使用 GUI 模式（移除 --headless）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1
```

**在 Isaac Sim 視窗中您會看到綠色球體標記目標位置！** 🎯

---

**配置位置**: `local_planner_env_cfg.py` 第 198-215 行（目標命令）和第 118-127 行（目標標記）

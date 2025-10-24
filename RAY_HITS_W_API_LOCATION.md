# 📍 ray_hits_w API 代碼位置說明

## 🎯 您問的這一行代碼在這裡

**文件位置**：
```
source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/local_planner/mdp/observations.py
```

**具體行號**：**第 44 行** ⭐

---

## 📝 完整代碼上下文

這段代碼在 `lidar_obs()` 函數中，用於處理 LiDAR 感測器的數據讀取：

```python:35:44:source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/local_planner/mdp/observations.py
# 嘗試多版本屬性存取（從新到舊）
if hasattr(data, "ray_hits_w") and hasattr(data, "pos_w"):
    # Isaac Sim 5.0+ / Isaac Lab 2025+：需手動計算距離
    # ray_hits_w: (num_envs, num_rays, 3) - 世界座標中的射線命中點
    # pos_w: (num_envs, 3) - 世界座標中的感測器位置
    hit_points = data.ray_hits_w  # (num_envs, num_rays, 3)
    sensor_pos = data.pos_w.unsqueeze(1)  # (num_envs, 1, 3) 擴展以廣播
    
    # 計算每條射線的距離
    distances = torch.norm(hit_points - sensor_pos, dim=-1)  # (num_envs, num_rays)
    print("✅ 使用 ray_hits_w API (2025+ / Sim 5.0) - 手動計算距離")
```

---

## 🔍 這段代碼的作用

### 功能
這段代碼實現了 **Isaac Sim 5.0 的 RayCaster API 兼容性**，因為新版 API 不再直接提供距離數據，需要手動計算。

### 為什麼需要這樣做？

**Isaac Sim 5.0 API 變更**：
- **舊 API** (Sim 4.x)：直接提供 `distances` 或 `ray_distances` 屬性
- **新 API** (Sim 5.0+)：只提供 `ray_hits_w`（命中點世界座標）和 `pos_w`（感測器位置）

### 計算邏輯

```
距離 = ||命中點位置 - 感測器位置||

具體步驟：
1. 獲取射線命中點：hit_points = data.ray_hits_w
   - 形狀：(num_envs, num_rays, 3)
   - 每條射線在世界座標系中的命中點 [x, y, z]

2. 獲取感測器位置：sensor_pos = data.pos_w.unsqueeze(1)
   - 形狀：(num_envs, 1, 3) → 廣播到 (num_envs, num_rays, 3)
   - 感測器在世界座標系中的位置 [x, y, z]

3. 計算歐幾里得距離：
   distances = torch.norm(hit_points - sensor_pos, dim=-1)
   - 結果形狀：(num_envs, num_rays)
   - 每條射線的距離值
```

---

## 🗂️ 完整函數結構

**函數名稱**：`lidar_obs()`

**位置**：第 21-66 行

**功能**：讀取 LiDAR 感測器數據並返回標準化的距離觀測

```python
def lidar_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """LiDAR 距離觀測（兼容 Isaac Lab 2023-2025+）
    
    Returns:
        LiDAR 點的距離數據，shape (num_envs, num_rays)
    """
    # 1. 獲取感測器
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    data = sensor.data
    
    # 2. 多版本 API 兼容（從新到舊嘗試）
    if hasattr(data, "ray_hits_w") and hasattr(data, "pos_w"):
        # ✅ Isaac Sim 5.0+ - 手動計算距離（第35-44行）
        ...
    elif hasattr(data, "hit_distances"):
        # Isaac Lab 2025.1
        ...
    elif hasattr(data, "distances"):
        # Isaac Lab 2024.1
        ...
    elif hasattr(data, "ray_distances"):
        # Isaac Lab ≤ 2023.1
        ...
    
    # 3. 標準化距離到 [0, 1]
    distances = distances / max_distance
    distances = torch.clamp(distances, 0.0, 1.0)
    
    return distances.squeeze(-1)
```

---

## 📊 API 版本兼容表

| Isaac Sim 版本 | API 屬性 | 是否需要手動計算距離 | 代碼行號 |
|---------------|---------|---------------------|---------|
| **5.0+** (2025+) | `ray_hits_w` + `pos_w` | ✅ 是（第42-43行） | 35-44 |
| **4.5** (2025.1) | `hit_distances` | ❌ 否 | 46-48 |
| **4.x** (2024.1) | `distances` | ❌ 否 | 49-51 |
| **3.x** (≤2023.1) | `ray_distances` | ❌ 否 | 52-54 |

---

## 🎯 在訓練中的使用

### 調用位置

這個觀測函數在環境配置中被註冊：

**文件**：`local_planner_env_cfg.py`  
**行號**：第 163-194 行

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # LiDAR 觀測 ⭐
        lidar_obs = ObsTerm(
            func=mdp.lidar_obs,          # ← 調用 observations.py 中的 lidar_obs()
            params={"sensor_cfg": SceneEntityCfg("lidar")},
        )
        
        # 機器人狀態
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        
        # 目標資訊
        goal_position_in_robot_frame = ObsTerm(
            func=mdp.goal_position_in_robot_frame,
            params={"command_name": "goal_command"},
        )
        distance_to_goal = ObsTerm(
            func=mdp.distance_to_goal,
            params={"command_name": "goal_command"},
        )
        
    policy: PolicyCfg = PolicyCfg()
```

### 訓練時的調用流程

```
訓練循環
  ↓
環境 step()
  ↓
ObservationManager.compute()
  ↓
調用所有 ObsTerm 函數
  ↓
mdp.lidar_obs() ← 執行 ray_hits_w 計算
  ↓
返回 LiDAR 距離數據 [num_envs, 360]
  ↓
組合成完整觀測 [num_envs, 369]
  ↓
傳給 Actor Network
```

---

## 🛠️ 如何查看和修改

### 查看代碼

```bash
# 方法1：使用 vim（跳到第44行）
vim +44 source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
local_planner/mdp/observations.py

# 方法2：使用 cat（查看21-66行）
cat source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
local_planner/mdp/observations.py | sed -n '21,66p'

# 方法3：使用 less（交互式查看）
less +44 source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
local_planner/mdp/observations.py
```

### 修改建議

如果您需要修改 LiDAR 數據處理邏輯，可以：

1. **修改距離計算方式**（第42-43行）
   ```python
   # 例如：使用曼哈頓距離而不是歐幾里得距離
   distances = torch.sum(torch.abs(hit_points - sensor_pos), dim=-1)
   ```

2. **添加距離過濾**（第60-63行後）
   ```python
   # 例如：將過遠的測量設為最大值
   distances[distances > 0.8] = 1.0
   ```

3. **修改標準化範圍**（第61-63行）
   ```python
   # 例如：使用對數標準化
   distances = torch.log(distances + 1.0) / torch.log(torch.tensor(max_distance + 1.0))
   ```

---

## 📚 相關文檔

### API 變更說明
- [md/RAYCASTER_API_FIX.md](md/RAYCASTER_API_FIX.md) - RayCaster API 修復記錄
- [md/HIT_DISTANCES_API_FIX.md](md/HIT_DISTANCES_API_FIX.md) - hit_distances 修復
- [md/RAY_HITS_MANUAL_CALCULATION.md](md/RAY_HITS_MANUAL_CALCULATION.md) - 手動計算方法

### 架構文檔
- [md/CODE_ARCHITECTURE_GUIDE.md](md/CODE_ARCHITECTURE_GUIDE.md) - 完整代碼架構
- [md/SIMULATION_SCENE_DESIGN.md](md/SIMULATION_SCENE_DESIGN.md) - LiDAR 配置

---

## 🔍 調試方法

### 查看實際使用的 API

運行訓練時，終端會顯示：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 4
```

**輸出示例**：
```
RayCasterData fields: ['ray_hits_w', 'pos_w', 'normals', ...]
✅ 使用 ray_hits_w API (2025+ / Sim 5.0) - 手動計算距離
```

這說明正在使用 Isaac Sim 5.0 的新 API。

### 驗證距離計算

在 `observations.py` 第44行後添加調試代碼：

```python
distances = torch.norm(hit_points - sensor_pos, dim=-1)
print("✅ 使用 ray_hits_w API (2025+ / Sim 5.0) - 手動計算距離")

# 調試：檢查距離範圍
print(f"  - 最小距離: {distances.min().item():.2f}m")
print(f"  - 最大距離: {distances.max().item():.2f}m")
print(f"  - 平均距離: {distances.mean().item():.2f}m")
```

---

## 🎯 總結

| 項目 | 信息 |
|-----|------|
| **文件路徑** | `source/isaaclab_tasks/.../mdp/observations.py` |
| **函數名稱** | `lidar_obs()` |
| **具體行號** | **第 44 行** ⭐ |
| **功能** | 手動計算 LiDAR 射線距離（Isaac Sim 5.0） |
| **計算公式** | `距離 = ||命中點 - 感測器位置||` |
| **輸入** | `ray_hits_w` [num_envs, 360, 3], `pos_w` [num_envs, 3] |
| **輸出** | `distances` [num_envs, 360] |
| **調用位置** | `local_planner_env_cfg.py` 第 166-169 行 |

---

**現在您知道這行代碼的具體位置和作用了！** 🎯

如果需要修改 LiDAR 數據處理邏輯，就編輯這個文件的第 35-44 行部分。


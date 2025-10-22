# 並行訓練環境與碰撞配置說明

## 🎮 您的觀察

### 觀察 1: 三台 Nova Carter 同時移動 ✅

**這是正常的！** 這是 Isaac Lab 的**並行訓練**機制。

### 觀察 2: 不同場景的牆壁似乎會被穿透 ⚠️

**這是一個需要注意的碰撞配置問題！**

---

## 🔍 並行訓練機制

### 什麼是並行訓練？

當您使用 `--num_envs 3` 啟動訓練時：

```
訓練場景 = 3個獨立環境並行運行

/World/envs/
├─ env_0/            # 環境 0
│   ├─ Robot/        # 機器人 1
│   ├─ StaticObstacles/
│   ├─ DynamicObstacles/
│   └─ GoalMarker/
│
├─ env_1/            # 環境 1  
│   ├─ Robot/        # 機器人 2
│   ├─ StaticObstacles/
│   ├─ DynamicObstacles/
│   └─ GoalMarker/
│
└─ env_2/            # 環境 2
    ├─ Robot/        # 機器人 3
    ├─ StaticObstacles/
    ├─ DynamicObstacles/
    └─ GoalMarker/
```

### 為什麼要並行訓練？

#### 優點：
1. ✅ **訓練速度快**: 同時收集3倍的經驗
2. ✅ **GPU利用率高**: 充分利用GPU並行計算能力
3. ✅ **多樣性**: 不同環境提供不同場景

#### 配置位置：

```bash
# 查看場景定義（第 38 行開始）
vim source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
    local_planner/local_planner_env_cfg.py +38
```

**關鍵代碼**（第 332 行）：
```python
@configclass
class LocalPlannerEnvCfg(ManagerBasedRLEnvCfg):
    # 場景設定
    scene: LocalPlannerSceneCfg = LocalPlannerSceneCfg(
        num_envs=1024,      # 默認1024個環境（會被覆蓋）
        env_spacing=15.0    # 環境間距15米
    )
```

**實際環境數量**（第 378 行）：
```python
def __post_init__(self):
    # ...
    self.scene.num_envs = 8  # 默認改為8個環境
```

**命令行覆蓋**：
```bash
# 您使用的命令可能是：
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 3  # ← 這裡指定3個環境
```

---

## ⚠️ 碰撞穿透問題

### 問題原因分析

您觀察到"不同場景的牆壁會被穿透"，這是因為**碰撞組（Collision Group）配置不當**。

#### 當前配置問題

**查看障礙物配置**（第 94-115 行）：

```python
# 靜態障礙物
static_obstacles = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/StaticObstacles",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(2.0, 2.0, 2.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 1.0)),
    # ⚠️ 注意：這裡沒有指定 collision_group！
)

# 動態障礙物
dynamic_obstacles = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/DynamicObstacles",
    spawn=sim_utils.SphereCfg(
        radius=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        # ⚠️ 注意：CollisionPropertiesCfg() 使用默認值
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 3.0, 1.0)),
)
```

### 碰撞組工作原理

Isaac Sim 使用**碰撞組（Collision Group）**來控制哪些物體可以碰撞：

```
collision_group = -1  # 與所有物體碰撞（地面通常用這個）
collision_group = 0   # 環境 0 專用
collision_group = 1   # 環境 1 專用
collision_group = 2   # 環境 2 專用
...
```

**默認行為**（沒指定collision_group時）:
- 不同環境的物體**可能會互相穿透**
- 因為 PhysX 不知道它們應該隔離

---

## 🔧 如何修復碰撞穿透問題

### 方案 1: 使用 {ENV_REGEX_NS} 路徑隔離（當前使用）

**優點**: 自動隔離，無需手動配置
**原理**: `{ENV_REGEX_NS}` 會展開為 `/World/envs/env_0`, `/World/envs/env_1` 等

```python
# 當前配置（已經正確）
static_obstacles = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/StaticObstacles",  # ✅ 每個環境獨立
    # ...
)
```

**這樣會創建**:
```
/World/envs/env_0/StaticObstacles  # 環境0的障礙物
/World/envs/env_1/StaticObstacles  # 環境1的障礙物
/World/envs/env_2/StaticObstacles  # 環境2的障礙物
```

### 方案 2: 明確設置碰撞組（更嚴格）

如果穿透問題仍然存在，可以明確設置碰撞組：

```python
# 在 spawn 配置中添加
static_obstacles = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/StaticObstacles",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(2.0, 2.0, 2.0),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_group=-1,  # 與所有物體碰撞
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 1.0)),
)
```

### 方案 3: 增加環境間距

```python
# 在第 332 行
scene: LocalPlannerSceneCfg = LocalPlannerSceneCfg(
    num_envs=3,
    env_spacing=20.0  # 從 15.0 增加到 20.0，避免重疊
)
```

---

## 📍 查看場景定義的位置

### 主要場景配置文件

```bash
# 🎯 核心場景定義（第 37-135 行）
vim source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
    local_planner/local_planner_env_cfg.py +37
```

### 關鍵代碼段

| 行號 | 內容 | 說明 |
|------|------|------|
| **38-55** | `LocalPlannerSceneCfg` 類定義 | 場景配置類聲明 |
| **42-55** | 地形配置 | 40×40m 平坦地面 |
| **58-77** | Nova Carter 配置 | **機器人USD引用** ⭐ |
| **80-92** | LiDAR 配置 | RayCaster 感測器 |
| **95-102** | 靜態障礙物 | **方塊USD引用** ⭐ |
| **105-115** | 動態障礙物 | 程序化球體 |
| **118-127** | 目標標記 | 綠色球體 |
| **130-133** | 光照 | 圓頂光源 |
| **332** | 環境數量 | `num_envs` 設定 |
| **378** | 默認環境數 | 設為 8 個 |

---

## 🔍 詳細檢查步驟

### 步驟 1: 查看完整場景配置

```bash
# 打開配置文件
cd /home/aa/IsaacLab
vim source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
    local_planner/local_planner_env_cfg.py

# 跳轉到場景定義
:37  # 按 :37 然後 Enter，跳到第37行
```

### 步驟 2: 查看特定組件

```python
# 機器人 USD（第 58-77 行）
robot = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/aa/isaacsim/usd/nova_carter.usd",  # ← USD 路徑
        activate_contact_sensors=False,
    ),
    # ... 初始狀態和執行器配置
)

# 靜態障礙物 USD（第 95-102 行）
static_obstacles = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/StaticObstacles",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  # ← USD 路徑
        scale=(2.0, 2.0, 2.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 1.0)),
)
```

### 步驟 3: 查看環境間距設置

```bash
# 搜索 env_spacing
grep -n "env_spacing" source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
    local_planner/local_planner_env_cfg.py
```

**輸出**:
```
332:    scene: LocalPlannerSceneCfg = LocalPlannerSceneCfg(num_envs=1024, env_spacing=15.0)
```

**15米間距**意味著：
- 環境 0 中心在 (0, 0)
- 環境 1 中心在 (15, 0)
- 環境 2 中心在 (30, 0)
- 依此類推...

---

## 🛠️ 修復碰撞穿透的方法

### 如果您看到環境間障礙物穿透

#### 方法 1: 增加環境間距（推薦）

```python
# 第 332 行
scene: LocalPlannerSceneCfg = LocalPlannerSceneCfg(
    num_envs=3,
    env_spacing=25.0  # 從 15.0 增加到 25.0
)
```

#### 方法 2: 減少環境數量

```bash
# 啟動時指定較少環境
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 2  # 減少到2個
    --headless
```

#### 方法 3: 添加明確的碰撞組（高級）

修改第 95-102 行的障礙物配置：

```python
static_obstacles = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/StaticObstacles",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(2.0, 2.0, 2.0),
        # 🔧 添加碰撞配置
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,  # 啟用碰撞
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 1.0)),
)
```

---

## 📊 環境布局示意圖

### 3 個並行環境的空間布局

```
俯視圖（env_spacing=15m）：

   Y
   ↑
   │
   │  env_0          env_1          env_2
   │    │              │              │
   │    ↓              ↓              ↓
   │  [🤖🎯🚧]      [🤖🎯🚧]      [🤖🎯🚧]
   │    │              │              │
   └────┼──────────────┼──────────────┼──────→ X
        0             15m            30m

圖例:
🤖 = Nova Carter
🎯 = 目標
🚧 = 障礙物

每個環境：
- 寬度: ~15m（env_spacing）
- 獨立運作
- 不應互相干擾
```

### 如果間距太小會發生什麼？

```
env_spacing = 10m（太小）

   env_0        env_1
     │            │
   [🤖🚧🎯]    [🤖🚧🎯]
     │            │
     └────────────┘
   ⚠️ 可能重疊！障礙物可能穿透
```

---

## 🔍 如何檢查當前配置

### 檢查環境數量和間距

```bash
cd /home/aa/IsaacLab

# 方法1: 直接查看配置
grep -A2 "LocalPlannerSceneCfg(" source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
    local_planner/local_planner_env_cfg.py | grep "num_envs\|env_spacing"

# 方法2: 查看 __post_init__ 中的設置
grep -A20 "def __post_init__" source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
    local_planner/local_planner_env_cfg.py | grep "num_envs"
```

### 檢查障礙物碰撞配置

```bash
# 查看靜態障礙物配置
sed -n '94,102p' source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
    local_planner/local_planner_env_cfg.py

# 查看動態障礙物配置
sed -n '105,115p' source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
    local_planner/local_planner_env_cfg.py
```

---

## 🎯 場景定義完整位置總結

### 核心場景配置

```bash
📁 source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
   local_planner/local_planner_env_cfg.py

關鍵部分：
─────────────────────────────────────────────────────
第 37-135 行:  LocalPlannerSceneCfg 類
├─ 第 42-55:   地形（平面）
├─ 第 58-77:   Nova Carter 機器人 ⭐
├─ 第 80-92:   LiDAR 感測器
├─ 第 95-102:  靜態障礙物（方塊）⭐
├─ 第 105-115: 動態障礙物（球體）⭐
├─ 第 118-127: 目標標記
└─ 第 130-133: 光照

第 332 行:     環境數量設定
第 342-380 行: __post_init__ 方法（運行時配置）
```

### 快速跳轉命令

```bash
# 查看機器人 USD 引用
vim +58 source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
    local_planner/local_planner_env_cfg.py

# 查看障礙物配置
vim +95 source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
    local_planner/local_planner_env_cfg.py

# 查看環境數量設定
vim +332 source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
    local_planner/local_planner_env_cfg.py
```

---

## 📝 修改場景的步驟

### 如果您想修改場景

1. **打開配置文件**:
```bash
vim source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/\
    local_planner/local_planner_env_cfg.py
```

2. **找到 LocalPlannerSceneCfg 類**（第 38 行）

3. **修改您想要的組件**:
   - 更換機器人 USD（第 61 行）
   - 修改障礙物（第 95-115 行）
   - 調整 LiDAR（第 80-92 行）

4. **修改環境設定**（第 332 行）:
   - `num_envs`: 並行環境數量
   - `env_spacing`: 環境間距

5. **保存並測試**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 2 --headless --max_iterations 10
```

---

## 💡 關鍵要點

### 關於並行訓練

1. ✅ **這是正常的**: 看到多個機器人是並行訓練機制
2. ✅ **提高效率**: 同時收集多份經驗
3. ✅ **GPU加速**: 充分利用GPU並行能力

### 關於碰撞穿透

1. ⚠️ **可能的問題**: 環境間距不足導致重疊
2. 🔧 **解決方法**: 增加 `env_spacing` 或減少 `num_envs`
3. 🔍 **檢查方法**: 在GUI模式下觀察，或檢查碰撞日誌

### 關於場景定義

1. 📁 **主文件**: `local_planner_env_cfg.py`
2. 📍 **核心行號**: 第 37-135 行（場景），第 332 行（環境數量）
3. 🎯 **USD 路徑**: 第 61 行（機器人），第 98 行（障礙物）

---

## 🧪 診斷並行環境

### 創建診斷腳本

```python
# test_parallel_envs.py
import torch

env_spacing = 15.0
num_envs = 3

print("並行環境空間布局：")
for i in range(num_envs):
    x_offset = i * env_spacing
    print(f"環境 {i}: 中心在 ({x_offset:.1f}, 0.0)")
    
print(f"\n總空間需求: {(num_envs-1) * env_spacing:.1f}m")
print(f"建議場景大小: {max(40, (num_envs-1) * env_spacing + 20):.1f}m")
```

運行:
```bash
./isaaclab.sh -p test_parallel_envs.py
```

---

**現在您知道在哪裡查看和修改場景定義了！** 🎬

核心文件就是：`source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/local_planner/local_planner_env_cfg.py`

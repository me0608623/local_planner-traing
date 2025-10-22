# Nova Carter 訓練模擬場景設計

## 🎬 場景總覽

訓練場景由以下組件構成：
1. **地形** - 平坦地面
2. **機器人** - Nova Carter（差速驅動）
3. **感測器** - LiDAR（360度掃描）
4. **障礙物** - 靜態和動態障礙物
5. **目標標記** - 綠色球體（視覺化）
6. **光照** - 圓頂光源

---

## 🤖 核心組件詳解

### 1. Nova Carter 機器人 ⭐

#### USD 模型文件
```python
usd_path="/home/aa/isaacsim/usd/nova_carter.usd"
```

**這是什麼？**
- NVIDIA Isaac Sim 提供的 Nova Carter 機器人完整3D模型
- 包含：
  - 機器人幾何形狀（視覺和碰撞）
  - 關節定義（輪子、轉向等）
  - 物理屬性（質量、慣性）
  - 材質和紋理

#### 機器人規格
```python
機器人配置:
├─ 位置路徑: "{ENV_REGEX_NS}/Robot"
├─ 初始位置: (0, 0, 0) - 原點
├─ 初始朝向: (1, 0, 0, 0) - 無旋轉
└─ 驅動方式: 差速驅動（兩個主輪）

主要驅動輪:
├─ 左輪: joint_wheel_left
├─ 右輪: joint_wheel_right
├─ 輪子半徑: 0.125m
├─ 輪距: 0.413m
├─ 最大線速度: 2.0 m/s
└─ 最大角速度: π rad/s

執行器配置:
├─ 類型: ImplicitActuatorCfg（隱式求解）
├─ 速度限制: 100.0 rad/s
├─ 力矩限制: 1000.0 N·m
├─ 剛度: 0.0（純速度控制）
└─ 阻尼: 10000.0（高阻尼，穩定控制）
```

**為什麼選擇這個模型？**
- ✅ NVIDIA 官方提供，質量有保證
- ✅ 包含完整的物理模型
- ✅ 適合導航任務
- ✅ 支援差速驅動控制

#### Nova Carter 結構
```
nova_carter.usd
├─ Robot/
│   ├─ chassis_link/          # 底盤
│   │   ├─ base_link/          # 基座（LiDAR安裝點）
│   │   ├─ joint_wheel_left    # 左主驅動輪 ⭐
│   │   ├─ joint_wheel_right   # 右主驅動輪 ⭐
│   │   ├─ joint_caster_*      # 萬向輪（支撐用）
│   │   └─ joint_swing_*       # 懸吊輪（支撐用）
│   └─ [其他組件]
└─ [視覺和碰撞網格]
```

---

### 2. LiDAR 感測器 📡

#### 配置詳解
```python
類型: RayCaster（光線投射模擬）
安裝位置: /World/envs/.*/Robot/Robot/chassis_link/base_link

掃描參數:
├─ 類型: 2D LiDAR（單通道）
├─ 水平視野: -180° ~ +180°（360度全方位）
├─ 垂直視野: 0° ~ 0°（平面掃描）
├─ 水平解析度: 1.0°（每度一個點）
├─ 光線數量: 360 條
├─ 最大距離: 10.0 米
├─ 漂移範圍: 0.0（無噪音）
└─ 調試可視化: False

檢測目標:
└─ 地面: /World/ground
```

**LiDAR 工作原理**:
1. 從機器人基座發射360條光線
2. 每條光線檢測到障礙物的距離
3. 返回360維的距離向量
4. 用於避障和導航

**為什麼使用 RayCaster？**
- ✅ GPU加速，計算快速
- ✅ 精確的幾何碰撞檢測
- ✅ 可配置的掃描模式
- ✅ 適合強化學習訓練

---

### 3. 地形設計 🌍

#### 平坦地面
```python
類型: 程序化生成平面
尺寸: 40m × 40m
材質配置:
├─ 靜摩擦係數: 1.0（不易滑動）
├─ 動摩擦係數: 1.0
├─ 恢復係數: 0.0（無彈性）
└─ 組合模式: multiply

位置: (0, 0, 0)
旋轉: 水平放置
可視化: 啟用
調試可視化: 關閉
```

**設計原理**:
- 簡單環境，專注於導航學習
- 高摩擦力確保穩定運動
- 足夠大的空間（40m×40m）供機器人活動

---

### 4. 障礙物配置 🚧

#### A. 靜態障礙物（方塊）
```python
USD模型: Props/Blocks/DexCube/dex_cube_instanceable.usd
來源: ISAAC_NUCLEUS_DIR（Isaac Sim資源庫）

配置:
├─ 原始大小: 0.1m × 0.1m × 0.1m
├─ 縮放: 2.0x（變成 0.2m × 0.2m × 0.2m）
├─ 初始位置: (5.0, 0.0, 1.0)
└─ 類型: 剛體（RigidObject）

用途:
└─ 模擬固定障礙物（如牆壁、柱子）
```

#### B. 動態障礙物（球體）
```python
類型: 程序化生成球體
配置:
├─ 半徑: 0.5m
├─ 質量: 1.0 kg
├─ 顏色: 紅色 (1.0, 0.0, 0.0)
├─ 初始位置: (3.0, 3.0, 1.0)
└─ 物理: 完整剛體模擬

特性:
├─ 可移動（受物理引擎控制）
├─ 可被推動
└─ 模擬動態障礙物（如移動的行人、車輛）
```

**障礙物重置機制**:
```python
# 在 EventCfg 中定義
reset_static_obstacles:
  - 隨機位置: x∈(2,8)m, y∈(-4,4)m, z=1m
  - 每個episode開始時重置

reset_dynamic_obstacles:
  - 隨機位置: x∈(2,8)m, y∈(-4,4)m, z=1m
  - 隨機速度: vx∈(-1,1)m/s, vy∈(-1,1)m/s
  
push_dynamic_obstacles:
  - 每3-5秒推動一次
  - 模擬動態環境
```

---

### 5. 目標標記 🎯

```python
類型: 動力學球體（Kinematic）
配置:
├─ 半徑: 0.3m
├─ 顏色: 綠色 (0.0, 1.0, 0.0)
├─ 碰撞: 禁用（不會干擾機器人）
├─ 物理: 運動學控制（不受重力影響）
└─ 初始位置: (8.0, 0.0, 0.3)

用途:
├─ 視覺化目標位置
├─ 幫助調試和觀察
└─ 不參與物理碰撞
```

---

### 6. 光照配置 💡

```python
類型: DomeLight（圓頂光）
配置:
├─ 顏色: 淡白色 (0.9, 0.9, 0.9)
├─ 強度: 500.0
└─ 路徑: /World/DomeLight

效果:
├─ 均勻照亮整個場景
├─ 提供良好的視覺效果
└─ 適合渲染和視覺化
```

---

## 🏗️ 場景層級結構

```
/World/
├─ ground                          # 地形
├─ DomeLight                       # 光照
└─ envs/                           # 環境容器
    └─ env_0, env_1, ..., env_N    # 並行環境
        ├─ Robot/                  # Nova Carter機器人
        │   └─ Robot/              # 機器人根節點
        │       └─ chassis_link/   # 底盤
        │           ├─ base_link/  # 基座（LiDAR安裝點）
        │           ├─ wheel_left  # 左輪
        │           └─ wheel_right # 右輪
        ├─ StaticObstacles/        # 靜態障礙物
        ├─ DynamicObstacles/       # 動態障礙物
        └─ GoalMarker/             # 目標標記
```

**{ENV_REGEX_NS} 是什麼？**
- 代表 `/World/envs/env_*`
- 允許多個環境並行訓練
- 例如：`num_envs=4` 時會創建 env_0, env_1, env_2, env_3

---

## 🎮 場景運作流程

### 1. 初始化（Episode Start）

```python
For each 並行環境:
  1. 重置機器人位置
     └─ 隨機位置: x∈(-2,2)m, y∈(-2,2)m
     └─ 隨機朝向: yaw∈(-π,π)
  
  2. 生成隨機目標
     └─ 距離: 5-10米
     └─ 方向: 隨機
  
  3. 重置障礙物
     └─ 靜態障礙物: 隨機位置
     └─ 動態障礙物: 隨機位置+速度
  
  4. 更新目標標記位置
     └─ 移動到新目標位置（綠球）
```

### 2. 訓練循環（Episode Running）

```python
For each time step:
  1. LiDAR 掃描
     └─ 發射360條光線
     └─ 檢測障礙物距離
     └─ 返回距離向量
  
  2. 獲取觀測
     └─ LiDAR數據（360維）
     └─ 目標相對位置（2維）
     └─ 機器人速度（2維）
     └─ 前一動作（2維）
  
  3. 策略輸出動作
     └─ 線速度指令
     └─ 角速度指令
  
  4. 差速驅動轉換
     └─ 計算左右輪速度
     └─ v_left = v_linear - v_angular * wheel_base/2
     └─ v_right = v_linear + v_angular * wheel_base/2
  
  5. 物理模擬
     └─ 應用輪速度
     └─ 模擬機器人運動
     └─ 更新動態障礙物
  
  6. 計算獎勵
     └─ 接近目標: +獎勵
     └─ 碰撞障礙: -懲罰
     └─ 靜止不動: -懲罰
  
  7. 檢查終止條件
     └─ 到達目標？
     └─ 碰撞？
     └─ 超時？
```

### 3. Episode 結束

```python
If 終止條件滿足:
  1. 記錄 episode 數據
     └─ 總獎勵
     └─ 步數
     └─ 終止原因
  
  2. 返回步驟1（重新初始化）
```

---

## 🔧 場景可調整參數

### A. 環境難度

```python
# 簡單環境
num_static_obstacles = 0      # 無靜態障礙物
num_dynamic_obstacles = 0     # 無動態障礙物
goal_distance = (2.0, 4.0)    # 近距離目標
episode_length_s = 40.0       # 更長時間

# 中等環境（當前默認）
num_static_obstacles = 1
num_dynamic_obstacles = 1
goal_distance = (5.0, 10.0)
episode_length_s = 30.0

# 困難環境
num_static_obstacles = 5
num_dynamic_obstacles = 3
goal_distance = (8.0, 15.0)
episode_length_s = 25.0
```

### B. 機器人性能

```python
# 修改速度限制
max_linear_speed = 2.0   # 線速度上限
max_angular_speed = π    # 角速度上限

# 修改輪子參數
wheel_radius = 0.125     # 輪半徑
wheel_base = 0.413       # 輪距
```

### C. 感測器配置

```python
# 修改 LiDAR
horizontal_res = 2.0     # 每2度一個點（180個點）
max_distance = 15.0      # 增加檢測範圍
drift_range = (0.0, 0.1) # 添加感測器噪音
```

---

## 📊 場景統計

### 默認配置下的場景規模

```
並行環境數: 4-8個（可調整）
地形大小: 40m × 40m = 1600m²
機器人數量: 每個環境1個
障礙物: 每環境2個（1靜態+1動態）
LiDAR光線: 360條/環境
總物理對象: ~20個/環境

GPU記憶體使用: ~4-8GB（取決於num_envs）
模擬頻率: 100Hz（dt=0.01s）
RL步頻: 25Hz（decimation=4）
```

---

## 🎯 場景設計原理

### 為什麼這樣設計？

#### 1. 簡單地形
- ✅ 專注於導航學習
- ✅ 避免地形複雜性干擾
- ✅ 計算效率高

#### 2. 2D LiDAR
- ✅ 提供360度環境感知
- ✅ 觀測維度合理（360維）
- ✅ 類似真實移動機器人

#### 3. 差速驅動
- ✅ Nova Carter的實際驅動方式
- ✅ 控制簡單（2維動作）
- ✅ 適合學習基本導航

#### 4. 稠密障礙物
- ✅ 提供避障挑戰
- ✅ 動態障礙物增加難度
- ✅ 更接近真實環境

#### 5. 隨機化
- ✅ 提高策略泛化能力
- ✅ 避免過擬合特定場景
- ✅ 學習魯棒的行為

---

## 🔍 USD 文件位置

### 當前使用的 USD 資源

```bash
# Nova Carter 機器人
/home/aa/isaacsim/usd/nova_carter.usd
└─ NVIDIA Isaac Sim 提供的官方模型

# 靜態障礙物（方塊）
${ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd
└─ Isaac Sim 資源庫中的方塊模型

# 地形、動態障礙物、目標標記
└─ 程序化生成（不使用 USD 文件）
```

### 如何查找其他 USD 資源？

```bash
# Isaac Sim 資源目錄
ls -la ~/isaacsim/usd/

# Nucleus 資源庫（線上）
# 通過 Isaac Sim Content Browser 瀏覽
```

### 可替換的機器人模型

```python
# 其他可用的機器人 USD
"/Isaac/Robots/Carter/carter_v1.usd"        # Carter v1
"/Isaac/Robots/Carter/carter_v2.usd"        # Carter v2  
"/Isaac/Robots/Jetbot/jetbot.usd"          # Jetbot
"/Isaac/Robots/Turtlebot/turtlebot3.usd"   # Turtlebot3

# 使用時需要相應調整關節名稱和參數
```

---

## 💡 自定義場景

### 如何修改場景？

```python
# 1. 更換機器人模型
robot = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/path/to/your/robot.usd",  # 改這裡
    ),
)

# 2. 添加更多障礙物
extra_obstacles = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/ExtraObstacles",
    spawn=sim_utils.UsdFileCfg(
        usd_path="${ISAAC_NUCLEUS_DIR}/Props/YourObject.usd",
    ),
)

# 3. 更改地形
# 可以使用 Isaac Lab 的地形生成器創建複雜地形
from isaaclab.terrains import TerrainImporterCfg

terrain = TerrainImporterCfg(
    terrain_type="perlin",  # 隨機地形
    terrain_generator=MyTerrainGenerator,
)
```

---

## 📚 相關文檔

- [代碼架構指南](CODE_ARCHITECTURE_GUIDE.md) - 完整代碼說明
- [訓練策略總結](../TRAINING_STRATEGY_SUMMARY.md) - 訓練策略
- [快速開始](../QUICK_START_GUIDE.md) - 立即開始

---

**總結**: 訓練場景以 Nova Carter 機器人為中心，配備2D LiDAR進行環境感知，在帶有隨機障礙物的平坦地形上學習目標導航任務。場景設計簡潔高效，專注於核心導航能力的學習。🎯

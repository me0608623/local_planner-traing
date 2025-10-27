# 🚀 下一步行動指南

> **基於37.5%成功率的穩固baseline，進入實際應用階段**

---

## 🎯 兩個主要方向

### 方向1：動態障礙物訓練 🔥

### 方向2：真實機器人部署 🤖

---

## 🔥 方向1：動態障礙物訓練

### 配置說明

我已創建 `LocalPlannerEnvCfg_DYNAMIC`：
- ✅ 基於DEBUG成功配置（極簡獎勵）
- ✅ 保持0.3-1.0米目標距離
- ✅ 新增1個動態障礙物（移動球體）
- ✅ 每3-5秒改變移動方向

### 訓練指令

**選項A：從頭訓練**（驗證動態障礙物的影響）

```bash
cd /home/aa/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Dynamic-v0 \
    --num_envs 16 \
    --max_iterations 5000 \
    --headless
```

**選項B：從靜態版本模型繼續**（遷移學習）

```bash
# 找到5000 iterations的模型
ls -lt logs/rsl_rl/local_planner_carter/ | head -3

# 從checkpoint繼續（假設目錄是最新的）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Dynamic-v0 \
    --num_envs 16 \
    --max_iterations 3000 \
    --load_run logs/rsl_rl/local_planner_carter/[最新目錄] \
    --checkpoint model_4999.pt \
    --headless
```

**預期結果**：
- 初期成功率：20-30%（比靜態版本略低，正常）
- 5000 iterations後：25-35%
- Agent需要學會預測和避開移動障礙物

**時間**：約1.5-2小時

---

## 🤖 方向2：真實機器人部署

### 步驟1：測試訓練好的模型（模擬器）

**找到最佳模型**：
```bash
cd /home/aa/IsaacLab
ls -lt logs/rsl_rl/local_planner_carter/ | head -3
```

**測試模型**（GUI模式，觀察行為）：
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-DEBUG-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/local_planner_carter/[最新目錄]/model_4999.pt
```

**觀察重點**：
- ✅ 機器人是否平滑移動到目標？
- ✅ 是否能避開障礙物？
- ✅ 是否會原地打轉或卡住？

### 步驟2：評估成功率

**多次測試**（統計成功率）：
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-DEBUG-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/local_planner_carter/[最新目錄]/model_4999.pt \
    --num_episodes 100
```

**記錄**：
- 總episodes：100
- 成功次數：應該約35-40次
- 平均到達時間
- 碰撞次數

### 步驟3：導出模型

**導出為.pt格式**（已經是，可直接使用）：
```
model_4999.pt 包含：
- Actor network權重
- Critic network權重
- 優化器狀態
```

**或導出為ONNX**（用於優化推理，可選）：
```python
import torch
from rsl_rl.modules import ActorCritic

# 載入模型
checkpoint = torch.load('logs/rsl_rl/local_planner_carter/[目錄]/model_4999.pt')

# 創建模型（需要匹配訓練時的架構）
model = ActorCritic(
    num_actor_obs=369,
    num_critic_obs=369,
    num_actions=2,
    actor_hidden_dims=[256, 256, 128],
    critic_hidden_dims=[256, 256, 128],
    activation='elu'
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 導出ONNX
dummy_input = torch.randn(1, 369)
torch.onnx.export(
    model.actor, 
    dummy_input, 
    'nova_carter_policy.onnx',
    input_names=['observation'],
    output_names=['action'],
    dynamic_axes={'observation': {0: 'batch'}, 'action': {0: 'batch'}}
)
print('✅ 模型已導出為 nova_carter_policy.onnx')
```

### 步驟4：真實機器人ROS2接口

**觀測處理**（將真實LiDAR轉換為模型輸入）：
```python
def process_lidar(lidar_msg):
    """處理ROS2 LaserScan消息為模型輸入"""
    # 1. 獲取LiDAR距離（360個點）
    ranges = np.array(lidar_msg.ranges)
    
    # 2. 歸一化到[0,1]（和訓練時一樣）
    max_range = 10.0  # 訓練時的max_distance
    ranges = np.clip(ranges / max_range, 0.0, 1.0)
    
    # 3. 處理inf和nan
    ranges = np.nan_to_num(ranges, nan=1.0, posinf=1.0)
    
    return ranges  # shape: (360,)
```

**完整ROS2 Node範例**：
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import torch
import numpy as np

class NovaCarterPolicyNode(Node):
    def __init__(self, model_path):
        super().__init__('nova_carter_policy')
        
        # 載入模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 訂閱
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal', self.goal_callback, 10)
        
        # 發布
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 狀態
        self.latest_lidar = None
        self.latest_velocity = np.zeros(6)
        self.goal_position = np.array([1.0, 0.0])
        
    def lidar_callback(self, msg):
        # 處理LiDAR
        self.latest_lidar = self.process_lidar(msg)
        # 執行推理
        self.execute_policy()
    
    def execute_policy(self):
        if self.latest_lidar is None:
            return
        
        # 構建觀測（和訓練時一樣：LiDAR + 速度 + 目標）
        obs = np.concatenate([
            self.latest_lidar,           # 360
            self.latest_velocity,        # 6
            self.goal_position,          # 2
            [np.linalg.norm(self.goal_position)]  # 1
        ])  # 總共369維
        
        # 模型推理
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.model.act_inference(obs_tensor)
        
        # 發布控制指令
        cmd = Twist()
        cmd.linear.x = float(action[0, 0]) * 2.0   # 縮放到-2~+2 m/s
        cmd.angular.z = float(action[0, 1]) * 3.14  # 縮放到-π~+π rad/s
        self.cmd_pub.publish(cmd)
```

**啟動方式**：
```bash
ros2 run your_package nova_carter_policy_node \
    --model logs/rsl_rl/local_planner_carter/[目錄]/model_4999.pt
```

### 步驟5：真實機器人測試流程

**測試階段1：靜止測試**
- 機器人不動，只測試LiDAR數據處理
- 驗證觀測處理是否正確

**測試階段2：小範圍測試**
- 設置0.3-0.5米的近距離目標
- 開放區域，無障礙物
- 觀察機器人是否能到達

**測試階段3：障礙物測試**
- 加入簡單靜態障礙物
- 驗證避障能力

**測試階段4：正常使用**
- 設置0.5-1.0米目標
- 複雜環境測試

---

## 📊 預期成果

### 動態障礙物訓練

**預期**：
- 成功率：25-35%（比靜態版本略低）
- 新能力：避開移動障礙物
- 訓練時間：約2小時

### 真實機器人

**Sim-to-Real Gap**：
- 模擬器成功率：37.5%
- 真實機器人預期：20-30%（會有下降）
- 主要差異：LiDAR噪音、地面摩擦力、延遲

**改善方法**：
- Domain Randomization（訓練時加入噪音）
- Fine-tuning（在真實機器人上微調）
- Sensor Calibration（校準感測器）

---

## 🎯 我的建議執行順序

### 第1步：測試模型（模擬器）

```bash
# 測試DEBUG模型表現
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-DEBUG-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/local_planner_carter/[最新5000iter目錄]/model_4999.pt
```

### 第2步：訓練動態障礙物版本

```bash
# 從靜態版本模型繼續
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Dynamic-v0 \
    --num_envs 16 \
    --max_iterations 3000 \
    --load_run logs/rsl_rl/local_planner_carter/[最新5000iter目錄] \
    --checkpoint model_4999.pt \
    --headless
```

### 第3步：準備真實機器人部署

- 創建ROS2 package
- 實作policy node
- 配置LiDAR接口

---

**您想先執行哪一步？建議先測試模型（play.py），看看訓練效果！** 🎯


# Nova Carter 訓練診斷指南

## 📊 訓練結果分析框架

### 關鍵指標解讀

#### 1. 獎勵指標
- **Mean reward**: 平均累積獎勵
  - 正值：策略表現良好
  - 負值：策略需要改進
  - 趨勢：應該隨訓練逐漸上升

#### 2. 成功率指標
- **Episode_Reward/reached_goal**: 到達目標的獎勵
  - 理想值：> 0，表示有成功案例
  - 0：從未到達目標
  
- **Episode_Termination/goal_reached**: 成功終止率
  - 理想值：> 0.1（至少10%成功率）
  - 0：策略完全未學會任務

#### 3. 終止條件分析
- **Episode_Termination/time_out**: 超時終止率
  - 1.0 (100%)：所有episode都超時，策略未學會
  - < 0.5：大部分能在時限內完成或碰撞

#### 4. 距離誤差
- **Metrics/goal_command/position_error**: 與目標的距離
  - 理想值：< 0.5 米
  - > 5 米：策略可能在原地打轉或隨機移動

## 🔍 常見問題診斷

### 問題 1: 獎勵始終為負值，未學會任務

**症狀**:
```
Mean reward: -2598.61
Episode_Reward/reached_goal: 0.0000
Episode_Termination/time_out: 1.0000
```

**可能原因**:

1. **獎勵函數設計問題**
   - 懲罰過重，導致策略畏縮不前
   - 獎勵稀疏，沒有足夠的中間獎勵引導
   - 距離獎勵尺度不當

2. **觀測空間問題**
   - LiDAR 數據未正確標準化
   - 目標相對位置計算錯誤
   - 觀測維度過高或過低

3. **動作空間問題**
   - 動作範圍過大或過小
   - 速度限制不合理
   - 動作噪聲過大

4. **超參數設置**
   - 學習率過高或過低
   - 批次大小不適當
   - 折扣因子 γ 設置不當

5. **環境配置問題**
   - 目標距離過遠
   - 障礙物過於密集
   - Episode 時間限制過短

### 問題 2: 訓練不穩定，獎勵波動大

**症狀**:
```
Mean value_function loss: 3154.6131 (過高)
Mean action noise std: 1.07 (可能過大)
```

**可能原因**:
- Value function 難以收斂
- 探索噪聲過大
- 環境隨機性過高

### 問題 3: 機器人不移動或原地打轉

**症狀**:
```
Mean episode length: 181.36 (但未到達目標)
position_error: 5.1919 (幾乎沒有接近目標)
```

**可能原因**:
- 動作懲罰過重
- 碰撞懲罰導致過度保守
- 觀測到目標的信息不清晰

## 🛠️ 診斷步驟

### 步驟 1: 檢查環境配置

```python
# 檢查目標距離設置
print(env.cfg.commands.goal_command.ranges.distance)

# 檢查最大episode步數
print(env.cfg.episode_length_s)

# 檢查動作範圍
print(env.cfg.actions.joint_vel.scale)
```

### 步驟 2: 分析獎勵函數權重

查看 `local_planner_env_cfg.py` 中的獎勵權重：

```python
@configclass
class RewardsCfg:
    # 檢查這些權重是否合理
    progress_to_goal_weight = 1.0      # 接近目標獎勵
    reached_goal_weight = 100.0        # 到達目標獎勵
    collision_penalty_weight = -10.0   # 碰撞懲罰
    obstacle_proximity_penalty_weight = -0.1  # 接近障礙物懲罰
```

### 步驟 3: 觀察訓練曲線

```bash
# 查看 TensorBoard 日誌
tensorboard --logdir logs/rsl_rl/
```

關注以下曲線：
- Mean reward 是否上升
- Episode length 是否合理
- Success rate 是否增加
- Value loss 是否收斂

### 步驟 4: 視覺化測試

```bash
# 使用訓練好的策略進行測試
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/local_planner_carter/*/model_*.pt
```

觀察機器人行為：
- 是否有明顯的目標導向行為？
- 是否能避開障礙物？
- 是否在原地打轉？

## 🔧 改進建議

### 建議 1: 調整獎勵函數

**增加引導性獎勵**:
```python
# 增加連續的距離減少獎勵
progress_to_goal_weight = 5.0  # 從 1.0 增加

# 減少過度懲罰
collision_penalty_weight = -5.0  # 從 -10.0 減少
standstill_penalty_weight = -0.1  # 從 -1.0 減少
```

**添加 Shaping 獎勵**:
- 考慮添加朝向目標的獎勵
- 添加距離減少的連續獎勵
- 減少稀疏獎勵的比重

### 建議 2: 調整環境難度

**簡化初始環境**:
```python
# 減少障礙物數量
scene.obstacles.num_obstacles = 5  # 從 10+ 減少

# 增加目標距離範圍
commands.goal_command.ranges.distance = (2.0, 5.0)  # 從 (5.0, 10.0) 減少

# 增加 episode 時間
episode_length_s = 30.0  # 從 20.0 增加
```

### 建議 3: 調整超參數

**學習率**:
```python
# rsl_rl_ppo_cfg.py
learning_rate = 5e-4  # 從 1e-3 減少，更穩定
```

**Batch size**:
```python
num_steps_per_env = 24  # 從 16 增加
mini_batch_size = 128   # 從 64 增加
```

**探索噪聲**:
```python
# 隨訓練減少噪聲
desired_kl = 0.01  # 從 0.02 減少，更保守的更新
```

### 建議 4: 觀測標準化

確保觀測被正確標準化：
```python
# 在 observations.py 中
def lidar_obs(env, sensor_cfg):
    distances = sensor.data.ray_hits_w[..., 0]
    # 標準化到 [0, 1]
    distances = distances / sensor_cfg.max_distance
    return distances
```

### 建議 5: Curriculum Learning

**階段式訓練**:
1. **階段 1**: 簡單環境（少障礙物，近目標）
2. **階段 2**: 中等環境（中等障礙物，中等距離）
3. **階段 3**: 完整環境（原始設置）

```python
# 實現難度調整
if iteration < 300:
    env.cfg.scene.obstacles.num_obstacles = 3
    env.cfg.commands.goal_command.ranges.distance = (2.0, 4.0)
elif iteration < 600:
    env.cfg.scene.obstacles.num_obstacles = 5
    env.cfg.commands.goal_command.ranges.distance = (3.0, 6.0)
else:
    # 使用完整難度
    pass
```

## 📈 成功訓練的指標

**良好訓練應該顯示**:
```
Mean reward: > -500 (理想 > 0)
Episode_Reward/reached_goal: > 0.1 (至少10%成功率)
Episode_Termination/goal_reached: > 0.1
Episode_Termination/time_out: < 0.8 (大部分不超時)
position_error: < 2.0 (平均距離目標<2米)
```

## 🔍 診斷工具腳本

創建診斷腳本來自動分析訓練結果：

```python
# scripts/analyze_training_results.py
import re

def analyze_training_log(log_text):
    """分析訓練日誌並提供診斷"""
    
    # 提取關鍵指標
    mean_reward = float(re.search(r'Mean reward: ([-\d.]+)', log_text).group(1))
    reached_goal = float(re.search(r'reached_goal: ([\d.]+)', log_text).group(1))
    time_out = float(re.search(r'time_out: ([\d.]+)', log_text).group(1))
    position_error = float(re.search(r'position_error: ([\d.]+)', log_text).group(1))
    
    # 診斷
    issues = []
    
    if mean_reward < -1000:
        issues.append("❌ 平均獎勵過低，策略未學會任務")
    
    if reached_goal == 0.0:
        issues.append("❌ 從未到達目標，需要簡化環境或調整獎勵")
    
    if time_out > 0.9:
        issues.append("❌ 幾乎所有episode超時，考慮增加時間限制")
    
    if position_error > 4.0:
        issues.append("❌ 距離目標太遠，機器人可能不移動或隨機移動")
    
    return issues
```

## 💡 快速診斷清單

- [ ] 檢查獎勵函數權重是否合理
- [ ] 確認觀測空間是否正確標準化
- [ ] 驗證動作範圍是否適當
- [ ] 檢查環境難度是否過高
- [ ] 查看 TensorBoard 訓練曲線
- [ ] 視覺化測試觀察機器人行為
- [ ] 考慮使用更簡單的環境開始
- [ ] 調整超參數（學習率、batch size）
- [ ] 實施 Curriculum Learning

## 📚 參考資源

- [PPO 超參數調優指南](https://github.com/openai/baselines)
- [強化學習除錯技巧](https://andyljones.com/posts/rl-debugging.html)
- [Isaac Lab 訓練最佳實踐](https://isaac-sim.github.io/IsaacLab/)

---

**記住**: 強化學習訓練是迭代過程，需要耐心調試和優化。從簡單環境開始，逐步增加難度！

# 訓練策略快速參考

## 🎯 核心訓練策略：PPO + 稠密獎勵

### 主要文件

| 文件 | 用途 | 重要性 |
|------|------|--------|
| **`mdp/rewards.py`** | 獎勵函數實現 | ⭐⭐⭐ **最重要** |
| **`rsl_rl_ppo_cfg.py`** | PPO演算法配置 | ⭐⭐ |
| **`local_planner_env_cfg.py`** | 環境配置（獎勵權重） | ⭐⭐ |
| **`train.py`** | 訓練腳本 | ⭐ |

---

## 📊 當前訓練策略

### 獎勵設計（稠密獎勵）

```python
總獎勵 = (
    + 接近目標獎勵     × 10.0     # 持續引導
    + 到達目標獎勵     × 100.0    # 最終目標  
    - 障礙物接近懲罰   × 5.0      # 安全警告
    - 碰撞懲罰         × 50.0     # 嚴重錯誤
    - 角速度懲罰       × 0.01     # 平滑運動
    - 靜止懲罰         × 0.1      # 鼓勵行動
)
```

### PPO 超參數

```python
學習率 (learning_rate) = 1e-3
PPO裁剪參數 (clip_param) = 0.2
折扣因子 (gamma) = 0.99
熵係數 (entropy_coef) = 0.01
網路架構 = [256, 256, 128]  # 3層MLP
```

---

## 🔧 快速調整指南

### 問題診斷

| 問題 | 症狀 | 解決方案 |
|------|------|---------|
| **不移動** | 機器人靜止 | 增加 `standstill_penalty` 到 -1.0 |
| **碰撞太多** | collision rate > 50% | 增加 `collision_penalty` 到 -100.0 |
| **很難到達目標** | success rate < 5% | 增加 `progress_to_goal` 到 20.0 |
| **訓練不穩定** | reward 劇烈波動 | 降低 `learning_rate` 到 3e-4 |
| **收斂太慢** | 500次後無進步 | 增加 `num_steps_per_env` 到 48 |

### 修改位置

```bash
# 修改獎勵權重
vim source/.../local_planner/local_planner_env_cfg.py
# 找到 RewardsCfg 類，修改 weight 參數

# 修改演算法參數
vim source/.../local_planner/agents/rsl_rl_ppo_cfg.py
# 找到 LocalPlannerPPORunnerCfg 類

# 修改獎勵函數邏輯（高級）
vim source/.../local_planner/mdp/rewards.py
```

---

## 📈 期望訓練曲線

### 正常訓練進度

```
迭代 0-500:
├─ Mean Reward: -2000 → -500
├─ Success Rate: 0% → 10%
└─ 學習基本導航

迭代 500-1500:
├─ Mean Reward: -500 → 0
├─ Success Rate: 10% → 40%
└─ 學習避障 + 目標導向

迭代 1500-3000:
├─ Mean Reward: 0 → 500+
├─ Success Rate: 40% → 70%+
└─ 策略精細化
```

---

## 🎓 訓練策略原理

### 為什麼這樣設計？

#### 1. 稠密獎勵 vs 稀疏獎勵

**稠密獎勵（當前使用）**:
- ✅ 每步都有反饋
- ✅ 學習更快
- ✅ 適合複雜任務

**稀疏獎勵**:
- ❌ 只在完成時給獎勵
- ❌ 學習很慢
- ❌ 難以收斂

#### 2. PPO vs 其他演算法

**PPO 優勢**:
- ✅ 穩定性好（裁剪機制）
- ✅ 樣本效率高
- ✅ 實現簡單
- ✅ 經過驗證

#### 3. 獎勵權重層級

```
達成目標 (100)     # 最重要
    ↓
避免碰撞 (50)      # 安全第一
    ↓
接近目標 (10)      # 持續引導
    ↓
保持距離 (5)       # 預防性安全
    ↓
運動品質 (0.1)     # 錦上添花
```

---

## 💡 實用技巧

### 1. 開始訓練前

```bash
# 先用少量迭代測試
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 2 \
    --headless \
    --max_iterations 10

# 確認沒有錯誤後再開始完整訓練
```

### 2. 監控關鍵指標

```python
必看指標:
✅ Mean reward (應該上升)
✅ Episode_Reward/reached_goal (應該增加)
✅ Episode_Termination/time_out (應該減少)
✅ Episode_Termination/goal_reached (應該增加)
```

### 3. 診斷工具

```bash
# 分析訓練結果
./isaaclab.sh -p scripts/analyze_training_log.py --stdin

# 檢查環境
./isaaclab.sh -p scripts/diagnose_tensor_device.py
```

### 4. 保存實驗記錄

```bash
# 修改參數前記錄當前設置
git add -A
git commit -m "實驗1: baseline, lr=1e-3, reward_weight=10"

# 訓練後記錄結果
# 在 commit message 中記錄最終 reward 和 success rate
```

---

## 🔬 進階調優

### A. 獎勵工程（Reward Engineering）

#### 當前獎勵函數問題識別

```bash
# 查看獎勵分解
# 訓練日誌中會顯示:
Episode_Reward/progress_to_goal: -125.65  # ❌ 負值！問題！
Episode_Reward/reached_goal: 0.0000       # ❌ 從未到達
Episode_Reward/collision_penalty: -1000   # ❌ 碰撞太多
```

#### 根據問題調整

```python
# 如果 progress_to_goal 是負值
→ 機器人沒有接近目標
→ 增加 progress_to_goal weight 或簡化環境

# 如果 collision_penalty 很大
→ 機器人碰撞太頻繁  
→ 增加 obstacle_proximity_penalty 或 collision_penalty

# 如果 reached_goal 始終為0
→ 任務太難
→ 縮短目標距離或延長時間
```

### B. Curriculum Learning（課程學習）

#### 階段式難度提升

```python
# 階段1: 簡單（0-500次迭代）
目標距離: 2-3米
障礙物: 0個
時間: 40秒

# 階段2: 中等（500-1500次）
目標距離: 3-6米
障礙物: 3個
時間: 35秒

# 階段3: 完整（1500-3000次）
目標距離: 5-10米
障礙物: 5-10個
時間: 30秒
```

### C. 超參數搜索

#### 推薦範圍

```python
learning_rate: [1e-4, 3e-4, 1e-3, 3e-3]
entropy_coef: [0.001, 0.01, 0.05]
clip_param: [0.1, 0.2, 0.3]
num_mini_batches: [2, 4, 8]
```

---

## 📚 相關文檔

- [完整代碼架構指南](md/CODE_ARCHITECTURE_GUIDE.md) - 詳細的技術說明
- [訓練診斷指南](md/TRAINING_DIAGNOSIS_GUIDE.md) - 問題排查
- [快速開始指南](QUICK_START_GUIDE.md) - 立即開始訓練

---

**記住**: 強化學習需要實驗和調整，沒有一成不變的最佳參數！🧪

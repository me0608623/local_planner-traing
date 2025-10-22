# Nova Carter 訓練快速開始指南

## ⚠️ 最重要的提醒

**所有命令必須使用 `./isaaclab.sh -p` 而不是系統 `python`！**

```bash
# ✅ 正確
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task ...

# ❌ 錯誤  
python scripts/reinforcement_learning/rsl_rl/train.py --task ...
```

## 🚀 5 分鐘快速開始

### 1. 首次訓練（簡化環境）

```bash
cd /home/aa/IsaacLab

# 使用簡化環境開始訓練
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-Easy-v0 \
    --num_envs 4 \
    --headless
```

**為什麼選擇簡化環境？**
- 目標更近（2-5m）
- 障礙物更少（3個）
- 更容易學會任務
- 更快看到訓練效果

### 2. 監控訓練進度

訓練過程中查看關鍵指標：
- **Mean reward**: 應該逐漸上升（目標 > -500）
- **Episode_Reward/reached_goal**: 成功率（目標 > 10%）
- **Episode_Termination/time_out**: 超時率（目標 < 80%）

### 3. 診斷訓練結果

如果訓練不理想，使用診斷工具：

```bash
# 方法1: 直接分析（使用示例）
./isaaclab.sh -p scripts/analyze_training_log.py

# 方法2: 粘貼您的訓練日誌
./isaaclab.sh -p scripts/analyze_training_log.py --stdin
# 然後粘貼訓練輸出，按 Ctrl+D

# 方法3: 分析日誌文件
./isaaclab.sh -p scripts/analyze_training_log.py \
    --file logs/rsl_rl/your_training.log
```

工具會自動告訴您：
- 🔍 發現的問題
- 💡 改進建議
- ⚙️ 配置調整方案

## 📊 Curriculum Learning（階段式訓練）

如果簡化環境還是太難，試試階段式訓練：

### 階段 1: 最簡單（建議 300 次迭代）

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-Curriculum-Stage1-v0 \
    --num_envs 4 \
    --headless \
    --max_iterations 300
```

**特點**: 1.5-3m 目標，50秒時間，最容易成功

### 階段 2: 中等難度（建議 300 次迭代）

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-Curriculum-Stage2-v0 \
    --num_envs 4 \
    --headless \
    --max_iterations 300 \
    --resume  # 從階段1繼續
```

**特點**: 3-6m 目標，5個障礙物

### 階段 3: 完整難度（建議 500 次迭代）

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 8 \
    --headless \
    --max_iterations 500 \
    --resume  # 從階段2繼續
```

## 🔍 常見問題快速解決

### 問題 1: ModuleNotFoundError

```
錯誤: ModuleNotFoundError: No module named 'isaaclab_tasks'
```

**解決**:
```bash
# 確保環境正確設置
cd /home/aa/IsaacLab
source isaaclab.sh -s
```

### 問題 2: 訓練獎勵始終為負

```
Mean reward: -2598.61
Episode_Reward/reached_goal: 0.0000
```

**解決**:
```bash
# 使用簡化環境
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-Easy-v0 \
    --num_envs 4 --headless
```

### 問題 3: PhysX tensor device 錯誤

```
[Error] [omni.physx.tensors.plugin] Incompatible device of velocity tensor
```

**解決**:
```bash
# 使用 Headless 模式（問題只在GUI模式出現）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 4 \
    --headless  # 確保添加此參數
```

### 問題 4: GPU 記憶體不足

```
錯誤: CUDA out of memory
```

**解決**:
```bash
# 減少並行環境數量
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 2 \  # 從 4 或 8 減少到 2
    --headless
```

## 📈 期望的訓練效果時間線

使用 **Easy 環境** 的預期進度：

| 迭代次數 | 期望獎勵 | 成功率 | 距離誤差 |
|---------|----------|--------|---------|
| 0-100 | -2000 → -1000 | 0% → 5% | 5m → 3m |
| 100-300 | -1000 → -500 | 5% → 15% | 3m → 2m |
| 300-500 | -500 → -200 | 15% → 30% | 2m → 1.5m |
| 500-1000 | -200 → 0+ | 30% → 50%+ | 1.5m → 1m |

使用 **標準環境** 需要的時間大約是 Easy 環境的 2-3 倍。

## 🛠️ 實用工具

### 診斷 PhysX 設備問題

```bash
./isaaclab.sh -p scripts/diagnose_tensor_device.py --full
```

### 視覺化訓練結果

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/*/*/model_*.pt
```

### 查看 TensorBoard

```bash
# 在另一個終端
tensorboard --logdir logs/rsl_rl/
# 然後在瀏覽器打開 http://localhost:6006
```

## 📚 詳細文檔

- [完整 README](README.md)
- [訓練診斷指南](md/TRAINING_DIAGNOSIS_GUIDE.md)
- [GUI vs Headless 問題](md/GUI_VS_HEADLESS_PHYSX_ANALYSIS.md)
- [NVIDIA 官方問題分析](md/NVIDIA_OFFICIAL_PHYSX_ISSUE_ANALYSIS.md)

## 💡 最佳實踐總結

1. **總是使用 `./isaaclab.sh -p`** - 不要用系統 python
2. **從簡化環境開始** - Easy 或 Curriculum Stage 1
3. **使用 Headless 模式** - 更穩定，避免 PhysX 錯誤
4. **定期使用診斷工具** - 及時發現問題
5. **耐心等待** - 強化學習需要時間
6. **階段式增加難度** - 不要直接用最難的環境

---

**祝您訓練成功！** 🚀

如有問題，請查看詳細文檔或使用診斷工具。

# 🚀 Nova Carter 正式訓練指令

## ⚠️ 重要提醒

**所有命令必須使用 `./isaaclab.sh -p`，不要使用系統 python！**

---

## 🎯 推薦的正式訓練指令

### 方案 1: 標準 Headless 訓練（推薦） ⭐

```bash
cd /home/aa/IsaacLab

# 激活環境
conda activate env_isaaclab

# 正式訓練（Headless 模式，最穩定）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 8 \
    --headless \
    --max_iterations 3000 \
    --seed 42
```

**特點**:
- ✅ Headless 模式（避免 GUI 模式的 PhysX 錯誤）
- ✅ 8 個並行環境（提高訓練效率）
- ✅ 3000 次迭代（充分訓練）
- ✅ 固定隨機種子（可重現結果）

**預計訓練時間**: 約 2-3 小時（取決於 GPU 性能）

---

### 方案 2: CPU 訓練（無 GPU 或兼容性最佳）

```bash
cd /home/aa/IsaacLab

conda activate env_isaaclab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-CPU-v0 \
    --num_envs 2 \
    --headless \
    --max_iterations 1500 \
    --seed 42
```

**特點**:
- ✅ CPU 模式（最高兼容性）
- ✅ 較少環境（CPU 運算限制）
- ⚠️ 訓練較慢

**預計訓練時間**: 約 6-8 小時

---

### 方案 3: 快速驗證訓練（測試用）

```bash
cd /home/aa/IsaacLab

conda activate env_isaaclab

# 快速測試（10分鐘）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 4 \
    --headless \
    --max_iterations 100 \
    --seed 42
```

**用途**:
- 驗證環境配置正確
- 檢查是否有錯誤
- 觀察初期訓練趨勢

---

### 方案 4: GPU 優化訓練（高性能）

```bash
cd /home/aa/IsaacLab

conda activate env_isaaclab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-v0 \
    --num_envs 16 \
    --headless \
    --max_iterations 3000 \
    --seed 42
```

**特點**:
- ✅ GPU 優化配置
- ✅ 更多並行環境（16個）
- ✅ 最快訓練速度

**需求**: 較好的 GPU（至少 8GB 顯存）

---

### 方案 5: GUI 模式訓練（可視化需求）

```bash
cd /home/aa/IsaacLab

conda activate env_isaaclab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-GUI-Fixed-v0 \
    --num_envs 2 \
    --max_iterations 1000 \
    --seed 42
```

**特點**:
- 📺 可以看到訓練過程
- 🟢 看到目標標記
- ⚠️ 較慢（GUI 渲染開銷）

**注意**: 不使用 `--headless` 參數

---

## 📊 完整參數說明

### 必需參數

```bash
--task                    # 環境任務名稱
--num_envs               # 並行環境數量
```

### 可選參數

```bash
--headless               # 無 GUI 模式（推薦）
--max_iterations         # 訓練迭代次數（默認 3000）
--seed                   # 隨機種子（可重現，默認 42）
--video                  # 錄製訓練視頻
--video_interval 500     # 每 500 步錄一次視頻
```

### 進階參數

```bash
--resume                 # 從檢查點繼續訓練
--checkpoint PATH        # 指定檢查點路徑
--run_name NAME          # 自定義運行名稱
```

---

## 🎯 根據您的情況選擇

### 如果您的 GPU 記憶體充足（≥ 8GB）

```bash
# ✅ 推薦：標準 GPU 訓練
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 8 \
    --headless \
    --max_iterations 3000 \
    --seed 42
```

### 如果您的 GPU 記憶體有限（4-6GB）

```bash
# ✅ 推薦：減少環境數量
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 4 \
    --headless \
    --max_iterations 3000 \
    --seed 42
```

### 如果您想看到訓練過程

```bash
# ✅ GUI 模式（使用修復配置）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-GUI-Fixed-v0 \
    --num_envs 2 \
    --max_iterations 1000 \
    --seed 42
```

### 如果遇到任何錯誤

```bash
# ✅ 最保守的配置（CPU模式）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-CPU-v0 \
    --num_envs 2 \
    --headless \
    --max_iterations 1500 \
    --seed 42
```

---

## 📈 訓練流程

### 完整訓練流程

```bash
# 步驟 1: 進入項目目錄
cd /home/aa/IsaacLab

# 步驟 2: 激活環境
conda activate env_isaaclab

# 步驟 3: 啟動訓練
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 8 \
    --headless \
    --max_iterations 3000 \
    --seed 42

# 步驟 4: 監控訓練（在另一個終端）
# 打開 TensorBoard
cd /home/aa/IsaacLab
tensorboard --logdir logs/rsl_rl/
# 瀏覽器打開 http://localhost:6006

# 步驟 5: 訓練完成後測試
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/local_planner_carter/*/model_*.pt
```

---

## 🔍 訓練期間監控

### 在終端中查看

訓練日誌會顯示：

```
Learning iteration 500/3000

Mean reward: -245.32              ← 應該逐漸上升
Episode_Reward/reached_goal: 0.15 ← 成功率（應該增加）
Episode_Termination/time_out: 0.65 ← 超時率（應該減少）
Episode_Termination/goal_reached: 0.15 ← 到達率（應該增加）
```

### 良好訓練的指標

```
前期（0-500次）:
  Mean reward: -2000 → -500
  Success rate: 0% → 5-10%

中期（500-1500次）:
  Mean reward: -500 → 0
  Success rate: 10% → 30-40%

後期（1500-3000次）:
  Mean reward: 0 → 500+
  Success rate: 40% → 60-70%+
```

---

## 🛑 如何停止訓練

### 正常停止

```bash
# 在訓練終端按
Ctrl + C

# 模型會自動保存到
logs/rsl_rl/local_planner_carter/[時間戳]/
```

### 從檢查點繼續

```bash
# 找到最新的檢查點
ls -lt logs/rsl_rl/local_planner_carter/*/

# 繼續訓練
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 8 \
    --headless \
    --resume \
    --checkpoint logs/rsl_rl/local_planner_carter/[時間戳]/model_*.pt
```

---

## 📊 訓練結果位置

### 保存位置

```
logs/rsl_rl/local_planner_carter/
└─ [年-月-日]/
    └─ [時-分-秒]/
        ├─ model_100.pt      # 第100次迭代的模型
        ├─ model_200.pt      # 第200次迭代的模型
        ├─ ...
        ├─ model_3000.pt     # 最終模型
        ├─ config.yaml       # 訓練配置
        └─ events.out.tfevents.*  # TensorBoard 日誌
```

---

## 🧪 訓練後測試

### 可視化訓練好的策略

```bash
# Play 模式（觀看機器人行為）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/local_planner_carter/*/model_3000.pt

# 使用最佳模型
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/local_planner_carter/*/model_*.pt \
    --num_episodes 10  # 運行10個episode
```

---

## 💡 訓練技巧

### 1. 使用 tmux 或 screen（長時間訓練）

```bash
# 安裝 tmux
sudo apt install tmux

# 創建 tmux 會話
tmux new -s nova_carter_training

# 在 tmux 中運行訓練
cd /home/aa/IsaacLab
conda activate env_isaaclab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 8 \
    --headless \
    --max_iterations 3000

# 離開 tmux（訓練繼續在背景）
Ctrl+B, 然後按 D

# 重新連接
tmux attach -t nova_carter_training
```

### 2. 保存訓練日誌

```bash
# 將輸出保存到文件
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 8 \
    --headless \
    --max_iterations 3000 \
    2>&1 | tee training_log_$(date +%Y%m%d_%H%M%S).txt
```

### 3. 多次實驗（不同隨機種子）

```bash
# 實驗 1
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 8 --headless --seed 42

# 實驗 2（不同隨機種子）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 8 --headless --seed 123

# 實驗 3
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 8 --headless --seed 456
```

---

## 🎓 根據訓練階段選擇

### 階段 1: 初次訓練（驗證設置）

```bash
# 短時間測試（100 次迭代，約 10 分鐘）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 4 \
    --headless \
    --max_iterations 100 \
    --seed 42
```

**檢查**:
- ✅ 沒有錯誤？
- ✅ Mean reward 有上升趨勢？
- ✅ 保存了檢查點？

→ **如果都正常，繼續下一階段**

### 階段 2: 中期訓練（1000 次迭代）

```bash
# 中期訓練（約 1 小時）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 8 \
    --headless \
    --max_iterations 1000 \
    --seed 42
```

**期望結果**:
- Mean reward: > -500
- Success rate: > 10%

### 階段 3: 完整訓練（3000 次迭代）

```bash
# 完整訓練（約 2-3 小時）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 8 \
    --headless \
    --max_iterations 3000 \
    --seed 42
```

**期望結果**:
- Mean reward: > 0
- Success rate: > 50%

---

## 🎬 訓練時記錄視頻（可選）

```bash
# 訓練並記錄視頻
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 8 \
    --headless \
    --max_iterations 3000 \
    --video \
    --video_interval 500 \
    --video_length 200
```

**視頻保存位置**: `logs/rsl_rl/local_planner_carter/*/videos/`

---

## 📋 訓練檢查清單

### 訓練前檢查

- [ ] 在 `/home/aa/IsaacLab` 目錄下
- [ ] 已激活 `conda activate env_isaaclab`
- [ ] 確認使用 `./isaaclab.sh -p` 而非 `python`
- [ ] 選擇了合適的 `--num_envs`（根據 GPU 記憶體）
- [ ] 選擇了 Headless 模式（`--headless`）

### 訓練中監控

- [ ] Mean reward 是否上升？
- [ ] Success rate 是否增加？
- [ ] Time out rate 是否減少？
- [ ] 沒有錯誤訊息？
- [ ] 模型是否定期保存？

### 訓練後評估

- [ ] 最終 Mean reward > 0？
- [ ] Success rate > 50%？
- [ ] 使用 Play 模式測試
- [ ] 可視化觀察行為是否合理

---

## ⚡ 快速啟動命令（複製貼上即可）

### 推薦：標準訓練

```bash
cd /home/aa/IsaacLab && conda activate env_isaaclab && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Navigation-LocalPlanner-Carter-v0 --num_envs 8 --headless --max_iterations 3000 --seed 42
```

### 快速測試

```bash
cd /home/aa/IsaacLab && conda activate env_isaaclab && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Navigation-LocalPlanner-Carter-v0 --num_envs 4 --headless --max_iterations 100 --seed 42
```

### CPU 訓練

```bash
cd /home/aa/IsaacLab && conda activate env_isaaclab && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Navigation-LocalPlanner-Carter-CPU-v0 --num_envs 2 --headless --max_iterations 1500 --seed 42
```

---

## 🔧 常見問題

### Q: 訓練很慢怎麼辦？

A: 
1. 減少 `--num_envs`
2. 檢查 GPU 使用率（`nvidia-smi`）
3. 確保使用 Headless 模式

### Q: 出現 CUDA out of memory？

A: 減少環境數量
```bash
--num_envs 4  # 或更少
```

### Q: 想從中斷的訓練繼續？

A: 使用 `--resume` 參數
```bash
--resume \
--checkpoint logs/rsl_rl/local_planner_carter/[時間]/model_*.pt
```

### Q: 如何比較不同配置？

A: 使用不同的 `--run_name`
```bash
--run_name "experiment_lr_3e-4"
--run_name "experiment_lr_1e-3"
```

---

## 📊 訓練完成後

### 查看最佳模型

```bash
# 查看所有保存的模型
ls -lh logs/rsl_rl/local_planner_carter/*/model_*.pt

# 通常最後的模型是最好的
# 或者查看 TensorBoard 找到 reward 最高的迭代
```

### 測試最佳模型

```bash
# 使用 Play 腳本
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/local_planner_carter/*/model_3000.pt
```

### 分析訓練結果

```bash
# 使用診斷工具分析最後的訓練日誌
./isaaclab.sh -p scripts/analyze_training_log.py --stdin
# 粘貼訓練輸出，按 Ctrl+D
```

---

## 🎯 正式訓練命令（推薦配置）

### 最終推薦命令 ⭐

```bash
cd /home/aa/IsaacLab

conda activate env_isaaclab

# 正式訓練（複製這個命令）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 8 \
    --headless \
    --max_iterations 3000 \
    --seed 42 \
    2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log
```

**這個命令會**:
- ✅ 訓練 3000 次迭代
- ✅ 使用 8 個並行環境
- ✅ Headless 模式（最穩定）
- ✅ 保存訓練日誌到文件
- ✅ 自動保存模型（每 100 次迭代）

**預計時間**: 2-3 小時（取決於 GPU）

---

**祝您訓練成功！** 🚀

訓練過程中如有問題，請參考：
- [訓練診斷指南](md/TRAINING_DIAGNOSIS_GUIDE.md)
- [快速開始指南](QUICK_START_GUIDE.md)

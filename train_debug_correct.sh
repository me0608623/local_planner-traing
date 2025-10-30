#!/bin/bash
# 正確配置的DEBUG訓練

cd /home/aa/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-DEBUG-v0 \
    --num_envs 16 \
    --max_iterations 1000 \
    --headless \
    > debug_correct_$(date +%Y%m%d_%H%M%S).log 2>&1



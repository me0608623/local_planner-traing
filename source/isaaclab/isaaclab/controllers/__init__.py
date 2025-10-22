# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for different controllers and motion-generators.

Controllers or motion generators are responsible for closed-loop tracking of a given command. The
controller can be a simple PID controller or a more complex controller such as impedance control
or inverse kinematics control. The controller is responsible for generating the desired joint-level
commands to be sent to the robot.
"""

from .differential_ik import DifferentialIKController
from .differential_ik_cfg import DifferentialIKControllerCfg
from .operational_space import OperationalSpaceController
from .operational_space_cfg import OperationalSpaceControllerCfg

# Pink IK Controller (需要 Pinocchio，如果未安裝則跳過)
# 對於不需要複雜運動學的任務（如輪式機器人導航），可以不安裝 Pinocchio
try:
    from .pink_ik import NullSpacePostureTask, PinkIKController, PinkIKControllerCfg
except ImportError as e:
    # Pinocchio 未安裝，跳過 Pink IK Controller
    # 這不影響其他控制器（如差速驅動、DifferentialIK 等）
    import warnings
    warnings.warn(
        f"Pink IK Controller 無法導入（需要 Pinocchio）: {e}\n"
        "如果您使用輪式機器人或不需要複雜運動學，可以忽略此警告。\n"
        "如果需要使用機械臂 IK 控制，請安裝 Pinocchio。",
        ImportWarning
    )

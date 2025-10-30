"""
最小可用環境配置（MIN）— 僅保留訓練必須元件：
Terrain(plane) + Nova Carter + 2D LiDAR + Goal

用途：
- 快速啟動/教學示例
- 降低干擾變數，專注策略基本能力

差異：
- 不包含動態障礙物與事件驅動
- 獎勵為極簡三項：progress/reached/standstill
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CommandTermCfg as CommandTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.navigation.local_planner.mdp as mdp


@configclass
class LocalPlannerSceneCfgMin(InteractiveSceneCfg):
    """最小場景：地面 + 機器人 + LiDAR + 目標標記"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/aa/isaacsim/usd/nova_carter.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "base": ImplicitActuatorCfg(
                joint_names_expr=["joint_wheel_left", "joint_wheel_right"],
                velocity_limit=100.0,
                effort_limit=1000.0,
                stiffness=0.0,
                damping=10000.0,
            ),
        },
    )

    lidar = RayCasterCfg(
        prim_path="/World/envs/.*/Robot/Robot/chassis_link/base_link",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        max_distance=10.0,
        drift_range=(0.0, 0.0),
        debug_vis=False,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ActionsCfg:
    base_velocity = mdp.DifferentialDriveActionCfg(
        asset_name="robot",
        left_wheel_joint_names=["joint_wheel_left"],
        right_wheel_joint_names=["joint_wheel_right"],
        wheel_radius=0.125,
        wheel_base=0.413,
        max_linear_speed=0.8,
        max_angular_speed=0.8,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        lidar_distances = ObsTerm(
            func=mdp.lidar_obs,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        goal_position = ObsTerm(
            func=mdp.goal_position_in_robot_frame,
            params={"command_name": "goal_command"},
        )
        goal_distance = ObsTerm(
            func=mdp.distance_to_goal,
            params={"command_name": "goal_command"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    goal_command = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="chassis_link",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(2.0, 6.0),
            pos_y=(-3.0, 3.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class RewardsCfg:
    """獎勵配置（修正版 v4 - 2025-10-30）
    
    版本歷史：
    
    v1（失敗 - 10000 iter）：
    - Mean Reward 190（虛高），Success 0%
    - 問題：原地朝向策略（Reward Hacking）
    
    v2（改善 - 5000 iter）：
    - Progress 0.0425（變正✅），Position Error 4.19m
    - 問題：Progress 太小，前進太慢
    
    v3（失敗 - 10000 iter）：
    - Progress -0.0104（變負❌），Position Error 3.84m
    - Standstill -1.16、Anti-idle -0.53（懲罰暴增）
    - 問題：懲罰過重，Agent 被壓制不敢動
    
    v4（當前 - 方案 A）：
    1. 保持 progress 權重（60）與 near_goal（20, radius 3m）
    2. 大幅降低所有懲罰項（平衡正負獎勵）
       - standstill: 4.0 → 1.0
       - anti_idle: 2.0 → 0.5
       - spin: 0.5 → 0.1
       - time: 0.01 → 0.005
    3. 讓正向獎勵（progress）主導，懲罰只做弱約束
    
    目標：
    - Progress 從負值回到正值（> 0.1）
    - Position Error < 3m
    - Agent 敢於大膽前進
    """
    progress_to_goal = RewTerm(
        func=mdp.progress_to_goal_reward,
        weight=60.0,  # ↑↑ v3: 從 30.0 提升（強力推動接近目標）
        params={"command_name": "goal_command"},
    )
    reached_goal = RewTerm(
        func=mdp.reached_goal_reward,
        weight=200.0,  # 保持
        params={"command_name": "goal_command", "threshold": 0.8},
    )
    near_goal_shaping = RewTerm(
        func=mdp.near_goal_shaping,
        weight=20.0,  # ↑ v3: 從 10.0 提升（擴大影響力）
        params={"command_name": "goal_command", "radius": 3.0},  # ↑ v3: 從 1.5m 擴大（更早生效）
    )
    heading_alignment = RewTerm(
        func=mdp.heading_alignment_reward,
        weight=1.0,  # ↓ 從 5.0（避免壓倒 progress，且改為條件式）
        params={"command_name": "goal_command", "v_min": 0.1},
    )
    standstill_penalty = RewTerm(
        func=mdp.standstill_penalty,
        weight=1.0,  # ↓↓ v4: 從 4.0 大幅降低（v3 懲罰過重，壓制探索）
    )
    anti_idle = RewTerm(
        func=mdp.anti_idle_penalty,
        weight=0.5,  # ↓ v4: 從 2.0 降低（減輕壓制）
        params={"v_threshold": 0.05},
    )
    spin_penalty = RewTerm(
        func=mdp.spin_penalty,
        weight=0.1,  # ↓ v4: 從 0.5 降低（僅保留弱約束）
        params={"w_threshold": 0.5, "v_threshold": 0.1},
    )
    time_penalty = RewTerm(
        func=mdp.time_penalty,
        weight=0.005,  # ↓ v4: 從 0.01 再降低（最小化時間壓力）
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"command_name": "goal_command", "threshold": 0.8},
    )


@configclass
class LocalPlannerEnvCfgMin(ManagerBasedRLEnvCfg):
    """最小可用環境（MIN）"""

    scene: LocalPlannerSceneCfgMin = LocalPlannerSceneCfgMin(num_envs=48, env_spacing=10.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 30.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.device = "cuda:0"
        self.viewer.eye = (10.0, 10.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)



# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
本地規劃器環境配置 - Nova Carter 動態避障導航任務
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CommandTermCfg as CommandTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab_tasks.manager_based.navigation.local_planner.mdp as mdp

##
# 場景定義
##


@configclass
class LocalPlannerSceneCfg(InteractiveSceneCfg):
    """本地規劃器場景配置：Nova Carter + LiDAR + 障礙物"""

    # 地面 - 使用平坦地形
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

    # Nova Carter 機器人
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/aa/isaacsim/usd/nova_carter.usd",  # ✅ 使用原始 USD
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "base": ImplicitActuatorCfg(
                joint_names_expr=["joint_wheel_left", "joint_wheel_right"],  # ✅ 修正關節名稱
                velocity_limit=100.0,
                effort_limit=1000.0,
                stiffness=0.0,
                damping=10000.0,
            ),
        },
    )

    # LiDAR 感測器 - 使用 RayCaster 模擬 2D LiDAR
    lidar = RayCasterCfg(
        prim_path="/World/envs/.*/Robot/Robot/chassis_link/base_link",  # ✅ 修正：路徑多一層 Robot
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,  # 2D LiDAR
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,  # 每度一個點，共 360 個點
        ),
        max_distance=10.0,
        drift_range=(0.0, 0.0),
        debug_vis=False,
    )

    # 靜態障礙物 - 方塊
    static_obstacles = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/StaticObstacles",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 1.0)),
    )

    # 動態障礙物 - 球體
    dynamic_obstacles = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DynamicObstacles",
        spawn=sim_utils.SphereCfg(
            radius=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 3.0, 1.0)),
    )

    # 目標標記（視覺化用）
    goal_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(8.0, 0.0, 0.3)),
    )

    # 光照
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP 設定
##


@configclass
class ActionsCfg:
    """動作空間配置 - 差速驅動控制"""

    # 線速度和角速度控制（差速驅動）
    # Nova Carter 主驅動輪：joint_wheel_left, joint_wheel_right
    # joint_caster_*, joint_swing_* 輪為輔助支撐輪，無需主動控制
    base_velocity = mdp.DifferentialDriveActionCfg(
        asset_name="robot",
        left_wheel_joint_names=["joint_wheel_left"],
        right_wheel_joint_names=["joint_wheel_right"],
        wheel_radius=0.125,  # 根據實際 Nova Carter 規格調整
        wheel_base=0.413,    # 根據實際 Nova Carter 規格調整
        max_linear_speed=2.0,
        max_angular_speed=math.pi,
    )


@configclass
class ObservationsCfg:
    """觀察空間配置 - LiDAR + 機器人狀態 + 目標資訊"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略觀察組"""

        # LiDAR 距離數據 (360 個點)
        lidar_distances = ObsTerm(
            func=mdp.lidar_obs,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
        )

        # 機器人當前速度（線速度 + 角速度）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

        # 目標相對位置（機器人座標系）
        goal_position = ObsTerm(
            func=mdp.goal_position_in_robot_frame,
            params={"command_name": "goal_command"},
        )

        # 目標距離和方向
        goal_distance = ObsTerm(
            func=mdp.distance_to_goal,
            params={"command_name": "goal_command"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # 觀察組
    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """指令配置 - 目標位置生成"""

    # 隨機目標位置指令
    goal_command = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="chassis_link",  # Nova Carter 的主體 link
        resampling_time_range=(10.0, 10.0),  # 每 10 秒重新生成目標
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(3.0, 10.0),
            pos_y=(-5.0, 5.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class RewardsCfg:
    """獎勵函數配置"""

    # 正向獎勵：接近目標
    progress_to_goal = RewTerm(
        func=mdp.progress_to_goal_reward,
        weight=10.0,
        params={"command_name": "goal_command"},
    )

    # 到達目標獎勵
    reached_goal = RewTerm(
        func=mdp.reached_goal_reward,
        weight=100.0,
        params={"command_name": "goal_command", "threshold": 0.5},
    )

    # 懲罰：與障礙物過近
    obstacle_proximity_penalty = RewTerm(
        func=mdp.obstacle_proximity_penalty,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "safe_distance": 1.0},
    )

    # 懲罰：碰撞
    collision_penalty = RewTerm(
        func=mdp.collision_penalty,
        weight=-50.0,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "collision_threshold": 0.3},
    )

    # 懲罰：過大的角速度（鼓勵平滑運動）
    ang_vel_penalty = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 懲罰：靜止不動
    standstill_penalty = RewTerm(
        func=mdp.standstill_penalty,
        weight=-0.1,
    )


@configclass
class TerminationsCfg:
    """終止條件配置"""

    # 時間限制
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 到達目標
    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"command_name": "goal_command", "threshold": 0.5},
    )

    # 碰撞（LiDAR 偵測到非常近的障礙物）
    collision = DoneTerm(
        func=mdp.collision_termination,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "collision_threshold": 0.3},
    )


@configclass
class EventCfg:
    """事件配置 - 場景重置與隨機化"""

    # 重置機器人位置
    reset_robot_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 重置動態障礙物位置
    reset_dynamic_obstacles = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (2.0, 8.0), "y": (-4.0, 4.0), "z": (1.0, 1.0)},
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
            "asset_cfg": SceneEntityCfg("dynamic_obstacles"),
        },
    )

    # 為動態障礙物施加隨機推力（模擬運動）
    push_dynamic_obstacles = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 5.0),  # 每 3-5 秒推一次
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("dynamic_obstacles"),
        },
    )


##
# 環境配置
##


@configclass
class LocalPlannerEnvCfg(ManagerBasedRLEnvCfg):
    """Nova Carter 本地規劃器環境 - 動態避障導航任務"""

    # 場景設定
    scene: LocalPlannerSceneCfg = LocalPlannerSceneCfg(num_envs=1024, env_spacing=15.0)

    # MDP 設定
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """後處理：設定模擬參數"""
        # 模擬設定
        self.decimation = 4  # 每 4 個物理步長執行一次 RL 步長
        self.episode_length_s = 30.0  # 每回合 30 秒
        self.sim.dt = 0.01  # 物理時間步長 10ms
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        
        # 🔧 明確設定GPU設備（解決PhysX張量設備匹配問題）
        self.sim.device = "cuda:0"
        
        # 🔧 增加GPU緩衝區容量（防止GPU記憶體不足導致回退到CPU）
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 512 * 1024
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 256 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 1024

        # 檢視器設定
        self.viewer.eye = (10.0, 10.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


##
# 簡化版環境（用於快速測試）
##


@configclass
class LocalPlannerEnvCfg_SIMPLE(LocalPlannerEnvCfg):
    """簡化版：較少環境數量，較短回合時間"""

    def __post_init__(self):
        super().__post_init__()
        # 覆寫設定
        self.scene.num_envs = 8
        self.episode_length_s = 15.0


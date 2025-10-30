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
        max_linear_speed=2.0,
        max_angular_speed=math.pi,
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
    progress_to_goal = RewTerm(
        func=mdp.progress_to_goal_reward,
        weight=50.0,
        params={"command_name": "goal_command"},
    )
    reached_goal = RewTerm(
        func=mdp.reached_goal_reward,
        weight=500.0,
        params={"command_name": "goal_command", "threshold": 0.5},
    )
    standstill_penalty = RewTerm(
        func=mdp.standstill_penalty,
        weight=-0.1,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"command_name": "goal_command", "threshold": 0.5},
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



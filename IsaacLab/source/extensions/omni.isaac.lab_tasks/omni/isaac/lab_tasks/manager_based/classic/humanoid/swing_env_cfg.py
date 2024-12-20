# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import torch
import omni.isaac.lab_tasks.manager_based.classic.humanoid.mdp as mdp
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
import omni.isaac.core.utils.prims as prim_utils
##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=7.0, restitution=0.0),
        debug_vis=False,
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.04,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(), 
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0), 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 1.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.0,    # Increase static friction (default is usually 1.0)
                dynamic_friction=2.0,   # Increase dynamic friction (default is usually 1.0)
                restitution=0.0        # Keep zero restitution for no bouncing
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.706, 0.315, 0.08)),
    )
    # robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Humanoid/humanoid_instanceable.usd",
            usd_path=f"/home/daniel/development/csc2626/final-project/IsaacLab/humanoid_w9.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.34),
            # joint_pos={
            #     ".*": 0.0,
            # },
            joint_pos={
                "lower_waist:0": 0.2143025 + 0,
                "lower_waist:1": 0.29417243 + 0,
                "right_upper_arm:0": (0.4 - 0.30267698) + 0.385398,
                "right_upper_arm:2": (0.7 + 0.46815178) - 0.615472907,
                "left_upper_arm:0": 0.3498035 - 0.785398,
                "left_upper_arm:2": 0.519223 - 0.815472907,
                "pelvis": 0.0 + 0.0,
                "right_lower_arm": 0.67 * (0.85204905 - 2),
                "left_lower_arm": 0.67 * (0.8723765 - 2),
                "right_thigh:0": 0.67 * (-0.14598082 + 0.0),
                "right_thigh:1": 0.67 * (-0.77993596 + 0.0),
                "right_thigh:2": 0.0 + 0.0,
                "left_thigh:0": 0.67 * (0.14598082 + 0.0),
                "left_thigh:1": 0.67 * (-0.836486 + 0.0),
                "left_thigh:2": 0.0 + 0.0,
                "right_shin": 3.33 * (-0.11140303 + 0.0),
                "left_shin": 3.33 * (-0.11913639 + 0.0),
                "right_foot:0": 0.0 + 0.0,
                "right_foot:1": 0.0 + 0.0,
                "left_foot:0": 0.0 + 0.0,
                "left_foot:1": 0.0 + 0.0,
            },
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness={
                    ".*_waist.*": 100.0,
                    ".*_upper_arm.*": 100.0,
                    "pelvis": 100.0,
                    ".*_lower_arm": 100.0,
                    ".*_thigh:0": 100.0,
                    ".*_thigh:1": 100.0,
                    ".*_thigh:2": 100.0,
                    ".*_shin": 120.0,
                    ".*_foot.*": 50.0,
                },
                damping={
                    ".*_waist.*": 20.0,
                    ".*_upper_arm.*": 20.0,
                    "pelvis": 20.0,
                    ".*_lower_arm": 20.0,
                    ".*_thigh:0": 20.0,
                    ".*_thigh:1": 20.0,
                    ".*_thigh:2": 20.0,
                    ".*_shin": 20.0,
                    ".*_foot.*": 20.0,
                },
            ),
        },
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            ".*_waist.*": 67.5,
            ".*_upper_arm.*": 67.5,
            "pelvis": 67.5,
            ".*_lower_arm": 45.0,
            ".*_thigh:0": 45.0,
            ".*_thigh:1": 135.0,
            ".*_thigh:2": 45.0,
            ".*_shin": 90.0,
            ".*_foot.*": 22.5,
        },
    )
    #['torso', 'head', 'lower_waist', 'right_upper_arm', 'left_upper_arm', 'pelvis', 'right_lower_arm', 'left_lower_arm', 'right_thigh', 'left_thigh', 'right_hand', 'left_hand', 'right_shin', 'left_shin', 'Club', 'right_foot', 'left_foot']
    # joint_positions = mdp.JointPositionActionCfg(
    #     use_default_offset=False,
    #     asset_name="robot",
    #     joint_names=[".*"],
    #     offset={
    #         "lower_waist:0": 0.8143025 + 0,
    #         "lower_waist:1": -0.42,#-0.89417243 + 0,
    #         "right_upper_arm:0": (0.4 - 0.30267698) + 0.385398,
    #         "right_upper_arm:2": (0.7 + 0.46815178) - 0.615472907,
    #         "left_upper_arm:0": 0.3498035 - 0.785398,
    #         "left_upper_arm:2": 0.519223 - 0.815472907,
    #         "pelvis": 0.0 + 0.0,
    #         "right_lower_arm": 0.67 * (0.85204905 - 2),
    #         "left_lower_arm": 0.67 * (0.8723765 - 2),
    #         "right_thigh:0": 0.67 * (-0.14598082 + 0.0),
    #         "right_thigh:1": 0.67 * (-0.77993596 + 0.0),
    #         "right_thigh:2": 0.0 + 0.0,
    #         "left_thigh:0": 0.67 * (0.14598082 + 0.0),
    #         "left_thigh:1": 0.67 * (-0.836486 + 0.0),
    #         "left_thigh:2": 0.0 + 0.0,
    #         "right_shin": 3.33 * (-0.11140303 + 0.0),
    #         "left_shin": 3.33 * (-0.11913639 + 0.0),
    #         "right_foot:0": 0.0 + 0.0,
    #         "right_foot:1": 0.0 + 0.0,
    #         "left_foot:0": 0.0 + 0.0,
    #         "left_foot:1": 0.0 + 0.0,
    #     },

    #     scale={
    #         ".*_waist.*": 1.5,
    #         ".*_upper_arm.*": 1.5,
    #         "pelvis": 1.5,
    #         ".*_lower_arm": 1.5,
    #         ".*_thigh:0": 1.5,
    #         ".*_thigh:1": 1.5,
    #         ".*_thigh:2": 1.5,
    #         ".*_shin": 2.0,
    #         ".*_foot.*": 1.0,
    #     },

    #     # scale={
    #     #     ".*_waist.*": 1.0,
    #     #     ".*_upper_arm.*": 1.0,
    #     #     "pelvis": 1.0,
    #     #     ".*_lower_arm": 0.67,
    #     #     ".*_thigh:0": 0.67,
    #     #     ".*_thigh:1": 0.67,
    #     #     ".*_thigh:2": 0.67,
    #     #     ".*_shin": 3.33,
    #     #     ".*_foot.*": 0.33,
    #     # },
    # )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""
        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        # base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        # base_up_proj = ObsTerm(func=mdp.base_up_proj)
        # base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot", "right_foot"])},
        )
        actions = ObsTerm(func=mdp.last_action)
        current_timestep = ObsTerm(func=mdp.episode_length)
       
        ball_pos = ObsTerm(
            func=mdp.root_pos_w, 
            params={"asset_cfg": SceneEntityCfg("ball")}
        )
        # ball_lin_vel = ObsTerm(
        #     func=mdp.root_lin_vel_w, 
        #     params={"asset_cfg": SceneEntityCfg("ball")}
        # )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            # "position_range": (-0.2, 0.2),
            # "velocity_range": (-0.1, 0.1),
        },
    )

    reset_ball = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {
                "x": (0.0, 0.0),  # Fixed x position
                "y": (0.0, 0.0),  # Fixed y position
                "z": (0.0, 0.0),    # Fixed z position
            },
            "velocity_range": {
                "linear": (0.0, 0.0),  # Zero initial velocity
                "angular": (0.0, 0.0)  # Zero initial angular velocity
            }
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    distance = RewTerm(func=mdp.ball_distance_bonus, weight=0.3, params={"threshold": 1.0})
    # velocity = RewTerm(func=mdp.ball_velocity_bonus, weight=0.3, params={"threshold": 0.1})
    # pose_match = RewTerm(func=mdp.pose_sequence_match_reward, weight=50.0)
    # velocity_match = RewTerm(func=mdp.pose_sequence_velocity_match_reward, weight=50.0)
    #zero_reward = RewTerm(func=mdp.zero_reward, weight=0.0)
    #pose_match = RewTerm(func=mdp.pose_match_bonus, weight=1.0, params={"threshold": 0.98})
    #joint_velocity_match = RewTerm(func=mdp.joint_sequence_velocity_match_reward, weight=1.0)
    #joint_match = RewTerm(func=mdp.joint_sequence_match_reward, weight=1.0)
    quaternion_match = RewTerm(func=mdp.quaternion_sequence_match_reward, weight=1.0)
    ee_match = RewTerm(func=mdp.ee_sequence_match_reward, weight=1.0)
    #pose_loc = RewTerm(func=mdp.pose_loc_bonus, weight=1.0, params={"threshold": 0.98})
    # (1) Reward for moving forward
    # progress = RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    # (2) Stay alive bonus
    #alive = RewTerm(func=mdp.is_alive, weight=0.5)
    # (3) Reward for non-upright posture
    #upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.05, params={"threshold": 0.85})
    # (4) Reward for moving in the right direction
    # move_to_target = RewTerm(
    #     func=mdp.move_to_target_bonus, weight=0.5, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
    # )
    # (5) Penalty for large action commands
    #action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    # (6) Penalty for energy consumption
    # energy = RewTerm(
    #     func=mdp.power_consumption,
    #     weight=-0.005,
    #     params={
    #         "gear_ratio": {
    #             ".*_waist.*": 67.5,
    #             ".*_upper_arm.*": 67.5,
    #             "pelvis": 67.5,
    #             ".*_lower_arm": 45.0,
    #             ".*_thigh:0": 45.0,
    #             ".*_thigh:1": 135.0,
    #             ".*_thigh:2": 45.0,
    #             ".*_shin": 90.0,
    #             ".*_foot.*": 22.5,
    #         }
    #     },
    # )
    # (7) Penalty for reaching close to joint limits
    # joint_limits = RewTerm(
    #     func=mdp.joint_limits_penalty_ratio,
    #     weight=-0.25,
    #     params={
    #         "threshold": 0.98,
    #         "gear_ratio": {
    #             ".*_waist.*": 67.5,
    #             ".*_upper_arm.*": 67.5,
    #             "pelvis": 67.5,
    #             ".*_lower_arm": 45.0,
    #             ".*_thigh:0": 45.0,
    #             ".*_thigh:1": 135.0,
    #             ".*_thigh:2": 45.0,
    #             ".*_shin": 90.0,
    #             ".*_foot.*": 22.5,
    #         },
    #     },
    # )
    


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.8})


@configclass
class SwingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid walking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 6.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0

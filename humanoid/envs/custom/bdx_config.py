from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# frame_stack = 15              # 堆叠15帧观察数据
# num_single_obs = 47          # 单帧观察维度为47
# num_observations = int(frame_stack * num_single_obs)  # 总观察维度
# num_actions = 12             # 动作空间维度为12（对应12个关节）
# num_envs = 4096             # 并行环境数量
# episode_length_s = 24        # 每个回合24秒

# pos_limit = 1.0             # 位置限制
# vel_limit = 1.0             # 速度限制
# torque_limit = 0.85         # 扭矩限制为最大值的85%

# file = '...XBot-L.urdf'     # 机器人URDF文件路径
# foot_name = "ankle_roll"    # 脚踝关节名称
# terminate_after_contacts_on = ['base_link']  # 机身接触地面时终止
# self_collisions = 0         # 启用自碰撞检测

# mesh_type = 'plane'         # 地形类型：平面
# static_friction = 0.6       # 静摩擦系数
# dynamic_friction = 0.6      # 动摩擦系数
# terrain_length = 8.         # 地形长度
# terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]  # 不同地形类型的比例

# noise_level = 0.6           # 噪声水平
# # 各种状态量的噪声比例
# noise_scales:
#     dof_pos = 0.05         # 关节位置噪声
#     dof_vel = 0.5          # 关节速度噪声
#     quat = 0.03           # 四元数噪声

# pos = [0.0, 0.0, 0.95]     # 初始位置
# default_joint_angles = {...} # 默认关节角度配置

# stiffness = {'leg_roll': 200.0, ...}  # 各关节刚度
# damping = {'leg_roll': 10, ...}       # 各关节阻尼
# action_scale = 0.25                   # 动作缩放比例
# decimation = 10                       # 控制频率降采样（100Hz）

# dt = 0.001                  # 仿真时间步长（1000Hz）
# up_axis = 1                 # 向上轴为z轴
#
# randomize_friction = True   # 随机化摩擦系数
# friction_range = [0.1, 2.0] # 摩擦系数范围
# push_robots = True         # 启用随机推动
# push_interval_s = 4        # 推动间隔
#
# ranges:
#     lin_vel_x = [-0.3, 0.6]    # 前进速度范围
#     lin_vel_y = [-0.3, 0.3]    # 侧向速度范围
#     ang_vel_yaw = [-0.3, 0.3]  # 转向速度范围
#
# base_height_target = 0.89      # 目标躯干高度
# tracking_sigma = 5             # 跟踪奖励系数
# scales: # 不同奖励项的权重
#     joint_pos = 1.6           # 关节位置奖励
#     tracking_lin_vel = 1.2    # 速度跟踪奖励
#     orientation = 1.          # 姿态奖励
#
# seed = 5                      # 随机种子
# class policy:
#     actor_hidden_dims = [512, 256, 128]   # Actor网络结构
#     critic_hidden_dims = [768, 256, 128]  # Critic网络结构
# class algorithm:
#     learning_rate = 1e-5      # 学习率
#     gamma = 0.994             # 折扣因子
#     num_mini_batches = 4      # 小批量数量


class HumanoidBotLCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 73
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 4096
        episode_length_s = 24     # episode length in seconds
        use_ref_actions = False   # speed up training by using reference actions

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bdx/go_bdx.urdf'

        name = "bdx"
        foot_name = "foot"
        knee_name = "knee"

        terminate_after_contacts_on = ['pelvis']
        penalize_contacts_on = ["pelvis"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.95]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_yaw': 0.,
            'left_hip_roll': 0.,
            'left_hip_pitch': 0.,
            'left_knee': 0.,
            'left_ankle': 0.,
            'right_hip_yaw': 0.,
            'right_hip_roll': 0.,
            'right_hip_pitch': 0.,
            'right_knee': 0.,
            'right_ankle': 0.,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            'hip_yaw': 200.0,  # 髋部偏航需要适中刚度
            'hip_roll': 250.0,  # 髋部滚动需要较大刚度保持稳定
            'hip_pitch': 350.0,  # 髋部俯仰需要大刚度支撑身体
            'knee': 350.0,  # 膝盖需要大刚度支撑身体
            'ankle': 200.0,  # 踝关节需要适中刚度保持平衡
        }
        damping = {
            'hip_yaw': 10.0,
            'hip_roll': 12.0,
            'hip_pitch': 12.0,
            'knee': 12.0,
            'ankle': 10.0,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.3, 0.6]   # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3] # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.89
        min_dist = 0.2
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.06        # m
        cycle_time = 0.64                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 700  # Forces above this value are penalized
        soft_dof_pos_limit = 0.7  # 根据关节限位±0.7853调整
        soft_dof_vel_limit = 28.0  # 略小于关节速度限制30
        soft_torque_limit = 22.0  # 略小于关节力矩限制23.7

        class scales:
            # reference motion tracking
            joint_pos = 0.2
            feet_clearance = 1.
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1.
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002
            action_rate = -0.001
            torques = -0.0002
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 0.25
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.


class SimBotLCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 3001  # number of policy updates

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'SimBot_ppo'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt

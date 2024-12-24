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


class BdXBotLCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        # change the observation dim
        # 表示堆叠的观察帧数量。这意味着智能体不仅能看到当前状态，还能看到之前14帧的历史状态
        frame_stack = 15
        # 特权观察的帧堆叠数量，这里是3帧。特权观察通常包含一些在实际部署时可能无法获得，但在训练时有助于学习的信息
        c_frame_stack = 3

        # 单个时间步的观察维度（实际情况调整）
        # 关节角度和角速度
        # 机器人姿态
        # 足端接触力
        # 重心位置和速度等
        # TODO
        num_single_obs = 41
        # 总的观察维度是615（15 * 41）。这是因为将15帧的观察连接在一起，使得智能体能够理解时序信息。
        num_observations = int(frame_stack * num_single_obs)
        # 单帧特权观察的维度是65
        # 地形信息
        # 完整的动力学状态
        # 参考轨迹等额外信息
        single_num_privileged_obs = 65
        # 总的特权观察维度是195（3 * 65）
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        # 动作空间维度为10，对于双足机器人，这通常对应于：
        # 左右腿的髋关节、膝关节和踝关节的位置或力矩控制
        # 可能还包括躯干的姿态控制
        num_actions = 10
        num_envs = 4096
        # 每个训练回合的持续时间为30秒
        # 这个时长需要足够长以让机器人学习稳定的步态
        # 但也不能太长以避免累积误差和训练效率问题
        episode_length_s = 24
        # 是否使用参考动作来加速训练
        # 设置为False表示完全依赖强化学习来学习动作，而不使用人工设计的参考动作
        # 使用参考动作可能会加快训练速度，但可能限制策略的探索空间
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

        # 如果这些部位发生碰撞就终止训练回合
        terminate_after_contacts_on = ['pelvis']

        # 对这些部位的接触给予惩罚
        penalize_contacts_on = ["pelvis"]

        # 是否启用自碰撞检测
        self_collisions = 0
        # 是否翻转视觉附件
        flip_visual_attachments = False
        # 是否用胶囊体替换圆柱体
        replace_cylinder_with_capsule = False
        # 是否固定机器人基座
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        # 地形类型设置为平面（也可以设置为'trimesh'用于更复杂的地形）
        mesh_type = 'plane'
        # mesh_type = 'trimesh'

        # 是否启用课程学习，逐步增加地形难度
        curriculum = True
        # rough terrain only:
        # 是否测量地形高度
        measure_heights = False
        # 静摩擦系数
        static_friction = 0.6
        # 动摩擦系数
        dynamic_friction = 0.6

        # 地形尺寸参数

        # 地形长度
        terrain_length = 8.
        # 地形宽度
        terrain_width = 8.
        # 地形行数（难度等级）
        num_rows = 20  # number of terrain rows (levels)
        # 地形列数（类型）
        num_cols = 20  # number of terrain cols (types)

        # 初始课程状态
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        # 不同类型地形的比例分布：
        # 依次代表：
        # - 平地 (20%)
        # - 障碍物 (20%)
        # - 不均匀地形 (40%)
        # - 上坡 (10%)
        # - 下坡 (10%)
        # - 上楼梯 (0%)
        # - 下楼梯 (0%)
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]

        # 弹性系数（碰撞恢复系数）
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
        pos = [0.0, 0.0, 0.0]

        default_joint_angles = {
            'left_hip_yaw': 0.,
            'left_hip_roll': 0.0,
            'left_hip_pitch': 0.0,
            'left_knee': 0.0,
            'left_ankle': 0.0,

            'right_hip_yaw': 0.0,
            'right_hip_roll': 0.0,
            'right_hip_pitch': 0.0,
            'right_knee': 0.0,
            'right_ankle': 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            # 需要适中刚度实现转向
            # 过大会影响灵活性
            # 过小会导致方向不稳定
            'hip_yaw': 40.0,  # 髋部偏航需要适中刚度
            # 需要较大刚度维持侧向稳定
            # 影响机器人的左右平衡
            # 过小会导致侧倾
            'hip_roll': 40.0,  # 髋部滚动需要较大刚度保持稳定
            # 需要最大刚度支撑身体重量
            # 影响前后平衡和推进力
            # 关系到步态稳定性
            'hip_pitch': 80.0,  # 髋部俯仰需要大刚度支撑身体
            # 需要大刚度支撑身体
            # 影响腿部伸展和屈曲
            # 关系到着地缓冲和推进
            'knee': 80.0,  # 膝盖需要大刚度支撑身体
            # 需要适中刚度保持平衡
            # 影响足底适应地形
            # 关系到步态平稳性
            'ankle': 80.0,  # 踝关节需要适中刚度保持平衡
        }
        # 阻尼
        # 一般设置为刚度的 4-6%
        # 过大会使运动迟缓
        # 过小会产生振荡
        # 需要根据实际运动效果调整
        damping = {
            'hip_yaw': 0.25,
            'hip_roll': 0.25,
            'hip_pitch': 0.25,
            'knee': 0.25,
            'ankle': 0.25,
        }
        # 0.2表示动作范围是默认角度的±20%
        # 较小的值可以：
        # 提高控制精度
        # 增加稳定性
        # 减少过大动作带来的风险
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # 设置为4表示：
        # 每4个仿真步长执行一次控制
        # 实现100Hz的控制频率
        # 平衡计算负载和控制精度
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        gravity = [0.0, 0.0, -9.81]
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
        # 命令维度：前进速度、横向速度、偏航角速度、朝向
        num_commands = 4
        # 重新采样时间（每8秒更换一次目标命令）
        resampling_time = 8.
        # 是否使用朝向命令
        heading_command = True

        class ranges:
            lin_vel_x = [-0.2, 0.5]   # min max [m/s]
            lin_vel_y = [-0.2, 0.2]   # min max [m/s]
            ang_vel_yaw = [-0.25, 0.25] # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.33
        min_dist = 0.15
        max_dist = 0.2

        # 运动参考参数
        # 关节位置目标范围(弧度)
        target_joint_pos_scale = 0.15    # rad
        # 足部目标高度(米)
        target_feet_height = 0.05        # m
        # 步态周期(秒)
        cycle_time = 0.7                # sec

        # 奖励限制参数
        # 将负奖励截断为0
        only_positive_rewards = True
        # 跟踪奖励系数
        tracking_sigma = 4
        # 最大接触力限制(N)
        max_contact_force = 700

        # 软约束限制
        soft_dof_pos_limit = 0.75  # 根据关节限位±0.7853调整
        soft_dof_vel_limit = 27.0  # 略小于关节速度限制30
        soft_torque_limit = 20.0  # 略小于关节力矩限制23.7

        class scales:
            # 运动跟踪相关权重
            # 关节位置跟踪权重
            joint_pos = 0.25
            # 足部离地间隙权重
            feet_clearance = 1.
            # 足部接触数量权重
            feet_contact_number = 1.2

            # 步态相关权重
            # 足部悬空时间权重
            feet_air_time = 1.
            # 足部滑动惩罚权重
            foot_slip = -0.05
            # 足部间距权重
            feet_distance = 0.2

            # 接触相关权重
            # 接触力惩罚权重
            feet_contact_forces = -0.01

            # 速度跟踪权重
            # 线速度跟踪权重
            tracking_lin_vel = 0.8
            # 角速度跟踪权重
            tracking_ang_vel = 0.4
            # 速度不匹配指数权重
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            # 低速奖励权重
            low_speed = 0.2
            # 严格速度跟踪权重
            track_vel_hard = 0.5

            # 姿态相关权重
            # 默认关节位置权重
            default_joint_pos = 0.5
            # 方向跟踪权重
            orientation = 1.2
            # 基座高度维持权重
            base_height = 0.3
            # 基座加速度权重
            base_acc = 0.2

            # 能量效率相关权重
            # 动作平滑度惩罚
            action_smoothness = -0.002
            # 关节力矩使用惩罚
            torques = -0.0003
            # 关节速度惩罚
            dof_vel = -5e-4
            # 关节加速度惩罚
            dof_acc = -1e-7
            # 碰撞惩罚
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


class BdXBotLCfgPPO(LeggedRobotCfgPPO):
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
        experiment_name = 'BdXBot_ppo'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt

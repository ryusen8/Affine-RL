import numpy as np

class MapArg:
    SCREEN_WIDTH_PX = 1200  # 假设你希望窗口宽度是 1200 像素
    SCREEN_HEIGHT_PX = 600  # 假设你希望窗口高度是 600 像素
    PADDING_PX = 20
    DT = 0.1
    SEED = 0
    MAP_SCALE = 5
    MIN_X = 0.0
    MAX_X = 100.0*MAP_SCALE # [0, 100]
    MIN_Y = -10.0*MAP_SCALE
    MAX_Y = 10.0*MAP_SCALE # [-10, 10]
    GOAL_POS = np.array([85.0, 0.0])*MAP_SCALE
    GOAL_RADIUS = 2.0*MAP_SCALE

    CIRCLE_POS = np.array([[25.0, 0.0],[75.0, 0.0]])*MAP_SCALE
    CIRCLE_RADIUS = 2.0*MAP_SCALE
    RECTANGLE_SIZE = np.array([25.0, 6.0])*MAP_SCALE
    RECT_Y = MAX_Y - RECTANGLE_SIZE[1]*0.5
    RECTANGLE_POS = np.array([[50.0*MAP_SCALE, RECT_Y],[50.0*MAP_SCALE, -RECT_Y]])
    

class AgentArg:
    NUM_LEADERS = 3
    GOAL_RADIUS = 2.0
    AGENT_RADIUS = 1.5

    NOMINAL_CONFIG = np.array([np.array([0.0, 0.0]),
                               np.array([-1.0, 1.0])*4*AGENT_RADIUS,
                               np.array([-1.0, -1.0])*4*AGENT_RADIUS])
    LEADER_SPAWN = NOMINAL_CONFIG + np.array([20.0, 0.0])

    LIDAR_NUM_RAYS = 16
    LIDAR_MAX_RANGE = 20
    LIDAR_FOV = 2*np.pi

    MAX_VEL = 6.0
    MIN_VEL = -6.0
    MAX_ACC = 2
    MIN_ACC = -2
    TOL_ERROR = 2*AGENT_RADIUS

    # 仿射变换参数
    MAX_ROT = np.pi
    MIN_ROT = -np.pi
    MAX_SCALE = 2.0
    MIN_SCALE = 0.5
    MAX_SHEAR = 1.0
    MIN_SHEAR = -1.0

    # PID控制真实领导者位移
    KP = 0.2
    KD = 0.1

class RewardArg:
    TOL_COLLIDE_TIMES = 12

    R_TOGOAL = 120
    R_GOAL = 500
    R_MOVE = 15
    R_DIR = 5

    P_SLOW = -50
    P_FAIL = -500
    P_DANGER = -0.08
    P_COLLIDE = -10
    P_FORM_ERROR = -0.2
    P_TIME = -0.5

class TrainArg:
    ENV_NAME = "affine_gym_env/AffineEnv"
    NUM_EP = 120_00 # 训练的总回合数
    EP_MAX_STEP = 600 # 每个回合的最大步数
    LOG_INTERVAL = 10 # 每隔多少个回合打印一次日志
    NUM_TEST_EP = 5 # 测试回合数
    SMOOTH = 15 # 奖励曲线平滑窗口大小

    # SAC 算法超参数
    ACTOR_LR = 1e-4
    CRITIC_LR = 1e-4
    ALPHA_LR = 1e-4 # 温度系数 alpha 的学习率,一般地，温度系数的学习率和网络参数的学习率保持一致
    GAMMA = 0.99 # 折扣因子
    TAU = 0.01 # 软更新因子
    ALPHA_INIT = 0.2 # 初始温度参数 (如果使用自动熵调整，此值会被覆盖)
    BUFFER_SIZE = 1_000_000 # 经验回放缓冲区容量
    BATCH_SIZE = 256 # 训练批次大小

    # 网络参数
    ACTOR_HIDDEN_SIZE = 256
    CRITIC_HIDDEN_SIZE = 256




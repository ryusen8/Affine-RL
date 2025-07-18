import numpy as np

class MapArg:
    SCREEN_WIDTH_PX = 1200  # 假设你希望窗口宽度是 1200 像素
    SCREEN_HEIGHT_PX = 600  # 假设你希望窗口高度是 600 像素
    PADDING_PX = 20
    DT = 0.1
    SEED = None
    MAP_SCALE = 5
    MIN_X = 0.0
    MAX_X = 100.0*MAP_SCALE # [0, 100]
    MIN_Y = -10.0*MAP_SCALE
    MAX_Y = 10.0*MAP_SCALE # [-10, 10]
    GOAL_POS = np.array([85.0, 0.0], dtype=np.float32)*MAP_SCALE
    GOAL_RADIUS = 2.0*MAP_SCALE

    CIRCLE_POS = np.array([[25.0, 0.0],[75.0, 0.0]], dtype=np.float32)*MAP_SCALE
    CIRCLE_RADIUS = 2.0*MAP_SCALE
    RECTANGLE_SIZE = np.array([25.0, 6.0], dtype=np.float32)*MAP_SCALE
    RECT_Y = MAX_Y - RECTANGLE_SIZE[1]*0.5
    RECTANGLE_POS = np.array([[50.0*MAP_SCALE, RECT_Y],[50.0*MAP_SCALE, -RECT_Y]], dtype=np.float32)
    COLOR = {
            "BACKGROUND": (255, 255, 255),
            "BOUNDARY": (0, 0, 0),
            "OBSTACLE": (150, 150, 150),
            "GOAL": (0, 255, 0),
            "LEADER": (0, 0, 255),
            "FIRST": (255, 0, 0),
            "LIDAR_RAY": (200, 200, 200), # 调浅一点
            "FIRST_TRAIL": (255, 150, 150),
            "LEADER_TRAIL": (150, 150, 255),
            "TARGET_FORMATION": ((173, 216, 230),(182,60,27)),
            "HUD_TEXT": (50, 50, 50),
        }

class AgentArg:
    NUM_LEADERS = 3
    GOAL_RADIUS = 2.0
    AGENT_RADIUS = 1.5

    NOMINAL_CONFIG = np.array([np.array([0.0, 0.0]),
                                np.array([-1.0, 1.0])*6*AGENT_RADIUS,
                                np.array([-1.0, -1.0])*6*AGENT_RADIUS], dtype=np.float32)
    LEADER_SPAWN = NOMINAL_CONFIG + np.array([20.0, 0.0], dtype=np.float32)

    LIDAR_NUM_RAYS = 16
    LIDAR_MAX_RANGE = 25
    LIDAR_FOV = 1.5*np.pi

    MAX_VEL = 12.0
    MIN_VEL = -12.0
    MAX_ACC = 4
    MIN_ACC = -4
    TOL_ERROR = 4*AGENT_RADIUS

    # 仿射变换参数
    MAX_ROT = np.pi
    MIN_ROT = -np.pi
    MAX_SCALE = 2.0
    MIN_SCALE = 0.4
    MAX_SHEAR = 1.5
    MIN_SHEAR = -1.5

    # PD控制真实领导者位移
    KP = 0.2
    KD = 0.1

class RewardArg:
    TOL_COLLIDE_TIMES = 6

    R_TOGOAL = 160
    R_GOAL = 800
    R_MOVE = 10
    R_DIR = 7

    P_SLOW = -50
    P_OVERSPEED = -3
    P_FAIL = -300
    P_DANGER = -2
    P_COLLIDE = -30
    P_FORM_ERROR = -0.3
    P_TIME = 0.0

class TrainArg:
    ENV_NAME = "affine_gym_env/AffineEnv"
    SEED = None
    NUM_EP = 80_00 # 训练的总回合数
    EP_MAX_STEP = 400 # 每个回合的最大步数
    LOG_INTERVAL = 10 # 每隔多少个回合打印一次日志.
    GIF_INTERVAL = 500
    GIF_FPS = 30
    NUM_TEST_EP = 2 # 测试回合数
    SMOOTH = 15 # 奖励曲线平滑窗口大小
    REWARD_SCALE = 0.01
    # SAC 算法超参数
    LR = 2e-4
    ACTOR_LR = LR
    CRITIC_LR = LR
    ALPHA_LR = LR # 温度系数 alpha 的学习率,一般地，温度系数的学习率和网络参数的学习率保持一致
    GAMMA = 0.99 # 折扣因子
    TAU = 0.005 # 软更新因子
    ALPHA_INIT = 0.2 # 初始温度参数 (如果使用自动熵调整，此值会被覆盖)
    BUFFER_SIZE = 1_000_000 # 经验回放缓冲区容量
    BATCH_SIZE = 512 # 训练批次大小
    LOG_STD_MIN = -5
    LOG_STD_MAX = 0.0
    # 网络参数
    ACTOR_HIDDEN_SIZE = 256
    CRITIC_HIDDEN_SIZE = 256




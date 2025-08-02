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
            
            "FIRST": (255, 35, 96),
            "LEADER": (0, 125, 255),
            "FOLLOWER": (34, 216, 99),
            "FIRST_TRAIL": (255, 150, 150),
            "LEADER_TRAIL": (150, 150, 255),
            "FOLLOWER_TRAIL": (150, 255, 150),
            "TARGET_FORMATION": ((150, 150, 255),(150, 255, 150)),

            "LIDAR_RAY": (200, 200, 200),
            "COMM_LINE": (135, 70, 205),           
            "HUD_TEXT": (50, 50, 50),
        }

class AgentArg:
    NUM_LEADERS = 3
    NUM_FOLLOWERS = 3
    NUM_AGENTS = NUM_LEADERS + NUM_FOLLOWERS
    GOAL_RADIUS = 2.0
    AGENT_RADIUS = 1.5
    # NOMINAL_CONFIG = np.array([np.array([0.0, 0.0]),
    #                             np.array([-1.0, 1.0])*6*AGENT_RADIUS,
    #                             np.array([-1.0, -1.0])*6*AGENT_RADIUS])
    # LEADER_SPAWN = NOMINAL_CONFIG + np.array([20.0, 0.0])
    r_leader = np.array([[np.sqrt(3), 0], [0, 1], [0, -1]], dtype=np.float32) - np.array([np.sqrt(3), 0], dtype=np.float32)
    r_follower = np.array([[-np.sqrt(3), 2], [-np.sqrt(3), 0], [-np.sqrt(3), -2]], dtype=np.float32) - np.array([np.sqrt(3), 0], dtype=np.float32)
    NOMINAL_CONFIG_BASE = np.vstack([r_leader, r_follower]).astype(np.float32)
    NOMINAL_CONFIG = NOMINAL_CONFIG_BASE * 6 * AGENT_RADIUS # 放大队形
    SPAWN_OFFSET = np.array([50.0, 0.0], dtype=np.float32)
    AGENT_SPAWN = NOMINAL_CONFIG + SPAWN_OFFSET

    LIDAR_NUM_RAYS = 16
    LIDAR_MAX_RANGE = 30
    LIDAR_FOV = 2*np.pi

    MAX_VEL = 15.0
    MIN_VEL = -15.0
    MAX_ACC = 4
    MIN_ACC = -4
    TOL_ERROR = 4*AGENT_RADIUS

    # 仿射变换参数
    MAX_ROT = np.pi
    MIN_ROT = -np.pi
    MAX_SCALE = 1.5
    MIN_SCALE = 0.5
    MAX_SHEAR = 1.5
    MIN_SHEAR = -1.5

    COLLISION_THRESHOLD = AGENT_RADIUS * 4
    DANGER_THRESHOLD = AGENT_RADIUS * 6

    # PD控制真实领导者位移
    KP_LEADER = 0.2 
    KD_LEADER = 0.1

    # 基于应力的控制律
    KP_FOLLOWER = 0.5
    KV_FOLLOWER = 2.0    
    STRESS_MATRIX = np.array([
        [0.3461,    -0.3461,    -0.3461,    0.0,        0.3461,     0.0],
        [-0.3461,   0.6854,     0.0069,     -0.0420,    -0.6015,    0.2973],
        [-0.3461,   0.0069,     0.6853,     0.0420,     -0.0908,    -0.2973],
        [0.0,       -0.0420,    0.0420,     0.0420,     -0.0420,    0.0],
        [0.3461,    -0.6015,    -0.0908,    -0.0420,    0.6855,     -0.2973],
        [0.0,       0.2973,     -0.2973,    0.0,        -0.2973,    0.2973]
    ], dtype=np.float32)

    NEIGHBORS = [
        np.array([1, 2, 4]),
        np.array([0, 2, 3, 4, 5]),
        np.array([0, 1, 3, 4, 5]),
        np.array([1, 2, 4]),
        np.array([0, 1, 2, 3, 5]),
        np.array([1, 2, 4]),
    ]

class RewardArg:
    R_GOAL = 200
    R_MOVE = 0.08
    R_NEAR = 200

    P_CRASH = -200
    P_AVOID = -50

class TrainArg:
    ENV_NAME = "affine_gym_env/AffineEnv"
    SEED = None
    NUM_EP = 60_00 # 训练的总回合数
    EP_MAX_STEP = 350 # 每个回合的最大步数
    LOG_INTERVAL = 10 # 每隔多少个回合打印一次日志.
    GIF_INTERVAL = 500
    GIF_FPS = 30
    NUM_TEST_EP = 5 # 测试回合数   
    SMOOTH = 15 # 奖励曲线平滑窗口大小
    REWARD_SCALE = 2 ** -4
    # SAC 算法超参数
    LR = 3e-4
    ACTOR_LR = LR
    CRITIC_LR = LR
    ALPHA_LR = LR # 温度系数 alpha 的学习率,一般地，温度系数的学习率和网络参数的学习率保持一致
    GAMMA = 0.99 # 折扣因子
    TAU = 0.01 # 软更新因子
    ALPHA_INIT = 0.2 # 初始温度参数 (如果使用自动熵调整，此值会被覆盖)
    BUFFER_SIZE = 10_0_0000 # 经验回放缓冲区容量
    BATCH_SIZE = 256 # 训练批次大小
    LOG_STD_MIN = -10.0
    LOG_STD_MAX = 1.0
    # 网络参数
    ACTOR_HIDDEN_SIZE = 256
    CRITIC_HIDDEN_SIZE = 256




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
    GOAL_RADIUS = 1.0*MAP_SCALE

    CIRCLE_POS = np.array([[25.0, 0.0],[75.0, 0.0]])*MAP_SCALE
    CIRCLE_RADIUS = 2.5*MAP_SCALE
    RECTANGLE_POS = np.array([[50.0, 6.25],[50.0, -6.25]])*MAP_SCALE
    RECTANGLE_SIZE = np.array([25.0, 7.5])*MAP_SCALE

class AgentArg:
    NUM_LEADERS = 3
    GOAL_RADIUS = 1.0
    AGENT_RADIUS = 1.0

    NOMINAL_CONFIG = np.array([np.array([0.0, 0.0]),
                               np.array([-1.0, 1.0])*4*AGENT_RADIUS,
                               np.array([-1.0, -1.0])*4*AGENT_RADIUS])
    LEADER_SPAWN = NOMINAL_CONFIG + np.array([10.0, 0.0])

    LIDAR_NUM_RAYS = 16
    LIDAR_MAX_RANGE = 5.0
    LIDAR_FOV = 2*np.pi

    MAX_VEL = 5.0
    MIN_VEL = -5.0
    MAX_ACC = 1.0
    MIN_ACC = -1.0

    # 仿射变换参数
    MAX_ROT = np.pi
    MIN_ROT = -np.pi
    MAX_SCALE = 2.0
    MIN_SCALE = 0.5
    MAX_SHEAR = 1.0
    MIN_SHEAR = -1.0

    # PID控制真实领导者位移
    KP = 0.1
    KD = 0.1

class RewardArg:
    R_DANGER = -0.1
    R_COLLISION = -0.5
    R_GOAL = 10
    R_MOVE = 0.05

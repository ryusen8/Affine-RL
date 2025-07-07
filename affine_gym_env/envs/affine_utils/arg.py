import numpy as np

class MapArg:
    SCREEN_WIDTH_PX = 1200  # 假设你希望窗口宽度是 1200 像素
    SCREEN_HEIGHT_PX = 600  # 假设你希望窗口高度是 600 像素
    PADDING_PX = 20
    DT = 0.1
    SEED = 0
    MIN_X = 0.0
    MAX_X = 100.0 # [0, 100]
    MIN_Y = -10.0
    MAX_Y = 10.0 # [-10, 10]
    GOAL_POS = np.array([85.0, 0.0])
    GOAL_RADIUS = 1.0
    CIRCLE_POS = np.array([[25.0, 0.0],[75.0, 0.0]])
    CIRCLE_RADIUS = 2.5
    RECTANGLE_POS = np.array([[50.0, 6.25],[50.0, -6.25]])
    RECTANGLE_SIZE = np.array([25.0, 7.5])

class AgentArg:
    NUM_REAL_LEADER = 3
    GOAL_RADIUS = 1.0
    AGENT_RADIUS = 1.0
    VIRTUAL_LEADER_SPAWN = np.array([10.0, 0.0])
    # 正三角形队形
    REAL_LEADER_SPAWN = np.array([VIRTUAL_LEADER_SPAWN+[4*AGENT_RADIUS, 0],
                                  VIRTUAL_LEADER_SPAWN+[2*AGENT_RADIUS, 2*np.sqrt(3)*AGENT_RADIUS],
                                  VIRTUAL_LEADER_SPAWN+[2*AGENT_RADIUS, -2*np.sqrt(3)*AGENT_RADIUS]])
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

    # 比例控制器控制真实领导者位移
    KP_POS = 1
    KP_VEL = 1

class RewardArg:
    R_DANGER = -0.1
    R_COLLISION = -0.5
    R_GOAL = 10
    R_MOVE = 0.2

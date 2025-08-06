import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", message=".*The obs returned.*")
warnings.filterwarnings("ignore", message=".*precision lowered by.*")
import pygame
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random

try: # 训练时用
    from .affine_utils.lidar import Lidar
    from .affine_utils.obstacles import Circle, Rectangle, Line
    from .affine_utils.arg import MapArg, AgentArg, RewardArg, RandomMapArg
except ImportError : # 测试时用
    from affine_utils.lidar import Lidar
    from affine_utils.obstacles import Circle, Rectangle, Line
    from affine_utils.arg import MapArg, AgentArg, RewardArg, RandomMapArg

class AffineEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    def __init__(self, render_mode=None, seed=MapArg.SEED, map_type='random'):
        super().__init__()
        self.map_type = map_type
        # --- 根据 map_type 选择参数 ---
        if self.map_type == 'random':
            self.map_arg = RandomMapArg
            AgentArg.AGENT_SPAWN = AgentArg.NOMINAL_CONFIG + self.map_arg.SPAWN_OFFSET
        else: # static
            self.map_arg = MapArg
            # 静态地图的出生点
            AgentArg.AGENT_SPAWN = AgentArg.NOMINAL_CONFIG + np.array([10.0 * MapArg.MAP_SCALE, 0.0], dtype=np.float32)

        self.dt = MapArg.DT
        self.min_x = self.map_arg.MIN_X
        self.max_x = self.map_arg.MAX_X # [0, 100]
        self.min_y = self.map_arg.MIN_Y
        self.max_y = self.map_arg.MAX_Y # [-10, 10]

        self.screen_width = MapArg.SCREEN_WIDTH_PX
        # self.screen_height = MapArg.SCREEN_HEIGHT_PX
        if self.map_type == 'random':
            # self.screen_height = int(MapArg.SCREEN_HEIGHT_PX * (self.map_arg.MAX_Y - self.map_arg.MIN_Y) / (MapArg.MAX_Y - MapArg.MIN_Y))
            self.screen_height = RandomMapArg.SCREEN_HEIGHT_PX
        else:
            self.screen_height = MapArg.SCREEN_HEIGHT_PX

        self.padding_px = 20 # 屏幕边距

        # 有效绘制区域的尺寸
        self.effective_screen_width = self.screen_width - 2 * self.padding_px
        self.effective_screen_height = self.screen_height - 2 * self.padding_px

        self.world_width = self.max_x - self.min_x
        self.world_height = self.max_y - self.min_y

        # 计算两个方向上的缩放因子
        scale_factor_x_needed = self.effective_screen_width / self.world_width
        scale_factor_y_needed = self.effective_screen_height / self.world_height

        # 实际使用的缩放因子：取两者中较小的一个，以确保所有内容都可见且不拉伸
        self.scale_factor = min(scale_factor_x_needed, scale_factor_y_needed)

        # 计算实际缩放后的世界内容在有效区域中的像素尺寸
        self.scaled_world_width_px = self.world_width * self.scale_factor
        self.scaled_world_height_px = self.world_height * self.scale_factor

        # 计算内容在有效区域中居中所需的偏移量（额外留白）
        self.horizontal_extra_padding_px = (self.effective_screen_width - self.scaled_world_width_px) / 2
        self.vertical_extra_padding_px = (self.effective_screen_height - self.scaled_world_height_px) / 2

        self.circle_radius = MapArg.CIRCLE_RADIUS
        self.circle_pos = MapArg.CIRCLE_POS
        self.rectangle_size = MapArg.RECTANGLE_SIZE
        self.rectangle_pos = MapArg.RECTANGLE_POS
        self.static_obstacles = [] # 初始化为空列表
        # self.static_obstacles = list(np.array([
        #     Rectangle(center=np.array(self.rectangle_pos[0], dtype=np.float32), size=np.array(self.rectangle_size, dtype=np.float32), angle=0.0),
        #     Rectangle(center=np.array(self.rectangle_pos[1], dtype=np.float32), size=np.array(self.rectangle_size, dtype=np.float32), angle=0.0),
        #     Circle(center=np.array(self.circle_pos[0], dtype=np.float32), radius=self.circle_radius),
        #     Circle(center=np.array(self.circle_pos[1], dtype=np.float32), radius=self.circle_radius),
        #     Rectangle(center=np.array([self.max_x/2, self.max_y+2.5], dtype=np.float32), # 上边界
        #                 size=np.array([self.max_x, 5], dtype=np.float32), angle=0.0),
        #     Rectangle(center=np.array([self.max_x/2, self.min_y-2.5], dtype=np.float32), # 下边界
        #                 size=np.array([self.max_x, 5], dtype=np.float32), angle=0.0),
        #     Rectangle(center=np.array([self.min_x-2.5, 0.0], dtype=np.float32), # 左边界
        #                 size=np.array([5, self.max_y - self.min_y], dtype=np.float32), angle=0.0),
        #     Rectangle(center=np.array([self.max_x+2.5, 0.0], dtype=np.float32), # 右边界
        #                 size=np.array([5, self.max_y - self.min_y], dtype=np.float32), angle=0.0),         
        # ], dtype=object))
        # self.num_obstacles = 8
        self.goal_radius = self.map_arg.GOAL_RADIUS
        self.goal_pos = self.map_arg.GOAL_POS

        self.num_agents = AgentArg.NUM_AGENTS
        self.num_leaders = AgentArg.NUM_LEADERS
        self.num_followers = AgentArg.NUM_FOLLOWERS
        # self.agent_spawn = np.array(AgentArg.AGENT_SPAWN, dtype=np.float32)
        self.agent_spawn = AgentArg.AGENT_SPAWN.copy().astype(np.float32)
        self.agent_pos = self.agent_spawn.copy()
        self.agent_radius = AgentArg.AGENT_RADIUS
        # 为所有智能体创建Lidar的障碍物表示
        self.agent_obstacle_circles = [Circle(center=np.array(pos, dtype=np.float32), radius=self.agent_radius) 
                                        for pos in self.agent_pos]

        self.lidar_num_rays = AgentArg.LIDAR_NUM_RAYS
        self.lidar_max_range = AgentArg.LIDAR_MAX_RANGE
        self.lidar_fov = AgentArg.LIDAR_FOV
        self.lidar = Lidar(n_rays=self.lidar_num_rays, max_range=self.lidar_max_range, fov=self.lidar_fov)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.action_space = spaces.Dict({
                "acc": spaces.Box(low=np.array([AgentArg.MIN_ACC, AgentArg.MIN_ACC], dtype=np.float32), 
                                    high=np.array([AgentArg.MAX_ACC, AgentArg.MAX_ACC], dtype=np.float32)),
                "rot": spaces.Box(low=np.float32(AgentArg.MIN_ROT), 
                                    high=np.float32(AgentArg.MAX_ROT)),
                "scale": spaces.Box(low=np.array([AgentArg.MIN_SCALE, AgentArg.MIN_SCALE], dtype=np.float32), 
                                    high=np.array([AgentArg.MAX_SCALE, AgentArg.MAX_SCALE], dtype=np.float32)),
                "shear": spaces.Box(low=np.array([AgentArg.MIN_SHEAR, AgentArg.MIN_SHEAR], dtype=np.float32), 
                                    high=np.array([AgentArg.MAX_SHEAR, AgentArg.MAX_SHEAR], dtype=np.float32)),
                },
                seed=seed)

            self.observation_space = spaces.Dict({
                # "dist_to_goal": spaces.Box(low=np.float32(0.0), high=np.float32(np.sqrt((self.max_x - self.min_x)**2 + (self.max_y - self.min_y)**2)), shape=(1,), dtype=np.float32),
                # "leader1_vel": spaces.Box(low=np.float32(-AgentArg.MAX_VEL), high=np.float32(AgentArg.MAX_VEL), shape=(2,), dtype=np.float32), 
                "other_rel_pos": spaces.Box(low=np.float32(0.0), high=np.float32(20.0), shape=(self.num_agents-1, 2), dtype=np.float32),
                "other_rel_vel": spaces.Box(low=np.float32(AgentArg.MIN_VEL), high=np.float32(AgentArg.MAX_VEL), shape=(self.num_agents-1, 2), dtype=np.float32),
                "agents_meas": spaces.Box(low=np.float32(0.0), high=np.float32(self.lidar_max_range), shape=(self.num_agents, self.lidar_num_rays), dtype=np.float32),
                },
                seed=seed)
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.font = None
        self.step_count = 0
        self.last_action = {}
        self.last_leader0_acc = np.zeros(2)

        # 定义颜色
        self.COLORS = MapArg.COLOR

        # 轨迹缓冲区设置
        self.trail_length = 100 # 轨迹点数
        # 为每个实际领导者创建一个独立的轨迹缓冲区
        self.agent_trails = [deque(maxlen=self.trail_length) for _ in range(self.num_agents)]
        
    def _generate_static_obstacles(self):
        self.static_obstacles = list(np.array([
            Rectangle(center=np.array(MapArg.RECTANGLE_POS[0], dtype=np.float32), size=np.array(MapArg.RECTANGLE_SIZE, dtype=np.float32), angle=0.0),
            Rectangle(center=np.array(MapArg.RECTANGLE_POS[1], dtype=np.float32), size=np.array(MapArg.RECTANGLE_SIZE, dtype=np.float32), angle=0.0),
            Circle(center=np.array(MapArg.CIRCLE_POS[0], dtype=np.float32), radius=MapArg.CIRCLE_RADIUS),
            Circle(center=np.array(MapArg.CIRCLE_POS[1], dtype=np.float32), radius=MapArg.CIRCLE_RADIUS),
        ], dtype=object))

    def _generate_random_obstacles(self):
        self.static_obstacles = []
        block_size = self.map_arg.OBSTACLE_BLOCK_SIZE
        
        # 定义几种俄罗斯方块形状（相对于簇中心的局部坐标）
        tetris_shapes = {
            'I': [np.array([0, -1.5]), np.array([0, -0.5]), np.array([0, 0.5]), np.array([0, 1.5])],
            'O': [np.array([-0.5, -0.5]), np.array([0.5, -0.5]), np.array([-0.5, 0.5]), np.array([0.5, 0.5])],
            'T': [np.array([-1, 0]), np.array([0, 0]), np.array([1, 0]), np.array([0, -1])],
            'L': [np.array([-1, -1]), np.array([-1, 0]), np.array([0, 0]), np.array([1, 0])],
            'S': [np.array([-1, 0]), np.array([0, 0]), np.array([0, -1]), np.array([1, -1])],
        }
        
        shape_keys = list(tetris_shapes.keys())
        
        # 在安全区域内生成障碍物簇
        safe_margin_x = 2 * (self.goal_pos[0] - self.agent_spawn[0][0]) / 10
        spawn_x_min = self.agent_spawn[:, 0].min() + safe_margin_x
        spawn_x_max = self.goal_pos[0] - safe_margin_x
        x_total_width = spawn_x_max - spawn_x_min
        x_segment_length = x_total_width / self.map_arg.NUM_OBSTACLE_CLUSTERS    

        for i in range(self.map_arg.NUM_OBSTACLE_CLUSTERS):
            # 随机选择一个形状
            shape_key = random.choice(shape_keys)
            blocks = tetris_shapes[shape_key]
            
            # # 随机选择位置和角度
            # center_x = random.uniform(spawn_x_min, spawn_x_max)
            # center_y = random.uniform(self.min_y + block_size[1]*2, self.max_y - block_size[1]*2)
            # 计算当前段的X轴范围
            segment_start_x = spawn_x_min + i * x_segment_length
            segment_end_x = segment_start_x + x_segment_length
            
            # 在该X段内随机选择一个X坐标
            center_x = random.uniform(segment_start_x, segment_end_x)
            
            # Y坐标仍然在整个有效Y轴范围内随机选择
            center_y = random.uniform(self.min_y + block_size[1]*2, self.max_y - block_size[1]*2)

            cluster_center = np.array([center_x, center_y], dtype=np.float32)
            angle = random.uniform(0, 2 * np.pi)
            
            rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle),  np.cos(angle)]])
            
            # 创建构成簇的矩形
            for block_local_pos in blocks:
                # 缩放并旋转局部位置
                rotated_pos = (block_local_pos * block_size) @ rot_matrix.T
                # 计算世界坐标
                rect_center = cluster_center + rotated_pos
                self.static_obstacles.append(
                    Rectangle(center=rect_center, size=block_size, angle=angle)
                )

    # --- 新增：生成边界 ---
    def _generate_boundaries(self):
        # 添加边界矩形
        boundary_thickness = 5.0 # 边界厚度
        self.static_obstacles.extend([
            Rectangle(center=np.array([(self.max_x+self.min_x)/2, self.max_y+boundary_thickness/2]), size=np.array([self.max_x-self.min_x, boundary_thickness]), angle=0.0), # 上
            Rectangle(center=np.array([(self.max_x+self.min_x)/2, self.min_y-boundary_thickness/2]), size=np.array([self.max_x-self.min_x, boundary_thickness]), angle=0.0), # 下
            Rectangle(center=np.array([self.min_x-boundary_thickness/2, 0]), size=np.array([boundary_thickness, self.max_y-self.min_y]), angle=0.0), # 左
            Rectangle(center=np.array([self.max_x+boundary_thickness/2, 0]), size=np.array([boundary_thickness, self.max_y-self.min_y]), angle=0.0), # 右
        ])

    def _get_obs(self):
        meas = []
        for i in range(self.num_agents):
            obstacles_for_this_agent = self.static_obstacles + \
                                        self.agent_obstacle_circles[:i] + \
                                        self.agent_obstacle_circles[i+1:]
            dist = self.lidar.scan(agent_pos=self.agent_pos[i], 
                                    agent_angle=0.0, 
                                    obstacles=obstacles_for_this_agent)
            meas.append(dist.astype(np.float32))

        return {
            # "dist_to_goal": np.linalg.norm(self.agent_pos[0] - self.goal_pos).reshape(1,).astype(np.float32),
            # "leader1_vel": self.agent_vel[0].astype(np.float32),
            "other_rel_pos": (self.agent_pos[1:] - self.agent_pos[0]).astype(np.float32),
            "other_rel_vel": (self.agent_vel[1:] - self.agent_vel[0]).astype(np.float32),
            "agents_meas": np.array(meas, dtype=np.float32),
        }
    
    def _get_info(self):
        return {"finish":self.finish, "fail":self.fail}

    def reset(self, seed=MapArg.SEED, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # --- 根据地图类型生成障碍物 ---
        if self.map_type == 'random':
            self._generate_random_obstacles()
        else:
            self._generate_static_obstacles()

        # 统一生成边界
        self._generate_boundaries()
        self.num_obstacles = len(self.static_obstacles)

        self.finish = False
        self.fail = False
        self.collide_count = 0
        self.crash = False
        self.reward_info = {}
        self.agent_pos = self.agent_spawn.copy().astype(np.float32)
        self.agent_acc = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.agent_vel = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.old_pos = self.agent_pos.copy().astype(np.float32)
        self.target_pos = self.agent_pos.copy().astype(np.float32) # 仿射变换的目标

        for i in range(self.num_agents):
            self.agent_obstacle_circles[i].center = self.agent_pos[i]
            # 重置轨迹缓冲区
            self.agent_trails[i].clear()
            self.agent_trails[i].append(self.agent_pos[i].copy()) # 保存初始位置

        # <--- 新增: 重置步数和上一动作信息 ---
        self.step_count = 0
        self.last_action = {"acc": np.zeros(2, dtype=np.float32), "rot": np.float32(0.0), "scale": np.ones(2, dtype=np.float32), "shear": np.zeros(2, dtype=np.float32)}
        self.last_leader0_acc.fill(0.0) # 重置为0
        self.current_jerk = 0.0

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._init_render()

        return obs, info

    def step(self, action):
        self.step_count += 1
        self.last_action = action
        self.old_pos = self.agent_pos.copy()

        acc = np.array(action["acc"], dtype=np.float32)
        rot = np.float32(action["rot"])
        scale = np.array(action["scale"], dtype=np.float32)
        shear = np.array(action["shear"], dtype=np.float32)
        acc = np.clip(acc, AgentArg.MIN_ACC, AgentArg.MAX_ACC).astype(np.float32)
        rot = np.clip(rot, AgentArg.MIN_ROT, AgentArg.MAX_ROT).astype(np.float32)
        scale = np.clip(scale, AgentArg.MIN_SCALE, AgentArg.MAX_SCALE).astype(np.float32)
        shear = np.clip(shear, AgentArg.MIN_SHEAR, AgentArg.MAX_SHEAR).astype(np.float32)

        self.current_jerk = np.linalg.norm(acc - self.last_leader0_acc)
        self.last_leader0_acc = acc.copy().astype(np.float32)

        # 更新领导者1的状态
        self.agent_acc[0] = acc
        self.agent_vel[0] += self.agent_acc[0] * self.dt
        self.agent_pos[0] += self.agent_vel[0] * self.dt
        
        # 计算仿射变换目标位置
        self.target_pos = self.apply_affine_transform(AgentArg.NOMINAL_CONFIG, rot, np.zeros(2, dtype=np.float32), scale, shear) + self.agent_pos[0]

        # 更新领导者2,3的状态
        self.agent_acc[1:self.num_leaders] = AgentArg.KP_LEADER * (self.target_pos[1:self.num_leaders] - self.agent_pos[1:self.num_leaders]) + \
                                            AgentArg.KD_LEADER * (self.agent_vel[0] - self.agent_vel[1:self.num_leaders])
        self.agent_vel[1:self.num_leaders] += self.agent_acc[1:self.num_leaders] * self.dt
        self.agent_pos[1:self.num_leaders] += self.agent_vel[1:self.num_leaders] * self.dt
        
        # 更新跟随者状态
        follower_indices = range(self.num_leaders, self.num_agents)
        temp_follower_acc = np.zeros((self.num_followers, 2), dtype=np.float32)

        for i in follower_indices:
            sum_term = np.zeros(2)
            gamma_i = 0.0
            # 遍历智能体i的邻居j
            for j in AgentArg.NEIGHBORS[i]:
                omega_ij = -AgentArg.STRESS_MATRIX[i, j]
                current_rel_pos = self.agent_pos[i] - self.agent_pos[j]
                vel_diff = self.agent_vel[i] - self.agent_vel[j]
                v_j_dot = self.agent_acc[j]

                term = omega_ij * (AgentArg.KP_FOLLOWER * (current_rel_pos) + AgentArg.KV_FOLLOWER * vel_diff - v_j_dot)
                sum_term += term
                gamma_i += omega_ij

            if abs(gamma_i) > 1e-6: # 防止除以零
                # 存储计算出的新加速度
                temp_follower_acc[i - self.num_leaders] = -sum_term / gamma_i

        # 同时更新所有跟随者的状态
        self.agent_acc[follower_indices] = temp_follower_acc.astype(np.float32)
        self.agent_vel[follower_indices] += self.agent_acc[follower_indices] * self.dt
        self.agent_pos[follower_indices] += self.agent_vel[follower_indices] * self.dt
        self.agent_pos = np.clip(self.agent_pos, [self.min_x, self.min_y], [self.max_x, self.max_y]).astype(np.float32)
        self.agent_vel = np.clip(self.agent_vel, AgentArg.MIN_VEL, AgentArg.MAX_VEL).astype(np.float32)

        for i in range(self.num_agents):
            self.agent_trails[i].append(self.agent_pos[i].copy()) 
            self.agent_obstacle_circles[i].center = self.agent_pos[i]

        obs = self._get_obs()
        rew, done = self.reward(obs)
        if self.render_mode == "human":
            self.render()
        info = self._get_info()
        return obs, rew, done, False, info

    # 在 affine_env.py 的 AffineEnv 类中

    def reward(self, obs):
        done = False

        total_reward = 0.0
        rew_goal = 0.0
        rew_near = 0.0
        rew_move = 0.0
        pen_crash = 0.0
        pen_avoid = 0.0
        pen_stay = 0.0
        # 计算到达目标的奖励
        leader_center = sum(self.agent_pos[:3]) / 3
        # dist_to_goal = obs["dist_to_goal"]
        dist_to_goal = np.linalg.norm(leader_center - self.goal_pos)
        if dist_to_goal < self.goal_radius:
            rew_goal = RewardArg.R_GOAL
            print(f"\nGoal reached at step {self.step_count}!")
            return rew_goal, True

        # 计算移动奖励
        old_leader_center = sum(self.old_pos[:3]) / 3
        old_dist_to_goal = np.linalg.norm(old_leader_center - self.goal_pos)
        rew_move = RewardArg.R_MOVE * (old_dist_to_goal - dist_to_goal)

        pen_stay = RewardArg.P_STAY * np.tanh( 0.005*(dist_to_goal-390) )
        # rew_near = RewardArg.R_NEAR * np.tanh(1/dist_to_goal)
        # rew_near = RewardArg.R_NEAR * np.tanh(dist_to_goal)
        
        # 计算碰撞惩罚
        min_dist_to_any_obstacle = np.min(obs["agents_meas"])
        if min_dist_to_any_obstacle < 2*AgentArg.AGENT_RADIUS:
            pen_crash = RewardArg.P_CRASH
            self.fail = True # 标记失败类型为碰撞
            done = True
            total_reward += pen_crash
            # self.reward_info = {"total_reward": total_reward, "pen_crash": pen_crash}
            return total_reward, done
        
        for i in range(self.num_agents):
            min_dist = np.min(obs["agents_meas"][i])
            if min_dist < AgentArg.DANGER_THRESHOLD:
                pen_avoid += 0.5 * RewardArg.P_AVOID * (1 / min_dist - 1 / AgentArg.DANGER_THRESHOLD)
        # for i in range(self.num_agents):
        #     num_static_obstacles = len(self.static_obstacles)

        #     min_dist_to_obstacles = np.min(obs["agents_meas"][i][:num_static_obstacles])
        #     min_dist_to_agents = np.min(obs["agents_meas"][i][num_static_obstacles:])
            
        #     # # 到静态障碍物的危险程度
        #     # if min_dist_to_obstacles < AgentArg.DANGER_THRESHOLD:
        #     #     pen_avoid += 0.5 * RewardArg.P_AVOID_OBST * (1 / min_dist_to_obstacles - 1 / AgentArg.DANGER_THRESHOLD) # ** 2                   
        #     # if min_dist_to_agents < 0.5*AgentArg.DANGER_THRESHOLD:
        #     #     pen_avoid += 0.5 * RewardArg.P_AVOID_AGENT * (1 / min_dist_to_agents - 1 / AgentArg.DANGER_THRESHOLD) # ** 2
        #     pen_avoid = 0.0
        #     # 碰撞惩罚
        #     if min_dist_to_obstacles < AgentArg.COLLISION_THRESHOLD or min_dist_to_agents < 0.5*AgentArg.COLLISION_THRESHOLD:
        #         pen_crash = RewardArg.P_CRASH
        #         done = True
        
        total_reward = np.float32(rew_goal + rew_move + rew_near + pen_crash + pen_avoid + pen_stay)

        self.reward_info = {
            "total_reward": total_reward,
            "rew_goal": rew_goal,
            "rew_near": rew_near,
            "rew_move": rew_move,
            "pen_stay": pen_stay,
            "pen_crash": pen_crash,
            "pen_avoid": pen_avoid,
        }
        return total_reward, done

    def _init_render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("AffineEnv")
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.font is None:
            pygame.font.init() # 确保字体模块已初始化
            self.font = pygame.font.SysFont("Arial", 18) # 使用常见的字体和大小
            
    def render(self):
        if self.screen is None and (self.render_mode == "human" or self.render_mode == "rgb_array"):
            self._init_render()

        canvas = self.screen
        canvas.fill(self.COLORS["BACKGROUND"])

        def world_to_screen(pos):
            # Pygame Y轴向下为正，所以需要翻转
            x_screen = int((pos[0] - self.min_x) * self.scale_factor +
                            self.padding_px + self.horizontal_extra_padding_px)
            y_screen = int(self.scaled_world_height_px - (pos[1] - self.min_y) * self.scale_factor +
                self.padding_px + self.vertical_extra_padding_px)
            return (x_screen, y_screen) # 返回元组

        # --- 绘制障碍物 ---
        for obs_obj in self.static_obstacles:
            if isinstance(obs_obj, Circle):
                center_screen = world_to_screen(obs_obj.center)
                radius_screen = int(obs_obj.radius * self.scale_factor)
                pygame.draw.circle(canvas, self.COLORS["OBSTACLE"], center_screen, radius_screen)
                
            elif isinstance(obs_obj, Rectangle):
                rect_center_screen = world_to_screen(obs_obj.center)
                half_width_screen = int(obs_obj.size[0] / 2 * self.scale_factor)
                half_height_screen = int(obs_obj.size[1] / 2 * self.scale_factor)

                corners_local = np.array([
                    [-half_width_screen, -half_height_screen],
                    [ half_width_screen, -half_height_screen],
                    [ half_width_screen,  half_height_screen],
                    [-half_width_screen,  half_height_screen]
                ])

                angle = -obs_obj.angle # 假设 Rectangle.angle 是逆时针
                rot_matrix = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]
                ])
                rotated_corners = (corners_local @ rot_matrix.T) + rect_center_screen
                pygame.draw.polygon(canvas, self.COLORS["OBSTACLE"], rotated_corners)

        # --- 绘制终点 ---
        goal_center_screen = world_to_screen(self.goal_pos)
        goal_radius_screen = int(self.goal_radius * self.scale_factor)
        pygame.draw.circle(canvas, self.COLORS["GOAL"], goal_center_screen, goal_radius_screen, 2)

        # --- 绘制通信拓扑 ---
        for i in range(self.num_agents):
            for j in AgentArg.NEIGHBORS[i]:
                if j > i: # 防止重复绘制
                    start_pos = world_to_screen(self.agent_pos[i])
                    end_pos = world_to_screen(self.agent_pos[j])
                    pygame.draw.line(canvas, self.COLORS["COMM_LINE"], start_pos, end_pos, 1)

        # --- 绘制轨迹 ---
        # 实际领导者轨迹
        for i in range(self.num_agents):
            if len(self.agent_trails[i]) > 1:
                points_screen = [world_to_screen(p) for p in self.agent_trails[i]]
                for j in range(len(points_screen) - 1):
                    alpha_factor = (j + 1) / self.trail_length
                    # current_color = tuple([int(c * alpha_factor + (255 * (1 - alpha_factor))) for c in self.COLORS["LEADER_TRAIL"]]) if i!=0 else tuple([int(c * alpha_factor + (255 * (1 - alpha_factor))) for c in self.COLORS["FIRST_TRAIL"]])
                    current_color = tuple([int(c * alpha_factor + (255 * (1 - alpha_factor))) for c in self.COLORS["FIRST_TRAIL"]]) if i==0 else \
                                    tuple([int(c * alpha_factor + (255 * (1 - alpha_factor))) for c in self.COLORS["LEADER_TRAIL"]]) if i==1 or i==2 else \
                                    tuple([int(c * alpha_factor + (255 * (1 - alpha_factor))) for c in self.COLORS["FOLLOWER_TRAIL"]])
                    pygame.draw.line(canvas, current_color, points_screen[j], points_screen[j+1], 2)
        
        # --- 绘制期望队形 ---
        if hasattr(self, 'target_pos') and self.target_pos is not None:
            agent_radius_screen = int(self.agent_radius * self.scale_factor)
            for i in range(self.num_agents-1):
                target_pos_screen = world_to_screen(self.target_pos[i+1])
                if i<self.num_leaders-1:
                    pygame.draw.circle(canvas, self.COLORS["TARGET_FORMATION"][0], target_pos_screen, agent_radius_screen, 2)
                else:
                    pygame.draw.circle(canvas, self.COLORS["TARGET_FORMATION"][1], target_pos_screen, agent_radius_screen, 2)

        # --- 绘制实际领导者 ---
        agent_radius_screen = int(self.agent_radius * self.scale_factor)
        lidar_scans = self._get_obs()["agents_meas"]
        for i in range(self.num_agents):
            agent_pos = self.agent_pos[i]
            agent_pos_screen = world_to_screen(self.agent_pos[i])

            # 绘制 Lidar 射线
            # lidar_distances = lidar_scans[i]
            # world_angles = self.lidar.relative_angles + 0.0 # 假设智能体朝向是0
            # for j in range(self.lidar_num_rays):
            #     angle = world_angles[j]
            #     distance = lidar_distances[j]
            #     ray_end_world = agent_pos + np.array([np.cos(angle), np.sin(angle)]) * distance
            #     ray_end_screen = world_to_screen(ray_end_world)
            #     pygame.draw.line(canvas, self.COLORS["LIDAR_RAY"], agent_pos_screen, ray_end_screen, 1)

            agent_color = self.COLORS["FIRST"] if i == 0 else \
                            self.COLORS["LEADER"] if i==1 or i==2 else \
                            self.COLORS["FOLLOWER"]            
            pygame.draw.circle(canvas, agent_color, agent_pos_screen, agent_radius_screen)
        
        # --- 在屏幕左上角绘制文本信息 ---
        if hasattr(self, 'last_action') and self.last_action:
            # 从 last_action 字典中获取动作分量
            acc = self.last_action.get("acc", [0, 0])
            rot_val = self.last_action.get("rot", 0)
            scale = self.last_action.get("scale", [1, 1])
            shear = self.last_action.get("shear", [0, 0])

            # 关键修复：如果 rot_val 是Numpy类型，使用 .item() 转换成标准Python数字
            if isinstance(rot_val, np.ndarray):
                rot = rot_val.item()
            else:
                rot = rot_val            
            # 创建要显示的文本列表
            info_texts = [
                f"Step: {self.step_count}",
                f"Acc: [{acc[0]:.2f}, {acc[1]:.2f}]",
                f"Rot: {rot:.2f}",
                f"Scale: [{scale[0]:.2f}, {scale[1]:.2f}]",
                f"Shear: [{shear[0]:.2f}, {shear[1]:.2f}]"
            ]

            # 动态生成奖励和惩罚文本，自动适应 reward_info 的内容
            reward_keys = [k for k in self.reward_info if k.startswith("rew_") or k.startswith("tot_")]
            penalty_keys = [k for k in self.reward_info if k.startswith("pen_")]

            rew_texts = [f"{k}: {self.reward_info[k]:.3f}" for k in reward_keys]
            pen_texts = [f"{k}: {self.reward_info[k]:.3f}" for k in penalty_keys]

            # 逐行渲染并绘制到画布上
            y_offset = 10
            for text in info_texts:
                text_surface = self.font.render(text, True, self.COLORS["HUD_TEXT"])
                canvas.blit(text_surface, (10, y_offset))
                y_offset += 22
            y_offset = 10
            for text in rew_texts:
                text_surface = self.font.render(text, True, self.COLORS["HUD_TEXT"])
                canvas.blit(text_surface, (200, y_offset))
                y_offset += 22
            y_offset = 10
            for text in pen_texts:
                text_surface = self.font.render(text, True, self.COLORS["HUD_TEXT"])
                canvas.blit(text_surface, (400, y_offset))
                y_offset += 22

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

    def apply_affine_transform(self,
        agents_coords: np.ndarray,
        rot: 0.0,
        trans: np.ndarray = np.array([0.0, 0.0], dtype=np.float32),
        scale: np.ndarray = np.array([1.0, 1.0], dtype=np.float32),
        shear: np.ndarray = np.array([0.0, 0.0], dtype=np.float32)
    ) -> np.ndarray:
        """
        对智能体的坐标进行仿射变换。

        Args:
            agents_coords: 一个 (N, 2) 的 NumPy 数组，表示 N 个智能体的 (x, y) 坐标。
            rot: 旋转角度（弧度）。
            trans: 一个 (2,) 的 NumPy 数组，表示平移向量 [tx, ty]。
            scale: 一个 (2,) 的 NumPy 数组，表示缩放因子 [sx, sy]。
            shear: 一个 (2,) 的 NumPy 数组，表示剪切因子 [shx, shy]。

        Returns:
            一个 (N, 2) 的 NumPy 数组，表示变换后的智能体坐标。
        """
            # --- 新增：健壮性处理 ---
        # 检查 rot 是否为 NumPy 数组，如果是，则提取其中的标量值
        if isinstance(rot, np.ndarray):
            rot = rot.item() # .item() 从单元素数组中提取标量

        # 检查 scale 和 shear，确保它们是扁平的 (2,) 向量，而不是 (1, 2) 或 (2, 1)
        scale = np.ravel(scale).astype(np.float32)
        shear = np.ravel(shear).astype(np.float32)
        # --- 结束新增 ---

        num_agents = agents_coords.shape[0]
        coords_homogeneous = np.hstack((agents_coords.astype(np.float32), np.ones((num_agents, 1), dtype=np.float32)))

        # --- 优化点：直接计算组合变换矩阵 (这部分保持不变) ---
        cos_r = np.cos(rot) # 现在 rot 保证是标量，cos_r 和 sin_r 也是标量
        sin_r = np.sin(rot)
        sx, sy = scale
        shx, shy = shear
        tx, ty = trans

        m11 = cos_r * sx + sin_r * shy * sx
        m12 = cos_r * shx * sy - sin_r * sy
        m13 = tx
        
        m21 = sin_r * sx - cos_r * shy * sx
        m22 = sin_r * shx * sy + cos_r * sy
        m23 = ty

        transformation_matrix = np.array([
            [m11, m12, m13],
            [m21, m22, m23],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        transformed_coords_homogeneous = (transformation_matrix @ coords_homogeneous.T).T
        transformed_coords = transformed_coords_homogeneous[:, :2].astype(np.float32)

        return transformed_coords

if __name__ == "__main__":
    import time

    # 创建环境实例，启用 human 渲染模式
    env = AffineEnv(render_mode="human")

    obs, info = env.reset()

    num_steps = 250 # 运行的步数
    total_reward = 0

    for step_count in range(num_steps):
        # 随机采样一个动作
        action = env.action_space.sample()

        # 执行一步
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # 打印信息
        if step_count % 50 == 0:
            print(f"\nStep: {step_count}")
            print(f"Reward: {reward:.2f}")
            print(f"Total Reward: {total_reward:.2f}")

        # 如果达到终止条件
        if terminated or truncated:
            print(f"\nEpisode finished at step {step_count} with total reward: {total_reward:.2f}")
            obs, info = env.reset() # 重置环境
            total_reward = 0
            time.sleep(1) # 暂停一下，方便观察重置

    env.close() # 关闭渲染窗口
    print("Simulation finished.")
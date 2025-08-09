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
    def __init__(self, seed=MapArg.SEED, map_type='random', device='cuda'):
        super().__init__()

        #* 随机地图和静态地图切换
        self.map_type = map_type
        if self.map_type == 'random':
            self.map_arg = RandomMapArg
            AgentArg.AGENT_SPAWN = AgentArg.NOMINAL_CONFIG + self.map_arg.SPAWN_OFFSET
        else:
            self.map_arg = MapArg
            AgentArg.AGENT_SPAWN = AgentArg.NOMINAL_CONFIG + np.array([10.0 * MapArg.MAP_SCALE, 0.0], dtype=np.float32)

        #* 地图参数
        self.dt = MapArg.DT
        self.min_x = self.map_arg.MIN_X
        self.max_x = self.map_arg.MAX_X
        self.min_y = self.map_arg.MIN_Y
        self.max_y = self.map_arg.MAX_Y

        #* 障碍物和目标参数
        self.circle_radius = MapArg.CIRCLE_RADIUS
        self.circle_pos = MapArg.CIRCLE_POS
        self.rectangle_size = MapArg.RECTANGLE_SIZE
        self.rectangle_pos = MapArg.RECTANGLE_POS
        self.static_obstacles = [] # 初始化为空列表
        self.goal_radius = self.map_arg.GOAL_RADIUS
        self.goal_pos = self.map_arg.GOAL_POS

        #* 智能体参数
        self.num_agents = AgentArg.NUM_AGENTS
        self.num_leaders = AgentArg.NUM_LEADERS
        self.num_followers = AgentArg.NUM_FOLLOWERS
        self.agent_spawn = AgentArg.AGENT_SPAWN.copy().astype(np.float32)
        self.agent_radius = AgentArg.AGENT_RADIUS
        # 将每个智能体表示为一个Circle对象，方便碰撞检测
        self.agent_obstacle_circles = [Circle(center=np.array(pos, dtype=np.float32), radius=self.agent_radius) 
                                        for pos in self.agent_spawn]

        #* 雷达参数
        self.lidar_num_rays = AgentArg.LIDAR_NUM_RAYS
        self.lidar_max_range = AgentArg.LIDAR_MAX_RANGE
        self.lidar_fov = AgentArg.LIDAR_FOV
        # self.lidar = Lidar(n_rays=self.lidar_num_rays, max_range=self.lidar_max_range, fov=self.lidar_fov)
        self.lidar = Lidar(n_rays=self.lidar_num_rays, max_range=self.lidar_max_range, fov=self.lidar_fov, device=device)

        #* 渲染参数
        if self.map_type == 'random':
            self.screen_height = RandomMapArg.SCREEN_HEIGHT_PX
        else:
            self.screen_height = MapArg.SCREEN_HEIGHT_PX
        self.screen_width = MapArg.SCREEN_WIDTH_PX
        self.padding_px = 20 # 屏幕边距
        # 定义绘制区域
        self.effective_screen_width = self.screen_width - 2 * self.padding_px
        self.effective_screen_height = self.screen_height - 2 * self.padding_px
        self.world_width = self.max_x - self.min_x
        self.world_height = self.max_y - self.min_y
        # 计算缩放因子
        scale_factor_x_needed = self.effective_screen_width / self.world_width
        scale_factor_y_needed = self.effective_screen_height / self.world_height
        self.scale_factor = min(scale_factor_x_needed, scale_factor_y_needed)
        # 计算实际缩放后的世界内容在有效区域中的像素尺寸
        self.scaled_world_width_px = self.world_width * self.scale_factor
        self.scaled_world_height_px = self.world_height * self.scale_factor
        # 计算内容在有效区域中居中所需的偏移量（额外留白）
        self.horizontal_extra_padding_px = (self.effective_screen_width - self.scaled_world_width_px) / 2
        self.vertical_extra_padding_px = (self.effective_screen_height - self.scaled_world_height_px) / 2

        self.render_fps = 30
        self.screen = None
        self.clock = None
        self.font = None
        self.COLORS = MapArg.COLOR
        # 轨迹设置
        self.trail_length = 100 # 轨迹点数
        self.agent_trails = [deque(maxlen=self.trail_length) for _ in range(self.num_agents)]

        #* 动作空间和观测空间
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
                }, seed=seed)
            self.observation_space = spaces.Dict({
                #? 是否有效
                "vector_to_goal": spaces.Box(low=-self.world_width, high=self.world_width, shape=(2,), dtype=np.float32),
                "leader1_vel": spaces.Box(low=np.float32(-AgentArg.MAX_VEL), high=np.float32(AgentArg.MAX_VEL), shape=(2,), dtype=np.float32), 
                
                "other_rel_pos": spaces.Box(low=np.float32(0.0), high=np.float32(20.0), shape=(self.num_agents-1, 2), dtype=np.float32),
                "other_rel_vel": spaces.Box(low=np.float32(AgentArg.MIN_VEL), high=np.float32(AgentArg.MAX_VEL), shape=(self.num_agents-1, 2), dtype=np.float32),
                "agents_meas": spaces.Box(low=np.float32(0.0), high=np.float32(self.lidar_max_range), shape=(self.num_agents, self.lidar_num_rays), dtype=np.float32),
                }, seed=seed)
            
    #* 生成固定障碍
    def _generate_static_obstacles(self):
        return list(np.array([
            Rectangle(center=np.array(MapArg.RECTANGLE_POS[0], dtype=np.float32), size=np.array(MapArg.RECTANGLE_SIZE, dtype=np.float32), angle=0.0),
            Rectangle(center=np.array(MapArg.RECTANGLE_POS[1], dtype=np.float32), size=np.array(MapArg.RECTANGLE_SIZE, dtype=np.float32), angle=0.0),
            Circle(center=np.array(MapArg.CIRCLE_POS[0], dtype=np.float32), radius=MapArg.CIRCLE_RADIUS),
            Circle(center=np.array(MapArg.CIRCLE_POS[1], dtype=np.float32), radius=MapArg.CIRCLE_RADIUS),
        ], dtype=object))
        
    #* 生成随机障碍
    def _generate_random_obstacles(self):
        random_obstacles = []
        block_size = self.map_arg.OBSTACLE_BLOCK_SIZE
        # 定义俄罗斯方块形状
        tetris_shapes = {
            'I': [np.array([0, -1.5]), np.array([0, -0.5]), np.array([0, 0.5]), np.array([0, 1.5])],
            'O': [np.array([-0.5, -0.5]), np.array([0.5, -0.5]), np.array([-0.5, 0.5]), np.array([0.5, 0.5])],
            'T': [np.array([-1, 0]), np.array([0, 0]), np.array([1, 0]), np.array([0, -1])],
            'L': [np.array([-1, -1]), np.array([-1, 0]), np.array([0, 0]), np.array([1, 0])],
            'S': [np.array([-1, 0]), np.array([0, 0]), np.array([0, -1]), np.array([1, -1])],
        }
        shape_keys = list(tetris_shapes.keys())
        # 定义障碍生成区域
        safe_margin_x = 2 * (self.goal_pos[0] - self.agent_spawn[0][0]) / 10
        spawn_x_min = self.agent_spawn[:, 0].min() + safe_margin_x
        spawn_x_max = self.goal_pos[0] - safe_margin_x
        x_total_width = spawn_x_max - spawn_x_min
        x_segment_length = x_total_width / self.map_arg.NUM_OBSTACLE_CLUSTERS    
        for i in range(self.map_arg.NUM_OBSTACLE_CLUSTERS):
            # 随机选择一个形状
            shape_key = random.choice(shape_keys)
            blocks = tetris_shapes[shape_key]
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
                random_obstacles.append(
                    Rectangle(center=rect_center, size=block_size, angle=angle)
                )
        return random_obstacles
    
    #* 生成边界
    def _generate_boundaries(self):
        boundary_thickness = 5.0 # 边界厚度
        return [
            Rectangle(center=np.array([(self.max_x+self.min_x)/2, self.max_y+boundary_thickness/2]), size=np.array([self.max_x-self.min_x, boundary_thickness]), angle=0.0), # 上
            Rectangle(center=np.array([(self.max_x+self.min_x)/2, self.min_y-boundary_thickness/2]), size=np.array([self.max_x-self.min_x, boundary_thickness]), angle=0.0), # 下
            Rectangle(center=np.array([self.min_x-boundary_thickness/2, 0]), size=np.array([boundary_thickness, self.max_y-self.min_y]), angle=0.0), # 左
            Rectangle(center=np.array([self.max_x+boundary_thickness/2, 0]), size=np.array([boundary_thickness, self.max_y-self.min_y]), angle=0.0), # 右
            ]
    
    #* 获得观测值
    def _get_obs(self):
        meas = []
        for i in range(self.num_agents):
            # 将该智能体从障碍物列表中排除
            other_agent_obstacles = self.agent_obstacle_circles[:i] + self.agent_obstacle_circles[i+1:]
            dist = self.lidar.scan(
                agent_pos=self.agent_pos[i],
                static_obstacles=self.static_obstacles,
                agent_obstacles=other_agent_obstacles
            )
            meas.append(dist)

        return {
            "vector_to_goal": (self.goal_pos - self.agent_pos[0]).astype(np.float32), # 领导者1到目标的向量
            "leader1_vel": self.agent_vel[0].astype(np.float32), # 领导者1的绝对速度
            "other_rel_pos": (self.agent_pos[1:] - self.agent_pos[0]).astype(np.float32), # 其他智能体相对于领导者1的位置
            "other_rel_vel": (self.agent_vel[1:] - self.agent_vel[0]).astype(np.float32), # 其他智能体相对于领导者1的速度
            "agents_meas": np.array(meas, dtype=np.float32), # 智能体的雷达测量值
        }
    
    #* 获得调试信息
    def _get_info(self):
        return {"finish":self.finish}

    #* 重置环境
    def reset(self, seed=MapArg.SEED, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 生成障碍物
        if self.map_type == 'random':
            obstacles = self._generate_random_obstacles()
        else:
            obstacles = self._generate_static_obstacles()
        # 生成边界
        boundaries = self._generate_boundaries()

        self.static_obstacles = obstacles + boundaries
        self.num_obstacles = len(self.static_obstacles)

        self.finish = False
        self.reward_info = {}
        self.agent_pos = self.agent_spawn.copy().astype(np.float32)
        self.agent_acc = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.agent_vel = np.zeros((self.num_agents, 2), dtype=np.float32)

        self.old_pos = self.agent_pos.copy().astype(np.float32)
        self.target_pos = self.agent_pos.copy().astype(np.float32)

        for i in range(self.num_agents):
            self.agent_obstacle_circles[i].center = self.agent_pos[i]
            self.agent_trails[i].clear() # 重置轨迹缓冲区
            self.agent_trails[i].append(self.agent_pos[i].copy()) # 保存初始位置

        self.step_count = 0
        obs = self._get_obs()
        info = self._get_info()

        self._init_render()

        return obs, info

    #* 计算动作
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

        # 更新领导者1的状态
        self.agent_acc[0] = acc
        self.agent_vel[0] += self.agent_acc[0] * self.dt
        self.agent_pos[0] += self.agent_vel[0] * self.dt

        # 更新领导者2,3的状态
        self.target_pos = self.apply_affine_transform(AgentArg.NOMINAL_CONFIG, rot, np.zeros(2, dtype=np.float32), scale, shear) + self.agent_pos[0]
        self.agent_acc[1:self.num_leaders] = (AgentArg.KP_LEADER * (self.target_pos[1:self.num_leaders] - self.agent_pos[1:self.num_leaders]) 
                                            + AgentArg.KD_LEADER * (self.agent_vel[0] - self.agent_vel[1:self.num_leaders]))
        self.agent_vel[1:self.num_leaders] += self.agent_acc[1:self.num_leaders] * self.dt
        self.agent_pos[1:self.num_leaders] += self.agent_vel[1:self.num_leaders] * self.dt

        # 更新跟随者状态
        follower_idx = range(self.num_leaders, self.num_agents)
        follower_acc = np.zeros((self.num_followers, 2), dtype=np.float32)
        for i in follower_idx:
            sum_term = np.zeros(2)
            gamma_i = 0.0
            for j in AgentArg.NEIGHBORS[i]:
                omega_ij = -AgentArg.STRESS_MATRIX[i, j]
                current_rel_pos = self.agent_pos[i] - self.agent_pos[j]
                vel_diff = self.agent_vel[i] - self.agent_vel[j]
                v_j_dot = self.agent_acc[j]
                term = omega_ij * (AgentArg.KP_FOLLOWER * (current_rel_pos) + AgentArg.KV_FOLLOWER * vel_diff - v_j_dot)
                sum_term += term
                gamma_i += omega_ij
            follower_acc[i - self.num_leaders] = -sum_term / gamma_i

        self.agent_acc[follower_idx] = follower_acc.astype(np.float32)
        self.agent_vel[follower_idx] += self.agent_acc[follower_idx] * self.dt
        self.agent_pos[follower_idx] += self.agent_vel[follower_idx] * self.dt

        self.agent_obstacle_circles[i].center = self.agent_pos[i]

        # 更新渲染轨迹
        for i in range(self.num_agents):
            self.agent_trails[i].append(self.agent_pos[i].copy()) 
            
        obs = self._get_obs()
        rew, done = self.reward(obs)
        self.render()
        info = self._get_info()
        return obs, rew, done, False, info

    #* 计算奖励
    def reward(self, obs):
        done = False

        total_reward = 0.0
        rew_goal = 0.0
        rew_near = 0.0
        rew_move = 0.0
        pen_crash = 0.0
        pen_danger = 0.0
        pen_stay = 0.0

        #* 到达目标的奖励
        leader_center = sum(self.agent_pos[:3]) / 3
        dist_to_goal = np.linalg.norm(leader_center - self.goal_pos)
        if dist_to_goal < self.goal_radius:
            rew_goal = RewardArg.R_GOAL
            print(f"\n==在第{self.step_count}步到达终点==")
            return rew_goal, True

        #* 移动奖励
        old_leader_center = sum(self.old_pos[:3]) / 3
        old_dist_to_goal = np.linalg.norm(old_leader_center - self.goal_pos)
        rew_move = RewardArg.R_MOVE * (old_dist_to_goal - dist_to_goal)
        pen_stay = RewardArg.P_STAY * np.tanh( 0.005*(dist_to_goal-390) ) # 防止在出生点附近逗留
        
        #* 碰撞惩罚
        min_dist_to_obstacles = np.min(obs["agents_meas"])
        if min_dist_to_obstacles < 2 * AgentArg.AGENT_RADIUS:
            pen_crash = RewardArg.P_CRASH
            self.fail = True
            done = True
            total_reward += pen_crash
            return total_reward, done
        
        #* 危险惩罚
        for i in range(self.num_agents):
            min_dist = np.min(obs["agents_meas"][i])
            if min_dist < AgentArg.DANGER_THRESHOLD:
                pen_danger += 0.5 * RewardArg.P_DANGER * (1 / min_dist - 1 / AgentArg.DANGER_THRESHOLD)
        
        total_reward = np.float32(rew_goal + rew_move + rew_near + pen_crash + pen_danger + pen_stay)

        self.reward_info = {
            "total_reward": total_reward,
            "rew_goal": rew_goal,
            "rew_near": rew_near,
            "rew_move": rew_move,
            "pen_stay": pen_stay,
            "pen_crash": pen_crash,
            "pen_danger": pen_danger,
        }
        return total_reward, done

    #* 初始化渲染
    def _init_render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("AffineEnv")
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.font is None:
            pygame.font.init() # 确保字体模块已初始化
            self.font = pygame.font.SysFont("Arial", 18) # 使用常见的字体和大小

    #* 渲染
    def render(self, render_lidar=True):
        if self.screen is None:
            self._init_render()

        canvas = self.screen
        canvas.fill(self.COLORS["BACKGROUND"])

        def world_to_screen(pos):
            # Pygame Y轴向下为正，所以需要翻转
            x_screen = int((pos[0] - self.min_x) * self.scale_factor +
                            self.padding_px + self.horizontal_extra_padding_px)
            y_screen = int(self.scaled_world_height_px - (pos[1] - self.min_y) * self.scale_factor +
                self.padding_px + self.vertical_extra_padding_px)
            return (x_screen, y_screen)

        #* 绘制障碍物
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

                angle = -obs_obj.angle
                rot_matrix = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]
                ])
                rotated_corners = (corners_local @ rot_matrix.T) + rect_center_screen
                pygame.draw.polygon(canvas, self.COLORS["OBSTACLE"], rotated_corners)

        #* 绘制终点
        goal_center_screen = world_to_screen(self.goal_pos)
        goal_radius_screen = int(self.goal_radius * self.scale_factor)
        pygame.draw.circle(canvas, self.COLORS["GOAL"], goal_center_screen, goal_radius_screen, 2)

        #* 绘制通信拓扑
        for i in range(self.num_agents):
            for j in AgentArg.NEIGHBORS[i]:
                if j > i: # 防止重复绘制
                    start_pos = world_to_screen(self.agent_pos[i])
                    end_pos = world_to_screen(self.agent_pos[j])
                    pygame.draw.line(canvas, self.COLORS["COMM_LINE"], start_pos, end_pos, 1)

        #* 绘制轨迹
        # 实际领导者轨迹
        for i in range(self.num_agents):
            if len(self.agent_trails[i]) > 1:
                points_screen = [world_to_screen(p) for p in self.agent_trails[i]]
                for j in range(len(points_screen) - 1):
                    alpha_factor = (j + 1) / self.trail_length
                    # current_color = tuple([int(c * alpha_factor + (255 * (1 - alpha_factor))) for c in self.COLORS["LEADER_TRAIL"]]) if i!=0 else tuple([int(c * alpha_factor + (255 * (1 - alpha_factor))) for c in self.COLORS["FIRST_TRAIL"]])
                    current_color = (tuple([int(c * alpha_factor + (255 * (1 - alpha_factor))) for c in self.COLORS["FIRST_TRAIL"]]) if i==0 else
                                    tuple([int(c * alpha_factor + (255 * (1 - alpha_factor))) for c in self.COLORS["LEADER_TRAIL"]]) if i==1 or i==2 else
                                    tuple([int(c * alpha_factor + (255 * (1 - alpha_factor))) for c in self.COLORS["FOLLOWER_TRAIL"]]))
                    pygame.draw.line(canvas, current_color, points_screen[j], points_screen[j+1], 2)
        
        #* 绘制期望位置
        if hasattr(self, 'target_pos') and self.target_pos is not None:
            agent_radius_screen = int(self.agent_radius * self.scale_factor)
            for i in range(self.num_agents-1):
                target_pos_screen = world_to_screen(self.target_pos[i+1])
                if i<self.num_leaders-1:
                    pygame.draw.circle(canvas, self.COLORS["TARGET_FORMATION"][0], target_pos_screen, agent_radius_screen, 2)
                else:
                    pygame.draw.circle(canvas, self.COLORS["TARGET_FORMATION"][1], target_pos_screen, agent_radius_screen, 2)

        #* 绘制智能体
        agent_radius_screen = int(self.agent_radius * self.scale_factor)
        lidar_scans = self._get_obs()["agents_meas"]
        for i in range(self.num_agents):
            agent_pos = self.agent_pos[i]
            agent_pos_screen = world_to_screen(self.agent_pos[i])

            #* 绘制 Lidar 射线
            if render_lidar:
                lidar_distances = lidar_scans[i]
                world_angles = self.lidar.relative_angles_np + 0.0 # 假设智能体朝向是0

                for j in range(self.lidar_num_rays):
                    angle = world_angles[j]
                    distance = lidar_distances[j]
                    if distance < self.lidar_max_range:
                        ray_end_world = agent_pos + np.array([np.cos(angle), np.sin(angle)]) * distance
                        ray_end_screen = world_to_screen(ray_end_world)
                        pygame.draw.line(canvas, self.COLORS["LIDAR_RAY"], agent_pos_screen, ray_end_screen, 1)

            agent_color = (self.COLORS["FIRST"] if i == 0 else
                            self.COLORS["LEADER"] if i==1 or i==2 else
                            self.COLORS["FOLLOWER"])          
            pygame.draw.circle(canvas, agent_color, agent_pos_screen, agent_radius_screen)
        
        #* 显示文本
        if hasattr(self, 'last_action') and self.last_action:
            acc = self.last_action.get("acc", [0, 0])
            rot_val = self.last_action.get("rot", 0)
            scale = self.last_action.get("scale", [1, 1])
            shear = self.last_action.get("shear", [0, 0])
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

            reward_keys = [k for k in self.reward_info if k.startswith("rew_") or k.startswith("tot_")]
            penalty_keys = [k for k in self.reward_info if k.startswith("pen_")]
            rew_texts = [f"{k}: {self.reward_info[k]:.3f}" for k in reward_keys]
            pen_texts = [f"{k}: {self.reward_info[k]:.3f}" for k in penalty_keys]

            # 调整文本位置
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

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.render_fps)

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

        if isinstance(rot, np.ndarray):
            rot = rot.item()

        scale = np.ravel(scale).astype(np.float32)
        shear = np.ravel(shear).astype(np.float32)

        num_agents = self.num_agents
        coords_homogeneous = np.hstack((agents_coords.astype(np.float32), np.ones((num_agents, 1), dtype=np.float32)))

        cos_r = np.cos(rot)
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

#* 测试环境
if __name__ == "__main__":
    import time

    env = AffineEnv()
    obs, info = env.reset()
    num_steps = 500 # 运行的步数
    total_reward = 0

    for step_count in range(num_steps):
        action = env.action_space.sample() # 随机采样动作
        if step_count >= 1:
            action['acc'] = np.array([0.2,0])
            action['rot'] = np.array([0])
            action['scale'] = np.array([0.5,2])
            action['shear'] = np.array([0,0])
        if step_count >= int(num_steps/3):
            action['acc'] = np.array([-0.1,0])
            action['rot'] = np.array([0])
            action['scale'] = np.array([2,0.5])
            action['shear'] = np.array([0,0])            

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        terminated = False 

        if terminated or truncated:
            obs, info = env.reset()
            total_reward = 0
            time.sleep(2)

    env.close()
    print("==环境测试结束==")
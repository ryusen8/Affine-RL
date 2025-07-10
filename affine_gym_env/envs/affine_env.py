import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from collections import deque # 导入 deque 用于轨迹保存

# 测试时用
# from affine_utils.lidar import Lidar
# from affine_utils.obstacles import Circle, Rectangle
# from affine_utils.arg import MapArg, AgentArg, RewardArg
# 训练时用
from .affine_utils.lidar import Lidar
from .affine_utils.obstacles import Circle, Rectangle, Line
from .affine_utils.arg import MapArg, AgentArg, RewardArg

class AffineEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, seed=MapArg.SEED):
        self.dt = MapArg.DT
        self.min_x = MapArg.MIN_X
        self.max_x = MapArg.MAX_X # [0, 100]
        self.min_y = MapArg.MIN_Y
        self.max_y = MapArg.MAX_Y # [-10, 10]

        self.screen_width = MapArg.SCREEN_WIDTH_PX
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
        # 如果 self.scaled_world_width_px < self.effective_screen_width，则有水平额外边距
        self.horizontal_extra_padding_px = (self.effective_screen_width - self.scaled_world_width_px) / 2
        # 如果 self.scaled_world_height_px < self.effective_screen_height，则有垂直额外边距
        self.vertical_extra_padding_px = (self.effective_screen_height - self.scaled_world_height_px) / 2

        self.circle_radius = MapArg.CIRCLE_RADIUS
        self.circle_pos = MapArg.CIRCLE_POS
        self.rectangle_size = MapArg.RECTANGLE_SIZE
        self.rectangle_pos = MapArg.RECTANGLE_POS
        self.obstacle_array = np.array([
            Rectangle(center=self.rectangle_pos[0], size=self.rectangle_size, angle=0.0),
            Rectangle(center=self.rectangle_pos[1], size=self.rectangle_size, angle=0.0),
            Circle(center=self.circle_pos[0], radius=self.circle_radius),
            Circle(center=self.circle_pos[1], radius=self.circle_radius),
            Line(p1=np.array([self.min_x, self.max_y]), p2=np.array([self.max_x, self.max_y])), # 上边界
            Line(p1=np.array([self.min_x, self.min_y]), p2=np.array([self.max_x, self.min_y])), # 下边界
            Line(p1=np.array([self.min_x, self.min_y]), p2=np.array([self.min_x, self.max_y])), # 左边界
            Line(p1=np.array([self.max_x, self.min_y]), p2=np.array([self.max_x, self.max_y])), # 右边界
        ])
        self.goal_radius = MapArg.GOAL_RADIUS
        self.goal_pos = MapArg.GOAL_POS

        self.num_leader = AgentArg.NUM_LEADERS
        self.leader_spawn = AgentArg.LEADER_SPAWN
        self.leader_pos = self.leader_spawn
        self.agent_radius = AgentArg.AGENT_RADIUS

        self.lidar_num_rays = AgentArg.LIDAR_NUM_RAYS
        self.lidar_max_range = AgentArg.LIDAR_MAX_RANGE
        self.lidar_fov = AgentArg.LIDAR_FOV

        self.lidar = Lidar(n_rays=self.lidar_num_rays, max_range=self.lidar_max_range, fov=self.lidar_fov)

        self.action_space = spaces.Dict({
            "acc": spaces.Box(low=np.array([AgentArg.MIN_ACC, AgentArg.MIN_ACC]), high=np.array([AgentArg.MAX_ACC, AgentArg.MAX_ACC]), dtype=np.float64),
            "rot": spaces.Box(low=AgentArg.MIN_ROT, high=AgentArg.MAX_ROT, dtype=np.float64),
            "scale": spaces.Box(low=np.array([AgentArg.MIN_SCALE, AgentArg.MIN_SCALE]), high=np.array([AgentArg.MAX_SCALE, AgentArg.MAX_SCALE]), dtype=np.float64),
            "shear": spaces.Box(low=np.array([AgentArg.MIN_SHEAR, AgentArg.MIN_SHEAR]), high=np.array([AgentArg.MAX_SHEAR, AgentArg.MAX_SHEAR]), dtype=np.float64),
            },
            seed=seed)
        
        self.observation_space = spaces.Dict({
            "leader_pos": spaces.Box(low=np.array([self.min_x,self.min_y]*self.num_leader).reshape(self.num_leader,2),
                                    high=np.array([self.max_x,self.max_y]*self.num_leader).reshape(self.num_leader,2), dtype=np.float64),

            "leader_vel": spaces.Box(low=np.array([AgentArg.MIN_VEL,AgentArg.MIN_VEL]*self.num_leader).reshape(self.num_leader,2),
                                    high=np.array([AgentArg.MAX_VEL,AgentArg.MAX_VEL]*self.num_leader).reshape(self.num_leader,2), dtype=np.float64), 

            "leader_meas": spaces.Box(low=0.0, high=self.lidar_max_range, shape=(self.num_leader, self.lidar_num_rays), dtype=np.float64),

            "formation_error": spaces.Box(low=-AgentArg.TOL_ERROR, high=AgentArg.TOL_ERROR, shape=(self.num_leader-1,), dtype=np.float64), 
            },
            seed=seed)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # 定义颜色
        self.COLORS = {
            "BACKGROUND": (255, 255, 255), # 白色
            "BOUNDARY": (0, 0, 0),         # 黑色
            "OBSTACLE": (150, 150, 150),   # 灰色
            "GOAL": (0, 255, 0),           # 绿色
            "LEADER": (0, 0, 255),    # 蓝色
            "FIRST": (255, 0, 0),
            "LIDAR_RAY": (150, 150, 150),    # 灰色
            "FIRST_TRAIL": (255, 150, 150),
            "LEADER_TRAIL": (150, 150, 255),    # 浅蓝，用于实际领导者轨迹
        }

        # 轨迹缓冲区设置
        self.trail_length = 100 # 轨迹点数
        # 为每个实际领导者创建一个独立的轨迹缓冲区
        self.leader_trails = [deque(maxlen=self.trail_length) for _ in range(self.num_leader)]

    def _get_obs(self):
        meas = []

        all_fixed_obstacles = list(self.obstacle_array)

        for i, leader_pos in enumerate(self.leader_pos):
            all_leaders_circles = [Circle(center=pos, radius=self.agent_radius) for pos in self.leader_pos]

            current_leader_obstacles_for_scan = []
            current_leader_obstacles_for_scan.extend(all_fixed_obstacles)
            for j, other_leader_circle in enumerate(all_leaders_circles):
                if i != j:
                    current_leader_obstacles_for_scan.append(other_leader_circle)

            # 假设智能体朝向角度是0，如果未来有朝向，需要在这里传入正确的角度
            dist = self.lidar.scan(agent_pos=leader_pos, agent_angle=0.0, obstacles=current_leader_obstacles_for_scan)
            meas.append(dist)

        return {
            "leader_pos": self.leader_pos,
            "leader_vel": self.leader_vel,
            "leader_meas": np.array(meas),
            "formation_error": np.linalg.norm(self.target_pos[1:] - self.leader_pos[1:], axis=1)
        }

    def _get_info(self):
        #TODO 加入碰撞计数器
        return {}

    def reset(self, seed=MapArg.SEED, options=None):
        super().reset(seed=seed)

        self.leader_pos = np.array(self.leader_spawn)
        self.leader_acc = np.zeros((self.num_leader, 2))
        self.leader_vel = np.zeros((self.num_leader, 2))

        # 重置轨迹缓冲区
        for i in range(self.num_leader):
            self.leader_trails[i].clear()
            self.leader_trails[i].append(self.leader_pos[i].copy()) # 保存初始位置

        # 保存上一时刻位置用来计算奖励
        self.old_pos = self.leader_pos
        self.target_pos = self.leader_pos
        # self.danger_count = 0
        self.collide_count = 0
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._init_render()

        return obs, info

    def step(self, action):
        acc = np.array(action["acc"], )
        rot = np.float64(action["rot"])
        scale = np.array(action["scale"], )
        shear = np.array(action["shear"], )
        acc = np.clip(acc, AgentArg.MIN_ACC, AgentArg.MAX_ACC)
        rot = np.clip(rot, AgentArg.MIN_ROT, AgentArg.MAX_ROT)
        scale = np.clip(scale, AgentArg.MIN_SCALE, AgentArg.MAX_SCALE)
        shear = np.clip(shear, AgentArg.MIN_SHEAR, AgentArg.MAX_SHEAR)
        self.old_pos = self.leader_pos.copy()
        # 更新领导者1的状态
        self.leader_acc[0] = acc
        self.leader_vel[0] += self.leader_acc[0] * self.dt
        self.leader_pos[0] += self.leader_vel[0] * self.dt
        # transl = self.leader_pos[0] - self.leader_spawn[0]
        transl = np.array([0.0, 0.0])
        self.target_pos = self.apply_affine_transform(AgentArg.NOMINAL_CONFIG, rot, transl, scale, shear) + self.leader_pos[0]
        # 更新领导者2和3的状态
        self.leader_acc[1:] = AgentArg.KP * (self.target_pos[1:] - self.leader_pos[1:]) + AgentArg.KD * (self.leader_vel[0] - self.leader_vel[1:])
        self.leader_vel[1:] += self.leader_acc[1:] * self.dt
        self.leader_pos[1:] += self.leader_vel[1:] * self.dt
        self.leader_pos = np.clip(self.leader_pos, [self.min_x, self.min_y], [self.max_x, self.max_y])
        self.leader_vel = np.clip(self.leader_vel, AgentArg.MIN_VEL, AgentArg.MAX_VEL)

        for i in range(self.num_leader):
            self.leader_trails[i].append(self.leader_pos[i].copy())
        obs = self._get_obs()
        rew, done = self.reward(obs)
        if self.render_mode == "human":
            self.render()
        info = self._get_info()
        return obs, rew, done, False, info

    def reward(self, obs):
        total_reward = 0
        rew_goal = 0.0
        self.danger_count = 0
        self.collide_rays = 0
        done = False
        leader_meas = obs["leader_meas"]
        formation_error = obs["formation_error"]
        
        #* 完成任务奖励 rew_goal
        if np.linalg.norm(self.leader_pos[0] - self.goal_pos) < self.goal_radius:
            rew_goal = RewardArg.R_GOAL
            done = True

        #* 移动和朝向奖励 rew_move, rew_dir
        move_length = np.linalg.norm(self.old_pos[0] - self.goal_pos) - np.linalg.norm(self.leader_pos[0] - self.goal_pos)
        dist_to_goal = np.linalg.norm(self.leader_pos[0] - self.goal_pos)
        goal_dir = self.goal_pos - self.old_pos[0]
        move_dir = self.leader_pos[0] - self.old_pos[0]

        rew_dir = RewardArg.R_DIR * np.dot(goal_dir, move_dir) / (np.linalg.norm(goal_dir) + 1e-6)
        rew_move = RewardArg.R_MOVE * (move_length - 0.1*dist_to_goal) + rew_dir
        
        #* 滞留惩罚 pen_slow
        leader_speed = np.linalg.norm(self.leader_vel[0])
        if np.abs(leader_speed) < 0.1*AgentArg.MAX_VEL and move_length < 0.5*self.agent_radius:
            pen_slow = RewardArg.P_SLOW

        #* 队形误差惩罚 pen_form_error
        pen_form_error = RewardArg.P_FORM_ERROR * np.sum(formation_error)

        #* 任务时间惩罚 pen_time
        pen_time = RewardArg.P_TIME

        #* 危险惩罚 pen_danger, pen_collide
        for i in range(self.num_leader):
            meas = leader_meas[i]
            # 危险区域：Lidar 探测到的距离小于一个阈值（例如，2倍智能体半径）
            self.danger_count += np.sum(meas < (self.agent_radius * 2))
            # 碰撞：Lidar 探测到的距离小于或等于智能体半径
            self.collide_rays += np.sum(meas <= self.agent_radius)
        if self.collide_rays > 2:
            self.collide_count += 1

        pen_danger = RewardArg.P_DANGER * self.danger_count
        pen_collide = RewardArg.P_COLLIDE * self.collide_rays
        
        if self.collide_count >= self.num_leader * RewardArg.TOL_COLLIDE_TIMES:
            pen_collide += RewardArg.P_FAIL
            done = True

        total_reward = rew_goal + rew_move + pen_collide + pen_danger + pen_form_error + pen_time + pen_slow
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
        for obs_obj in self.obstacle_array: # 改名为 obs_obj 避免与 obs 混淆
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

        # --- 绘制地图边界 ---
        pygame.draw.rect(canvas, self.COLORS["BOUNDARY"],
                        (int(self.padding_px + self.horizontal_extra_padding_px),
                        int(self.padding_px + self.vertical_extra_padding_px),
                        int(self.scaled_world_width_px),
                        int(self.scaled_world_height_px)), 2)
        # --- 绘制终点 ---
        goal_center_screen = world_to_screen(self.goal_pos)
        goal_radius_screen = int(self.goal_radius * self.scale_factor)
        pygame.draw.circle(canvas, self.COLORS["GOAL"], goal_center_screen, goal_radius_screen, 2)

        # --- 绘制轨迹 ---
        # 实际领导者轨迹
        for i in range(self.num_leader):
            if len(self.leader_trails[i]) > 1:
                points_screen = [world_to_screen(p) for p in self.leader_trails[i]]
                for j in range(len(points_screen) - 1):
                    alpha_factor = (j + 1) / self.trail_length
                    current_color = tuple([int(c * alpha_factor + (255 * (1 - alpha_factor))) for c in self.COLORS["LEADER_TRAIL"]]) if i!=0 else tuple([int(c * alpha_factor + (255 * (1 - alpha_factor))) for c in self.COLORS["FIRST_TRAIL"]])
                    pygame.draw.line(canvas, current_color, points_screen[j], points_screen[j+1], 2)

        # --- 绘制实际领导者 ---
        agent_radius_screen = int(self.agent_radius * self.scale_factor)
        for i in range(self.num_leader):
            leader_pos = self.leader_pos[i]
            leader_screen = world_to_screen(leader_pos)
            # 绘制 Lidar 射线
            lidar_distances = self._get_obs()["leader_meas"][i]
            # agent_angle=0.0，所以这里也保持一致
            world_angles = self.lidar.relative_angles + 0.0 # 假设智能体朝向是0

            for j in range(self.lidar_num_rays):
                angle = world_angles[j]
                distance = lidar_distances[j]
                ray_end_world = leader_pos + np.array([np.cos(angle), np.sin(angle)]) * distance
                ray_end_screen = world_to_screen(ray_end_world)
                pygame.draw.line(canvas, self.COLORS["LIDAR_RAY"], leader_screen, ray_end_screen, 1)

            pygame.draw.circle(canvas, self.COLORS["LEADER"] if i!=0 else self.COLORS["FIRST"], leader_screen, agent_radius_screen)

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
        rot: float = 0.0,
        trans: np.ndarray = np.array([0.0, 0.0]),
        scale: np.ndarray = np.array([1.0, 1.0]),
        shear: np.ndarray = np.array([0.0, 0.0])
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
        num_agents = agents_coords.shape[0]

        # 1. 创建齐次坐标
        # 将 (x, y) 转换为 (x, y, 1) 以便进行矩阵乘法
        coords_homogeneous = np.hstack((agents_coords, np.ones((num_agents, 1))))

        # 2. 构建变换矩阵 (3x3)

        # 2.1 旋转矩阵
        cos_rot = np.cos(rot)
        sin_rot = np.sin(rot)
        rot_matrix = np.array([
            [cos_rot, -sin_rot, 0],
            [sin_rot,  cos_rot, 0],
            [0,        0,       1]
        ])

        # 2.2 缩放矩阵
        scale_matrix = np.array([
            [scale[0], 0,        0],
            [0,        scale[1], 0],
            [0,        0,        1]
        ])

        # 2.3 剪切矩阵
        shear_matrix = np.array([
            [1,         shear[0], 0],
            [shear[1],  1,        0],
            [0,         0,        1]
        ])

        # 2.4 平移矩阵
        trans_matrix = np.array([
            [1, 0, trans[0]],
            [0, 1, trans[1]],
            [0, 0, 1]
        ])

        # 3. 组合所有变换（注意顺序：缩放 -> 剪切 -> 旋转 -> 平移）
        # 通常的顺序是先应用线性变换（缩放、剪切、旋转），最后应用平移
        # 这里我们按照常见的变换顺序，将它们组合成一个单一的变换矩阵
        # 最终变换矩阵 M = T * R * Sh * S
        transformation_matrix = trans_matrix @ rot_matrix @ shear_matrix @ scale_matrix

        # 4. 应用变换
        # 将齐次坐标与变换矩阵相乘
        transformed_coords_homogeneous = (transformation_matrix @ coords_homogeneous.T).T

        # 5. 还原为二维坐标
        # 丢弃齐次坐标的最后一列 (1)
        transformed_coords = transformed_coords_homogeneous[:, :2]

        return transformed_coords

if __name__ == "__main__":
    import time

    # 创建环境实例，启用 human 渲染模式
    env = AffineEnv(render_mode="human")

    obs, info = env.reset()

    # 打印初始观测，确认数据结构
    print("Initial Observation Keys:", obs.keys())
    print("Leader X Pos:", obs["leader_x"])
    print("Leader Y Pos:", obs["leader_y"])
    print("Leader Lidar Meas Shape:", obs["leader_meas"].shape)

    num_steps = 500 # 运行的步数
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
            # env.leader_pos 是最新的位置
            print(f"Leader Lidar Meas Min: {np.min(obs['leader_meas']):.2f}, Max: {np.max(obs['leader_meas']):.2f}")

        # 如果达到终止条件
        if terminated or truncated:
            print(f"\nEpisode finished at step {step_count} with total reward: {total_reward:.2f}")
            obs, info = env.reset() # 重置环境
            total_reward = 0
            time.sleep(1) # 暂停一下，方便观察重置

    env.close() # 关闭渲染窗口
    print("Simulation finished.")
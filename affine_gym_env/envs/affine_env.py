import numpy as np
import pygame
import torch
import gymnasium as gym
from gymnasium import spaces
from collections import deque # 导入 deque 用于轨迹保存

# 确保这些导入路径正确
from .affine_utils.lidar import Lidar
from .affine_utils.obstacles import Circle, Rectangle
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
        ])
        self.goal_radius = MapArg.GOAL_RADIUS
        self.goal_pos = MapArg.GOAL_POS

        self.num_real_leader = AgentArg.NUM_REAL_LEADER
        self.virtual_leader_spawn = AgentArg.VIRTUAL_LEADER_SPAWN
        self.real_leader_spawn = AgentArg.REAL_LEADER_SPAWN
        self.virtual_leader_pos = self.virtual_leader_spawn
        self.real_leader_pos = self.real_leader_spawn
        self.agent_radius = AgentArg.AGENT_RADIUS

        self.lidar_num_rays = AgentArg.LIDAR_NUM_RAYS
        self.lidar_max_range = AgentArg.LIDAR_MAX_RANGE
        self.lidar_fov = AgentArg.LIDAR_FOV

        self.lidar = Lidar(n_rays=self.lidar_num_rays, max_range=self.lidar_max_range, fov=self.lidar_fov)


        self.action_space = spaces.Dict({
            "acc": spaces.Box(low=np.array([AgentArg.MIN_ACC, AgentArg.MIN_ACC]), high=np.array([AgentArg.MAX_ACC, AgentArg.MAX_ACC]), dtype=np.float32),
            "rot": spaces.Box(low=AgentArg.MIN_ROT, high=AgentArg.MAX_ROT, shape=(), dtype=np.float32),
            "scale": spaces.Box(low=np.array([AgentArg.MIN_SCALE, AgentArg.MIN_SCALE]), high=np.array([AgentArg.MAX_SCALE, AgentArg.MAX_SCALE]), dtype=np.float32),
            "shear": spaces.Box(low=np.array([AgentArg.MIN_SHEAR, AgentArg.MIN_SHEAR]), high=np.array([AgentArg.MAX_SHEAR, AgentArg.MAX_SHEAR]), dtype=np.float32),
            },
            seed=seed)
        self.observation_space = spaces.Dict({
            "virtual_pos": spaces.Box(low=np.array([self.min_x, self.min_y], dtype=np.float32), high=np.array([self.max_x, self.max_y], dtype=np.float32)),
            "leader_x": spaces.Box(low=self.min_x, high=self.max_x, shape=(self.num_real_leader, ), dtype=np.float32),
            "leader_y": spaces.Box(low=self.min_y, high=self.max_y, shape=(self.num_real_leader, ), dtype=np.float32),
            "leader_meas": spaces.Box(low=0.0, high=self.lidar_max_range, shape=(self.num_real_leader, self.lidar_num_rays), dtype=np.float32),
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
            "VIRTUAL_LEADER": (255, 0, 0), # 红色
            "REAL_LEADER": (0, 0, 255),    # 蓝色
            "LIDAR_RAY": (200, 200, 0),    # 黄色
            "VIRTUAL_TRAIL": (255, 150, 150), # 浅红，用于虚拟领导者轨迹
            "REAL_TRAIL": (150, 150, 255),    # 浅蓝，用于实际领导者轨迹
        }

        # 轨迹缓冲区设置
        self.trail_length = 100 # 轨迹点数
        self.virtual_leader_trail = deque(maxlen=self.trail_length)
        # 为每个实际领导者创建一个独立的轨迹缓冲区
        self.real_leader_trails = [deque(maxlen=self.trail_length) for _ in range(self.num_real_leader)]


    def _get_obs(self):
        meas = []

        all_fixed_obstacles = list(self.obstacle_array)

        for i, leader_pos in enumerate(self.real_leader_pos):
            all_real_leaders_circles = [Circle(center=pos, radius=self.agent_radius) for pos in self.real_leader_pos]

            current_leader_obstacles_for_scan = []
            current_leader_obstacles_for_scan.extend(all_fixed_obstacles)
            for j, other_leader_circle in enumerate(all_real_leaders_circles):
                if i != j:
                    current_leader_obstacles_for_scan.append(other_leader_circle)

            # 假设智能体朝向角度是0，如果未来有朝向，需要在这里传入正确的角度
            dist = self.lidar.scan(agent_pos=leader_pos, agent_angle=0.0, obstacles=current_leader_obstacles_for_scan)
            meas.append(dist)

        return {
            "virtual_pos": self.virtual_leader_pos,
            "leader_x": self.real_leader_pos[:,0],
            "leader_y": self.real_leader_pos[:,1],
            "leader_meas": np.array(meas, dtype=np.float32)
        }

    def _get_info(self):
        #TODO 加入碰撞计数器
        return {}

    def reset(self, seed=MapArg.SEED, options=None):
        super().reset(seed=seed)

        self.virtual_leader_pos = np.array(self.virtual_leader_spawn, dtype=np.float32)
        self.real_leader_pos = np.array(self.real_leader_spawn, dtype=np.float32)

        self.virtual_leader_acc = np.array([0., 0.], dtype=np.float32)
        self.virtual_leader_vel = np.array([0., 0.], dtype=np.float32)
        self.real_leader_acc = np.zeros((self.num_real_leader, 2), dtype=np.float32)
        self.real_leader_vel = np.zeros((self.num_real_leader, 2), dtype=np.float32)

        # 重置轨迹缓冲区
        self.virtual_leader_trail.clear()
        self.virtual_leader_trail.append(self.virtual_leader_pos.copy()) # 保存初始位置
        for i in range(self.num_real_leader):
            self.real_leader_trails[i].clear()
            self.real_leader_trails[i].append(self.real_leader_pos[i].copy()) # 保存初始位置

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._init_render()

        return obs, info

    def step(self, action):
        acc = np.array(action["acc"], dtype=np.float32)
        rot = float(action["rot"])
        scale = np.array(action["scale"], dtype=np.float32)
        shear = np.array(action["shear"], dtype=np.float32)

        # 保存当前位置用于计算奖励（例如移动奖励）
        old_virtual_leader_pos = self.virtual_leader_pos.copy()

        self.virtual_leader_acc = acc

        # 严格限制加速度在 MAX_ACC 和 MIN_ACC 之间，针对每个分量
        self.virtual_leader_acc = np.clip(self.virtual_leader_acc, AgentArg.MIN_ACC, AgentArg.MAX_ACC)


        transl = self.virtual_leader_vel * self.dt + 0.5 * self.virtual_leader_acc * self.dt**2

        self.virtual_leader_pos += transl
        self.virtual_leader_vel += self.virtual_leader_acc * self.dt

        # 严格限制虚拟领导者速度在 MAX_VEL 范围内
        self.virtual_leader_vel = np.clip(self.virtual_leader_vel, -AgentArg.MAX_VEL, AgentArg.MAX_VEL) # 假设MAX_VEL也适用于负方向

        real_leader_target_pos = self.apply_affine_transform(self.real_leader_pos, rot, transl, scale, shear)

        pos_error = real_leader_target_pos - self.real_leader_pos
        vel_error = self.virtual_leader_vel - self.real_leader_vel

        self.real_leader_acc = AgentArg.KP_POS * pos_error + AgentArg.KP_VEL * vel_error

        # 限制实际领导者加速度在 MAX_ACC 和 MIN_ACC 之间，针对每个分量
        self.real_leader_acc = np.clip(self.real_leader_acc, AgentArg.MIN_ACC, AgentArg.MAX_ACC)

        self.real_leader_pos += self.real_leader_vel * self.dt + 0.5 * self.real_leader_acc * self.dt**2
        self.real_leader_vel += self.real_leader_acc * self.dt

        # 限制实际领导者速度在 MAX_VEL 范围内
        self.real_leader_vel = np.clip(self.real_leader_vel, -AgentArg.MAX_VEL, AgentArg.MAX_VEL)

        # 边界检查 (可选，确保智能体在地图内)
        self.real_leader_pos[:, 0] = np.clip(self.real_leader_pos[:, 0], self.min_x, self.max_x)
        self.real_leader_pos[:, 1] = np.clip(self.real_leader_pos[:, 1], self.min_y, self.max_y)
        self.virtual_leader_pos = np.clip(self.virtual_leader_pos, np.array([self.min_x, self.min_y]), np.array([self.max_x, self.max_y]))

        # 更新轨迹缓冲区
        self.virtual_leader_trail.append(self.virtual_leader_pos.copy())
        for i in range(self.num_real_leader):
            self.real_leader_trails[i].append(self.real_leader_pos[i].copy())

        obs = self._get_obs()
        rew, done = self.reward(obs, old_virtual_leader_pos) # 传入 old_virtual_leader_pos
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, rew, done, False, info

    # reward 函数现在接受 old_virtual_leader_pos
    def reward(self, obs, old_virtual_leader_pos):
        rew_goal = 0.0
        done = False

        virtual_pos = obs["virtual_pos"]
        leader_meas = obs["leader_meas"]

        dist_to_goal = np.linalg.norm(virtual_pos - self.goal_pos)
        if dist_to_goal <= self.goal_radius:
            rew_goal = RewardArg.R_GOAL
            done = True

        # 使用虚拟领导者实际移动的距离来计算移动奖励
        step_length = np.linalg.norm(virtual_pos - old_virtual_leader_pos)
        rew_move = RewardArg.R_MOVE * step_length

        danger_count = 0
        collision_count = 0

        for i in range(self.num_real_leader):
            meas = leader_meas[i]
            # 危险区域：Lidar 探测到的距离小于一个阈值（例如，2倍智能体半径）
            danger_count += np.sum(meas < (self.agent_radius * 2 ))
            # 碰撞：Lidar 探测到的距离小于或等于智能体半径
            collision_count += np.sum(meas <= self.agent_radius)

        rew_danger = RewardArg.R_DANGER * danger_count
        rew_collision = RewardArg.R_COLLISION * collision_count

        total_reward = rew_goal + rew_move + rew_danger + rew_collision
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
        # 虚拟领导者轨迹
        if len(self.virtual_leader_trail) > 1:
            points_screen = [world_to_screen(p) for p in self.virtual_leader_trail]
            # 渐变效果：从浅到深
            for i in range(len(points_screen) - 1):
                # 越旧的轨迹点越透明（这里通过颜色变浅模拟）
                alpha_factor = (i + 1) / self.trail_length 
                current_color = tuple([int(c * alpha_factor + (255 * (1 - alpha_factor))) for c in self.COLORS["VIRTUAL_TRAIL"]])
                # Pygame draw.line 不直接支持 alpha，所以我们用混合颜色近似
                pygame.draw.line(canvas, current_color, points_screen[i], points_screen[i+1], 1)

        # 实际领导者轨迹
        for i in range(self.num_real_leader):
            if len(self.real_leader_trails[i]) > 1:
                points_screen = [world_to_screen(p) for p in self.real_leader_trails[i]]
                for j in range(len(points_screen) - 1):
                    alpha_factor = (j + 1) / self.trail_length
                    current_color = tuple([int(c * alpha_factor + (255 * (1 - alpha_factor))) for c in self.COLORS["REAL_TRAIL"]])
                    pygame.draw.line(canvas, current_color, points_screen[j], points_screen[j+1], 1)

        # --- 绘制虚拟领导者 ---
        virtual_leader_screen = world_to_screen(self.virtual_leader_pos)
        pygame.draw.circle(canvas, self.COLORS["VIRTUAL_LEADER"], virtual_leader_screen, 5)

        # --- 绘制实际领导者 ---
        agent_radius_screen = int(self.agent_radius * self.scale_factor)
        for i in range(self.num_real_leader):
            real_leader_pos = self.real_leader_pos[i]
            real_leader_screen = world_to_screen(real_leader_pos)
            pygame.draw.circle(canvas, self.COLORS["REAL_LEADER"], real_leader_screen, agent_radius_screen)

            # 绘制 Lidar 射线
            lidar_distances = self._get_obs()["leader_meas"][i]
            # Lidar 射线是从智能体中心发射的，朝向是 agent_angle (这里假设为0)
            # 这里的 Lidar.relative_angles 是相对于智能体朝向的角度
            # 如果智能体有朝向，你需要在 Lidar.scan 时传入真实的 agent_angle，
            # 并且这里的 world_angles 也要加上该智能体的真实朝向。
            # 目前你的 _get_obs 仍然使用 agent_angle=0.0，所以这里也保持一致
            world_angles = self.lidar.relative_angles + 0.0 # 假设智能体朝向是0

            for j in range(self.lidar_num_rays):
                angle = world_angles[j]
                distance = lidar_distances[j]
                ray_end_world = real_leader_pos + np.array([np.cos(angle), np.sin(angle)]) * distance
                ray_end_screen = world_to_screen(ray_end_world)
                pygame.draw.line(canvas, self.COLORS["LIDAR_RAY"], real_leader_screen, ray_end_screen, 1)


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
    import numpy as np

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
    print("Virtual Leader Pos:", obs["virtual_pos"])
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
            print(f"Virtual Leader Pos: {obs['virtual_pos']}")
            # env.real_leader_pos 是最新的位置
            print(f"Leader Lidar Meas Min: {np.min(obs['leader_meas']):.2f}, Max: {np.max(obs['leader_meas']):.2f}")

        # 如果达到终止条件
        if terminated or truncated:
            print(f"\nEpisode finished at step {step_count} with total reward: {total_reward:.2f}")
            obs, info = env.reset() # 重置环境
            total_reward = 0
            time.sleep(1) # 暂停一下，方便观察重置

    env.close() # 关闭渲染窗口
    print("Simulation finished.")
import numpy as np
from dataclasses import dataclass
from .obstacles import Circle, Rectangle

class Lidar:
    """
    一个用于2D环境的Lidar传感器。

    该传感器可以从智能体的位置发射多条射线，并计算与环境中
    圆形和矩形障碍物的交点距离。
    """
    def __init__(self, n_rays: int, max_range: float, fov: float = 2 * np.pi):
        """
        初始化Lidar传感器。

        Args:
            n_rays (int): 射线的数量。
            max_range (float): 传感器的最大测量距离。
            fov (float): 传感器的视场角范围 (Field of View)，默认为360度。
        """
        self.n_rays = n_rays
        self.max_range = max_range
        self.relative_angles = np.linspace(-fov / 2, fov / 2, n_rays, endpoint=False)

    def scan(self, agent_pos: np.ndarray, agent_angle: float, obstacles: list) -> np.ndarray:
        """
        执行一次扫描，返回每条射线的测量距离。

        Args:
            agent_pos (np.ndarray): 智能体的当前位置 [x, y]。
            agent_angle (float): 智能体的当前朝向角度 (弧度)，x轴正方向为0。
            obstacles (list): 环境中所有障碍物 (Circle或Rectangle对象) 的列表。

        Returns:
            np.ndarray: 一个包含每条射线测量距离的一维数组。
                        如果没有检测到障碍物，距离为max_range。
        """
        distances = np.full(self.n_rays, self.max_range, dtype=np.float32)
        world_angles = self.relative_angles + agent_angle
        ray_directions = np.c_[np.cos(world_angles), np.sin(world_angles)]

        for obstacle in obstacles:
            if isinstance(obstacle, Circle):
                obstacle_dists = self._ray_intersect_circle(agent_pos, ray_directions, obstacle)
            elif isinstance(obstacle, Rectangle):
                obstacle_dists = self._ray_intersect_rectangle(agent_pos, ray_directions, obstacle)
            else:
                continue

            distances = np.minimum(distances, obstacle_dists)

        return distances

    def _ray_intersect_circle(self, ray_origin: np.ndarray, ray_directions: np.ndarray, circle: Circle) -> np.ndarray:
        """矢量化计算所有射线与单个圆形障碍物的交点距离。"""
        L = ray_origin - circle.center
        b = 2 * np.einsum('ij,j->i', ray_directions, L)
        c = np.dot(L, L) - circle.radius**2
        delta = b**2 - 4 * c

        distances = np.full(self.n_rays, self.max_range)
        mask = delta >= 0
        if not np.any(mask):
            return distances

        sqrt_delta = np.sqrt(delta[mask])
        t1 = (-b[mask] - sqrt_delta) / 2.0
        t2 = (-b[mask] + sqrt_delta) / 2.0
        t = np.where(t1 > 0, t1, t2)

        valid_t = t[t > 0]
        valid_mask_indices = np.where(mask)[0][t > 0]

        distances[valid_mask_indices] = np.minimum(valid_t, self.max_range)

        return distances

    def _ray_intersect_rectangle(self, ray_origin: np.ndarray, ray_directions: np.ndarray, rect: Rectangle) -> np.ndarray:
        """矢量化计算所有射线与单个旋转矩形的交点距离 (Slab方法)。"""
        # 1. 坐标系变换：将射线原点和方向变换到矩形的局部坐标系中
        local_origin = ray_origin - rect.center

        # 我们需要将世界坐标系旋转 -rect.angle 度，以对齐矩形。
        angle_to_rotate = -rect.angle
        cos_a = np.cos(angle_to_rotate)
        sin_a = np.sin(angle_to_rotate)

        # CORRECTED: 这是用于旋转行向量 [x, y] 的正确矩阵
        # [x', y'] = [x, y] @ [[cos, sin], [-sin, cos]]
        rotation_matrix = np.array([[cos_a, sin_a], 
                                    [-sin_a, cos_a]])

        # 将射线的原点和方向都应用相同的旋转
        local_origin_rotated = local_origin @ rotation_matrix
        local_directions = ray_directions @ rotation_matrix

        # 2. Slab Test: 计算射线与轴对齐包围盒(AABB)的交点
        half_size = rect.size / 2.0
        inv_directions = np.divide(1.0, local_directions, where=local_directions!=0, out=np.full_like(local_directions, np.inf))

        t_near = ((-half_size) - local_origin_rotated) * inv_directions
        t_far = (half_size - local_origin_rotated) * inv_directions
        
        t_near = np.nan_to_num(t_near, nan=np.inf)
        t_far = np.nan_to_num(t_far, nan=np.inf)

        t_min = np.minimum(t_near, t_far)
        t_max = np.maximum(t_near, t_far)

        t_entry = np.max(t_min, axis=1)
        t_exit = np.min(t_max, axis=1)

        # 3. 确定有效交点
        distances = np.full(self.n_rays, self.max_range)
        mask = (t_entry < t_exit) & (t_exit > 0)

        final_t = t_entry[mask]
        valid_t = final_t[final_t > 0]
        valid_mask_indices = np.where(mask)[0][final_t > 0]

        distances[valid_mask_indices] = np.minimum(valid_t, self.max_range)

        return distances

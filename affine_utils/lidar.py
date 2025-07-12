import numpy as np
from dataclasses import dataclass
from .obstacles import Circle, Rectangle, Line

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
            obstacles (list): 环境中所有障碍物 (Circle, Rectangle, Line对象) 的列表。
        """
        distances = np.full(self.n_rays, self.max_range, dtype=np.float32)

        # 计算所有射线的绝对方向
        ray_angles = agent_angle + self.relative_angles
        ray_directions = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=1)

        for obstacle in obstacles:
            current_distances = np.full(self.n_rays, self.max_range, dtype=np.float32)

            if isinstance(obstacle, Circle):
                current_distances = self._ray_intersect_circle(agent_pos, ray_directions, obstacle)
            elif isinstance(obstacle, Rectangle):
                current_distances = self._ray_intersect_rectangle(agent_pos, ray_directions, obstacle)
            elif isinstance(obstacle, Line):
                current_distances = self._ray_intersect_line(agent_pos, ray_directions, obstacle)
            else:
                continue # 忽略未知类型的障碍物

            distances = np.minimum(distances, current_distances)

        return distances

    def _ray_intersect_circle(self, ray_origin: np.ndarray, ray_directions: np.ndarray, circle: Circle) -> np.ndarray:
        """矢量化计算所有射线与单个圆形的交点距离。"""
        # 将射线原点平移到以圆心为原点
        local_origin = ray_origin - circle.center

        # 构造二次方程 At^2 + Bt + C = 0
        # A = |D|^2 (D是射线方向向量)
        A = np.sum(ray_directions**2, axis=1) # D是单位向量，所以A通常是1.0

        # B = 2 * D . L (L是平移后的射线原点向量)
        # local_origin 是 (2,) 维数组，ray_directions 是 (N, 2) 维数组
        # 这里的点积操作会自动广播 local_origin
        B = 2 * np.sum(ray_directions * local_origin, axis=1)

        # C = |L|^2 - R^2
        # local_origin**2 结果仍是 (2,) 维数组。对其求和，不再指定 axis=1。
        C = np.sum(local_origin**2) - circle.radius**2 # 修正：移除 axis=1

        # 计算判别式 delta = B**2 - 4 * A * C
        # 注意: C 现在是一个标量，但 A 和 B 是 N 维数组，NumPy 会自动广播 C
        delta = B**2 - 4 * A * C

        distances = np.full(self.n_rays, self.max_range, dtype=np.float32)

        # 只有当delta >= 0 时才有实数解 (射线与圆有交点)
        mask = delta >= 0
        
        if np.any(mask):
            sqrt_delta = np.sqrt(delta[mask])
            
            # 计算两个可能的交点距离
            t1 = (-B[mask] - sqrt_delta) / (2 * A[mask])
            t2 = (-B[mask] + sqrt_delta) / (2 * A[mask])

            # 选择第一个正的交点作为有效距离
            # t_candidates 包含 t1 和 t2
            t_candidates = np.stack([t1, t2], axis=1)
            
            # 过滤掉非正的t值（即在射线起点后方的交点）
            t_candidates[t_candidates < 0] = np.inf # 将负值替换为无穷大，以便min函数忽略
            
            # 找到每条射线最小的正t值（即最近的有效交点）
            final_t = np.min(t_candidates, axis=1)
            
            # 将结果应用到原始的距离数组中，确保不超过max_range
            distances[mask] = np.minimum(final_t, self.max_range)

        return distances

    def _ray_intersect_rectangle(self, ray_origin: np.ndarray, ray_directions: np.ndarray, rect: Rectangle) -> np.ndarray:
        """矢量化计算所有射线与单个旋转矩形的交点距离 (Slab方法)。"""
        distances = np.full(self.n_rays, self.max_range, dtype=np.float32)

        # 1. 坐标系变换：将射线原点和方向变换到矩形的局部坐标系中
        # ray_origin 是 (2,)，需要扩展到 (N, 2) 才能与 ray_directions 的处理保持一致
        # 以便后续 local_origin_rotated 也是 (N, 2)
        local_origin_single = ray_origin - rect.center
        # 修正：将单个原点扩展为与射线数量相同的 N 个原点，方便后续的矢量化操作
        local_origin_expanded = np.tile(local_origin_single, (self.n_rays, 1))

        # 我们需要将世界坐标系旋转 -rect.angle 度，以对齐矩形。
        angle_to_rotate = -rect.angle
        cos_a = np.cos(angle_to_rotate)
        sin_a = np.sin(angle_to_rotate)

        # 用于旋转行向量 [x, y] 的旋转矩阵
        rotation_matrix = np.array([[cos_a, sin_a],
                                    [-sin_a, cos_a]])

        # 将射线的原点和方向都应用相同的旋转
        # local_origin_expanded 是 (N, 2)，rotation_matrix 是 (2, 2)，结果是 (N, 2)
        local_origin_rotated = local_origin_expanded @ rotation_matrix # 现在 local_origin_rotated 维度正确 (N, 2)
        local_directions = ray_directions @ rotation_matrix

        # 2. Slab Test: 计算射线与轴对齐包围盒(AABB)的交点
        half_size = rect.size / 2.0

        # 处理 local_directions 中接近 0 的情况，避免除以 0
        inv_directions = np.full_like(local_directions, np.inf)
        # np.isclose 用于浮点数比较，比 == 更安全，避免除零错误
        non_zero_mask_x = ~np.isclose(local_directions[:, 0], 0.0)
        non_zero_mask_y = ~np.isclose(local_directions[:, 1], 0.0)

        inv_directions[non_zero_mask_x, 0] = 1.0 / local_directions[non_zero_mask_x, 0]
        inv_directions[non_zero_mask_y, 1] = 1.0 / local_directions[non_zero_mask_y, 1]

        # 这些行现在可以正确执行，因为 local_origin_rotated 已经是 (N, 2) 数组
        t_near_x = ((-half_size[0]) - local_origin_rotated[:, 0]) * inv_directions[:, 0]
        t_far_x = ((half_size[0]) - local_origin_rotated[:, 0]) * inv_directions[:, 0]

        t_near_y = ((-half_size[1]) - local_origin_rotated[:, 1]) * inv_directions[:, 1]
        t_far_y = ((half_size[1]) - local_origin_rotated[:, 1]) * inv_directions[:, 1]
        
        # 确保 t_near_x/y 总是小于 t_far_x/y (通过交换来实现)
        t_min_x = np.minimum(t_near_x, t_far_x)
        t_max_x = np.maximum(t_near_x, t_far_x)
        t_min_y = np.minimum(t_near_y, t_far_y)
        t_max_y = np.maximum(t_near_y, t_far_y)

        # 射线的进入点是所有 t_min 中的最大值
        t_entry = np.maximum(t_min_x, t_min_y)
        # 射线的退出点是所有 t_max 中的最小值
        t_exit = np.minimum(t_max_x, t_max_y)

        # 3. 确定有效交点
        # 条件1: t_entry < t_exit (射线穿过AABB)
        # 条件2: t_exit > 0 (射线有正向交点，即使起点在内部，出口也必须在前方)
        mask_valid_intersection = (t_entry < t_exit) & (t_exit > 0)
        
        # 对于有效交点，选择实际距离
        # 如果 t_entry >= 0，说明射线从AABB外部进入，取 t_entry (最近的入口点)
        # 如果 t_entry < 0 且 t_exit > 0，说明射线起点在AABB内部，从内部射出，取 t_exit (最近的出口点)
        
        # 初始化最终距离数组
        final_t_values = np.full(self.n_rays, np.inf)

        # 射线从外部进入矩形
        mask_external_entry = mask_valid_intersection & (t_entry >= 0)
        final_t_values[mask_external_entry] = t_entry[mask_external_entry]

        # 射线从矩形内部射出
        mask_internal_exit = mask_valid_intersection & (t_entry < 0) & (t_exit >= 0)
        final_t_values[mask_internal_exit] = t_exit[mask_internal_exit]
        
        # 最终的距离不能超过 max_range
        distances = np.minimum(final_t_values, self.max_range)

        return distances

    def _ray_intersect_line(self, ray_origin: np.ndarray, ray_directions: np.ndarray, line: Line) -> np.ndarray:
        """矢量化计算所有射线与单个线段的交点距离。"""
        # 射线: P = ray_origin + t * ray_directions
        # 线段: Q = line.p1 + u * v_line,  其中 v_line = line.p2 - line.p1

        v_line = line.p2 - line.p1
        origin_diff = line.p1 - ray_origin

        denominator = np.cross(ray_directions, v_line)

        t_numerator_scalar = np.cross(origin_diff, v_line)
        t_numerator = np.full(self.n_rays, t_numerator_scalar, dtype=np.float32) # 修正：确保它是一个 (N,) 数组

        u_numerator = np.cross(origin_diff, ray_directions)

        distances = np.full(self.n_rays, self.max_range, dtype=np.float32)

        # 避免除以零 (射线与线段平行)
        # np.isclose 用于浮点数比较，比 == 更安全
        parallel_mask = np.isclose(denominator, 0.0)

        # 对于非平行的情况
        non_parallel_mask = ~parallel_mask
        if np.any(non_parallel_mask):
            # t_values 和 u_values 现在将是正确形状的数组，可以被掩码索引
            t_values = t_numerator[non_parallel_mask] / denominator[non_parallel_mask]
            u_values = u_numerator[non_parallel_mask] / denominator[non_parallel_mask]

            # 检查交点是否在射线上方 (t > 0) 且在线段范围内 (0 <= u <= 1)
            valid_intersection_mask = (t_values > 0) & (u_values >= 0) & (u_values <= 1)
            
            # 将有效距离应用到 distances 数组中
            # 先找到 non_parallel_mask 中对应 valid_intersection_mask 的索引
            original_indices = np.where(non_parallel_mask)[0][valid_intersection_mask]
            
            distances[original_indices] = np.minimum(t_values[valid_intersection_mask], self.max_range)
        
        return distances

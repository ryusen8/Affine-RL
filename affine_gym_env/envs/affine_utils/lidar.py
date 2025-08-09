# lidar.py (Corrected Version)

import numpy as np
import numba

# 导入仅用于类型提示和isinstance检查
try:
    from .obstacles import Circle, Rectangle
except ModuleNotFoundError:
    from affine_gym_env.envs.affine_utils.obstacles import Circle, Rectangle

# ===================================================================
# 1. Numba 加速的 "纯计算" 函数
# ===================================================================

@numba.jit(nopython=True, cache=True, fastmath=True)
def _numba_ray_intersect_circle(ray_origin: np.ndarray,
                                ray_directions: np.ndarray,
                                circle_center: np.ndarray,
                                circle_radius: float,
                                n_rays: int,
                                max_range: float) -> np.ndarray:
    """计算所有射线与单个圆形障碍物的交点距离 (Numba加速版)"""
    distances = np.full(n_rays, max_range, dtype=np.float32)
    L = ray_origin - circle_center
    c = np.dot(L, L) - circle_radius**2

    for i in range(n_rays):
        direction = ray_directions[i]
        b = 2 * np.dot(direction, L)
        delta = b**2 - 4 * c

        if delta >= 0:
            sqrt_delta = np.sqrt(delta)
            t1 = (-b - sqrt_delta) / 2.0
            
            if t1 > 1e-6:
                if t1 < distances[i]:
                    distances[i] = t1
            else:
                t2 = (-b + sqrt_delta) / 2.0
                if t2 > 1e-6:
                    if t2 < distances[i]:
                        distances[i] = t2
                        
    return distances

@numba.jit(nopython=True, cache=True, fastmath=True)
def _numba_ray_intersect_rectangle(ray_origin: np.ndarray,
                                   ray_directions: np.ndarray,
                                   rect_center: np.ndarray,
                                   rect_size: np.ndarray,
                                   rect_angle: float,
                                   n_rays: int,
                                   max_range: float) -> np.ndarray:
    """计算所有射线与单个旋转矩形的交点距离 (Numba加速版, 修正了除零错误)"""
    distances = np.full(n_rays, max_range, dtype=np.float32)
    half_size = rect_size / 2.0
    
    local_origin = ray_origin - rect_center
    angle_to_rotate = -rect_angle
    cos_a = np.cos(angle_to_rotate)
    sin_a = np.sin(angle_to_rotate)
    
    local_origin_rotated = np.empty_like(local_origin)
    local_origin_rotated[0] = local_origin[0] * cos_a - local_origin[1] * sin_a
    local_origin_rotated[1] = local_origin[0] * sin_a + local_origin[1] * cos_a

    for i in range(n_rays):
        dir_x, dir_y = ray_directions[i, 0], ray_directions[i, 1]
        local_dir_x = dir_x * cos_a - dir_y * sin_a
        local_dir_y = dir_x * sin_a + dir_y * cos_a
        
        # === Slab Test (修正版) ===
        t_near = -np.inf
        t_far = np.inf

        # --- 处理 X-Slab ---
        if abs(local_dir_x) < 1e-6:
            # 射线平行于X-slab的边界
            if local_origin_rotated[0] < -half_size[0] or local_origin_rotated[0] > half_size[0]:
                # 射线在slab之外，不可能相交
                continue # 直接跳到下一条射线
        else:
            # 计算与X-slab两个平面的交点
            t1 = (-half_size[0] - local_origin_rotated[0]) / local_dir_x
            t2 = (half_size[0] - local_origin_rotated[0]) / local_dir_x
            if t1 > t2:
                t1, t2 = t2, t1 # 确保 t1 是较近的交点
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

        # --- 处理 Y-Slab ---
        if abs(local_dir_y) < 1e-6:
            # 射线平行于Y-slab的边界
            if local_origin_rotated[1] < -half_size[1] or local_origin_rotated[1] > half_size[1]:
                # 射线在slab之外，不可能相交
                continue
        else:
            t1 = (-half_size[1] - local_origin_rotated[1]) / local_dir_y
            t2 = (half_size[1] - local_origin_rotated[1]) / local_dir_y
            if t1 > t2:
                t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

        # --- 判断最终交点 ---
        if t_near < t_far and t_far > 1e-6:
            # t_near 是射线的入口点
            final_t = t_near
            if final_t > 1e-6: # 确保交点在射线前方
                if final_t < distances[i]:
                    distances[i] = final_t
                    
    return distances

# ===================================================================
# 2. Lidar 类 (调度器), 无需修改
# ===================================================================
class Lidar:
    """
    一个用于2D环境的Lidar传感器。
    它负责准备数据并调用Numba加速的函数进行计算。
    """
    def __init__(self, n_rays: int, max_range: float, fov: float = 2 * np.pi):
        self.n_rays = n_rays
        self.max_range = float(max_range)
        self.relative_angles = np.linspace(-fov / 2, fov / 2, n_rays, endpoint=False).astype(np.float32)

    def scan(self, agent_pos: np.ndarray, agent_angle: float, obstacles: list) -> np.ndarray:
        final_distances = np.full(self.n_rays, self.max_range, dtype=np.float32)
        world_angles = self.relative_angles + np.float32(agent_angle)
        ray_directions = np.c_[np.cos(world_angles), np.sin(world_angles)].astype(np.float32)
        agent_pos_f32 = agent_pos.astype(np.float32)

        for obstacle in obstacles:
            if isinstance(obstacle, Circle):
                obstacle_dists = _numba_ray_intersect_circle(
                    agent_pos_f32, ray_directions, 
                    obstacle.center.astype(np.float32), 
                    np.float32(obstacle.radius), 
                    self.n_rays, self.max_range
                )
            elif isinstance(obstacle, Rectangle):
                obstacle_dists = _numba_ray_intersect_rectangle(
                    agent_pos_f32, ray_directions,
                    obstacle.center.astype(np.float32), 
                    obstacle.size.astype(np.float32), 
                    np.float32(obstacle.angle),
                    self.n_rays, self.max_range
                )
            else:
                continue
            
            np.minimum(final_distances, obstacle_dists, out=final_distances)

        return final_distances
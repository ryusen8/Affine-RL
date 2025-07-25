import numpy as np
import numba
# 确保你的 obstacles.py 也在同一个文件夹
from .obstacles import Circle, Rectangle, Line 

# ===================================================================
# 1. 创建独立的、可被Numba编译的 "纯计算" 函数
#    这些函数不属于任何类，并且只接收基础数据类型。
# ===================================================================

@numba.jit(nopython=True, cache=True)
def _numba_ray_intersect_circle(ray_origin, ray_directions, circle_center, circle_radius, n_rays, max_range):
    """计算所有射线与单个圆形障碍物的交点距离 (Numba加速版)"""
    distances = np.full(n_rays, max_range, dtype=np.float32)
    L = ray_origin.astype(np.float32) - circle_center.astype(np.float32)
    c = np.dot(L, L) - circle_radius**2

    # 循环遍历每条射线
    for i in range(n_rays):
        direction = ray_directions[i].astype(np.float32)
        b = 2 * np.dot(direction, L)
        delta = b**2 - 4 * c

        if delta >= 0:
            sqrt_delta = np.sqrt(delta)
            t1 = (-b - sqrt_delta) / 2.0
            
            # 我们只需要最近的、在射线前方的交点
            if t1 > 1e-6: # 1e-6 是一个小的容差，避免浮点数误差
                if t1 < distances[i]:
                    distances[i] = np.float32(t1)
            else:
                t2 = (-b + sqrt_delta) / 2.0
                if t2 > 1e-6:
                    if t2 < distances[i]:
                        distances[i] = np.float32(t2)
                        
    return distances

@numba.jit(nopython=True, cache=True)
def _numba_ray_intersect_rectangle(ray_origin, ray_directions, rect_center, rect_size, rect_angle, n_rays, max_range):
    """计算所有射线与单个旋转矩形的交点距离 (Numba加速版)"""
    distances = np.full(n_rays, max_range, dtype=np.float32)
    half_size = rect_size.astype(np.float32) / 2.0
    
    # 坐标系变换
    local_origin = ray_origin.astype(np.float32) - rect_center.astype(np.float32)
    angle_to_rotate = -rect_angle
    cos_a = np.cos(angle_to_rotate)
    sin_a = np.sin(angle_to_rotate)
    
    # Numba中手动进行旋转变换
    local_origin_rotated_x = local_origin[0] * cos_a - local_origin[1] * sin_a
    local_origin_rotated_y = local_origin[0] * sin_a + local_origin[1] * cos_a
    
    # 循环处理每条射线
    for i in range(n_rays):
        # 旋转射线方向
        dir_x = ray_directions[i, 0]
        dir_y = ray_directions[i, 1]
        local_dir_x = dir_x * cos_a - dir_y * sin_a
        local_dir_y = dir_x * sin_a + dir_y * cos_a

        # Slab Test
        t_near = np.zeros(2, dtype=np.float32)
        t_far = np.zeros(2, dtype=np.float32)

        # 处理X方向
        if abs(local_dir_x) < 1e-6:
            if -half_size[0] > local_origin_rotated_x or half_size[0] < local_origin_rotated_x:
                continue # 射线平行于slab且在外部，不可能相交
        else:
            t_near[0] = (-half_size[0] - local_origin_rotated_x) / local_dir_x
            t_far[0] = (half_size[0] - local_origin_rotated_x) / local_dir_x
        
        # 处理Y方向
        if abs(local_dir_y) < 1e-6:
            if -half_size[1] > local_origin_rotated_y or half_size[1] < local_origin_rotated_y:
                continue
        else:
            t_near[1] = (-half_size[1] - local_origin_rotated_y) / local_dir_y
            t_far[1] = (half_size[1] - local_origin_rotated_y) / local_dir_y

        # 交换 t_near 和 t_far 保证 t_near < t_far
        if t_near[0] > t_far[0]: t_near[0], t_far[0] = t_far[0], t_near[0]
        if t_near[1] > t_far[1]: t_near[1], t_far[1] = t_far[1], t_near[1]

        t_entry = max(t_near[0], t_near[1])
        t_exit = min(t_far[0], t_far[1])

        if t_entry < t_exit and t_exit > 1e-6:
            final_t = t_entry
            if final_t > 1e-6:
                if final_t < distances[i]:
                    distances[i] = np.float32(final_t)
                    
    return distances

# 你也可以为 _ray_intersect_line 写一个类似的 _numba_ray_intersect_line 函数

# ===================================================================
# 2. 修改 Lidar 类，让它作为“调度器”，调用上面的纯计算函数
# ===================================================================
class Lidar:
    """
    一个用于2D环境的Lidar传感器。
    它负责准备数据并调用Numba加速的函数进行计算。
    """
    def __init__(self, n_rays: int, max_range: float, fov: float = 2 * np.pi):
        self.n_rays = n_rays
        self.max_range = float(max_range) # 确保是浮点数
        self.relative_angles = np.linspace(-fov / 2, fov / 2, n_rays, endpoint=False).astype(np.float32)

    def scan(self, agent_pos: np.ndarray, agent_angle: float, obstacles: list) -> np.ndarray:
        distances = np.full(self.n_rays, self.max_range, dtype=np.float32)
        world_angles = self.relative_angles + np.float32(agent_angle)
        # 提前计算所有射线方向
        ray_directions = np.c_[np.cos(world_angles), np.sin(world_angles)].astype(np.float32)

        for obstacle in obstacles:
            obstacle_dists = np.full(self.n_rays, self.max_range, dtype=np.float32)
            
            # --- 这里是核心修改 ---
            # 根据障碍物类型，提取出基础数据，然后调用对应的Numba函数
            if isinstance(obstacle, Circle):
                obstacle_dists = _numba_ray_intersect_circle(
                    agent_pos.astype(np.float32), ray_directions, 
                    obstacle.center.astype(np.float32), np.float32(obstacle.radius), 
                    self.n_rays, np.float32(self.max_range)
                )
            elif isinstance(obstacle, Rectangle):
                obstacle_dists = _numba_ray_intersect_rectangle(
                    agent_pos.astype(np.float32), ray_directions,
                    obstacle.center.astype(np.float32), obstacle.size.astype(np.float32), np.float32(obstacle.angle),
                    self.n_rays, np.float32(self.max_range)
                )
            # elif isinstance(obstacle, Line):
            #     # 调用 _numba_ray_intersect_line
            #     ...
            else:
                continue
            # --- 修改结束 ---

            distances = np.minimum(distances, obstacle_dists.astype(np.float32))

        return distances.astype(np.float32)
# myenv.py

import numpy as np
import warnings
from dataclasses import dataclass
import torch

# 忽略特定警告
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", message=".*The obs returned.*")
warnings.filterwarnings("ignore", message=".*precision lowered by.*")

import pygame
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random

# SB3 兼容性检查
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv


# =============================================================================
# 1. 参数整合类 (Args Class)
# =============================================================================
class Args:
    def __init__(self, map_type="static", seed=None):
        self.seed = seed
        self.dt = 0.1
        self.screen_width_px = 1200
        self.padding_px = 20
        self.color = {
            "background": (255, 255, 255),
            "boundary": (0, 0, 0),
            "obstacle": (150, 150, 150),
            "goal": (0, 255, 0),
            "first": (255, 35, 96),
            "leader": (0, 125, 255),
            "follower": (34, 216, 99),
            "first_trail": (255, 150, 150),
            "leader_trail": (150, 150, 255),
            "follower_trail": (150, 255, 150),
            "target_formation": ((150, 150, 255), (150, 255, 150)),
            "lidar_ray": (200, 200, 200),
            "comm_line": (135, 70, 205),
            "hud_text": (50, 50, 50),
        }
        if map_type == "random":
            self.screen_height_px = 800
            self.map_scale = 5.0
            self.min_x, self.max_x = 0.0, 100.0 * self.map_scale
            self.min_y, self.max_y = -20.0 * self.map_scale, 20.0 * self.map_scale
            self.spawn_offset = np.array([15.0 * self.map_scale, 0.0], dtype=np.float32)
            self.goal_pos = np.array([90.0 * self.map_scale, 0.0], dtype=np.float32)
            self.goal_radius = 2.0 * self.map_scale
            self.num_obstacle_clusters = 6
            self.obstacle_block_size = (
                np.array([3, 3], dtype=np.float32) * self.map_scale
            )
        else:  # static
            self.screen_height_px = 600
            self.map_scale = 2.0
            self.min_x, self.max_x = 0.0, 100.0 * self.map_scale
            self.min_y, self.max_y = -10.0 * self.map_scale, 10.0 * self.map_scale
            self.spawn_offset = np.array([10.0 * self.map_scale, 0.0], dtype=np.float32)
            self.goal_pos = np.array([90.0, 0.0], dtype=np.float32) * self.map_scale
            self.goal_radius = 1.0 * self.map_scale
            self.circle_pos = (
                np.array([[25.0, 0.0], [75.0, 0.0]], dtype=np.float32) * self.map_scale
            )
            self.circle_radius = 2.0 * self.map_scale
            self.rectangle_size = (
                np.array([25.0, 6.0], dtype=np.float32) * self.map_scale
            )
            rect_y = self.max_y - self.rectangle_size[1] * 0.5
            self.rectangle_pos = np.array(
                [[50.0 * self.map_scale, rect_y], [50.0 * self.map_scale, -rect_y]],
                dtype=np.float32,
            )

        self.num_leaders, self.num_followers = 3, 3
        self.num_agents = self.num_leaders + self.num_followers
        self.agent_radius = 1.5
        r_leader = np.array(
            [[np.sqrt(3), 0], [0, 1], [0, -1]], dtype=np.float32
        ) - np.array([np.sqrt(3), 0])
        r_follower = np.array(
            [[-np.sqrt(3), 2], [-np.sqrt(3), 0], [-np.sqrt(3), -2]], dtype=np.float32
        ) - np.array([np.sqrt(3), 0])
        self.nominal_config_base = np.vstack([r_leader, r_follower]).astype(np.float32)
        self.nominal_config = self.nominal_config_base * 6 * self.agent_radius
        self.agent_spawn = self.nominal_config + self.spawn_offset
        self.lidar_num_rays, self.lidar_max_range, self.lidar_fov = 16, 50.0, 2 * np.pi
        self.max_vel, self.min_vel = 15.0, -15.0
        self.max_acc, self.min_acc = 2.5, -2.5
        self.max_rot, self.min_rot = np.pi / 2, -np.pi / 2
        self.max_scale, self.min_scale = 1.5, 0.2
        self.max_shear, self.min_shear = 1.0, -1.0
        self.danger_threshold = 2 * (self.agent_radius * 4)
        self.kp_leader, self.kd_leader = 1.5, 2.2
        self.kp_follower, self.kv_follower = 0.6, 2.5
        self.stress_matrix = np.array(
            [
                [0.3461, -0.3461, -0.3461, 0.0, 0.3461, 0.0],
                [-0.3461, 0.6854, 0.0069, -0.0420, -0.6015, 0.2973],
                [-0.3461, 0.0069, 0.6853, 0.0420, -0.0908, -0.2973],
                [0.0, -0.0420, 0.0420, 0.0420, -0.0420, 0.0],
                [0.3461, -0.6015, -0.0908, -0.0420, 0.6855, -0.2973],
                [0.0, 0.2973, -0.2973, 0.0, -0.2973, 0.2973],
            ],
            dtype=np.float32,
        )
        self.neighbors = [
            np.array([1, 2, 4]),
            np.array([0, 2, 3, 4, 5]),
            np.array([0, 1, 3, 4, 5]),
            np.array([1, 2, 4]),
            np.array([0, 1, 2, 3, 5]),
            np.array([1, 2, 4]),
        ]
        self.r_goal, self.r_move = 500.0, 2.5
        self.p_stay, self.p_crash, self.p_danger = -0.8, -300.0, -80.0
        self.ep_max_step = 450


# =============================================================================
# 2. PyTorch Lidar 模块
# =============================================================================
@dataclass
class Rectangle:
    center: np.ndarray
    size: np.ndarray
    angle: float


@dataclass
class Circle:
    center: np.ndarray
    radius: float


@torch.no_grad()
def _torch_rotate_vector(v, a):
    c, s = torch.cos(a), torch.sin(a)
    r = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    return torch.matmul(r, v.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def _torch_cast_rays_to_spheres(ro, rd, sc, sr, mr):
    if sc.shape[0] == 0:
        return torch.full((rd.shape[0],), mr, device=ro.device)
    n_r, n_s = rd.shape[0], sc.shape[0]
    ro = ro.unsqueeze(0).unsqueeze(0).expand(n_r, n_s, -1)
    rd = rd.unsqueeze(1).expand(-1, n_s, -1)
    sc = sc.unsqueeze(0).expand(n_r, -1, -1)
    sr = sr.unsqueeze(0).expand(n_r, -1)
    L = ro - sc
    b = 2 * torch.sum(rd * L, -1)
    c = torch.sum(L * L, -1) - sr**2
    d = b**2 - 4 * c
    m = d >= 0
    sd = torch.sqrt(torch.where(m, d, 0.0))
    t1 = (-b - sd) / 2.0
    t2 = (-b + sd) / 2.0
    t1 = torch.where((t1 > 1e-6) & m, t1, mr)
    t2 = torch.where((t2 > 1e-6) & m, t2, mr)
    mdp = torch.min(t1, t2)
    md, _ = torch.min(mdp, 1)
    return md


@torch.no_grad()
def _torch_cast_rays_to_rectangles(ro, rd, rc, rs, ra, mr):
    if rc.shape[0] == 0:
        return torch.full((rd.shape[0],), mr, device=ro.device)
    n_r, n_re = rd.shape[0], rc.shape[0]
    ro = ro.unsqueeze(0).unsqueeze(0).expand(n_r, n_re, -1)
    rd = rd.unsqueeze(1).expand(-1, n_re, -1)
    rc = rc.unsqueeze(0).expand(n_r, -1, -1)
    rs = rs.unsqueeze(0).expand(n_r, -1, -1)
    ra = ra.unsqueeze(0).expand(n_r, -1)
    p_o_l = _torch_rotate_vector(ro - rc, -ra)
    r_d_l = _torch_rotate_vector(rd, -ra)
    s_r_d = r_d_l + 1e-8
    t_s = (-rs / 2.0 - p_o_l) / s_r_d
    t_f = (rs / 2.0 - p_o_l) / s_r_d
    t_min, _ = torch.min(torch.stack([t_s, t_f]), 0)
    t_max, _ = torch.max(torch.stack([t_s, t_f]), 0)
    t_n, _ = torch.max(t_min, -1)
    t_f, _ = torch.min(t_max, -1)
    c_m = (t_f > t_n) & (t_f > 1e-6)
    d = torch.where(c_m & (t_n > 1e-6), t_n, mr)
    md, _ = torch.min(d, 1)
    return md


class Lidar:
    def __init__(self, n, mr, fov=2 * np.pi, dev="cuda"):
        self.n_rays, self.max_range, self.device = n, float(mr), torch.device(dev)
        self.relative_angles_np = np.linspace(-fov / 2, fov / 2, n, endpoint=False)
        r_a = torch.from_numpy(self.relative_angles_np).to(self.device).float()
        self.base_ray_directions = torch.stack(
            [torch.cos(r_a), torch.sin(r_a)], -1
        ).float()

    @torch.no_grad()
    def scan(self, ap, so, ao):
        apt = torch.from_numpy(ap).to(self.device).float()
        sc = [o for o in so if isinstance(o, Circle)]
        sr = [o for o in so if isinstance(o, Rectangle)]
        ac = sc + ao
        if ac:
            scn = np.array([c.center for c in ac])
            srn = np.array([c.radius for c in ac])
            sc_t = torch.from_numpy(scn).to(self.device).float()
            sr_t = torch.from_numpy(srn).to(self.device).float()
            cd = _torch_cast_rays_to_spheres(
                apt, self.base_ray_directions, sc_t, sr_t, self.max_range
            )
        else:
            cd = torch.full((self.n_rays,), self.max_range, device=self.device)
        if sr:
            rcn = np.array([r.center for r in sr])
            rsn = np.array([r.size for r in sr])
            ran = np.array([r.angle for r in sr])
            rc_t = torch.from_numpy(rcn).to(self.device).float()
            rs_t = torch.from_numpy(rsn).to(self.device).float()
            ra_t = torch.from_numpy(ran).to(self.device).float()
            rd = _torch_cast_rays_to_rectangles(
                apt, self.base_ray_directions, rc_t, rs_t, ra_t, self.max_range
            )
        else:
            rd = torch.full((self.n_rays,), self.max_range, device=self.device)
        return torch.min(cd, rd).cpu().numpy()


# =============================================================================
# 3. 环境
# =============================================================================
class Affobstavoid(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, map_type="static", device="cuda"):
        super().__init__()
        self.args = Args(map_type=map_type)
        self.map_type, self.render_mode, self.device = map_type, render_mode, device
        self.static_obstacles = []
        self.agent_obstacle_circles = [
            Circle(pos, self.args.agent_radius) for pos in self.args.agent_spawn
        ]
        self.lidar = Lidar(
            self.args.lidar_num_rays,
            self.args.lidar_max_range,
            self.args.lidar_fov,
            self.device,
        )

        # 定义动作和观测空间
        action_low = np.array(
            [-self.args.max_acc] * 2
            + [-self.args.max_rot]
            + [-self.args.max_scale] * 2
            + [-self.args.max_shear] * 2,
            dtype=np.float32,
        )
        action_high = np.array(
            [self.args.max_acc] * 2
            + [self.args.max_rot]
            + [self.args.max_scale] * 2
            + [self.args.max_shear] * 2,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32
        )
        world_width = self.args.max_x - self.args.min_x
        obs_low = np.concatenate(
            [
                np.full(2, -world_width),
                np.full(2, -self.args.max_vel),
                np.full((self.args.num_agents - 1) * 2, -world_width),
                np.full((self.args.num_agents - 1) * 2, -2 * self.args.max_vel),
                np.zeros(self.args.num_agents * self.args.lidar_num_rays),
            ]
        )
        obs_high = np.concatenate(
            [
                np.full(2, world_width),
                np.full(2, self.args.max_vel),
                np.full((self.args.num_agents - 1) * 2, world_width),
                np.full((self.args.num_agents - 1) * 2, 2 * self.args.max_vel),
                np.full(
                    self.args.num_agents * self.args.lidar_num_rays,
                    self.args.lidar_max_range,
                ),
            ]
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        self._setup_rendering_scaling()
        self.screen, self.clock, self.font = None, None, None
        self.agent_trails = [deque(maxlen=100) for _ in range(self.args.num_agents)]

    def _setup_rendering_scaling(self):
        self.eff_w = self.args.screen_width_px - 2 * self.args.padding_px
        self.eff_h = self.args.screen_height_px - 2 * self.args.padding_px
        self.world_w, self.world_h = (
            self.args.max_x - self.args.min_x,
            self.args.max_y - self.args.min_y,
        )
        self.scale_factor = min(self.eff_w / self.world_w, self.eff_h / self.world_h)
        self.scaled_w, self.scaled_h = (
            self.world_w * self.scale_factor,
            self.world_h * self.scale_factor,
        )
        self.pad_x = (self.eff_w - self.scaled_w) / 2
        self.pad_y = (self.eff_h - self.scaled_h) / 2

    def _generate_static_obstacles(self):
        return [
            Rectangle(self.args.rectangle_pos[0], self.args.rectangle_size, 0.0),
            Rectangle(self.args.rectangle_pos[1], self.args.rectangle_size, 0.0),
            Circle(self.args.circle_pos[0], self.args.circle_radius),
            Circle(self.args.circle_pos[1], self.args.circle_radius),
        ]

    def _generate_random_obstacles(self):
        obs, b_s = [], self.args.obstacle_block_size
        shapes = {
            "I": [
                np.array([0, -1.5]),
                np.array([0, -0.5]),
                np.array([0, 0.5]),
                np.array([0, 1.5]),
            ],
            "O": [
                np.array([-0.5, -0.5]),
                np.array([0.5, -0.5]),
                np.array([-0.5, 0.5]),
                np.array([0.5, 0.5]),
            ],
            "T": [
                np.array([-1, 0]),
                np.array([0, 0]),
                np.array([1, 0]),
                np.array([0, -1]),
            ],
            "L": [
                np.array([-1, -1]),
                np.array([-1, 0]),
                np.array([0, 0]),
                np.array([1, 0]),
            ],
            "S": [
                np.array([-1, 0]),
                np.array([0, 0]),
                np.array([0, -1]),
                np.array([1, -1]),
            ],
        }
        keys = list(shapes.keys())
        s_m_x = (self.args.goal_pos[0] - self.args.agent_spawn[0][0]) * 0.1
        s_x_min, s_x_max = (
            self.args.agent_spawn[:, 0].min() + s_m_x,
            self.args.goal_pos[0] - s_m_x,
        )
        x_len = (s_x_max - s_x_min) / self.args.num_obstacle_clusters
        for i in range(self.args.num_obstacle_clusters):
            key = random.choice(keys)
            blocks = shapes[key]
            c_x = random.uniform(s_x_min + i * x_len, s_x_min + (i + 1) * x_len)
            y_m = b_s[1] * 4
            c_y = random.uniform(self.args.min_y + y_m, self.args.max_y - y_m)
            c_cen = np.array([c_x, c_y])
            ang = random.uniform(0, 2 * np.pi)
            r_mat = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            for b_l_p in blocks:
                r_pos = (b_l_p * b_s) @ r_mat.T
                r_cen = c_cen + r_pos
                obs.append(Rectangle(r_cen, b_s, ang))
        return obs

    def _generate_boundaries(self):
        t = self.world_w / 100.0
        return [
            Rectangle(
                np.array(
                    [(self.args.max_x + self.args.min_x) / 2, self.args.max_y + t / 2]
                ),
                np.array([self.world_w, t]),
                0.0,
            ),
            Rectangle(
                np.array(
                    [(self.args.max_x + self.args.min_x) / 2, self.args.min_y - t / 2]
                ),
                np.array([self.world_w, t]),
                0.0,
            ),
            Rectangle(
                np.array([self.args.min_x - t / 2, 0]), np.array([t, self.world_h]), 0.0
            ),
            Rectangle(
                np.array([self.args.max_x + t / 2, 0]), np.array([t, self.world_h]), 0.0
            ),
        ]

    def _get_obs(self):
        meas = np.array(
            [
                self.lidar.scan(
                    self.agent_pos[i],
                    self.static_obstacles,
                    self.agent_obstacle_circles[:i]
                    + self.agent_obstacle_circles[i + 1 :],
                )
                for i in range(self.args.num_agents)
            ]
        )
        return np.concatenate(
            [
                (self.args.goal_pos - self.agent_pos[0]),
                self.agent_vel[0],
                (self.agent_pos[1:] - self.agent_pos[0]).flatten(),
                (self.agent_vel[1:] - self.agent_vel[0]).flatten(),
                meas.flatten(),
            ]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.args.seed = seed
        obs = (
            self._generate_random_obstacles()
            if self.map_type == "random"
            else self._generate_static_obstacles()
        )
        self.static_obstacles = obs + self._generate_boundaries()
        self.agent_pos = self.args.agent_spawn.copy()
        self.agent_acc = np.zeros((self.args.num_agents, 2))
        self.agent_vel = np.zeros((self.args.num_agents, 2))
        self.current_formation_angle = 0.0
        for i in range(self.args.num_agents):
            self.agent_obstacle_circles[i].center = self.agent_pos[i]
            self.agent_trails[i].clear()
            self.agent_trails[i].append(self.agent_pos[i].copy())
        self.step_count = 0
        self.reward_info = {}
        self.last_action_dict = {}
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.step_count += 1
        self.old_pos = self.agent_pos.copy()
        acc, rot, scale, shear = action[0:2], action[2], action[3:5], action[5:7]
        self.last_action_dict = {"acc": acc, "rot": rot, "scale": scale, "shear": shear}
        self.current_formation_angle += (
            np.clip(rot, self.args.min_rot, self.args.max_rot) * self.args.dt
        )
        self.agent_acc[0] = np.clip(acc, self.args.min_acc, self.args.max_acc)
        self.agent_vel[0] += self.agent_acc[0] * self.args.dt
        self.agent_pos[0] += self.agent_vel[0] * self.args.dt
        self.target_pos = (
            self.apply_affine_transform(
                self.args.nominal_config,
                self.current_formation_angle,
                scale=np.clip(scale, self.args.min_scale, self.args.max_scale),
                shear=np.clip(shear, self.args.min_shear, self.args.max_shear),
            )
            + self.agent_pos[0]
        )
        l_idx = slice(1, self.args.num_leaders)
        self.agent_acc[l_idx] = self.args.kp_leader * (
            self.target_pos[l_idx] - self.agent_pos[l_idx]
        ) + self.args.kd_leader * (self.agent_vel[0] - self.agent_vel[l_idx])
        self.agent_vel[l_idx] += self.agent_acc[l_idx] * self.args.dt
        self.agent_pos[l_idx] += self.agent_vel[l_idx] * self.args.dt
        f_idx = range(self.args.num_leaders, self.args.num_agents)
        for i in f_idx:
            s_t, g_i = np.zeros(2), 0.0
            for j in self.args.neighbors[i]:
                o_ij = -self.args.stress_matrix[i, j]
                term = o_ij * (
                    self.args.kp_follower * (self.agent_pos[i] - self.agent_pos[j])
                    + self.args.kv_follower * (self.agent_vel[i] - self.agent_vel[j])
                    - self.agent_acc[j]
                )
                s_t += term
                g_i += o_ij
            self.agent_acc[i] = -s_t / g_i if g_i != 0 else np.zeros(2)
        self.agent_vel[f_idx] += self.agent_acc[f_idx] * self.args.dt
        self.agent_pos[f_idx] += self.agent_vel[f_idx] * self.args.dt
        for i in range(self.args.num_agents):
            self.agent_obstacle_circles[i].center = self.agent_pos[i]
            self.agent_trails[i].append(self.agent_pos[i].copy())
        obs = self._get_obs()
        rew, term = self.reward(obs)
        trunc = self.step_count >= self.args.ep_max_step
        if self.render_mode == "human":
            self.render()
        return obs, rew, term, trunc, self._get_info()

    def reward(self, obs):
        l_cen = np.mean(self.agent_pos[: self.args.num_leaders], axis=0)
        d_goal = np.linalg.norm(l_cen - self.args.goal_pos)
        if d_goal < self.args.goal_radius:
            print(f"\n== Goal reached at step {self.step_count} ==")
            return float(self.args.r_goal), True
        old_l_cen = np.mean(self.old_pos[: self.args.num_leaders], axis=0)
        old_d_goal = np.linalg.norm(old_l_cen - self.args.goal_pos)
        r_move = self.args.r_move * (old_d_goal - d_goal)
        p_stay = self.args.p_stay * (1 - np.tanh(0.1 * (old_d_goal - d_goal) + 1))
        lidar_start = 4 + (self.args.num_agents - 1) * 4
        all_lidar = obs[lidar_start:]
        min_dist = np.min(all_lidar)
        if min_dist < self.args.agent_radius * 2:
            p_crash = self.args.p_crash
            self.reward_info = {"total": p_crash, "crash": p_crash}
            return float(p_crash), True
        d_mask = all_lidar < self.args.danger_threshold
        p_danger = (
            self.args.p_danger
            * np.mean(1 - (all_lidar[d_mask] / self.args.danger_threshold))
            if np.any(d_mask)
            else 0.0
        )
        total = r_move + p_stay + p_danger
        self.reward_info = {
            "total": total,
            "move": r_move,
            "stay": p_stay,
            "danger": p_danger,
        }
        return float(total), False

    def _init_render(self):
        if self.render_mode and self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("AffineEnv")
                self.screen = pygame.display.set_mode(
                    (self.args.screen_width_px, self.args.screen_height_px)
                )
            else:
                self.screen = pygame.Surface(
                    (self.args.screen_width_px, self.args.screen_height_px)
                )
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 16)

    def render(self):
        if self.render_mode is None:
            return
        if self.screen is None:
            self._init_render()
        canvas = self.screen
        canvas.fill(self.args.color["background"])

        def w2s(pos):
            x = int(
                (pos[0] - self.args.min_x) * self.scale_factor
                + self.args.padding_px
                + self.pad_x
            )
            y = int(
                self.scaled_h
                - (pos[1] - self.args.min_y) * self.scale_factor
                + self.args.padding_px
                + self.pad_y
            )
            return (x, y)

        for obs in self.static_obstacles:
            if isinstance(obs, Circle):
                pygame.draw.circle(
                    canvas,
                    self.args.color["obstacle"],
                    w2s(obs.center),
                    int(obs.radius * self.scale_factor),
                )
            elif isinstance(obs, Rectangle):
                c, s = np.cos(-obs.angle), np.sin(-obs.angle)
                w, h = obs.size[0] * self.scale_factor, obs.size[1] * self.scale_factor
                corners = np.array(
                    [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
                ) @ np.array([[c, -s], [s, c]])
                pygame.draw.polygon(
                    canvas, self.args.color["obstacle"], corners + w2s(obs.center)
                )
        pygame.draw.circle(
            canvas,
            self.args.color["goal"],
            w2s(self.args.goal_pos),
            int(self.args.goal_radius * self.scale_factor),
            3,
        )
        for i in range(self.args.num_agents):
            if len(self.agent_trails[i]) > 1:
                pygame.draw.aalines(
                    canvas,
                    self.args.color["leader_trail"],
                    False,
                    [w2s(p) for p in self.agent_trails[i]],
                )
        for i in range(self.args.num_agents):
            color = (
                self.args.color["first"]
                if i == 0
                else (
                    self.args.color["leader"]
                    if i < self.args.num_leaders
                    else self.args.color["follower"]
                )
            )
            pygame.draw.circle(
                canvas,
                color,
                w2s(self.agent_pos[i]),
                int(self.args.agent_radius * self.scale_factor),
            )
        if self.reward_info:
            y_off = 10
            for k, v in self.reward_info.items():
                canvas.blit(
                    self.font.render(f"{k}:{v:.2f}", True, self.args.color["hud_text"]),
                    (10, y_off),
                )
                y_off += 20
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

    def _get_info(self):
        return {}

    def apply_affine_transform(
        self, coords, rot, trans=np.zeros(2), scale=np.ones(2), shear=np.zeros(2)
    ):
        c_h = np.hstack((coords, np.ones((self.args.num_agents, 1))))
        c, s, sx, sy, shx, shy, tx, ty = (
            np.cos(rot),
            np.sin(rot),
            scale[0],
            scale[1],
            shear[0],
            shear[1],
            trans[0],
            trans[1],
        )
        T = np.array(
            [
                [c * sx + s * shy * sx, c * shx * sy - s * sy, tx],
                [s * sx - c * shy * sx, s * shx * sy + c * sy, ty],
                [0.0, 0.0, 1.0],
            ]
        )
        return (T @ c_h.T).T[:, :2]

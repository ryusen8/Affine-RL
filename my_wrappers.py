import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict

class FlattenDictObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.keys = sorted(self.observation_space.spaces.keys()) # 确保您的字典键顺序正确

        flat_obs_dim = 0
        low_bounds = []
        high_bounds = []

        for key in self.keys:
            space = self.observation_space.spaces[key]
            if isinstance(space, gym.spaces.Box):
                flat_obs_dim += np.prod(space.shape)
                low_bounds.extend(space.low.flatten().tolist())
                high_bounds.extend(space.high.flatten().tolist())
            else:
                raise NotImplementedError(f"Observation space dict contains unsupported space type: {type(space)}. Only Box spaces within Dict are currently supported by this wrapper.")
        
        self.observation_space = gym.spaces.Box(
            low=np.array(low_bounds, dtype=np.float32),
            high=np.array(high_bounds, dtype=np.float32),
            dtype=np.float32
        )

    def observation(self, obs):
        flattened_obs = []
        for key in self.keys:
            flattened_obs.extend(obs[key].flatten().tolist())
        return np.array(flattened_obs, dtype=np.float32)

class FlattenDictAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.keys = sorted(self.action_space.spaces.keys()) # 确保与环境定义的一致

        flat_action_dim = 0
        low_bounds = []
        high_bounds = []

        for key in self.keys:
            space = self.action_space.spaces[key]
            if isinstance(space, gym.spaces.Box):
                flat_action_dim += np.prod(space.shape)
                low_bounds.extend(space.low.flatten().tolist())
                high_bounds.extend(space.high.flatten().tolist())
            else:
                raise NotImplementedError(f"Action space dict contains unsupported space type: {type(space)}. Only Box spaces within Dict are currently supported by this wrapper.")
        
        self.action_space = gym.spaces.Box(
            low=np.array(low_bounds, dtype=np.float32),
            high=np.array(high_bounds, dtype=np.float32),
            dtype=np.float32
        )

    def action(self, action_flat):
        action_dict = {}
        current_idx = 0
        for key in self.keys:
            space = self.env.action_space.spaces[key] # 获取原始环境的子空间定义
            size = np.prod(space.shape)
            action_dict[key] = np.array(action_flat[current_idx : current_idx + int(size)], dtype=np.float32).reshape(space.shape)
            current_idx += int(size)
        return action_dict
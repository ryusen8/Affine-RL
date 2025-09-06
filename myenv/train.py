# train.py

import os
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# 从您的环境文件中导入Env类
from myenv import Affobstavoid

# =============================================================================
# 1. 训练配置
# =============================================================================
# --- 核心参数 ---
TOTAL_TIMESTEPS = 1_000_000
NUM_CPU = 4
MAP_TYPE = 'random'

# --- 路径设置 ---
log_dir = "logs/"
models_dir = "models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# --- 模型保存频率 ---
SAVE_FREQ = max(20000 // NUM_CPU, 1)

# =============================================================================
# 2. 主训练逻辑
# =============================================================================
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"================ 使用设备: {device} ================")
    print(f"================ 并行环境数: {NUM_CPU} ================")

    # --- 创建向量化环境 ---
    env_kwargs = {'map_type': MAP_TYPE, 'device': device, 'render_mode': None}
    
    vec_env = make_vec_env(
        Affobstavoid,
        n_envs=NUM_CPU,
        seed=0,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv
    )

    # [FIXED] 移除了 gym.wrappers.RescaleAction 包装器。
    # Stable Baselines 3 的 SAC 算法会自动处理非归一化动作空间的缩放。
    # 其内部策略网络使用 tanh 将动作输出压缩到 [-1, 1]，
    # 然后在与环境交互时，框架会自动将此动作反向缩放到环境的实际动作范围。

    # --- 设置模型保存的回调函数 ---
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=models_dir,
        name_prefix="sac_affine_model"
    )

    # --- 定义 SAC 模型 ---
    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=2e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.99,
        tau=0.01,
        verbose=1,
        tensorboard_log=log_dir,
        device=device
    )

    print("================ 开始训练 ================")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True
    )

    final_model_path = os.path.join(models_dir, "sac_affine_final")
    model.save(final_model_path)
    print(f"================ 训练完成，最终模型保存在: {final_model_path} ================")

    vec_env.close()
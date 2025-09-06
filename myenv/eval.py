# evaluate.py

import os
import torch
import time
import gymnasium as gym

from stable_baselines3 import SAC

# 从您的环境文件中导入AffineEnv类
from myenv import Affobstavoid

# =============================================================================
# 1. 评估配置
# =============================================================================

# --- 要加载的模型路径 ---
# !! 重要 !!: 请将此路径修改为您想要测试的模型文件
# 它可以是最终模型 "sac_affine_final.zip"
# 也可以是训练过程中保存的检查点，例如 "sac_affine_model_120000_steps.zip"
MODEL_PATH = "D:\\Main Code\\Python\\affine_rl\\models\\sac_affine_final.zip"

# --- 测试参数 ---
NUM_TEST_EPISODES = 10       # 运行多少个测试回合
MAP_TYPE = 'random'          # 在哪种地图上测试: 'random' 或 'static'

# =============================================================================
# 2. 主评估逻辑
# =============================================================================
if __name__ == '__main__':
    # --- 检查模型文件是否存在 ---
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件未找到，请检查路径: {MODEL_PATH}")
        exit()

    # --- 确定设备 ---
    # 评估时使用CPU通常足够，且可以避免潜在的CUDA内存问题
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"================ 使用设备: {device} ================")
    
    # --- 创建单个、可见的环境 ---
    # 注意:
    # 1. 这里我们不再使用向量化环境(VecEnv)，因为我们只想看一个实例的表现。
    # 2. `render_mode` 必须设置为 "human" 才能看到窗口。
    env = Affobstavoid(
        render_mode="human",
        map_type=MAP_TYPE,
        device=device
    )
    
    # --- 加载模型 ---
    print(f"正在从 {MODEL_PATH} 加载模型...")
    try:
        model = SAC.load(MODEL_PATH, env=env, device=device)
        print("模型加载成功！")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        env.close()
        exit()

    # --- 运行评估循环 ---
    total_rewards = []
    for episode in range(NUM_TEST_EPISODES):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        
        print(f"\n=============== 开始测试回合 {episode + 1}/{NUM_TEST_EPISODES} ===============")

        while not terminated and not truncated:
            # `deterministic=True` 表示我们总是选择模型认为最优的动作，而不是像训练时那样进行随机探索
            action, _states = model.predict(obs, deterministic=True)
            
            # 与环境交互
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward

        print(f"回合结束。总奖励: {episode_reward:.2f}")
        total_rewards.append(episode_reward)
        
        # 在回合结束后暂停2秒，以便观察最终状态
        time.sleep(2)
        
    # --- 清理并报告结果 ---
    env.close()
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    print("\n=============== 评估完成 ===============")
    print(f"总共测试了 {NUM_TEST_EPISODES} 个回合。")
    print(f"平均奖励: {avg_reward:.2f}")
    print("==========================================")
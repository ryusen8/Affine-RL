import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import os
from datetime import datetime
from tqdm import tqdm
from my_wrappers import DictActionWrapper, DictObservationWrapper
import affine_gym_env

import sys
import os

print("--- sys.path for debugging ---")
for p in sys.path:
    print(p)
print("--- End sys.path for debugging ---")

# 确认当前工作目录
print(f"Current working directory: {os.getcwd()}")

# --- 0. 参数配置类 ---
class SACConfig:
    def __init__(self, env_name="Pendulum-v1"):
        self.env_name = env_name
        self.num_episodes = 20 # 训练的总回合数
        self.max_steps_per_episode = 500 # 每个回合的最大步数
        self.log_interval = 10 # 每隔多少个回合打印一次日志
        self.save_results = True # 是否保存训练结果 (模型和奖励曲线)

        # SAC 算法超参数
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.alpha_lr = 3e-4 # 温度系数 alpha 的学习率,一般地，温度系数的学习率和网络参数的学习率保持一致
        self.gamma = 0.99 # 折扣因子
        self.tau = 0.005 # 软更新因子
        self.alpha = 0.2 # 初始温度参数 (如果使用自动熵调整，此值会被覆盖)
        self.buffer_capacity = 1_000_000 # 经验回放缓冲区容量
        self.batch_size = 256 # 训练批次大小

        # Actor 网络参数
        self.actor_hidden_size = 256
        self.log_std_min = -20
        self.log_std_max = 2

        # Critic 网络参数
        self.critic_hidden_size = 256

        # 绘图参数
        self.smoothing_window = 10 # 奖励曲线平滑窗口大小

        # 测试参数
        self.num_test_episodes = 5 # 测试回合数

# --- 1. 定义网络结构 ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, log_std_min, log_std_max):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.log_std_linear = nn.Linear(hidden_size, action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()
        # Q1 network
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2_q1 = nn.Linear(hidden_size, hidden_size)
        self.fc3_q1 = nn.Linear(hidden_size, 1)

        # Q2 network
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2_q2 = nn.Linear(hidden_size, hidden_size)
        self.fc3_q2 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = torch.relu(self.fc1_q1(sa))
        q1 = torch.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)

        q2 = torch.relu(self.fc1_q2(sa))
        q2 = torch.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)
        return q1, q2

# --- 2. 经验回放缓冲区 ---
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (torch.tensor(np.array(states), dtype=torch.float32).to(self.device),
                torch.tensor(np.array(actions), dtype=torch.float32).to(self.device),
                torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(self.device),
                torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device),
                torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(self.device))

    def __len__(self):
        return len(self.buffer)

# --- 3. SAC 算法核心 ---
class SAC:
    def __init__(self, state_dim, action_dim, action_space, device, config: SACConfig):
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.batch_size = config.batch_size
        self.device = device

        self.actor = Actor(state_dim, action_dim, config.actor_hidden_size, config.log_std_min, config.log_std_max).to(device)
        self.critic = Critic(state_dim, action_dim, config.critic_hidden_size).to(device)
        self.critic_target = Critic(state_dim, action_dim, config.critic_hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # 自动熵调整
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)

        self.replay_buffer = ReplayBuffer(config.buffer_capacity, device)

        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32, device=device)
        self.action_bias = torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32, device=device)

    def select_action(self, state, evaluate=False):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        if evaluate is False:
            action, _ = self.actor.sample(state)
        else:
            mean, _ = self.actor.forward(state)
            action = torch.tanh(mean) 
        
        # 在转换为 NumPy 数组之前分离梯度
        return (action * self.action_scale + self.action_bias).squeeze(0).detach().cpu().numpy()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        # 将动作缩放回 [-1, 1] 范围，以匹配 Actor 网络的输出范围
        action_batch_scaled = (action_batch - self.action_bias) / self.action_scale

        # --- Critic Loss ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state_batch)
            # 确保传递给 Critic 目标网络的动作是原始的、未缩放的动作（即环境中的实际动作）
            next_q1_target, next_q2_target = self.critic_target(next_state_batch, next_action * self.action_scale + self.action_bias) 
            min_next_q_target = torch.min(next_q1_target, next_q2_target) - self.alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * self.gamma * min_next_q_target

        current_q1, current_q2 = self.critic(state_batch, action_batch) # 使用回放缓冲区中的原始动作批次
        critic_loss = torch.mean((current_q1 - target_q).pow(2)) + torch.mean((current_q2 - target_q).pow(2))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Loss ---
        new_action, log_prob = self.actor.sample(state_batch)
        # 确保传递给 Critic 的动作是原始的、未缩放的动作
        q1_new_action, q2_new_action = self.critic(state_batch, new_action * self.action_scale + self.action_bias) 
        min_q_new_action = torch.min(q1_new_action, q2_new_action)
        
        actor_loss = ((self.alpha * log_prob) - min_q_new_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha (温度) Loss ---
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # --- 更新目标网络 ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# --- 4. 训练函数 ---
def train_sac(config: SACConfig):
    env = gym.make(config.env_name, render_mode="rgb_array")
    
    # 检查并应用 Dict Wrappers
    if isinstance(env.observation_space, gym.spaces.Dict):
        print("Applying DictObservationWrapper...")
        env = DictObservationWrapper(env)
    if isinstance(env.action_space, gym.spaces.Dict):
        print("Applying DictActionWrapper...")
        env = DictActionWrapper(env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_space = env.action_space

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")
    if str(device) == 'cpu':
        print("WARNING: CUDA (GPU) is not available. Training will be performed on CPU, which might be slow.")
        
    agent = SAC(state_dim, action_dim, action_space, device, config)

    episode_rewards = []
    
    total_steps = 0
    
    # 使用 tqdm 包装训练循环
    with tqdm(total=config.num_episodes, desc=f"Training {config.env_name} (SAC)") as pbar:
        for episode in range(config.num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            for step in range(config.max_steps_per_episode):
                action = agent.select_action(state)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.replay_buffer.push(state, action, reward, next_state, done)
                agent.update()

                state = next_state
                episode_reward += reward
                total_steps += 1

                if done:
                    break
            
            episode_rewards.append(episode_reward)

            # 更新 tqdm 进度条的后缀信息
            pbar.set_postfix({
                'reward': f'{episode_reward:.2f}', 
                'avg_reward': f'{np.mean(episode_rewards[-config.log_interval:]):.2f}' if len(episode_rewards) >= config.log_interval else 'N/A',
                'steps': total_steps
            })
            pbar.update(1) # 更新进度条1步

            # if (episode + 1) % config.log_interval == 0:
            #     avg_reward_log = np.mean(episode_rewards[-config.log_interval:])
            #     print(f"Episode: {episode + 1}/{config.num_episodes}, Avg Reward: {avg_reward_log:.2f}, Total Steps: {total_steps}, Alpha: {agent.alpha.item():.4f}")

    env.close()

    # --- 保存结果到带时间戳和奖励的文件夹 ---
    saved_model_path = None # 用于存储最终的模型路径
    if config.save_results:
        # 计算最后log_interval回合的平均奖励作为文件夹名的一部分
        final_avg_reward = np.mean(episode_rewards[-config.log_interval:]) if len(episode_rewards) >= config.log_interval else np.mean(episode_rewards)
        
        # 获取当前时间并格式化
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # 创建文件夹名，使其与模型文件命名规则一致
        # 将小数点替换为下划线，以避免文件名中的特殊字符问题
        folder_name = f"{config.env_name}_SAC_{timestamp}_Reward{final_avg_reward:.2f}".replace('.', '_')
        
        # 创建完整路径
        save_dir = os.path.join("train_results", folder_name)
        os.makedirs(save_dir, exist_ok=True)

        # 模型文件名
        model_filename = "sac_actor_model.pth"
        saved_model_path = os.path.join(save_dir, model_filename)
        torch.save(agent.actor.state_dict(), saved_model_path) 
        print(f"Model saved to {saved_model_path}")

        # 奖励曲线图文件名
        plot_rewards_filename = "reward_curve.png"
        plot_rewards_path = os.path.join(save_dir, plot_rewards_filename)
        plot_rewards(episode_rewards, config.env_name, config.smoothing_window, plot_rewards_path) # 传入平滑窗口
        print(f"Reward curve saved to {plot_rewards_path}")
    else:
        # 如果不保存结果，仍然绘制曲线显示
        plot_rewards(episode_rewards, config.env_name, config.smoothing_window)

    return episode_rewards, saved_model_path # 返回奖励列表和保存的模型路径

# --- 5. 绘制奖励曲线 ---
# 增加一个 save_path 参数来保存图片，并增加 smoothing_window 参数
def plot_rewards(rewards, env_name, smoothing_window=1, save_path=None):
    plt.figure(figsize=(12, 7)) # 稍微大一点的图
    plt.plot(rewards, label='Raw Rewards', alpha=0.6, color='blue') # 原始奖励曲线

    # 计算平滑后的奖励
    if len(rewards) >= smoothing_window:
        smoothed_rewards = np.convolve(rewards, np.ones(smoothing_window)/smoothing_window, mode='valid')
        plt.plot(np.arange(len(smoothed_rewards)) + smoothing_window - 1, smoothed_rewards, label=f'Smoothed Rewards)', color='red', linewidth=2)
    else:
        print(f"Not enough data for smoothing with window {smoothing_window}. Plotting raw rewards only.")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"SAC Training Reward Curve for {env_name}")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close() # 关闭图形，防止在非交互式环境中显示
    else:
        plt.show()

# --- 6. 测试模型并渲染 ---
def test_model(config: SACConfig, model_path_override=None):
    env = gym.make(config.env_name, render_mode="human")
    
    # 检查并应用 Dict Wrappers (与训练时保持一致)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = DictObservationWrapper(env)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = DictActionWrapper(env)
        
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_space = env.action_space

    test_device = torch.device("cpu")
    
    # 确定模型加载路径
    model_to_load = model_path_override
    if not model_to_load: # 如果没有显式指定，则尝试查找最新模型
        print("No specific model path provided for testing. Attempting to find the latest model in 'results' directory.")
        latest_run_dir = None
        if os.path.exists("results"):
            run_dirs = [os.path.join("results", d) for d in os.listdir("results") if os.path.isdir(os.path.join("results", d))]
            if run_dirs:
                # Sort by modification time (most recent first)
                run_dirs.sort(key=os.path.getmtime, reverse=True) 
                latest_run_dir = run_dirs[0]
                model_to_load = os.path.join(latest_run_dir, "sac_actor_model.pth") # 确保文件名一致
                print(f"Found latest model: {model_to_load}")
            else:
                print("No run directories found in 'results'. Cannot test.")
                env.close()
                return
        else:
            print("'results' directory not found. Cannot test.")
            env.close()
            return
            
    if not os.path.exists(model_to_load):
        print(f"Error: Model file not found at {model_to_load}. Skipping test.")
        env.close()
        return

    print(f"\n--- Testing model: {model_to_load} on {config.env_name} (using device: {test_device}) ---")

    agent = SAC(state_dim, action_dim, action_space, test_device, config) 
    agent.actor.load_state_dict(torch.load(model_to_load, map_location=test_device))
    agent.actor.eval() # Set actor to evaluation mode

    test_rewards = []
    for episode in range(config.num_test_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(env.spec.max_episode_steps): 
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            env.render()
            if done:
                break
        test_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}")

    avg_test_reward = np.mean(test_rewards)
    print(f"Average Test Reward over {config.num_test_episodes} episodes: {avg_test_reward:.2f}")
    env.close()

# --- 7. 主运行逻辑 ---
if __name__ == "__main__":
    # 实例化配置类
    config = SACConfig(env_name="affine_gym_env/AffineEnv")
    # 可以通过修改 config 实例的属性来调整参数
    # config.num_episodes = 500
    # config.lr = 1e-3
    # config.smoothing_window = 20 # 调整平滑窗口大小

    print(f"Starting training for {config.env_name} with configuration:")
    for attr, value in vars(config).items():
        print(f"  {attr}: {value}")

    # 训练 SAC 算法
    # train_sac 现在会返回保存的模型路径
    rewards, saved_model_path = train_sac(config)
    
    # 测试模型并渲染
    # 如果模型成功保存，就用保存的路径进行测试
    if saved_model_path:
        test_model(config, model_path_override=saved_model_path)
    else:
        # 如果没有保存，或者 saved_model_path 为 None，则尝试查找最新模型
        test_model(config) # 会触发 test_model 内部的最新模型查找逻辑
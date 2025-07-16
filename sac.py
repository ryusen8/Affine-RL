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
from affine_gym_env.envs.affine_utils.arg import TrainArg
import imageio # <--- 新增: 导入 imageio 库用于创建 GIF

# --- 0. 参数配置类 ---
class SACConfig:
    def __init__(self, env_name="Pendulum-v1"):
        self.env_name = env_name
        self.num_episodes = TrainArg.NUM_EP # 训练的总回合数
        self.max_steps_per_episode = TrainArg.EP_MAX_STEP # 每个回合的最大步数
        self.log_interval = TrainArg.LOG_INTERVAL # 每隔多少个回合打印一次日志
        self.save_results = True # 是否保存训练结果 (模型和奖励曲线)

        # SAC 算法超参数
        self.actor_lr = TrainArg.ACTOR_LR
        self.critic_lr = TrainArg.CRITIC_LR
        self.alpha_lr = TrainArg.ALPHA_LR # 温度系数 alpha 的学习率,一般地，温度系数的学习率和网络参数的学习率保持一致
        self.gamma = TrainArg.GAMMA # 折扣因子
        self.tau = TrainArg.TAU # 软更新因子
        self.alpha = TrainArg.ALPHA_INIT # 初始温度参数 (如果使用自动熵调整，此值会被覆盖)
        self.buffer_capacity = TrainArg.BUFFER_SIZE # 经验回放缓冲区容量
        self.batch_size = TrainArg.BATCH_SIZE # 训练批次大小

        # Actor 网络参数
        self.actor_hidden_size = TrainArg.ACTOR_HIDDEN_SIZE
        self.log_std_min = -20
        self.log_std_max = 2

        # Critic 网络参数
        self.critic_hidden_size = TrainArg.CRITIC_HIDDEN_SIZE

        # 绘图参数
        self.smoothing_window = TrainArg.SMOOTH # 奖励曲线平滑窗口大小

        # 测试参数
        self.num_test_episodes = TrainArg.NUM_TEST_EP # 测试回合数
        # <--- 新增: GIF 保存相关的参数 ---
        self.gif_save_interval = TrainArg.GIF_INTERVAL # 每隔多少个回合保存一次 GIF
        self.gif_fps = 60 # 生成的 GIF 的帧率
        self.reward_scale = TrainArg.REWARD_SCALE
# --- 1. 定义网络结构 ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, log_std_min, log_std_max):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size) # 新增: 第一个归一化层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size) # 新增: 第二个归一化层
        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.log_std_linear = nn.Linear(hidden_size, action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        # x = torch.relu(self.fc1(state))
        # x = torch.relu(self.fc2(x))
        x = self.fc1(state)
        x = self.ln1(x) # 应用
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = self.ln2(x) # 应用
        x = torch.relu(x)
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
        # log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()
        # Q1 network
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.ln1_q1 = nn.LayerNorm(hidden_size) # 新增
        self.fc2_q1 = nn.Linear(hidden_size, hidden_size)
        self.ln2_q1 = nn.LayerNorm(hidden_size) # 新增
        self.fc3_q1 = nn.Linear(hidden_size, 1)

        # Q2 network
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_size)
        self.ln1_q2 = nn.LayerNorm(hidden_size) # 新增
        self.fc2_q2 = nn.Linear(hidden_size, hidden_size)
        self.ln2_q2 = nn.LayerNorm(hidden_size) # 新增
        self.fc3_q2 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        # q1 = torch.relu(self.fc1_q1(sa))
        # q1 = torch.relu(self.fc2_q1(q1))
        # q1 = self.fc3_q1(q1)

        # q2 = torch.relu(self.fc1_q2(sa))
        # q2 = torch.relu(self.fc2_q2(q2))
        # q2 = self.fc3_q2(q2)

        # --- 修改点: 在 Q1 中应用层归一化 ---
        q1 = self.fc1_q1(sa)
        q1 = self.ln1_q1(q1)
        q1 = torch.relu(q1)
        q1 = self.fc2_q1(q1)
        q1 = self.ln2_q1(q1)
        q1 = torch.relu(q1)
        q1 = self.fc3_q1(q1)

        # --- 修改点: 在 Q2 中应用层归一化 ---
        q2 = self.fc1_q2(sa)
        q2 = self.ln1_q2(q2)
        q2 = torch.relu(q2)
        q2 = self.fc2_q2(q2)
        q2 = self.ln2_q2(q2)
        q2 = torch.relu(q2)
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

def test_and_save_gif(agent, config, episode, save_path):
    """
    用当前模型测试一个回合，并将渲染过程保存为 GIF。
    
    参数:
    - agent: 正在训练的 SAC agent 实例。
    - config: SACConfig 实例。
    - episode: 当前的回合数，用于打印日志。
    - save_path: GIF 文件的完整保存路径。
    """
    frames = []
    # 创建一个用于测试和渲染的环境，渲染模式为 "rgb_array" 以便获取图像帧
    test_env = gym.make(config.env_name, render_mode="rgb_array")
    
    # 确保应用与训练时相同的环境装饰器
    if isinstance(test_env.observation_space, gym.spaces.Dict):
        test_env = DictObservationWrapper(test_env)
    if isinstance(test_env.action_space, gym.spaces.Dict):
        test_env = DictActionWrapper(test_env)

    try:
        state, _ = test_env.reset()
        
        # 关键：将 actor 网络设置为评估模式
        agent.actor.eval()
        
        # 运行一个完整的回合
        for step in range(config.max_steps_per_episode):
            # 渲染当前帧并存入列表
            frame = test_env.render()
            frames.append(frame)
            
            # 选择确定性动作进行评估
            action = agent.select_action(state, evaluate=True)
            next_state, _, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            state = next_state
            
            if done:
                break
        
        print(f"\nEpisode {episode}: Saving preview GIF to {save_path}...")
        # 使用 imageio 保存帧序列为 GIF
        imageio.mimsave(save_path, frames, fps=config.gif_fps)
        print("GIF saved successfully.")

    except Exception as e:
        print(f"Error during GIF generation for episode {episode}: {e}")
    finally:
        # 无论成功与否，都要关闭测试环境
        test_env.close()
        # 关键：将 actor 网络恢复为训练模式
        agent.actor.train()

# --- 4. 训练函数 ---
def train_sac(config: SACConfig):
    env = gym.make(config.env_name, render_mode="rgb_array")
    
    # 检查并应用 Dict Wrappers
    if isinstance(env.observation_space, gym.spaces.Dict):
        # print("Applying DictObservationWrapper...")
        env = DictObservationWrapper(env)
    if isinstance(env.action_space, gym.spaces.Dict):
        # print("Applying DictActionWrapper...")
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

    # 存放gif、模型和奖励曲线的文件夹
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"AffineEnv_SAC_{run_timestamp}"
    save_dir = os.path.join("test_model", folder_name)
    os.makedirs(save_dir, exist_ok=True)

    run_gif_dir = os.path.join(save_dir, f"train_preview")
    os.makedirs(run_gif_dir, exist_ok=True)
    print(f"Training preview GIFs will be saved in: {run_gif_dir}")   

    # 使用 tqdm 包装训练循环
    with tqdm(total=config.num_episodes, desc=f"{config.env_name} w/ SAC") as pbar:
        for episode in range(config.num_episodes):
            state, _ = env.reset()
            episode_steps = 0
            episode_reward = 0
            for step in range(config.max_steps_per_episode):
                action = agent.select_action(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                scaled_reward = reward * config.reward_scale
                done = terminated or truncated

                agent.replay_buffer.push(state, action, scaled_reward, next_state, done)
                agent.update()

                state = next_state
                episode_reward += reward
                total_steps += 1
                episode_steps += 1

                if done:
                    break

            episode_rewards.append(episode_reward)

            # 更新 tqdm 进度条的后缀信息
            pbar.set_postfix({
                'rew': f'{episode_reward:.2f}', 
                'avg_rew': f'{np.mean(episode_rewards[-config.log_interval:]):.2f}' if len(episode_rewards) >= config.log_interval else 'N/A',
                'stp_used': episode_steps,
                # 'finish': info['finish'],
                # 'fail': info['fail'],
            })
            pbar.update(1) # 更新进度条1步
            # <--- 新增: 定期测试并保存 GIF ---
            # 检查是否达到保存间隔，并且不是第0个回合
            if (episode + 1) % config.gif_save_interval == 0 and episode > 0:
                gif_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                # 构造 GIF 文件名：时间戳 + episode轮数.gif
                gif_filename = f"{gif_timestamp}_ep{episode + 1}.gif"
                gif_path = os.path.join(run_gif_dir, gif_filename)
                
                # 调用测试和保存函数
                test_and_save_gif(agent, config, episode + 1, gif_path)

    env.close()
    
    # --- 保存结果到带时间戳和奖励的文件夹 ---    
    saved_model_path = None # 用于存储最终的模型路径
    if config.save_results:
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
    env = gym.make(config.env_name, render_mode="rgb_array")

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
        print("No specific model path provided for testing. Attempting to find the latest model in 'test_model' directory.")
        latest_run_dir = None
        if os.path.exists("test_model"):
            run_dirs = [os.path.join("test_model", d) for d in os.listdir("test_model") if os.path.isdir(os.path.join("test_model", d))]
            if run_dirs:
                # Sort by modification time (most recent first)
                run_dirs.sort(key=os.path.getmtime, reverse=True) 
                latest_run_dir = run_dirs[0]
                model_to_load = os.path.join(latest_run_dir, "sac_actor_model.pth") # 确保文件名一致
                print(f"Found latest model: {model_to_load}")
            else:
                print("No run directories found in 'test_model'. Cannot test.")
                env.close()
                return
        else:
            print("'test_model' directory not found. Cannot test.")
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
    frames = [] # <--- 新增: 用于存储第一个回合的帧

    for episode in range(config.num_test_episodes):
        state, _ = env.reset()
        episode_reward = 0
        max_steps = env.spec.max_episode_steps if env.spec.max_episode_steps else config.max_steps_per_episode
        for step in range(max_steps): 
            # <--- 修改: 仅在第一个回合捕获帧 ---
            if episode == 0:
                frame = env.render()
                frames.append(frame)
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

    # <--- 新增: 在所有测试回合结束后，保存第一个回合的 GIF ---
    if frames:
        # 确定 GIF 的保存路径，与模型在同一个文件夹下
        model_dir = os.path.dirname(model_to_load)
        gif_path = os.path.join(model_dir, "final_test_preview.gif")
        
        try:
            print(f"\nSaving final test preview GIF to: {gif_path}")
            imageio.mimsave(gif_path, frames, fps=config.gif_fps)
            print("GIF saved successfully.")
        except Exception as e:
            print(f"Error saving final test GIF: {e}")

    avg_test_reward = np.mean(test_rewards)
    print(f"Average Test Reward over {config.num_test_episodes} episodes: {avg_test_reward:.2f}")
    env.close()

# --- 7. 主运行逻辑 ---
if __name__ == "__main__":
    # 实例化配置类
    config = SACConfig(env_name=TrainArg.ENV_NAME)

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
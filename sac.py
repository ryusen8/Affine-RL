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
from my_wrappers import DictActionWrapper, DictObservationWrapper, NormalizeObservationWrapper
import affine_gym_env
from affine_gym_env.envs.affine_utils.arg import TrainArg
import imageio # <--- 新增: 导入 imageio 库用于创建 GIF

# --- 0. 参数配置类 ---
class SACConfig:
    def __init__(self, env_name="affine_gym_env/AffineEnv"):
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
        self.log_std_min = TrainArg.LOG_STD_MIN
        self.log_std_max = TrainArg.LOG_STD_MAX

        # Critic 网络参数
        self.critic_hidden_size = TrainArg.CRITIC_HIDDEN_SIZE

        # 绘图参数
        self.smoothing_window = TrainArg.SMOOTH # 奖励曲线平滑窗口大小

        # 测试参数
        self.num_test_episodes = TrainArg.NUM_TEST_EP # 测试回合数
        # <--- 新增: GIF 保存相关的参数 ---
        self.gif_save_interval = TrainArg.GIF_INTERVAL # 每隔多少个回合保存一次 GIF
        self.gif_fps = TrainArg.GIF_FPS # 生成的 GIF 的帧率
        self.reward_scale = TrainArg.REWARD_SCALE

# --- 1. 定义网络结构 ---
class Actor(nn.Module):
    """
    升级版的 Actor 网络，借鉴了 ElegantRL 的设计。
    - 分离式网络结构：共享的状态特征提取 + 独立的均值/标准差头。
    - 混合激活函数：结合使用 ReLU 和 Hardswish。
    - 集成层归一化（LayerNorm）以增强稳定性。
    """
    def __init__(self, state_dim, action_dim, hidden_size, log_std_min, log_std_max):
        super(Actor, self).__init__()
        
        # 1. 共享的状态特征提取网络 (State Feature Extractor)
        #    负责将高维的状态映射到一个有意义的特征向量。
        self.net_state = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size), # 在激活函数前进行归一化
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # 2. 计算动作均值(mean)的“头”网络
        #    使用 Hardswish 增加非线性表达能力。
        self.net_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Hardswish(),
            nn.Linear(hidden_size // 2, action_dim)
        )
        
        # 3. 计算动作对数标准差(log_std)的“头”网络
        self.net_log_std = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Hardswish(),
            nn.Linear(hidden_size // 2, action_dim)
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 为数值稳定的 log_prob 计算做准备
        self.soft_plus = nn.Softplus()
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        """
        前向传播，计算均值和对数标准差。
        """
        # 首先通过共享网络提取状态特征
        state_feature = self.net_state(state)
        # 然后基于特征分别计算均值和对数标准差
        mean = self.net_mean(state_feature)
        log_std = self.net_log_std(state_feature)
        
        # 裁剪 log_std 到一个合理的范围，防止标准差过大或过小
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def sample(self, state):
        """
        采样一个动作，并计算其对数概率（使用数值稳定方法）。
        """
        mean, log_std = self.forward(state)
        
        # 1. 创建高斯噪声
        noise = torch.randn_like(mean)
        # 2. 通过重参数化技巧得到应用 tanh 之前的动作
        pre_tanh_action = mean + log_std.exp() * noise
        
        # 3. 计算高斯分布的 log_prob (数值稳定版)
        log_prob = -(log_std + self.log_sqrt_2pi + noise.pow(2) / 2)

        # 4. 应用 tanh 变换的修正 (数值稳定版)
        log_prob -= 2 * (np.log(2.0) - pre_tanh_action - self.soft_plus(-2.0 * pre_tanh_action))

        # 5. 对多维动作的 log_prob 求和，使其成为一个标量
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=1, keepdim=True)

        # 6. 计算最终动作
        action = torch.tanh(pre_tanh_action)
        
        return action, log_prob

class Critic(nn.Module):
    """
    升级版的 Critic 网络，遵循 Twin-Critic 设计。
    - 包含两个独立的Q网络 (Q1, Q2) 以缓解Q值高估。
    - 每个网络内部借鉴 ElegantRL 的结构，并集成层归一化。
    """
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()

        # --- Q1 网络 ---
        self.net_q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Hardswish(), # 在最后一层前使用 Hardswish
            nn.Linear(hidden_size, 1)
        )

        # --- Q2 网络 ---
        # 结构与 Q1 完全相同，但不共享任何权重。
        self.net_q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Hardswish(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        """
        前向传播，计算两个Q值。
        """
        # 将状态和动作拼接成一个输入向量
        sa_input = torch.cat([state, action], dim=1)
        
        # 分别通过两个网络计算Q值
        q1 = self.net_q1(sa_input)
        q2 = self.net_q2(sa_input)
        
        return q1, q2

# --- 2. 经验回放缓冲区 ---
class ReplayBuffer:
    """
    一个高性能的经验回放缓冲区，借鉴了 ElegantRL 的设计思想。
    - 在初始化时预分配连续的 PyTorch 张量内存。
    - 使用循环指针进行高效的数据写入。
    - 采样时使用高级索引，避免 Python 循环。
    """
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device):
        self.capacity = capacity
        self.device = device

        # 预分配内存。所有数据都直接存储在目标设备上 (CPU 或 GPU)
        # 1. 存储状态 (s_t)
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        # 2. 存储下一个状态 (s_{t+1})
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        # 3. 存储动作 (a_t)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        # 4. 存储奖励 (r_t) 和完成标志 (done_t)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

        # 循环指针和当前大小
        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """
        将一个 transition 存入缓冲区。
        输入都应该是 NumPy 数组或标量。
        """
        # 使用循环指针直接写入预分配的张量
        # 我们在这里进行从 NumPy 到 Torch Tensor 并移动到设备的转换
        self.states[self.ptr] = torch.from_numpy(state).to(self.device)
        self.actions[self.ptr] = torch.from_numpy(action).to(self.device)
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = torch.from_numpy(next_state).to(self.device)
        self.dones[self.ptr] = done
        
        # 更新指针和大小
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        """
        从缓冲区中随机采样一个批次的数据。
        """
        # 1. 用 torch.randint 快速生成一批随机索引 (在 GPU 上生成，如果 device 是 cuda)
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        # 2. 利用高级索引，一步到位地从缓冲区中取出所有数据
        #    这个过程没有 CPU-GPU 数据传输，因为数据已经在目标设备上了
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self) -> int:
        """返回当前缓冲区中的样本数量。"""
        return self.size

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

        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )

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
        env = DictObservationWrapper(env)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = DictActionWrapper(env)

    # env = NormalizeObservationWrapper(env)
    # env = DictActionWrapper(env)    

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
    # print(f"Training preview GIFs will be saved in: {run_gif_dir}")

    # # --- 新增: 初始随机探索 ---
    # initial_explore_steps = config.batch_size * 10 # 比如收集10个batch的随机数据
    # print(f"Collecting initial random samples for {initial_explore_steps} steps...")
    # state, _ = env.reset()
    # for _ in range(initial_explore_steps):
    #     random_action = env.action_space.sample() # 使用环境的随机采样
    #     next_state, reward, terminated, truncated, info = env.step(random_action)
    #     done = terminated or truncated
    #     scaled_reward = reward * config.reward_scale
    #     agent.replay_buffer.push(state, random_action, scaled_reward, next_state, done)
    #     if done:
    #         state, _ = env.reset()
    #     else:
    #         state = next_state
    # print("Initial random sampling complete.")

    # total_steps = initial_explore_steps # 更新总步数计数器
    # 使用 tqdm 包装训练循环
    desc = f"{config.env_name} w/ SAC"
    gradient_steps_per_step = 100
    with tqdm(total=config.num_episodes, desc=None) as pbar:
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
                # episode_reward += reward
                episode_reward += scaled_reward
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
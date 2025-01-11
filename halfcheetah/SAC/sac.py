import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from collections import deque, namedtuple
import random
import wandb
import time

def check_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nGPU is available!")
        print(f"Using: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        return device
    else:
        print("\nNo GPU available. Using CPU instead.")
        return torch.device("cpu")

def set_seeds(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class WandBConfig:
    def __init__(self):
        self.api_key = "a0fc75f04fa27bc24039cf264e6500367853626f"
        self.project_name = "saccheetah"
        
    def setup(self):
        os.environ["WANDB_API_KEY"] = self.api_key
        wandb.init(project=self.project_name)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action * self.max_action, log_prob

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class SAC:
    def __init__(
        self, 
        state_dim,
        action_dim,
        max_action,
        device,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        buffer_size=1000000,
        batch_size=256
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.episode_rewards = []
        self.cumulative_samples = 0
        self.memory = ReplayBuffer(buffer_size)

    def save_model(self, episode, save_dir="checkpoints"):
        """Save model checkpoints"""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'episode': episode,
            'episode_rewards': self.episode_rewards
        }
        torch.save(checkpoint, f"{save_dir}/sac_checkpoint_episode_{episode}.pt")
        
    def load_model(self, checkpoint_path):
        """Load model checkpoints"""
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        return checkpoint['episode'], checkpoint['episode_rewards']

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            mean, _ = self.actor(state)
            return torch.tanh(mean).cpu().data.numpy().flatten()
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(self.batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state_batch)
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward_batch + (1 - done_batch) * self.gamma * (target_q - self.alpha * next_log_pi)

        # Critic update
        current_q1, current_q2 = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        action, log_pi = self.actor.sample(state_batch)
        q1, q2 = self.critic(state_batch, action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_pi - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

def train_sac(
    save_interval=100,  # Save every 100 episodes
    max_episodes=1000,
    max_steps=1000,
    eval_freq=20
):
    set_seeds(45)
    wandb_config = WandBConfig()
    wandb_config.setup()
    
    env = gym.make("HalfCheetah-v4")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    device = check_gpu()
    print(f"\nTraining will run on: {device}")

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        buffer_size=1000000,
        batch_size=256
    )
    
    training_start_time = time.time()
    print("\nStarting training...")
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, float(done))
            
            if len(agent.memory) > agent.batch_size:
                agent.train(agent.memory)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            agent.cumulative_samples += 1
            
            if done:
                break
        
        agent.episode_rewards.append(episode_reward)
        
        metrics = {
            "Reward/Episode": episode_reward,
            "Reward/Cumulative_Mean": np.mean(agent.episode_rewards),
            "Reward/Cumulative_Std": np.std(agent.episode_rewards),
            "Efficiency/Episode_Length": episode_length,
            "Efficiency/Cumulative_Samples": agent.cumulative_samples,
            "Stability/Reward_Variability": np.std(agent.episode_rewards[-50:]) if len(agent.episode_rewards) >= 50 else 0,
            "Episode": episode
        }
        wandb.log(metrics)
        
        if (episode + 1) % eval_freq == 0:
            avg_reward = np.mean(agent.episode_rewards[-eval_freq:])
            print(f"Episode {episode+1}: Average Reward = {avg_reward:.2f}")
        
        # Save model checkpoints
        if (episode + 1) % save_interval == 0:
            agent.save_model(episode + 1)
            current_time = time.time()
            training_duration = current_time - training_start_time
            wandb.log({
                "Computation/Training_Time_Seconds": training_duration,
                "Computation/Episodes_Completed": episode + 1
            })

    agent.save_model(max_episodes)
    wandb.finish()
    return agent

if __name__ == "__main__":
    agent = train_sac()

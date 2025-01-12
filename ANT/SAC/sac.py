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

def set_seed(seed):
    """
    Set random seeds for reproducibility across PyTorch, NumPy, and Python's random module.
    
    Args:
        seed (int): Seed value to set for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  
# Actor Network
class WandBConfig:
    def __init__(self):
        self.api_key = "a0fc75f04fa27bc24039cf264e6500367853626f"
        self.project_name = "sacant"
        
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
        batch_size=512
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
        
        # Add trackers for Q-values
        self.q1_values = []
        self.q2_values = []

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
        
        self.q1_values.append(current_q1.mean().item())
        self.q2_values.append(current_q2.mean().item())
        
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
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q1_value': self.q1_values[-1],
            'q2_value': self.q2_values[-1]
        }

def train_sac(
    save_interval=100,
    max_episodes=10000,
    max_steps=1000,
    eval_freq=20,
    seed=45
):
    set_seed(seed)
    
    wandb_config = WandBConfig()
    wandb_config.setup()
    
    env = gym.make("Ant-v4")
    
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
    
    episode_q1_values = []
    episode_q2_values = []
    
    for episode in range(max_episodes):
        # state, _ = env.reset()
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
  
            agent.memory.push(state, action, reward, next_state, float(done))
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                train_info = agent.train(agent.memory)
                if train_info:
                    episode_losses.append(train_info)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            agent.cumulative_samples += 1
            
            if done:
                break
        
        agent.episode_rewards.append(episode_reward)
        
        # Calculate average Q-values for the episode
        if episode_losses:
            avg_q1 = np.mean([loss['q1_value'] for loss in episode_losses])
            avg_q2 = np.mean([loss['q2_value'] for loss in episode_losses])
            episode_q1_values.append(avg_q1)
            episode_q2_values.append(avg_q2)
        
        metrics = {
            "Reward/Episode": episode_reward,
            "Reward/Cumulative_Mean": np.mean(agent.episode_rewards),
            "Reward/Cumulative_Std": np.std(agent.episode_rewards),
            "Efficiency/Episode_Length": episode_length,
            "Efficiency/Cumulative_Samples": agent.cumulative_samples,
            "Stability/Reward_Variability": np.std(agent.episode_rewards[-50:]) if len(agent.episode_rewards) >= 50 else 0,
            "Q_Values/Q1": avg_q1 if episode_losses else 0,
            "Q_Values/Q2": avg_q2 if episode_losses else 0,
            "Q_Values/Q1_Q2_Difference": abs(avg_q1 - avg_q2) if episode_losses else 0,
            "Episode": episode
        }
        
        if episode_losses:
            metrics.update({
                "Loss/Critic": np.mean([loss['critic_loss'] for loss in episode_losses]),
                "Loss/Actor": np.mean([loss['actor_loss'] for loss in episode_losses])
            })
        
        wandb.log(metrics)
        

        if (episode + 1) % eval_freq == 0:
            avg_reward = np.mean(agent.episode_rewards[-eval_freq:])
            print(f"Episode {episode+1}: Average Reward = {avg_reward:.2f}")
        

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

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

# print(torch.version.cuda)

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
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(512, action_dim)
        self.log_std = nn.Linear(512, action_dim)
        self.max_action = max_action
        
        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=1)
                torch.nn.init.constant_(layer.bias, 0)
        
        torch.nn.init.orthogonal_(self.mean.weight, gain=0.01)
        torch.nn.init.constant_(self.mean.bias, 0)
        torch.nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        torch.nn.init.constant_(self.log_std.bias, -1)  # Initialize to small std
        
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        # Tighter bounds on log_std for more stable exploration
        log_std = torch.clamp(log_std, -10, 1)
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

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Increase network capacity and add layer normalization
        def create_critic_network():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
        
        self.q1 = create_critic_network()
        self.q2 = create_critic_network()
        
        # Initialize weights
        for net in [self.q1, self.q2]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight, gain=1)
                    torch.nn.init.constant_(layer.bias, 0)
        
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

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def __call__(self, x):
        self.update(x)
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

# SAC Agent
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
        lr_actor=3e-4,
        lr_critic=3e-4,
        buffer_size=1000000,
        batch_size=256
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.episode_rewards = []
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-5)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-5)
        
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.train_steps = 0
        self.memory = ReplayBuffer(buffer_size)
        self.q1_values = []
        self.q2_values = []
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='max', factor=0.5, patience=100, min_lr=1e-5
        )
        self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-5
        )
        
        # reward normalization
        self.return_scale = RunningMeanStd()
        
        # update frequency
        self.update_frequency = 1
        self.updates_per_step = 2
        
    def train(self, replay_buffer):
        """
        Train the SAC agent with empty array handling
        """
        if len(replay_buffer) < self.batch_size:
            return None  # Return None if buffer is not full enough
        
        self.train_steps += 1

        # Sample from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(self.batch_size)

        if len(state_batch) == 0 or len(action_batch) == 0:
            return None
            
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # normalize rewards for stability (clip to prevent extreme values)
        reward_batch = torch.clamp(reward_batch, -50, 50)
        
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state_batch)
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)
            
            alpha = self.log_alpha.exp()
            target_q = reward_batch + (1 - done_batch) * self.gamma * (target_q - alpha * next_log_pi)
            target_q = torch.clamp(target_q, -100, 100)

        current_q1, current_q2 = self.critic(state_batch, action_batch)
        critic_loss = F.huber_loss(current_q1, target_q) + F.huber_loss(current_q2, target_q)
        
        q1_mean = current_q1.mean().item() if current_q1.numel() > 0 else 0
        q2_mean = current_q2.mean().item() if current_q2.numel() > 0 else 0
        self.q1_values.append(q1_mean)
        self.q2_values.append(q2_mean)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        metrics = {
            'critic_loss': critic_loss.item() if critic_loss.numel() > 0 else 0,
            'actor_loss': None,
            'alpha_loss': None,
            'alpha': alpha.item(),
            'q1_value': q1_mean,
            'q2_value': q2_mean,
            'target_q_mean': target_q.mean().item() if target_q.numel() > 0 else 0,
            'log_pi_mean': None
        }

        if self.train_steps % 2 == 0:
            # Update temperature parameter
            action_new, log_pi = self.actor.sample(state_batch)
            alpha = self.log_alpha.exp()
            
            if log_pi.numel() > 0:  # Check if log_pi is not empty
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
    
                q1, q2 = self.critic(state_batch, action_new)
                if q1.numel() > 0 and q2.numel() > 0:
                    q = torch.min(q1, q2)
                    actor_loss = (alpha * log_pi - q).mean()
                    
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                    self.actor_optimizer.step()
                    
                    metrics.update({
                        'actor_loss': actor_loss.item(),
                        'alpha_loss': alpha_loss.item(),
                        'log_pi_mean': log_pi.mean().item()
                    })


        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                if torch.isfinite(param).all():  # Only update if parameters are finite
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return metrics

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



def train_sac(
    save_interval=100,
    max_episodes=10000,
    max_steps=1000,
    eval_freq=20,
    seed=42
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
        tau=0.005,  # Increased for better stability
        alpha=0.2,
        lr_actor=3e-4,
        lr_critic=3e-4,
        buffer_size=1000000,
        batch_size=256  # Reduced batch size for better generalization
    )
    
    training_start_time = time.time()
    print("\nStarting training...")
    
    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, float(done))
            
            if len(agent.memory) > agent.batch_size:
                train_info = agent.train(agent.memory)
                if train_info:
                    episode_losses.append(train_info)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        agent.episode_rewards.append(episode_reward)
        
        # Calculate average metrics for the episode
        if episode_losses:
            avg_critic_loss = np.mean([loss['critic_loss'] for loss in episode_losses])
            avg_q1 = np.mean([loss['q1_value'] for loss in episode_losses])
            avg_q2 = np.mean([loss['q2_value'] for loss in episode_losses])
            avg_alpha = np.mean([loss['alpha'] for loss in episode_losses])
            
            # Calculate average actor loss only for steps where it was updated
            actor_losses = [loss['actor_loss'] for loss in episode_losses if loss['actor_loss'] is not None]
            avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
            
            metrics = {
                "Loss/Critic": avg_critic_loss,
                "Loss/Actor": avg_actor_loss,  # This will be 0 if no actor updates occurred
                "Q_Values/Q1": avg_q1,
                "Q_Values/Q2": avg_q2,
                "Q_Values/Q1_Q2_Difference": abs(avg_q1 - avg_q2),
                "Temperature/Alpha": avg_alpha,
                "Reward/Episode": episode_reward,
                "Reward/Cumulative_Mean": np.mean(agent.episode_rewards),
                "Reward/Cumulative_Std": np.std(agent.episode_rewards),
                "Efficiency/Episode_Length": episode_length,
                "Stability/Reward_Variability": np.std(agent.episode_rewards[-50:]) if len(agent.episode_rewards) >= 50 else 0,
                "Episode": episode
            }
            
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

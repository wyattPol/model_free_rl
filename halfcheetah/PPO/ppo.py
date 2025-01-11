import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import wandb
import os
import random
import time

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


class WandB:
    os.environ["WANDB_API_KEY"] = "a0fc75f04fa27bc24039cf264e6500367853626f"
    project_name = "ppocheetah"

set_seed(45)
os.environ["WANDB_API_KEY"] = "a0fc75f04fa27bc24039cf264e6500367853626f"
wandb.init(project=WandB.project_name)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        x = self.net(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        return self.net(state)

class PPO:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.clip_ratio = 0.2
        self.gamma = 0.99
        self.lam = 0.95
        self.batch_size = 64
        self.n_epochs = 10
        
        # Tracking metrics for analysis
        self.episode_rewards = []
        self.episode_lengths = []
        self.cumulative_samples = 0
        
    def get_action(self, state):
        mean, std = self.actor(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().numpy()[0], log_prob.detach()[0]
    
    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, advantages, returns):
        for _ in range(self.n_epochs):
            for idx in range(0, len(states), self.batch_size):
                batch_states = states[idx:idx + self.batch_size]
                batch_actions = actions[idx:idx + self.batch_size]
                batch_old_log_probs = old_log_probs[idx:idx + self.batch_size]
                batch_advantages = advantages[idx:idx + self.batch_size]
                batch_returns = returns[idx:idx + self.batch_size]
                
                # Actor loss
                mean, std = self.actor(batch_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                values = self.critic(batch_states).squeeze()
                critic_loss = ((values - batch_returns) ** 2).mean()
                
                # Update networks
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
    
    def train(self, max_episodes=1000, steps_per_episode=1000, seed=45, save_interval=500):
        # Create a directory to save models if it doesn't exist
        os.makedirs('saved_models', exist_ok=True)
        
        # Track training start time for computational efficiency metrics
        training_start_time = time.time()
        
        # Reset tracking metrics
        self.episode_rewards = []
        self.episode_lengths = []
        
        for episode in range(max_episodes):
            state, _ = self.env.reset(seed=seed)  
            episode_reward = 0
            episode_length = 0
            
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            
            for step in range(steps_per_episode):
                action, log_prob = self.get_action(state)
                value = self.critic(state).item()
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Track cumulative samples for sample efficiency
            self.cumulative_samples += episode_length
            
            # Compute advantages and returns
            next_value = self.critic(state).item()
            advantages, returns = self.compute_gae(rewards, values, next_value, dones)
            
            states = torch.FloatTensor(np.array(states))
            actions = torch.FloatTensor(np.array(actions))
            old_log_probs = torch.FloatTensor(log_probs)
            
            self.update(states, actions, old_log_probs, advantages, returns)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            metrics = {
                "Reward/Episode": episode_reward,
                "Reward/Cumulative_Mean": np.mean(self.episode_rewards),
                "Reward/Cumulative_Std": np.std(self.episode_rewards),
                
                # Learning Efficiency
                "Efficiency/Episode_Length": episode_length,
                "Efficiency/Cumulative_Samples": self.cumulative_samples,
                
                # Policy Stability
                "Stability/Reward_Variability": np.std(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else 0,
                "Episode": episode
            }
            
            wandb.log(metrics)
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward}, Length: {episode_length}")
            
            if episode > 0 and episode % save_interval == 0:
                self.save_model(episode)
                current_time = time.time()
                training_duration = current_time - training_start_time
                wandb.log({
                    "Computation/Training_Time_Seconds": training_duration,
                    "Computation/Episodes_Completed": episode
                })
        
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        final_metrics = {
            "Final/Total_Training_Time": total_training_time,
            "Final/Total_Episodes": max_episodes,
            "Final/Mean_Reward": np.mean(self.episode_rewards),
            "Final/Max_Reward": np.max(self.episode_rewards),
            "Final/Total_Samples": self.cumulative_samples
        }
        wandb.log(final_metrics)
        
        # Save final model
        self.save_model(max_episodes)
        
        return self.episode_rewards

    def save_model(self, episode):
        """
        Save the actor and critic models
        
        Args:
            episode (int): Current episode number for filename
        """
        actor_path = f'saved_models/actor_ep{episode}.pth'
        critic_path = f'saved_models/critic_ep{episode}.pth'
        
        # Save actor model
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
        }, actor_path)
        
        # Save critic model
        torch.save({
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, critic_path)
        
        print(f"Models saved at episode {episode}")
        wandb.save(actor_path)
        wandb.save(critic_path)

   
    def load_model(self, actor_path, critic_path):
        """
        Load previously saved actor and critic models
        
        Args:
            actor_path (str): Path to saved actor model
            critic_path (str): Path to saved critic model
        """
        # Load actor model
        actor_checkpoint = torch.load(actor_path)
        self.actor.load_state_dict(actor_checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])
        
        # Load critic model
        critic_checkpoint = torch.load(critic_path)
        self.critic.load_state_dict(critic_checkpoint['model_state_dict'])
        self.critic_optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])
        
        print(f"Models loaded from {actor_path} and {critic_path}")
        
# Training
def main():
    wandb.init(
        project="[[ppocheetah]]",
        config={
            "algorithm": "PPO",
            "environment": "HalfCheetah-v4",
            "max_episodes": 5000,
            "steps_per_episode": 1000,
            "learning_rate_actor": 3e-4,
            "learning_rate_critic": 1e-3,
            "clip_ratio": 0.2,
            "gamma": 0.99,
            "lambda": 0.95
        }
    )

    env = gym.make('HalfCheetah-v4')
    set_seed(45)
    agent = PPO(env)
    rewards = agent.train(seed=45, max_episodes=5000, save_interval=500)
    wandb.finish()

if __name__ == "__main__":
    main()

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
    project_name = "ppoant"

set_seed(45)
os.environ["WANDB_API_KEY"] = "a0fc75f04fa27bc24039cf264e6500367853626f"
wandb.init(project=WandB.project_name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        # Initialize mean layer with smaller weights
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        
        # Learnable but initialized to larger negative value for smaller initial std
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)
        
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        x = self.net(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std).clamp(min=1e-3, max=1)  # Clamp std for stability
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        
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
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-5, eps=1e-5)
        
        self.clip_ratio = 0.2  
        self.gamma = 0.99
        self.lam = 0.95
        self.batch_size = 128  
        self.n_epochs = 8
        self.max_grad_norm = 0.5  
        self.vf_coef = 0.5
        
        # Tracking metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.cumulative_samples = 0
        
    def normalize_rewards(self, rewards):
        return (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    def get_action(self, state):
        with torch.no_grad():
            mean, std = self.actor(state)
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0) 
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().numpy()[0], log_prob.detach()[0]
    
    def update(self, states, actions, old_log_probs, advantages, returns):
   
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        for _ in range(self.n_epochs):
            # Generate random indices for minibatches
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Actor loss
                mean, std = self.actor(batch_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
                
                # Critic loss
                values = self.critic(batch_states).squeeze()
                critic_loss = self.vf_coef * ((values - batch_returns) ** 2).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
    
    def train(self, max_episodes=100000, steps_per_episode=1000, seed=45):
        set_seed(seed)
        
        for episode in range(max_episodes):
            state, _ = self.env.reset(seed=seed + episode)
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
        
            next_value = self.critic(state).item()
            advantages, returns = self.compute_gae(rewards, values, next_value, dones)
            self.update(states, actions, log_probs, advantages, returns)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            if episode > 0 and episode % 200 == 0:
                self.save_model(episode)
            

            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
                
                wandb.log({
                   
                    "reward": episode_reward,
                    "avg_reward": avg_reward,
                    
                })
        
        return self.episode_rewards
        
        
    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            # Temporal Difference (TD) Error
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            
            # GAE
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            
            # Insert the calculated advantage at the beginning (for efficient reverse order)
            advantages.insert(0, gae)

        #compute returns
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)

        # norm ads
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    


    def save_model(self, episode):
        """
        Save the actor and critic models
        
        Args:
            episode (int): Current episode number for filename
        """

        actor_path = f'saved_models/actor_ep{episode}.pth'
        critic_path = f'saved_models/critic_ep{episode}.pth'
        
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
        }, actor_path)
        
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

def main():
    wandb.init(
        project="ppo_ant",
        config={
            "algorithm": "PPO",
            "environment": "Ant-v4",
            "max_episodes": 10000,
            "steps_per_episode": 2048,
            "learning_rate_actor": 1e-5,
            "learning_rate_critic": 3e-5,
            "clip_ratio": 0.2,
            "gamma": 0.99,
            "lambda": 0.95
        }
    )
    
    env = gym.make('Ant-v4')
    agent = PPO(env)
    rewards = agent.train()
    wandb.finish()

if __name__ == "__main__":
    main()

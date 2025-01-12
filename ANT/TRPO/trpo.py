import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
import gymnasium as gym
import wandb
import time
from collections import deque
import random

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class WandBConfig:
    def __init__(self):
        self.api_key = "a0fc75f04fa27bc24039cf264e6500367853626f"
        self.project_name = "trpo_ant"
        
    def setup(self):
        os.environ["WANDB_API_KEY"] = self.api_key
        wandb.init(project=self.project_name)

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0)

    def forward(self, states):
        x = self.net(states)
        mean = self.mean(x)
        std = self.log_std.exp()
        return mean, std

    def get_action(self, state, deterministic=False):
        mean, std = self.forward(state)
        if deterministic:
            return mean
        dist = Normal(mean, std)
        action = dist.sample()
        return action

    def get_log_prob(self, states, actions):
        mean, std = self.forward(states)
        dist = Normal(mean, std)
        return dist.log_prob(actions).sum(-1)

class Value(nn.Module):
    def __init__(self, state_dim):
        super(Value, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

    def forward(self, states):
        return self.net(states).squeeze(-1)

class TRPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        gamma=0.99,
        tau=0.97,
        max_kl=0.01,
        damping=0.1,
        batch_size=2048,
        value_lr=3e-4
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_kl = max_kl
        self.damping = damping
        self.batch_size = batch_size

        self.policy = Policy(state_dim, action_dim).to(device)
        self.value = Value(state_dim).to(device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=value_lr)
        
        self.running_state = RunningMeanStd(shape=state_dim)
        self.running_reward = RunningMeanStd(shape=())

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy.get_action(state, deterministic=evaluate)
        return action.cpu().numpy()[0]

    def train(self, memory):
        batch = memory.get_batch()
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)

        with torch.no_grad():
            values = self.value(states)
            next_values = self.value(next_states)
            
            # GAE calculation
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = next_values[t]
                else:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(10):
            value_loss = (self.value(states) - returns).pow(2).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        with torch.no_grad():
            old_log_probs = self.policy.get_log_prob(states, actions)

        def get_kl():
            mean, std = self.policy(states)
            old_mean, old_std = self.policy(states.detach())
            old_dist = Normal(old_mean, old_std)
            dist = Normal(mean, std)
            return torch.distributions.kl.kl_divergence(old_dist, dist).mean()

        def surrogate_loss():
            log_probs = self.policy.get_log_prob(states, actions)
            ratio = torch.exp(log_probs - old_log_probs)
            return -(ratio * advantages).mean()

        loss = surrogate_loss()
        grads = torch.autograd.grad(loss, self.policy.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

        def Fvp(v):
            kl = get_kl()
            kl_grad = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            kl_grad = torch.cat([grad.view(-1) for grad in kl_grad])
            Fv = torch.autograd.grad(torch.sum(kl_grad * v), self.policy.parameters())
            Fv = torch.cat([grad.contiguous().view(-1) for grad in Fv])
            return Fv + self.damping * v

        def conjugate_gradient(Av_func, b, nsteps=10, residual_tol=1e-10):
            x = torch.zeros_like(b)
            r = b.clone()
            p = b.clone()
            rdotr = torch.dot(r, r)
            for i in range(nsteps):
                Ap = Av_func(p)
                alpha = rdotr / torch.dot(p, Ap)
                x += alpha * p
                r -= alpha * Ap
                new_rdotr = torch.dot(r, r)
                beta = new_rdotr / rdotr
                p = r + beta * p
                rdotr = new_rdotr
                if rdotr < residual_tol:
                    break
            return x

        stepdir = conjugate_gradient(Fvp, -loss_grad)

        # Line search
        shs = 0.5 * torch.dot(stepdir, Fvp(stepdir))
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm

        def update_model(step):
            idx = 0
            for param in self.policy.parameters():
                size = param.numel()
                param.data.add_(step[idx:idx + size].view(param.shape))
                idx += size

        expected_improve = -torch.dot(loss_grad, fullstep)
        prev_params = torch.cat([param.data.view(-1) for param in self.policy.parameters()])
        
        success, new_loss = self.line_search(prev_params, fullstep, expected_improve, surrogate_loss)
        
        return {
            'policy_loss': new_loss.item(),
            'value_loss': value_loss.item(),
            'kl_div': get_kl().item()
        }

    def line_search(self, prev_params, fullstep, expected_improve, surrogate_loss_func, max_backtracks=10):
        fval = surrogate_loss_func()
        
        for stepfrac in [.5**x for x in range(max_backtracks)]:
            step = stepfrac * fullstep
            idx = 0
            for param in self.policy.parameters():
                size = param.numel()
                param.data.copy_(prev_params[idx:idx + size].view(param.shape))
                param.data.add_(step[idx:idx + size].view(param.shape))
                idx += size
                
            new_loss = surrogate_loss_func()
            actual_improve = fval - new_loss
            if actual_improve > 0:
                return True, new_loss
                
            idx = 0
            for param in self.policy.parameters():
                size = param.numel()
                param.data.copy_(prev_params[idx:idx + size].view(param.shape))
                idx += size
                
        return False, fval

    def save_model(self, episode, save_dir="checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'episode': episode
        }, f"{save_dir}/trpo_checkpoint_episode_{episode}.pt")

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def push(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def get_batch(self):
        batch = {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'next_states': np.array(self.next_states),
            'dones': np.array(self.dones)
        }
        self.clear()
        return batch

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def __len__(self):
        return len(self.states)

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


def train_trpo(
    max_episodes=10000,
    max_steps=1000,
    save_interval=1000,
    eval_freq=10,
    seed=42
):
    set_seed(seed)
    
    wandb_config = WandBConfig()
    wandb_config.setup()
    
    env = gym.make("Ant-v4")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    device = check_gpu()
    print(f"\nTraining will run on: {device}")
    
    agent = TRPO(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        gamma=0.99,  # Keep this
        tau=0.95,    # Reduced from 0.97
        max_kl=0.005, # Reduced from 0.01
        damping=0.2,  # Increased from 0.1
        batch_size=4096, # Increased from 2048
        value_lr=1e-4   # Reduced from 3e-4
    )
    memory = Memory()
    
    training_start_time = time.time()
    episode_rewards = []
    print("\nStarting training...")
    
    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            memory.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
  
        if len(memory) >= agent.batch_size:
            train_info = agent.train(memory)
            
            metrics = {
                "Episode": episode,
                "Reward/Episode": episode_reward,
                "Reward/Average": np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),
                "Length/Episode": episode_length,
                "Loss/Policy": train_info['policy_loss'],
                "Loss/Value": train_info['value_loss'],
                "Policy/KL_Divergence": train_info['kl_div'],
                "Training/Steps": episode * max_steps + step,
                "Training/Time_Hours": (time.time() - training_start_time) / 3600
            }
            
            wandb.log(metrics)
        
        if (episode + 1) % eval_freq == 0:
            avg_reward = np.mean(episode_rewards[-eval_freq:])
            print(f"Episode {episode+1}: Average Reward = {avg_reward:.2f}")
        
        if (episode + 1) % save_interval == 0:
            agent.save_model(episode + 1)
    
    agent.save_model(max_episodes)
    wandb.finish()
    env.close()
    
    return agent

if __name__ == "__main__":
    agent = train_trpo()

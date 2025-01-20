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
from copy import deepcopy


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
        self.api_key = "84e8f4da75becf38a7bfe16d8de8ec9f6b62337f"
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
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5) 
        
        # Initialize weights using orthogonal initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0)

    def forward(self, states):
        x = self.net(states)
        mean = self.mean(x)
        std = self.log_std.exp().clamp(min=1e-6)  # Ensure std is positive
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
        
        # Initialize weights
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
        max_kl=0.005,
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

    # def get_entropy(self, states):
    #     """Calculate policy entropy for given states"""
    #     with torch.no_grad():
    #         mean, std = self.policy(states)
    #         dist = Normal(mean, std)
    #         entropy = dist.entropy().mean()
    #         return entropy.item()

    def get_entropy(self, states):
        """Calculate policy entropy for given states"""
        with torch.no_grad():
            mean, std = self.policy(states)
            # Add small epsilon to std for numerical stability
            std = std.clamp(min=1e-6)  # Ensure std is positive
            dist = Normal(mean, std)
            entropy = dist.entropy().mean()
            # Safeguard against negative entropy
            return max(0.0, entropy.item())

    def get_kl(self, states, old_dist):
        """Compute the KL divergence between old and new distributions"""
        mean, std = self.policy(states)
        dist = Normal(mean, std)
        kl = torch.distributions.kl_divergence(old_dist, dist).sum(-1).mean()
        return kl

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
            entropy = self.get_entropy(states)
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
            log_probs = self.policy.get_log_prob(states, actions)

            # Compute old log probs (store this for later comparison)
            old_log_probs = log_probs.detach() 
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
          
        for _ in range(10):
            value_loss = (self.value(states) - returns).pow(2).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)  # Add this line
            self.value_optimizer.step()

        with torch.no_grad():
            old_mean, old_std = self.policy(states)
            old_dist = Normal(old_mean, old_std)

        def surrogate_loss():
            log_probs = self.policy.get_log_prob(states, actions)
            ratio = torch.exp(log_probs - old_log_probs)
            return -(ratio * advantages).mean()

        loss = surrogate_loss()
        grads = torch.autograd.grad(loss, self.policy.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

        def Fvp(v):
            # Here, we are passing states and old_dist to get_kl
            kl = self.get_kl(states, old_dist)  # Fixed: pass states and old_dist to get_kl
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
        
        success, new_loss = self.line_search(prev_params, fullstep, expected_improve, surrogate_loss,states, old_dist)
        
        return {
            'policy_loss': new_loss.item() if success else loss.item(),
            'value_loss': value_loss.item(),
            'kl_div': self.get_kl(states, old_dist).item(),  # Fixed: use get_kl here
            'entropy': entropy
        }


    def line_search(self, prev_params, fullstep, expected_improve, surrogate_loss_func, states, old_dist, max_backtracks=10):
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

            # Now pass the correct arguments to get_kl
            kl = self.get_kl(states, old_dist)  # Fixed: use states and old_dist

            # Add acceptance criteria based on both improvement and KL
            improve = fval - new_loss
            if improve > 0 and kl <= self.max_kl * 1.5:  # Allow some slack in KL
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
        # Validate inputs
        if np.any(np.isnan(state)) or np.any(np.isnan(action)) or np.isnan(reward) or np.any(np.isnan(next_state)):
            print("Warning: NaN detected in memory push!")
            return
            
        # Clip reward for stability
        reward = np.clip(reward, -10.0, 10.0)
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def get_batch(self):
        if len(self.states) == 0:
            return None
            
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
    max_episodes=20000,
    max_steps=1000,
    save_interval=500,
    eval_freq=10,
    seed=42,
    early_stop_threshold=-150
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
        device=device
    )
    memory = Memory()
    
    # Initialize monitoring
    training_start_time = time.time()
    episode_rewards = []
    entropies = []  # Track entropy values
    best_average_reward = float('-inf')
    no_improvement_count = 0
    
    print("\nStarting training...")
    
    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_entropies = []  # Track entropy for this episode
        
        # Episode loop
        for step in range(max_steps):
            try:
                # Get action and entropy
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    mean, std = agent.policy(state_tensor)
                    dist = Normal(mean, std)
                    entropy = dist.entropy().mean().item()
                    episode_entropies.append(entropy)
                
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                memory.push(state, action, reward, next_state, float(done))
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
                    
            except Exception as e:
                print(f"Error during step {step}: {e}")
                break
        
        episode_rewards.append(episode_reward)
        avg_episode_entropy = np.mean(episode_entropies) if episode_entropies else 0
        entropies.append(avg_episode_entropy)
        
        # Train if we have enough samples
        if len(memory) >= agent.batch_size:
            try:
                train_info = agent.train(memory)
                
                # Log training metrics
                if (episode + 1) % eval_freq == 0:
                    avg_reward = np.mean(episode_rewards[-eval_freq:])
                    avg_entropy = np.mean(entropies[-eval_freq:])
                    
                    print(f"\nEpisode {episode+1}:")
                    print(f"Average Reward = {avg_reward:.2f}")
                    print(f"Policy Loss = {train_info['policy_loss']:.4f}")
                    print(f"Value Loss = {train_info['value_loss']:.4f}")
                    print(f"KL Divergence = {train_info['kl_div']:.4f}")
                    print(f"Average Entropy = {avg_entropy:.4f}")
                    
                    # Log to wandb if you're using it
                    if wandb.run is not None:
                        wandb.log({
                            "reward": avg_reward,
                            "policy_loss": train_info['policy_loss'],
                            "value_loss": train_info['value_loss'],
                            "kl_div": train_info['kl_div'],
                            "entropy": avg_entropy
                        })
                    
                    # Check for improvement
                    if avg_reward > best_average_reward:
                        best_average_reward = avg_reward
                        no_improvement_count = 0
                        agent.save_model(episode, "best_model")
                    else:
                        no_improvement_count += 1
                    
                    # Early stopping
                    if no_improvement_count >= 500 and avg_reward < early_stop_threshold:
                        print("\nStopping early due to lack of improvement...")
                        break
                    
            except Exception as e:
                print(f"Error during training: {e}")
                continue
        
        if (episode + 1) % save_interval == 0:
            agent.save_model(episode)

    wandb.finish()
    env.close()
    return agent

if __name__ == "__main__":
    agent = train_trpo()

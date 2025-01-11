"""
code from other surce for reference
"""
from IPython.display import clear_output
from tqdm import trange
import matplotlib.pyplot as plt
import gym
import torch

# from ppo.Runner import *
# from ppo.Policy import *
# from ppo.PPO import *
# from ppo.utils import *
# from ppo.Sampler import *
# from ppo.Network import *

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class AsArray:
    """ 
       Converts lists of interactions to ndarray.
    """
    def __call__(self, trajectory):
      # Modify trajectory inplace. 
      for k, v in filter(lambda kv: kv[0] != "state",
                         trajectory.items()):
        trajectory[k] = np.asarray(v)


class NormalizeAdvantages:
    """ Normalizes advantages to have zero mean and variance 1. """
    def __call__(self, trajectory):
        adv = trajectory["advantages"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        trajectory["advantages"] = adv


class GAE:
    """ Generalized Advantage Estimator. """
    def __init__(self, policy, gamma=0.99, lambda_=0.95):
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_
    
    def __call__(self, trajectory):
        value_target = policy.act(trajectory['state']['latest_observation'])['values'][0]
        env_steps = trajectory['state']['env_steps']
        rewards = torch.tensor(trajectory['rewards'], dtype=torch.float32)
        dones = torch.tensor(trajectory['resets'], dtype=torch.float32)
        is_not_done = 1 - dones
        trajectory['values'] = torch.tensor(trajectory['values'],dtype=torch.float32)
        trajectory['advantages'] = []
        trajectory['value_targets'] = []
        gae = 0
        for step in reversed(range(env_steps)):
            if step==env_steps - 1:
                delta = rewards[step] + self.gamma*value_target*is_not_done[step] - trajectory['values'][step]
            else:
                delta = rewards[step] + self.gamma*trajectory['values'][step + 1]*is_not_done[step] -\
                        trajectory['values'][step]
            
            gae = delta + self.gamma*self.lambda_*is_not_done[step]*gae
            trajectory['advantages'].insert(0, gae)
            trajectory['value_targets'].insert(0, gae + trajectory['values'][step])
        trajectory['advantages'] = torch.tensor(trajectory['advantages'], dtype=torch.float32)
        trajectory['value_targets'] = torch.tensor(trajectory['value_targets'], dtype=torch.float32)

class TrajectorySampler:
    """ Samples minibatches from trajectory for a number of epochs. """
    def __init__(self, runner, num_epochs, num_minibatches, transforms=None):
        self.runner = runner
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.transforms = transforms or []
        self.minibatch_count = 0
        self.epoch_count = 0
        self.trajectory = self.runner.get_next()
        for transform in self.transforms:
                transform(self.trajectory)
    def shuffle_trajectory(self):
        """ Shuffles all elements in trajectory.
            Should be called at the beginning of each epoch.
        """
        pass
    
    def get_next(self):
        """ Returns next minibatch.  """
        if self.epoch_count==self.num_epochs:
            self.trajectory = self.runner.get_next()
            for transform in self.transforms:
                transform(self.trajectory)
            self.epoch_count = 0
        minibatch_dict = {}
        rand_inds = np.random.randint(0, self.trajectory['state']['env_steps'], self.num_minibatches)
        for key, value in self.trajectory.items():
            if key!='state':
                if len(value)==2:
                    minibatch_dict[key] = self.trajectory[key][rand_inds,:]
                else:
                    minibatch_dict[key] = self.trajectory[key][rand_inds]
        self.epoch_count += 1
        return minibatch_dict

class EnvRunner:
    """ Reinforcement learning runner in an environment with given policy """

    def __init__(self, env, policy, nsteps, transforms=None, step_var=None):
        self.env = env
        self.policy = policy
        self.nsteps = nsteps
        self.transforms = transforms or []
        self.step_var = step_var if step_var is not None else 0
        self.state = {"latest_observation": self.env.reset()[0]}


    @property
    def nenvs(self):
        """ Returns number of batched envs or `None` if env is not batched """
        return getattr(self.env.unwrapped, "nenvs", None)

    def reset(self):
        """ Resets env and runner states. """
        self.state["latest_observation"],_ = self.env.reset()
        self.policy.reset()

    def get_next(self):
        """ Runs the agent in the environment.  """
        trajectory = defaultdict(list, {"actions": []})
        observations = []
        rewards = []
        resets = []
        self.state["env_steps"] = self.nsteps

        for i in range(self.nsteps):
            observations.append(self.state["latest_observation"])
            act = self.policy.act(self.state["latest_observation"])
            if "actions" not in act:
                raise ValueError("result of policy.act must contain 'actions' "
                                 f"but has keys {list(act.keys())}")
            for key, val in act.items():
                trajectory[key].append(val)

            obs, rew, done, _ = self.env.step(trajectory["actions"][-1])[:4] 

            self.state["latest_observation"] = obs
            rewards.append(rew)
            resets.append(done)
            self.step_var += self.nenvs or 1

            # Only reset if the env is not batched. Batched envs should
            # auto-reset.
            if not self.nenvs and np.all(done):
                self.state["env_steps"] = i + 1
                self.state["latest_observation"] ,_= self.env.reset()

        trajectory.update(
            observations=observations,
            rewards=rewards,
            resets=resets)
        trajectory["state"] = self.state

        for transform in self.transforms:
            transform(trajectory)
        return trajectory
    
class PPO:
    def __init__(self, policy, optimizer,
                   cliprange=0.2,
                   value_loss_coef=0.25,
                   max_grad_norm=0.5):
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        # Note that we don't need entropy regularization for this env.
        self.max_grad_norm = max_grad_norm
    
    def policy_loss(self, trajectory, act):
        """ Computes and returns policy loss on a given trajectory. """
        log_probs_all = act['distribution'].log_prob(torch.tensor(trajectory['actions']))
        log_old_probs_all = torch.tensor(trajectory['log_probs'])
        ratio = (log_probs_all - log_old_probs_all).exp()
        J_pi = ratio*trajectory['advantages'].detach()
        self.advantages_np = trajectory['advantages'].detach().mean().numpy()
        J_pi_clipped = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange)*trajectory['advantages'].detach()
        return -torch.mean(torch.min(J_pi, J_pi_clipped))
      
    def value_loss(self, trajectory, act):
        """ Computes and returns value loss on a given trajectory. """
        self.values_np = trajectory['values'].detach().mean().cpu().numpy()
        L_simple = (act['values'] - trajectory['value_targets'].detach())**2
        L_clipped = (trajectory['values'] + torch.clamp(act['values'] - trajectory['values'],
                    -self.cliprange, self.cliprange) - trajectory['value_targets'].detach())**2
        return torch.mean(torch.max(L_simple, L_clipped))
    
      
    def loss(self, trajectory):
        act = self.policy.act(trajectory["observations"], training=True)
        policy_loss = self.policy_loss(trajectory, act)
        value_loss = self.value_loss(trajectory, act)
        self.policy_loss_np = policy_loss.detach().numpy()
        self.value_loss_np = value_loss.detach().numpy()
        self.ppo_loss_np = self.policy_loss_np + self.value_loss_coef * self.value_loss_np
        return policy_loss + self.value_loss_coef * value_loss
      
    def step(self, trajectory):
        """ Computes the loss function and performs a single gradient step. """
        self.optimizer.zero_grad()
        self.loss(trajectory).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.total_norm = 0
        for p in self.policy.model.parameters():
            param_norm = p.grad.data.norm(2)
            self.total_norm += param_norm.item() ** 2
        self.total_norm = self.total_norm ** (1. / 2)

class Policy:
    def __init__(self, model):
        self.model = model
    
    def act(self, inputs, training=False):
        inputs = torch.tensor(inputs, dtype=torch.float32)
        (mus, sigmas), values = self.model(inputs)
        dist = torch.distributions.MultivariateNormal(mus, torch.diag_embed(sigmas, 0))
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        if training:
            return {'distribution': dist,
                    'values': values}
        else:
            return {'actions': actions.detach().numpy(),
                    'log_probs': log_probs.detach().numpy(),
                    'values': values.detach().numpy()}

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PolicyNetwork(nn.Module):
    
    def __init__(self, shape_in, action_shape, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.dense1 = layer_init(nn.Linear(shape_in, hidden_size))
        self.dense2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.dense3_mu = layer_init(nn.Linear(hidden_size, action_shape), std=0.01)
        self.dense3_std = layer_init(nn.Linear(hidden_size, action_shape), std=0.0)
        
    def forward(self, inputs):
        hid = torch.tanh(self.dense2(torch.tanh(self.dense1(inputs))))
        mu = self.dense3_mu(hid)
        sigma = torch.exp(self.dense3_std(hid))
        return mu, sigma

class ValueNetwork(nn.Module):
    
    def __init__(self, shape_in, hidden_size=64):
        super(ValueNetwork, self).__init__()
        self.dense1 = layer_init(nn.Linear(shape_in, hidden_size))
        self.dense2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.dense3 = layer_init(nn.Linear(hidden_size, 1), std =1.0)
    def forward(self, inputs):
        hid = torch.tanh(self.dense2(torch.tanh(self.dense1(inputs))))
        return self.dense3(hid)
    
    
class Network(nn.Module):
    def __init__(self, shape_in, action_shape, hidden_size=64):
        super(Network, self).__init__()
        self.policy = PolicyNetwork(shape_in, action_shape, hidden_size)
        self.value = ValueNetwork(shape_in, hidden_size)
    def forward(self, inputs):
        return self.policy(inputs), self.value(inputs)



def make_ppo_runner(env, policy, num_runner_steps=2048,
                    gamma=0.99, lambda_=0.95, 
                    num_epochs=16, num_minibatches=64):
    """ Creates runner for PPO algorithm. """
    runner_transforms = [AsArray(),
                         GAE(policy, gamma=gamma, lambda_=lambda_)]
    runner = EnvRunner(env, policy, num_runner_steps, 
                       transforms=runner_transforms)
    sampler_transforms = [NormalizeAdvantages()]
    sampler = TrajectorySampler(runner, num_epochs=num_epochs, 
                                num_minibatches=num_minibatches,
                                transforms=sampler_transforms)
    return sampler



def plot_tools(legend, position, data_y):
    plt.subplot(2,4,position)
    plt.plot(data_y, label=legend)
    plt.title(legend); plt.grid(); plt.legend() 
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))


def evaluate(env, agent, n_games=1, render=False):
    """Plays an a game from start till done, returns per-game rewards """
    agent.train(False)
    game_rewards = []
    done_counter = 0
    for _ in range(n_games):
        state,_ = env.reset()
        total_reward = 0
        while True:
            if render:
                env.render()
            state = torch.tensor(state, dtype=torch.float32)
            (mus, sigmas), _ = agent(state)
            dist = torch.distributions.MultivariateNormal(mus, torch.diag_embed(sigmas, 0))
            action = dist.sample().cpu().detach().numpy()
            state, reward, done, info = env.step(action)[:4] 
            total_reward += reward
            if done:
                break
        game_rewards.append(total_reward)
    agent.train(True)
    return game_rewards



if __name__ == '__main__':
    env = gym.make('HalfCheetah-v4', ctrl_cost_weight=0.1, reset_noise_scale=0.1, exclude_current_positions_from_observation=True)
    model = Network(shape_in=17, action_shape=6)
    policy = Policy(model)
    runner = make_ppo_runner(env, policy)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-05)

    ppo = PPO(policy, optimizer) 
    num_steps = []
    rewards = []
    value_losses = []
    policy_losses = []
    values = []
    grad_norms = []
    advantages = []
    ppo_losses = []
    for i in trange(100_000):
        trajectory = runner.get_next()
        ppo.step(trajectory)
        value_losses.append(ppo.values_np)
        policy_losses.append(ppo.policy_loss_np)
        values.append(ppo.values_np)
        grad_norms.append(ppo.total_norm)
        advantages.append(ppo.advantages_np)
        ppo_losses.append(ppo.ppo_loss_np)
        if i%100==0:
            clear_output(True)
            num_steps.append(runner.runner.step_var)
            
            
            rewards.append(np.mean(evaluate(env, model, n_games=1)))
            
            plt.figure(figsize=[20,10])
            
            plt.subplot(2,4,1)
            plt.plot(num_steps, rewards, label='Reward')
            plt.title("Rewards"); plt.grid(); plt.legend()
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

            plot_tools('Values', 2, values)
            plot_tools('Value loss', 3, value_losses)
            plot_tools('Policy loss', 4, policy_losses)
            plot_tools('PPO loss', 5, ppo_losses)
            plot_tools('Grad_norm_L2', 6, grad_norms) 
            plot_tools('Advantages', 7, advantages)

            plt.show()

    env.close()

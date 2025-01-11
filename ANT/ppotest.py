import gymnasium as gym
import torch
import numpy as np
from torch.distributions import Normal
import time
import argparse

from ppo import Actor, Critic, PPO

def evaluate_policy(actor, env, num_episodes=5, render=True):
    """
    Evaluate the trained policy for a specified number of episodes
    """
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            if render:
                env.render()
                time.sleep(0.01) 
            
         
            with torch.no_grad():
                mean, std = actor(state)
                dist = Normal(mean, std)
                action = dist.sample()
                action = torch.clamp(action, -1.0, 1.0)
            
          
            state, reward, terminated, truncated, _ = env.step(action.numpy()[0])
            done = terminated or truncated
            episode_reward += reward
            step += 1
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} - Reward: {episode_reward:.2f}, Steps: {step}")
    
    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained PPO agent on Ant-v4')
    parser.add_argument('--actor_checkpoint', 
                    type=str, 
                    default='saved_models/actor_ep2000.pth', 
                    help='Path to actor checkpoint')

    parser.add_argument('--critic_checkpoint', 
                        type=str, 
                        default='saved_models/critic_ep2000.pth', 
                        help='Path to critic checkpoint')

    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to evaluate')
    args = parser.parse_args()
    

    env = gym.make('Ant-v4', render_mode="human")

    agent = PPO(env)
    
    # load checkpoints
    try:
        actor_checkpoint = torch.load(args.actor_checkpoint)
        agent.actor.load_state_dict(actor_checkpoint['model_state_dict'])
        print(f"Successfully loaded actor checkpoint from {args.actor_checkpoint}")
        
        if args.critic_checkpoint:
            critic_checkpoint = torch.load(args.critic_checkpoint)
            agent.critic.load_state_dict(critic_checkpoint['model_state_dict'])
            print(f"Successfully loaded critic checkpoint from {args.critic_checkpoint}")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    agent.actor.eval()
    agent.critic.eval()
    
    # eva
    try:
        evaluate_policy(agent.actor, env, num_episodes=args.episodes)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    finally:
        env.close()

if __name__ == "__main__":
    main()

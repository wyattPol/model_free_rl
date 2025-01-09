import os
import torch
import gymnasium as gym
import numpy as np
from collections import deque
import time

# Import the model classes from your training script
from sac import Actor, Critic, SAC  # Assuming your training code is in sac.py

def evaluate_model(checkpoint_path, num_episodes=5, render=True):
    """
    Evaluate a trained SAC model and optionally render the environment.
    
    Args:
        checkpoint_path (str): Path to the saved model checkpoint
        num_episodes (int): Number of episodes to evaluate
        render (bool): Whether to render the environment
    """
    # Create environment
    env = gym.make("HalfCheetah-v4", render_mode="human" if render else None)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device
    )
    
    # Load checkpoint with appropriate device mapping
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        episode = checkpoint['episode']
        episode_rewards = checkpoint['episode_rewards']
        
        # Load state dicts with device mapping
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        print(f"Loaded checkpoint from episode {episode}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None
    
    # Ensure models are in eval mode
    agent.actor.eval()
    agent.critic.eval()
    agent.critic_target.eval()
    
    # Evaluation loop
    eval_rewards = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            with torch.no_grad():
                action = agent.select_action(state, evaluate=True)  # Use evaluate=True for deterministic actions
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            if render:
                env.render()
                time.sleep(0.01)  # Add small delay to make visualization smoother
        
        eval_rewards.append(episode_reward)
        print(f"Episode {ep + 1}/{num_episodes}: Reward = {episode_reward:.2f}")
    
    env.close()
    
    # Print evaluation statistics
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"\nEvaluation Results over {num_episodes} episodes:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return mean_reward, std_reward

def evaluate_multiple_checkpoints(checkpoint_dir, episode_interval=100, num_episodes=3):
    """
    Evaluate multiple checkpoints to see progression of learning.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoint files
        episode_interval (int): Interval between checkpoints to evaluate
        num_episodes (int): Number of episodes to evaluate per checkpoint
    """
    # Get all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    results = []
    for checkpoint_file in checkpoint_files:
        if int(checkpoint_file.split('_')[-1].split('.')[0]) % episode_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            print(f"\nEvaluating checkpoint: {checkpoint_file}")
            
            mean_reward, std_reward = evaluate_model(
                checkpoint_path=checkpoint_path,
                num_episodes=num_episodes,
                render=True
            )
            
            if mean_reward is not None:
                results.append({
                    'checkpoint': checkpoint_file,
                    'mean_reward': mean_reward,
                    'std_reward': std_reward
                })
    
    return results

if __name__ == "__main__":
    # Example usage
    checkpoint_dir = "checkpoints"  # Directory containing your saved checkpoints
    
    # Option 1: Evaluate a single checkpoint
    # try:
    #     latest_checkpoint = max(
    #         [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')],
    #         key=lambda x: int(x.split('_')[-1].split('.')[0])
    #     )
    #     checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    #     print(f"Evaluating latest checkpoint: {latest_checkpoint}")
    #     evaluate_model(checkpoint_path, num_episodes=5, render=True)
    # except Exception as e:
    #     print(f"Error loading checkpoints: {e}")
    #     print("Please ensure the checkpoint directory exists and contains .pt files")
    
    # Option 2: Evaluate multiple checkpoints to see progression
    print("\nEvaluating multiple checkpoints...")
    results = evaluate_multiple_checkpoints(
        checkpoint_dir=checkpoint_dir,
        episode_interval=100,  # Evaluate every 100th episode
        num_episodes=3
    )
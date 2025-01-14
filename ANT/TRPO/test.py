import os
import torch
import gymnasium as gym
import numpy as np
import time

from trpo import TRPO, Policy, Value  

def evaluate_model(checkpoint_path, num_episodes=5, render=True):
    """
    Evaluate a trained TRPO model and optionally render the environment.
    
    Args:
        checkpoint_path (str): Path to the saved model checkpoint
        num_episodes (int): Number of episodes to evaluate
        render (bool): Whether to render the environment
    """

    env = gym.make("Ant-v4", render_mode="human" if render else None)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    agent = TRPO(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        gamma=0.99,
        tau=0.97,
        max_kl=0.005,      
        damping=0.1,     
        batch_size=2048,  
        value_lr=3e-4    
    )
    

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        episode = checkpoint['episode']
        
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        agent.value.load_state_dict(checkpoint['value_state_dict'])
        
        print(f"Loaded checkpoint from episode {episode}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None
    
    agent.policy.eval()
    agent.value.eval()
    
    eval_rewards = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action deterministically for evaluation
            with torch.no_grad():
                action = agent.select_action(state, evaluate=True)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward = reward / 10.0 
            episode_reward += reward
            state = next_state
            
            if render:
                env.render()
                time.sleep(0.01)  # Add small delay to make visualization smoother
        
        eval_rewards.append(episode_reward)
        print(f"Episode {ep + 1}/{num_episodes}: Reward = {episode_reward:.2f}")
    
    env.close()
    
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

    checkpoint_dir = "checkpoints"  # Directory containing your saved checkpoints
    
    # Option 1: Evaluate a single checkpoint
    try:
        latest_checkpoint = max(
            [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Evaluating latest checkpoint: {latest_checkpoint}")
        evaluate_model(checkpoint_path, num_episodes=5, render=True)
    except Exception as e:
        print(f"Error loading checkpoints: {e}")
        print("Please ensure the checkpoint directory exists and contains .pt files")
    
    # Option 2: Evaluate multiple checkpoints to see progression
    # print("\nEvaluating multiple checkpoints...")
    # results = evaluate_multiple_checkpoints(
    #     checkpoint_dir=checkpoint_dir,
    #     episode_interval=100,  # Evaluate every 100th episode
    #     num_episodes=3
    # )

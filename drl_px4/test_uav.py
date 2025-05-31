import rclpy
import numpy as np
import torch  # Added for device checking
from drl_px4.uav_env import UAVEnv
from drl_px4.sac_per import SAC_PER
import time
from datetime import datetime

def get_timestamp():
    """Return a timestamp string in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    # Initialize ROS2
    rclpy.init()
    env = None  # Initialize env to None for proper cleanup in finally block
    try:
        # Initialize environment
        env = UAVEnv()
        print("Test environment created")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Determine device for model loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Load the trained model
        model_path = "sac_per_uav_interrupted_20250502_162628.zip"  # Update as needed
        try:
            model = SAC_PER.load(model_path, env=env, device=device)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            return

        # Test parameters
        num_episodes = 50  # Increased for better statistical reliability
        max_steps_per_episode = 200
        success_threshold = 0.5  # XY distance to consider reaching the target
        landing_threshold = 0.1  # Z distance to consider landed
        successes = 0
        total_rewards = []
        distances_to_goal = []
        landing_successes = 0
        landed_episodes = []  # Track landing success per episode
        steps_per_episode = []
        termination_reasons = []

        # Run test episodes
        for episode in range(num_episodes):
            print(f"\nStarting test episode {episode + 1}/{num_episodes}")
            obs, info = env.reset()  # Capture info from reset
            episode_reward = 0.0
            steps = 0
            done = False
            truncated = False
            episode_termination_reason = "unknown"

            while not (done or truncated):
                # Predict action using the trained model
                action, _ = model.predict(obs, deterministic=True)
                print(f"Step {steps + 1}: Action = {action}")

                # Step the environment
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1

                # Log current state
                xy_dist = np.sqrt((env.node.pos[0] - env.goal[0])**2 + (env.node.pos[1] - env.goal[1])**2)
                z_dist = abs(env.node.pos[2] - env.goal[2])
                print(f"Position: {env.node.pos[:3]}, XY dist: {xy_dist:.2f}, Z dist: {z_dist:.2f}, Reward: {reward:.2f}")

                # Check for landing and success
                if done and "reason" in info:
                    episode_termination_reason = info["reason"]
                    if episode_termination_reason == "landed":
                        landing_successes += 1
                        landed_episodes.append(True)
                        if xy_dist < success_threshold:
                            successes += 1
                        break
                    else:
                        landed_episodes.append(False)

                if steps >= max_steps_per_episode:
                    truncated = True
                    episode_termination_reason = "max_steps_reached"
                    landed_episodes.append(False)
                    print("Episode truncated: Max steps reached")

            # Log episode results
            final_xy_dist = np.sqrt((env.node.pos[0] - env.goal[0])**2 + (env.node.pos[1] - env.goal[1])**2)
            final_z_dist = abs(env.node.pos[2] - env.goal[2])
            distances_to_goal.append(final_xy_dist)
            total_rewards.append(episode_reward)
            steps_per_episode.append(steps)
            termination_reasons.append(episode_termination_reason)
            print(f"Episode {episode + 1} finished: Reward = {episode_reward:.2f}, "
                  f"Final XY dist = {final_xy_dist:.2f}, Final Z dist = {final_z_dist:.2f}, "
                  f"Steps = {steps}, Landed = {env.has_landed}, Termination reason = {episode_termination_reason}")

        # Compute and log statistics
        success_rate = (successes / num_episodes * 100) if num_episodes > 0 else 0.0
        landing_success_rate = (landing_successes / num_episodes * 100) if num_episodes > 0 else 0.0
        avg_reward = np.mean(total_rewards) if total_rewards else float('nan')
        reward_std = np.std(total_rewards) if total_rewards else float('nan')
        avg_distance = np.mean(distances_to_goal) if distances_to_goal else float('nan')
        avg_steps = np.mean(steps_per_episode) if steps_per_episode else float('nan')
        print("\nTest Summary:")
        print(f"Success rate (reached target within {success_threshold}m in XY): {success_rate:.2f}%")
        print(f"Landing success rate: {landing_success_rate:.2f}%")
        print(f"Average episode reward: {avg_reward:.2f} (std: {reward_std:.2f})")
        print(f"Average final XY distance to goal: {avg_distance:.2f}")
        print(f"Average steps per episode: {avg_steps:.2f}")
        print(f"Total episodes: {num_episodes}")

        # Save test results to a file
        timestamp = get_timestamp()
        results_file = f"test_results_{timestamp}.txt"
        try:
            with open(results_file, "w") as f:
                f.write(f"Test Results - Timestamp: {timestamp}\n")
                f.write(f"Model: {model_path}\n")
                f.write(f"Total Episodes: {num_episodes}\n")
                f.write(f"Success Rate (XY dist < {success_threshold}m): {success_rate:.2f}%\n")
                f.write(f"Landing Success Rate: {landing_success_rate:.2f}%\n")
                f.write(f"Average Episode Reward: {avg_reward:.2f} (std: {reward_std:.2f})\n")
                f.write(f"Average Final XY Distance to Goal: {avg_distance:.2f}\n")
                f.write(f"Average Steps per Episode: {avg_steps:.2f}\n")
                f.write("\nEpisode Details:\n")
                for i in range(num_episodes):
                    f.write(f"Episode {i+1}: Reward = {total_rewards[i]:.2f}, "
                            f"Final XY Dist = {distances_to_goal[i]:.2f}, "
                            f"Steps = {steps_per_episode[i]}, "
                            f"Landed = {landed_episodes[i]}, "
                            f"Termination Reason = {termination_reasons[i]}\n")
            print(f"Saved test results to {results_file}")
        except Exception as e:
            print(f"Failed to save test results to {results_file}: {e}")

    finally:
        # Cleanup
        if env is not None:
            env.close()  # Use close() instead of destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
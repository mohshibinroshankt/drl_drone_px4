import rclpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from drl_px4.drone_env import DroneEnv
from drl_px4.per_sac import SAC_PER
from datetime import datetime

def get_timestamp():
    """Return a timestamp string in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%SS")

def plot_trajectories(trajectories, goal, timestamp):
    # 2D Trajectory Plot
    plt.figure(figsize=(8, 6))
    for i, traj in enumerate(trajectories):
        plt.plot(traj[:, 0], traj[:, 1], label=f'Episode {i+1}')
    plt.scatter([goal[0]], [goal[1]], color='red', marker='*', s=200, label='Goal (7, 0)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Drone 2D Trajectories')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'trajectory_2d_{timestamp}.png')
    plt.close()

    # 3D Trajectory Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i, traj in enumerate(trajectories):
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f'Episode {i+1}')
    ax.scatter([goal[0]], [goal[1]], [goal[2]], color='red', marker='*', s=200, label='Goal (7, 0, 0)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Drone 3D Trajectories')
    ax.legend()
    plt.savefig(f'trajectory_3d_{timestamp}.png')
    plt.close()

    # Distance to Goal Plot
    plt.figure(figsize=(8, 6))
    for i, dists in enumerate([np.sqrt(np.sum((traj - goal)**2, axis=1)) for traj in trajectories]):
        plt.plot(dists, label=f'Episode {i+1}')
    plt.xlabel('Step')
    plt.ylabel('Distance to Goal (m)')
    plt.title('Distance to Goal Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'distance_to_goal_{timestamp}.png')
    plt.close()

    # Minimum LIDAR Distance Plot
    plt.figure(figsize=(8, 6))
    for i, min_dists in enumerate([traj[:, 3] for traj in trajectories]):  # Assuming min LIDAR stored
        plt.plot(min_dists, label=f'Episode {i+1}')
    plt.xlabel('Step')
    plt.ylabel('Minimum LIDAR Distance (m)')
    plt.title('Minimum LIDAR Distance Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'min_lidar_distance_{timestamp}.png')
    plt.close()

def main():
    # Initialize ROS2
    rclpy.init()
    try:
        # Initialize environment
        env = DroneEnv()
        print("Test environment created")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Load the trained model
        # Replace with the path to your saved model
        model_path = "sac_per_drone_error_20250423_173035S.zip"  # Update to your latest model
        try:
            model = SAC_PER.load(model_path, env=env, device="cuda")
            print(f"Loaded model from {model_path}")
            # Verify observation space compatibility
            if model.observation_space.shape != (44,):
                raise ValueError(f"Model observation space {model.observation_space.shape} does not match environment (44,)")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            return

        # Test parameters
        num_episodes = 10
        max_steps_per_episode = 200
        success_threshold = 0.5  # XY distance to consider reaching the target
        landing_threshold = 0.1  # Z distance to consider landed
        successes = 0
        landing_successes = 0
        total_rewards = []
        distances_to_goal = []
        trajectories = []

        # Run test episodes
        for episode in range(num_episodes):
            print(f"\nStarting test episode {episode + 1}/{num_episodes}")
            obs, _ = env.reset()
            if obs.shape != (44,):
                raise ValueError(f"Initial observation shape {obs.shape} does not match expected (44,)")
            episode_reward = 0.0
            steps = 0
            done = False
            truncated = False
            trajectory = []
            min_lidar_dists = []

            while not (done or truncated):
                # Predict action using the trained model (deterministic for optimal path)
                if obs.shape != (44,):
                    raise ValueError(f"Observation shape {obs.shape} does not match expected (44,) at episode {episode + 1}, step {steps + 1}")
                action, _ = model.predict(obs, deterministic=True)
                print(f"Step {steps + 1}: Action = {action}")

                # Step the environment
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1

                # Log current state
                xy_dist = np.sqrt((env.pos[0] - env.goal[0])**2 + (env.pos[1] - env.goal[1])**2)
                z_dist = abs(env.pos[2] - env.goal[2])
                min_lidar = np.min(env.lidar_data)
                print(f"Position: {env.pos[:3]}, XY dist: {xy_dist:.2f}, Z dist: {z_dist:.2f}, "
                      f"Min LIDAR: {min_lidar:.2f}, Reward: {reward:.2f}")

                # Store trajectory data (x, y, z, min_lidar)
                trajectory.append([env.pos[0], env.pos[1], env.pos[2], min_lidar])
                min_lidar_dists.append(min_lidar)

                # Check for landing
                if env.has_landed:
                    print("Drone has landed and disarmed")
                    landing_successes += 1
                    if xy_dist < success_threshold:
                        successes += 1
                    break

                if steps >= max_steps_per_episode:
                    truncated = True
                    print("Episode truncated: Max steps reached")

            # Log episode results
            final_xy_dist = np.sqrt((env.pos[0] - env.goal[0])**2 + (env.pos[1] - env.goal[1])**2)
            final_z_dist = abs(env.pos[2] - env.goal[2])
            distances_to_goal.append(final_xy_dist)
            total_rewards.append(episode_reward)
            trajectories.append(np.array(trajectory))
            print(f"Episode {episode + 1} finished: Reward = {episode_reward:.2f}, "
                  f"Final XY dist = {final_xy_dist:.2f}, Final Z dist = {final_z_dist:.2f}, "
                  f"Steps = {steps}, Landed = {env.has_landed}")

        # Compute and log statistics
        success_rate = successes / num_episodes * 100
        landing_success_rate = landing_successes / num_episodes * 100
        avg_reward = np.mean(total_rewards) if total_rewards else 0.0
        avg_distance = np.mean(distances_to_goal) if distances_to_goal else 0.0
        print("\nTest Summary:")
        print(f"Success rate (reached target within {success_threshold}m in XY): {success_rate:.2f}%")
        print(f"Landing success rate: {landing_success_rate:.2f}%")
        print(f"Average episode reward: {avg_reward:.2f}")
        print(f"Average final XY distance to goal: {avg_distance:.2f}")
        print(f"Total episodes: {num_episodes}")

        # Plot trajectories and metrics
        timestamp = get_timestamp()
        plot_trajectories(trajectories, env.goal, timestamp)
        print(f"Saved trajectory plots: trajectory_2d_{timestamp}.png, trajectory_3d_{timestamp}.png, "
              f"distance_to_goal_{timestamp}.png, min_lidar_distance_{timestamp}.png")

        # Save test results to a file
        results_file = f"test_results_{timestamp}.txt"
        with open(results_file, "w") as f:
            f.write(f"Test Results - Timestamp: {timestamp}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Total Episodes: {num_episodes}\n")
            f.write(f"Success Rate (XY dist < {success_threshold}m): {success_rate:.2f}%\n")
            f.write(f"Landing Success Rate: {landing_success_rate:.2f}%\n")
            f.write(f"Average Episode Reward: {avg_reward:.2f}\n")
            f.write(f"Average Final XY Distance to Goal: {avg_distance:.2f}\n")
            f.write("\nEpisode Details:\n")
            for i in range(num_episodes):
                f.write(f"Episode {i+1}: Reward = {total_rewards[i]:.2f}, "
                        f"Final XY Dist = {distances_to_goal[i]:.2f}, "
                        f"Landed = {i < landing_successes}\n")
        print(f"Saved test results to {results_file}")

    finally:
        if 'env' in locals():
            env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()